"""CPU checks for Control-Time Spiking; no GPU, no downloads, no trained weights.

Fast unit tests always run. Full-model tests need the cached bert-base-uncased
(or CTS_TEST_BERT) and build a random SDT-V3 in a temporary directory. The
2-step training smoke needs local LIBERO LeRobot data in CTS_TEST_LIBERO_DATA.
    python -m pytest tests/cts -q
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from turbovla.cts.data import window_steps
from turbovla.cts.diagnostics import constant_input_resolution
from turbovla.cts.neuron import Carried, CTSNeuron, DendriticMemory
from turbovla.cts.state import carry_numel, detach_carry, reset_samples, wrap_carried
from turbovla.models.components.spike_fusion import FunctionalPLIF
from turbovla.models.components.wta_operators import BipolarPLIF
from turbovla.models.configuration import CTSConfig, TurboVLAConfig

from tests.cts import assets

CODE = assets.CODE
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


class NeuronTest(unittest.TestCase):
    def test_plif_hard_equals_base_neurons(self):
        torch.manual_seed(0)
        for base, bipolar in ((BipolarPLIF(), True), (FunctionalPLIF(), False)):
            with torch.no_grad():
                base.decay_logit.fill_(0.3)
                base.log_threshold.fill_(math.log(0.4))
            node = CTSNeuron(16, bipolar=bipolar, charge="plif", reset="hard", decay_logit=0.3,
                             log_threshold=math.log(0.4))
            u_base = u_node = None
            valid = torch.rand(4, 7, 1) > 0.2
            for _ in range(6):
                current = torch.randn(4, 7, 16)
                s_base, u_base = base(current, u_base, valid=valid)
                s_node, u_node = node(current, u_node, valid=valid)
                self.assertTrue(torch.equal(s_base, s_node))
                self.assertTrue(torch.equal(u_base, u_node))

    def test_integrator_resolution_grows_with_timesteps(self):
        def rmse(form, steps):
            charge, reset = form.split("_")
            node = CTSNeuron(1, bipolar=True, charge=charge, reset=reset, residual_clamp=True)
            return constant_input_resolution(node, steps, grid=801, span=1.0)["rmse"]
        # The base EMA neuron saturates: 4x the steps buys little. IF-soft keeps improving.
        self.assertGreater(rmse("plif_hard", 16) / rmse("plif_hard", 4), 0.8)
        self.assertLess(rmse("if_soft", 16) / rmse("if_soft", 4), 0.5)
        self.assertLess(rmse("if_soft", 4), rmse("plif_hard", 4))

    def test_carry_retention_and_guard(self):
        node = CTSNeuron(3, bipolar=True, carry=True, retention_init=0.5)
        membrane = torch.full((2, 3), 0.2)
        current = torch.zeros(2, 3)
        _, carried = node(current, Carried(membrane))
        self.assertTrue(torch.allclose(carried, membrane * 0.5))
        _, plain = node(current, membrane)
        self.assertTrue(torch.allclose(plain, membrane))
        with self.assertRaises(RuntimeError):
            CTSNeuron(3, bipolar=True)(current, Carried(membrane))
        carried.sum().backward()
        self.assertIsNotNone(node.retention_logit.grad)

    def test_soft_reset_keeps_residual_within_threshold(self):
        node = CTSNeuron(1, bipolar=True, charge="if", reset="soft", decay_logit=10.0)
        membrane = None
        for value in (0.7, 0.7, 0.7):
            spike, membrane = node(torch.tensor([[value]]), membrane)
            self.assertLessEqual(float(membrane.abs().max()), 0.5 + 1e-6)

    def test_memory_restart_matches_episode_start(self):
        memory = DendriticMemory(4, gain_init=1.0)
        write = torch.randn(2, 5, 4)
        _, state = memory(write, None)
        self.assertTrue(torch.equal(state["trace"], write))
        restarted = {"trace": torch.randn(2, 5, 4), "started": torch.tensor([False, True])}
        _, state = memory(write, restarted)
        self.assertTrue(torch.equal(state["trace"][0], write[0]))
        self.assertFalse(torch.equal(state["trace"][1], write[1]))


class StateTest(unittest.TestCase):
    def carry(self):
        return {"instruction": (torch.ones(2, 3), torch.ones(2, 3, dtype=torch.bool), None),
                "query_index": 4,
                "projection": {"vision": {"input": torch.ones(2, 5)}, "text": {"input": torch.ones(2, 3)}},
                "fusion": [{"cross": {"attn": {"q": torch.ones(2, 2)},
                                      "memory": {"trace": torch.ones(2, 4),
                                                 "started": torch.ones(2, dtype=torch.bool)}},
                            "text": {"self": torch.ones(2, 3)}}]}

    def test_reset_zeroes_nested_text_state_but_keeps_instruction(self):
        reset = reset_samples(self.carry(), torch.tensor([False, True]))
        self.assertEqual(float(reset["projection"]["text"]["input"][0].abs().sum()), 0.0)
        self.assertEqual(float(reset["fusion"][0]["text"]["self"][0].abs().sum()), 0.0)
        self.assertFalse(bool(reset["fusion"][0]["cross"]["memory"]["started"][0]))
        self.assertTrue(bool(reset["fusion"][0]["cross"]["memory"]["started"][1]))
        self.assertTrue(torch.equal(reset["instruction"][0], torch.ones(2, 3)))
        self.assertEqual(reset["query_index"], 4)

    def test_wrap_copies_and_marks(self):
        carry = self.carry()
        wrapped = wrap_carried(carry["projection"])
        self.assertIsInstance(wrapped["vision"]["input"], Carried)
        wrapped["vision"]["input"] = None
        self.assertIsNotNone(carry["projection"]["vision"]["input"])
        self.assertEqual(wrap_carried(None), {})

    def test_detach_and_count(self):
        carry = self.carry()
        carry["projection"]["vision"]["input"].requires_grad_(True)
        self.assertFalse(detach_carry(carry)["projection"]["vision"]["input"].requires_grad)
        self.assertEqual(carry_numel(carry), 10 + 6 + 4 + 8 + 2 + 6)


class DataTest(unittest.TestCase):
    def test_window_steps(self):
        steps, present, in_episode = window_steps(start=5, length=40, burn_in=2, window=2, stride=12)
        self.assertEqual(steps, [0, 0, 5, 17])
        self.assertEqual(present, [False, False, True, True])
        self.assertEqual(in_episode, [True, True, True, True])
        steps, present, in_episode = window_steps(start=35, length=40, burn_in=1, window=2, stride=12)
        self.assertEqual(steps, [23, 35, 39])
        self.assertEqual(in_episode, [True, True, False])


class ConfigTest(unittest.TestCase):
    def test_roundtrip_and_old_checkpoints(self):
        config = TurboVLAConfig()
        payload = config.to_dict()
        self.assertEqual(TurboVLAConfig.from_mapping(payload).cts, CTSConfig())
        payload.pop("cts")
        self.assertFalse(TurboVLAConfig.from_mapping(payload).cts.enabled)


class RecipeTest(unittest.TestCase):
    def test_every_recipe_resolves_to_the_budget(self):
        sys.path.insert(0, str(CODE))
        from experiments.cts import train as launcher
        from turbovla.training import train_mixed

        for name, recipe in launcher.RECIPES["recipes"].items():
            options = SimpleNamespace()
            options.effective_world_size, options.windows_per_gpu, options.seed = 8, None, 42
            options.init_checkpoint = "/init.pth" if recipe.get("requires_init_checkpoint") else None
            options.resume_mode, options.extra, options.num_workers = "none", [], 4
            options.run_dir, options.vision_checkpoint, options.text_projection_init = "/runs/x", "/v.pth", "/t.pth"
            options.lr_schedule_steps = options.max_steps = 40000
            options.warmup_steps, options.log_freq, options.save_steps = 0, 20, 1000
            options.extra_save_steps = [20]
            options.vision_gradient_checkpointing = options.projection_gradient_checkpointing = False
            options.vision_checkpoint_scope = "all"
            entry, arguments, layout = launcher.resolve(name, options)
            self.assertEqual(layout["loss_queries_per_step"], 256, name)
            with mock.patch.object(sys, "argv", ["train", *arguments]):
                if entry == "cts":
                    from turbovla.cts import train as cts_train
                    args = cts_train.parse_args()
                    self.assertTrue(args.cts_enabled)
                    settings = {**launcher.RECIPES["cts_defaults"], **recipe.get("cts", {})}
                    self.assertEqual(args.cts_window, settings["window"], name)
                    self.assertEqual(args.cts_neuron, settings["neuron"], name)
                    self.assertEqual(args.cts_memory, settings["memory"], name)
                    self.assertEqual(args.cts_carry_head, settings["carry_head"], name)
                    self.assertEqual(args.batch_size * args.grad_accum_steps * 8 * args.cts_window, 256, name)
                else:
                    args = train_mixed.parse_args_with_mixed_suite_stats()
                    self.assertEqual(args.batch_size * args.grad_accum_steps * 8, 256, name)
            expected_t = recipe.get("base_overrides", {}).get("--cross_timesteps", "4")
            self.assertEqual(args.cross_timesteps, int(expected_t), name)
            self.assertEqual(args.max_steps, recipe.get("max_steps", 40000), name)
            if recipe["entry"] == "ann":
                self.assertEqual((args.fusion_backbone_type, args.vision_encoder_type, args.text_encoder_type,
                                  args.action_head_type), ("hybrid", "dinov3", "bert", "ann"), name)
                self.assertIsNone(args.text_projection_init, name)
            else:
                self.assertEqual(args.fusion_backbone_type, "spike_text_only", name)
                self.assertEqual(args.action_head_type, "full_spike_state8_shiftmax_continuous_inputs", name)
            self.assertTrue(args.libero_light_augmentation, name)
            self.assertEqual(args.joint_scope_audit, not recipe.get("requires_init_checkpoint", False), name)


class AnalysisTest(unittest.TestCase):
    def test_failure_taxonomy(self):
        from experiments.cts.diagnose import classify
        base = {"success": False, "final_goal": None}
        self.assertEqual(classify({**base, "queries": [{"goal": [False, False]}] * 3}), "none")
        self.assertEqual(classify({**base, "queries": [{"goal": [False, False]}, {"goal": [True, False]},
                                                       {"goal": [True, False]}]}), "stalled")
        self.assertEqual(classify({**base, "queries": [{"goal": [True, False]}, {"goal": [False, False]}]}),
                         "regressed")
        self.assertEqual(classify({"success": True, "queries": []}), "success")

    def test_operation_counter(self):
        from experiments.cts.efficiency import OpCounter, spike_level
        self.assertEqual(spike_level(torch.tensor([0.0, 1.0, -1.0, 0.0]), 8), 0.5)
        self.assertEqual(spike_level(torch.tensor([0.0, 0.25, 0.5, 0.0]), 8), 0.75)  # integer spikes 1, 2
        self.assertIsNone(spike_level(torch.randn(16), 8))
        model = torch.nn.Module()
        model.action_head, model.vision_encoder = torch.nn.Linear(4, 3), torch.nn.Linear(4, 3)
        counter = OpCounter(model, 8)
        with torch.no_grad(), counter:
            model.action_head(torch.tensor([[1.0, 0.0, 0.0, 1.0]]))  # 12 MACs at spike rate 0.5
            model.vision_encoder(torch.randn(1, 4))
        counter.close()
        self.assertEqual((counter.parts["head"]["ac"], counter.parts["head"]["mac"]), (6.0, 0))
        self.assertEqual((counter.parts["vision_encoder"]["ac"], counter.parts["vision_encoder"]["mac"]), (0.0, 12))

    def test_statistics(self):
        from experiments.cts.aggregate import mcnemar_exact, wilson
        low, high = wilson(452, 500)
        self.assertLess(low, 0.904)
        self.assertGreater(high, 0.904)
        self.assertAlmostEqual(mcnemar_exact(0, 0), 1.0)
        self.assertLess(mcnemar_exact(2, 20), 0.001)


@unittest.skipIf(assets.bert_path() is None, "bert-base-uncased is not cached")
class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sdt = assets.random_sdtv3(Path(cls.tmp.name))
        cls.bert = assets.bert_path()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def build(self, **cts):
        from turbovla.models.turbovla import build_turbovla
        torch.manual_seed(0)
        return build_turbovla(assets.smoke_args(self.sdt, self.bert, **cts))

    def test_disabled_cts_forms_reproduce_base(self):
        base = self.build().eval()
        cts = self.build(enabled=True, neuron="plif_hard", carry_projection=False, carry_fusion=False,
                         carry_head=False, memory=False).eval()
        cts.load_state_dict(base.state_dict(), strict=True)
        base.train()
        cts.train()
        batch = assets.smoke_batch()
        with torch.no_grad():
            torch.manual_seed(1)
            expected = base(*batch)
            torch.manual_seed(1)
            actual = cts(*batch)
        self.assertEqual(float((expected - actual).abs().max()), 0.0)

    def test_window_gradients_and_cold_restart(self):
        from turbovla.cts.apply import cts_parameter_names
        queries = [assets.smoke_batch(seed=j) for j in range(3)]
        present = torch.tensor([[False, True, True], [True, True, True]])
        for extra in ({}, {"memory_vision": True}):
            with self.subTest(**extra):
                model = self.build(enabled=True, **extra).train()
                actions = model(None, None, None, cts_window=dict(queries=queries, present=present, burn_in=1,
                                                                  state_dropout=0.5))
                self.assertEqual(tuple(actions.shape), (2, 2, 12, 7))
                actions.square().mean().backward()
                parameters = dict(model.named_parameters())
                names = cts_parameter_names(model)
                self.assertGreater(len(names), 0)
                self.assertEqual(any(".cts_memory_v." in n for n in names), bool(extra))
                self.assertEqual([n for n in names if parameters[n].grad is None], [])
                model.eval()
                with torch.no_grad():
                    warm = model(None, None, None, cts_window=dict(queries=queries, present=present, burn_in=1))
                    cold, _ = model.forward_cts(*queries[1], None)
                # Sample 0 is absent before its episode start: its first loss query is an exact cold start.
                self.assertEqual(float((warm[0, 0] - cold[0]).abs().max()), 0.0)
                self.assertGreater(float((warm[1, 0] - cold[1]).abs().max()), 0.0)


@unittest.skipIf(assets.bert_path() is None or assets.libero_root() is None,
                 "needs cached BERT and CTS_TEST_LIBERO_DATA=<dir with *_lerobot suites>")
class TrainingSmokeTest(unittest.TestCase):
    def test_two_steps_on_cpu(self):
        data = assets.libero_root()
        suite = sorted(data.glob("*_no_noops_1.0.0_lerobot"))[0]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sdt = assets.random_sdtv3(tmp)
            bert = assets.bert_path()
            command = [
                sys.executable, "-m", "turbovla.cts.train",
                "--dataset_dir", str(suite), "--dataset_dirs", str(suite), "--dataset_format", "lerobot",
                "--lerobot_gripper_convention", "binary_open", "--normalize_binary_gripper", "false",
                "--stats_path", str(CODE / "experiments/libero/configs/libero_all4_stats.json"),
                "--stats_key", "libero_all4_no_noops",
                "--text_layout_path", str(CODE / "experiments/libero/configs/online_text_layout.json"),
                "--dinov3_path", str(assets.processor_dir(tmp)), "--bert_path", bert, "--no_allow_hf_download",
                "--text_encoder_type", "bert", "--text_model_path", bert, "--text_timesteps", "1",
                "--text_mask_version", "legacy", "--text_sentence_mask_version", "legacy",
                "--text_head_mask_version", "corrected",
                "--vision_encoder_type", "sdtv3_19m", "--vision_model_path", str(sdt),
                "--vision_pretrained_checkpoint", str(sdt), "--vision_image_size", "224",
                "--fusion_backbone_type", "spike_text_only", "--text_only_spike_projections",
                "--cross_attention_type", "spike_sdsa", "--cross_timesteps", "2",
                "--action_head_type", "full_spike_state8_shiftmax_continuous_inputs", "--continuous_state",
                "--spike_state_encoding", "perdim_ann", "--num_state_tokens", "8", "--spike_state_population", "32",
                "--checkpoint_dir", str(tmp / "checkpoints"), "--checkpoint_prefix", "smoke",
                "--batch_size", "2", "--grad_accum_steps", "1", "--max_steps", "2", "--warmup_steps", "1",
                "--lr_schedule_steps", "2", "--precision", "fp32", "--dinov3_precision", "fp32",
                "--float32_matmul_precision", "highest", "--no_freeze_backbones", "--freeze_text_encoder",
                "--num_workers", "0", "--log_freq", "1", "--shuffle_buffer", "4", "--save_final",
                "--approve_spike_fusion_training", "--libero_light_augmentation",
                "--pretrained_init_ckpt", "unused", "--no_require_feature_enhancer_preload",
                "--no_load_text_projection_from_init",
                "--cts_smoke_test", "--cts_window", "2", "--cts_burn_in", "1", "--cts_lr_scale", "10",
            ]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"}
            result = subprocess.run(command, cwd=CODE, env=env, capture_output=True, text=True, timeout=3600)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"event"')]
            self.assertEqual(events[-1]["event"], "done")
            self.assertTrue((tmp / "checkpoints" / "smoke_2.pth").is_file())
            checkpoint = torch.load(tmp / "checkpoints" / "smoke_2.pth", map_location="cpu", weights_only=False)
            self.assertIn("ema_model_state_dict", checkpoint)
            self.assertTrue(checkpoint["model_config"]["cts"]["enabled"])
            self.assertEqual(checkpoint["training_scope"], "cts_smoke_test")


if __name__ == "__main__":
    unittest.main()
