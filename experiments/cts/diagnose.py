"""CTS diagnostics (E0a/E0b/E0c), for base and CTS checkpoints alike.

  neurons   E0a  learned g/theta, firing rate, silent fraction and realized
                 levels per layer on training windows, plus the analytic
                 constant-input probe (levels, dead zone, decode RMSE).
  flips     E0b  spike flips caused by a small perturbation (pixel noise,
                 TF32, int-k weights, bf16 carry), per layer and over queries,
                 and the resulting action change.
  failures  E0c  failure taxonomy from rollout traces (--trace_dir):
                 none / stalled / regressed.

Examples
  python experiments/cts/diagnose.py neurons --checkpoint ema.pth --out neurons.json
  python experiments/cts/diagnose.py flips --checkpoint ema.pth --mode tf32 --out flips.json
  python experiments/cts/diagnose.py failures --trace_dir eval/E1/traces --out failures.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

CODE = Path(__file__).resolve().parents[2]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


# ------------------------------------------------------------------ loading

def load_model(path: str, weights: str, device):
    import torch

    from turbovla.models.configuration import TurboVLAConfig
    from turbovla.models.turbovla import build_turbovla

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if weights == "auto":
        weights = "ema" if checkpoint.get("export_kind") == "ema_evaluation_only" else "raw"
    key = "ema_model_state_dict" if weights == "ema" else "model_state_dict"
    config = TurboVLAConfig.from_mapping(checkpoint["model_config"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    model = build_turbovla(config)
    model.load_state_dict(checkpoint[key], strict=True)
    model.to(device).float().eval().requires_grad_(False)
    return model, SimpleNamespace(**checkpoint["args"]), weights


def windows(args, window: int, batch_size: int, workers: int, seed: int):
    from torch.utils.data import DataLoader

    from turbovla.cts.data import LiberoQueryWindowDataset, window_collate

    dataset = LiberoQueryWindowDataset(
        args.dataset_dir, dataset_dirs=args.dataset_dirs, stats_path=args.stats_path,
        stats_key=args.stats_key, normalize_binary_gripper=args.normalize_binary_gripper,
        gripper_convention=args.lerobot_gripper_convention, LOCAL_DINOV3_PATH=args.dinov3_path,
        chunk_size=args.chunk_size, rank=0, world_size=1, seed=seed, local_files_only=True,
        expected_image_size=args.expected_image_size, vision_encoder_type=args.vision_encoder_type,
        vision_image_size=args.vision_image_size, shuffle_buffer=args.shuffle_buffer,
        step_mix_buffer_size=8, enable_dinov3_teacher=False, light_augmentation=False,
        window=window, burn_in=0, stride=getattr(args, "cts_stride", 12),
    )
    return iter(DataLoader(dataset, batch_size=batch_size, num_workers=workers, collate_fn=window_collate))


def run_queries(model, batch, device, carry_hook=None):
    """Run the queries of one window in order; stateless models see each query cold."""
    per_query, present, _, _ = batch
    carry = None
    for index, (samples, instructions, states, _, _) in enumerate(per_query):
        pixels = {"pixel_values": samples["pixel_values"].to(device)}
        states = states.to(device)
        if model.config.cts.enabled:
            prediction, carry = model.forward_cts(instructions, pixels, states, carry)
            if carry_hook is not None:
                carry = carry_hook(carry)
            if not bool(present[:, index].all()):
                from turbovla.cts.state import reset_samples
                carry = reset_samples(carry, present[:, index].to(device))
        else:
            prediction = model(instructions, pixels, states)
        yield index, prediction


# --------------------------------------------------------------- commands

def cmd_neurons(options):
    import torch

    from turbovla.cts.diagnostics import NeuronRecorder, constant_input_resolution, neuron_modules

    device = torch.device(options.device)
    model, args, weights = load_model(options.checkpoint, options.weights, device)
    recorder = NeuronRecorder(model)
    iterator = windows(args, options.window, options.batch_size, options.num_workers, options.seed)
    with torch.no_grad():
        for _ in range(options.batches):
            for _index, _prediction in run_queries(model, next(iterator), device):
                recorder.end_query()
    recorder.close()
    report = recorder.report()
    steps = model.config.interaction.cross_timesteps
    probe = {}
    for name, module in neuron_modules(model):
        cpu = copy.deepcopy(module).cpu().float()
        probe[name] = {"one_query": constant_input_resolution(cpu, steps, 1, grid=options.probe_grid)}
        if model.config.cts.enabled:
            probe[name]["over_queries"] = constant_input_resolution(
                cpu, steps, options.window, grid=options.probe_grid)
    report.update(checkpoint=options.checkpoint, weights=weights, timesteps=steps,
                  cts=model.config.cts.__dict__, probe=probe)
    write(options.out, report)
    for group, values in report["groups"].items():
        print(f"{group:18s} layers={values['layers']:3d} rate={values['rate']:.3f} "
              f"silent={values['silent']:.3f} levels={values['levels']:.1f}")


def cmd_flips(options):
    import torch

    from turbovla.cts.diagnostics import SpikeTape, flip_profile
    from turbovla.cts.robustness import fake_quantize_weights
    from turbovla.cts.state import map_tensors

    device = torch.device(options.device)
    model, args, weights = load_model(options.checkpoint, options.weights, device)
    batch = next(windows(args, options.window, options.batch_size, options.num_workers, options.seed))
    tape = SpikeTape(model)

    def record(perturbed_model, carry_hook=None):
        per_query = []
        with torch.no_grad():
            for _, prediction in run_queries(perturbed_model, batch, device, carry_hook):
                per_query.append((tape.take(), prediction.float().cpu()))
        return per_query

    clean = record(model)
    carry_hook = None
    if options.mode == "pixel":
        generator = torch.Generator().manual_seed(options.seed)
        per_query = batch[0]
        noisy = []
        for samples, instructions, states, targets, masks in per_query:
            pixels = samples["pixel_values"]
            noise = torch.randn(pixels.shape, generator=generator) * options.epsilon
            noisy.append(({**samples, "pixel_values": pixels + noise}, instructions, states, targets, masks))
        batch = (noisy, *batch[1:])
        perturbed = record(model)
    elif options.mode == "tf32":
        if device.type != "cuda":
            raise ValueError("TF32 needs a CUDA device")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        perturbed = record(model)
    elif options.mode == "weights":
        quantized = copy.deepcopy(model)
        tape.close()
        tape = SpikeTape(quantized)
        fake_quantize_weights(quantized, options.bits)
        perturbed = record(quantized)
    elif options.mode == "bf16_carry":
        if not model.config.cts.enabled:
            raise ValueError("bf16_carry needs a CTS checkpoint")

        def carry_hook(carry):
            return {key: value if key in {"instruction", "query_index"} else map_tensors(
                value, lambda t: t.to(torch.bfloat16).to(t.dtype) if t.is_floating_point() else t)
                for key, value in carry.items()}
        perturbed = record(model, carry_hook)
    else:
        raise ValueError(options.mode)
    tape.close()

    queries = []
    for index, ((tape_a, actions_a), (tape_b, actions_b)) in enumerate(zip(clean, perturbed)):
        profile = flip_profile(tape_a, tape_b)
        flips = sum(v["flips"] for v in profile.values())
        entries = sum(v["entries"] for v in profile.values())
        groups = {}
        for layer in profile.values():
            group = groups.setdefault(layer["group"], {"flips": 0, "entries": 0})
            group["flips"] += layer["flips"]
            group["entries"] += layer["entries"]
        queries.append({
            "query": index, "flip_rate": flips / max(entries, 1),
            "groups": {g: v["flips"] / max(v["entries"], 1) for g, v in groups.items()},
            "action_l1": float((actions_a - actions_b).abs().mean()),
            "action_linf": float((actions_a - actions_b).abs().max()),
            "layers": profile if options.per_layer else None,
        })
    report = {"checkpoint": options.checkpoint, "weights": weights, "mode": options.mode,
              "epsilon": options.epsilon, "bits": options.bits, "cts": model.config.cts.__dict__,
              "queries": queries}
    write(options.out, report)
    for query in queries:
        print(f"q{query['query']}: flip_rate={query['flip_rate']:.4f} action_l1={query['action_l1']:.4f} "
              + " ".join(f"{g}={v:.4f}" for g, v in query["groups"].items()))


def classify(trace: dict) -> str:
    """none: no goal predicate ever held; regressed: a predicate held, then was undone
    and the episode failed; stalled: progress kept but incomplete."""
    if trace["success"]:
        return "success"
    history = [query["goal"] for query in trace["queries"] if query.get("goal") is not None]
    if trace.get("final_goal") is not None:
        history.append(trace["final_goal"])
    if not history:
        return "unknown"
    ever = [any(step[i] for step in history) for i in range(len(history[0]))]
    if not any(ever):
        return "none"
    final = history[-1]
    if any(was and not now for was, now in zip(ever, final)):
        return "regressed"
    return "stalled"


def cmd_failures(options):
    from collections import Counter, defaultdict

    root = Path(options.trace_dir)
    per_suite = defaultdict(Counter)
    per_task = defaultdict(Counter)
    for path in sorted(root.rglob("*.json")):
        trace = json.loads(path.read_text())
        label = classify(trace)
        per_suite[trace["suite"]][label] += 1
        per_task[f"{trace['suite']}/{trace['task_id']:02d}"][label] += 1
    report = {"trace_dir": str(root), "suites": {k: dict(v) for k, v in per_suite.items()},
              "tasks": {k: dict(v) for k, v in sorted(per_task.items())}}
    write(options.out, report)
    for suite, counts in report["suites"].items():
        failures = sum(v for k, v in counts.items() if k != "success")
        print(f"{suite:15s} failures={failures:4d} " + " ".join(
            f"{k}={v}" for k, v in sorted(counts.items()) if k != "success"))


def write(path, payload):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("neurons", "flips"):
        sub = commands.add_parser(name)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--weights", choices=["auto", "ema", "raw"], default="auto")
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--window", type=int, default=4, help="consecutive queries per sample")
        sub.add_argument("--batch_size", type=int, default=4)
        sub.add_argument("--num_workers", type=int, default=2)
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--out")
    neurons = commands.choices["neurons"]
    neurons.add_argument("--batches", type=int, default=8)
    neurons.add_argument("--probe_grid", type=int, default=2001)
    flips = commands.choices["flips"]
    flips.add_argument("--mode", choices=["pixel", "tf32", "weights", "bf16_carry"], default="pixel")
    flips.add_argument("--epsilon", type=float, default=1e-3, help="pixel noise std (normalized units)")
    flips.add_argument("--bits", type=int, default=8)
    flips.add_argument("--per_layer", action="store_true")
    failures = commands.add_parser("failures")
    failures.add_argument("--trace_dir", required=True)
    failures.add_argument("--out")
    options = parser.parse_args()
    {"neurons": cmd_neurons, "flips": cmd_flips, "failures": cmd_failures}[options.command](options)


if __name__ == "__main__":
    main()
