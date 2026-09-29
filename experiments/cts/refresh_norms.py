"""Export CTS EMA weights with WTA norms refreshed on training windows.

Same contract as experiments/libero/refresh_wta_norms.py (recorded LeRobot
training data, fixed EMA parameters, only BN accumulators change), except that
the refresh forward runs query windows through forward_window with no burn-in:
every query of a window contributes, from the cold first query to warm ones,
so the statistics cover the state ages the policy meets when deployed.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from turbovla.cts.data import LiberoQueryWindowDataset, window_collate
from turbovla.evaluation.policy import configure_transformers_offline
from turbovla.models.components.wta_operators import verify_running_norms
from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.turbovla import build_turbovla
from turbovla.training.trainer import set_seed
from turbovla.training.wta_norms import training_stat_refresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8, help="windows per batch")
    parser.add_argument("--window", type=int, default=4, help="queries per refresh window")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    options = parser.parse_args()
    if min(options.batches, options.batch_size, options.window) < 1 or options.num_workers < 0:
        raise ValueError("Positive batches/batch_size/window and nonnegative workers required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run EMA-stat export on one device, not under torchrun")
    source = Path(options.checkpoint).resolve()
    output = Path(options.output).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output == source or output.exists() or temporary.exists():
        raise FileExistsError("EMA export must use a fresh output; training checkpoints are never overwritten")
    configure_transformers_offline()
    checkpoint = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    config = TurboVLAConfig.from_mapping(checkpoint["model_config"])
    args = SimpleNamespace(**checkpoint["args"])
    if not config.cts.enabled or config.interaction.backbone_type != "spike_text_only":
        raise ValueError("Expected a CTS spike-driven-vla checkpoint; use refresh_wta_norms.py otherwise")
    if args.dataset_format != "lerobot" or getattr(args, "dataset_split", "train") != "train":
        raise ValueError("Expected recorded LeRobot training data")
    numerics = checkpoint.get("numerics", {})
    if numerics.get("float32_matmul_precision") != "highest" or numerics.get("cudnn_allow_tf32") is not False:
        raise ValueError("Expected the highest/fp32, TF32-disabled training policy")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    set_seed(args.seed)
    device = torch.device(options.device)
    model = build_turbovla(config)
    model.load_state_dict(checkpoint["ema_model_state_dict"], strict=True)
    model.to(device).float().eval().requires_grad_(False)
    dataset = LiberoQueryWindowDataset(
        args.dataset_dir, dataset_dirs=args.dataset_dirs,
        stats_path=args.stats_path, stats_key=args.stats_key,
        normalize_binary_gripper=args.normalize_binary_gripper,
        gripper_convention=args.lerobot_gripper_convention,
        LOCAL_DINOV3_PATH=args.dinov3_path, chunk_size=args.chunk_size,
        rank=0, world_size=1, seed=args.seed, local_files_only=True,
        expected_image_size=args.expected_image_size,
        vision_encoder_type=args.vision_encoder_type, vision_image_size=args.vision_image_size,
        shuffle_buffer=args.shuffle_buffer, step_mix_buffer_size=args.cts_mix_buffer_windows,
        enable_dinov3_teacher=False, light_augmentation=False,
        window=options.window, burn_in=0, stride=args.cts_stride,
    )
    iterator = iter(DataLoader(dataset, batch_size=options.batch_size, num_workers=options.num_workers,
                               collate_fn=window_collate, pin_memory=device.type == "cuda"))
    with training_stat_refresh(model):
        for index in range(options.batches):
            per_query, present, _, _ = next(iterator)
            queries = [(instructions, {"pixel_values": samples["pixel_values"].to(device)}, states.to(device))
                       for samples, instructions, states, _, _ in per_query]
            predictions = model.forward_window(queries, present.to(device), burn_in=0)
            if not torch.isfinite(predictions).all():
                raise RuntimeError("Non-finite output during EMA training-stat refresh")
            if index == 0 or (index + 1) % 16 == 0 or index + 1 == options.batches:
                print(json.dumps({"event": "training_stat_refresh", "batch": index + 1,
                                  "batches": options.batches}), flush=True)
    norms = verify_running_norms(model)
    metadata = {
        "version": 1, "kind": "masked_per_timestep_synchronized_batch_norm",
        "raw": {"status": "absent"},
        "ema": {"status": "refreshed_with_training_data", "source": "training_data",
                "source_checkpoint": str(source), "dataset_dirs": args.dataset_dirs,
                "seed": args.seed, "batches": options.batches, "batch_size": options.batch_size,
                "refresh_window": options.window, "refresh_stride": args.cts_stride,
                "optimizer_steps": 0, "vision_bn_refreshed": True, "layers": norms},
    }
    export = {key: checkpoint[key] for key in
              ("model_config", "args", "numerics", "global_step", "ema_decay", "training_scope")}
    export.update(ema_model_state_dict={key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                  wta_normalization=metadata, export_kind="ema_evaluation_only",
                  cts_training=checkpoint.get("cts_training"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as stream:
        torch.save(export, stream)
    temporary.replace(output)
    print(json.dumps({"event": "ema_export_complete", "output": str(output),
                      "norm_layers": len(norms), "optimizer_steps": 0}), flush=True)


if __name__ == "__main__":
    main()
