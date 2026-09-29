#!/usr/bin/env python3
"""Query-window trainer for Control-Time Spiking (C3).

Everything that is not about control time is the base recipe, reused from
turbovla.training: model and teacher construction, GroundingDINO and
calibrated-projection preloads, the joint-scope audit, pi0.5 AdamW with EMA,
the sample-based LR schedule and the checkpoint format (EMA weights and WTA
normalization metadata included). What changes:

* a sample is a window of Q = burn_in + K consecutive control queries of one
  demonstration, `stride` frames apart (LiberoQueryWindowDataset);
* the model runs the whole window in one forward (DDP sees one forward per
  backward); burn-in queries run without gradients and only warm up the state;
* action loss and KD average over the K loss queries;
* the schedule counts loss queries: global_batch = batch * accum * world * K,
  so --batch_size 16 with K=2 matches the base 32 samples per GPU.

Run through experiments/cts/train.py (recipes) or directly with torchrun:
    torchrun --nproc_per_node 8 -m turbovla.cts.train <base flags> --cts_window 2 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.configuration import CTSConfig
# train_mixed imports pi05, which patches the trainer with pi0.5 AdamW + EMA and EMA saving.
from ..training import train_mixed, trainer
from ..training.trainer import (
    build_distillation_teacher, build_scheduler, cleanup_distributed, get_latest_checkpoint,
    load_interaction_from_init_checkpoint, load_text_projection_from_init_checkpoint,
    masked_l1_loss, masked_mse_loss, masked_text_alignment_loss, reduce_mean, set_seed,
    setup_distributed, unwrap_model,
)
from .apply import cts_parameter_names
from .data import LiberoQueryWindowDataset, window_collate

SPIKE_BACKBONES = {"spike_temporal", "spike_wta", "spike_text_only"}


# --------------------------------------------------------------------------- args

def cts_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    defaults = CTSConfig()
    model = parser.add_argument_group("CTS model (CTSConfig)")
    for name in CTSConfig.__dataclass_fields__:
        if name == "enabled":
            continue
        default = getattr(defaults, name)
        if isinstance(default, bool):
            model.add_argument(f"--cts_{name}", action=argparse.BooleanOptionalAction, default=default)
        else:
            model.add_argument(f"--cts_{name}", type=type(default), default=default)
    train = parser.add_argument_group("CTS query-window training")
    train.add_argument("--cts_window", type=int, default=2, help="K loss queries per window")
    train.add_argument("--cts_burn_in", type=int, default=2, help="no-grad warm-up queries per window")
    train.add_argument("--cts_stride", type=int, default=12,
                       help="frames between queries; equals the deployed open-loop chunk")
    train.add_argument("--cts_state_dropout", type=float, default=0.1,
                       help="probability of resetting a sample's state before a loss query")
    train.add_argument("--cts_windows_per_episode", choices=["length_over_window", "length"],
                       default="length_over_window",
                       help="length_over_window keeps the base number of loss samples per epoch")
    train.add_argument("--cts_mix_buffer_windows", type=int, default=None,
                       help="window shuffle buffer; default keeps the base step buffer in loss samples")
    train.add_argument("--cts_lr_scale", type=float, default=1.0,
                       help="LR multiplier for the new CTS parameters (retention, dendritic memory)")
    train.add_argument("--cts_init_checkpoint", type=str, default=None,
                       help="Pilot only: start from a trained policy; only CTS parameters may be missing")
    train.add_argument("--cts_init_weights", choices=["ema", "raw"], default="ema")
    train.add_argument("--cts_smoke_test", action="store_true",
                       help="Plumbing check: waives asset preloads, <=3 steps, checkpoint marked as smoke test")
    return parser


def parse_args():
    known, remaining = cts_parser().parse_known_args()
    if any(argument in {"-h", "--help"} for argument in remaining):
        cts_parser().print_help()
    original = sys.argv
    sys.argv = [original[0], *remaining]
    try:
        args = train_mixed.parse_args_and_record()
    finally:
        sys.argv = original
    for name, value in vars(known).items():
        setattr(args, name, value)
    args.cts_enabled = True
    return args


def validate_args(args, world_size):
    if args.cts_window < 1 or args.cts_burn_in < 0 or args.cts_stride < 1:
        raise ValueError("--cts_window >= 1, --cts_burn_in >= 0 and --cts_stride >= 1 are required")
    if not 0 <= args.cts_state_dropout < 1:
        raise ValueError("--cts_state_dropout must be in [0, 1)")
    if args.cts_lr_scale <= 0:
        raise ValueError("--cts_lr_scale must be positive")
    if args.fusion_backbone_type != "spike_text_only":
        raise ValueError("CTS is defined on the spike_text_only base (spike-driven-vla)")
    if not args.approve_spike_fusion_training:
        raise RuntimeError("Spike fusion training requires --approve_spike_fusion_training")
    if args.float32_matmul_precision != "highest" or args.precision != "fp32":
        raise ValueError("The spike-driven-vla recipe requires highest/fp32 consistently")
    if args.cts_smoke_test:
        if args.max_steps > 3:
            raise ValueError("--cts_smoke_test is limited to 3 steps")
    elif args.cts_init_checkpoint is None:
        if not args.require_feature_enhancer_preload or not args.text_projection_init:
            raise ValueError("Requires common GroundingDINO and calibrated text projection initialization")
    if args.cts_init_checkpoint is not None and args.joint_scope_audit:
        raise ValueError("The joint-scope audit certifies fresh training; drop it for --cts_init_checkpoint")
    if min(args.distill_feature_weight, args.distill_action_weight, args.distill_text_weight) < 0:
        raise ValueError("distillation weights must be non-negative")
    for name, fallback in (("head_lr", "lr"), ("head_weight_decay", "weight_decay"),
                           ("vision_encoder_lr", "dinov3_lr"),
                           ("vision_encoder_weight_decay", "dinov3_weight_decay")):
        if getattr(args, name) is None:
            setattr(args, name, getattr(args, fallback))
    if args.vision_image_size is None:
        args.vision_image_size = 224 if args.vision_encoder_type == "sdtv3_19m" else args.expected_image_size
    if args.lr_schedule_steps is None:
        args.lr_schedule_steps = args.max_steps
    if args.lr_schedule_steps <= args.warmup_steps:
        raise ValueError("lr_schedule_steps must be greater than warmup_steps")
    if args.cts_mix_buffer_windows is None:
        args.cts_mix_buffer_windows = max(1, args.step_mix_buffer_size // args.cts_window)
    args.cts_loss_queries_per_step = args.batch_size * args.grad_accum_steps * world_size * args.cts_window


# ------------------------------------------------------------------- model/optim

def load_policy_init(model, args):
    """Pilot warm start from a trained base policy. Only CTS parameters may be new."""
    checkpoint = torch.load(args.cts_init_checkpoint, map_location="cpu", weights_only=False)
    key = "ema_model_state_dict" if args.cts_init_weights == "ema" else "model_state_dict"
    if key not in checkpoint:
        raise KeyError(f"{args.cts_init_checkpoint} has no {key}")
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in checkpoint[key].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = set(cts_parameter_names(model))
    bad_missing = [name for name in missing if name not in allowed]
    if bad_missing or unexpected:
        raise RuntimeError(f"Policy init mismatch: missing={bad_missing[:8]} unexpected={list(unexpected)[:8]}")
    return {"checkpoint": args.cts_init_checkpoint, "weights": args.cts_init_weights,
            "new_cts_tensors": len(missing)}


def build_optimizer(model, args):
    """Base param groups (pi0.5 AdamW + EMA); CTS parameters optionally get their own LR."""
    optimizer, summary = trainer.build_param_group_optimizer(model, args)
    if args.cts_lr_scale == 1.0:
        return optimizer, summary
    params = dict(model.named_parameters())
    cts = {id(params[name]) for name in cts_parameter_names(model) if params[name].requires_grad}
    moved = []
    for group in optimizer.param_groups:
        kept = []
        for param in group["params"]:
            (moved if id(param) in cts else kept).append(param)
        group["params"] = kept
    optimizer.param_groups[:] = [group for group in optimizer.param_groups if group["params"]]
    lr = args.head_lr * args.cts_lr_scale
    optimizer.add_param_group({"params": moved, "lr": lr, "weight_decay": 0.0, "name": "cts_no_decay"})
    summary = [{"name": group["name"], "lr": group["lr"], "weight_decay": group["weight_decay"],
                "params": sum(p.numel() for p in group["params"])} for group in optimizer.param_groups]
    return optimizer, summary


def build_window_dataset(args, rank, world_size, distillation_enabled):
    dataset_format = train_mixed._resolve_dataset_format(args.dataset_dirs, args.dataset_format)
    if dataset_format != "lerobot":
        raise ValueError("Query windows need whole episodes; use the LeRobot input format")
    if args.lerobot_gripper_convention is None:
        raise ValueError("LeRobot input requires an explicit --lerobot_gripper_convention")
    return LiberoQueryWindowDataset(
        args.dataset_dir,
        dataset_dirs=args.dataset_dirs, stats_path=args.stats_path, stats_key=args.stats_key,
        normalize_binary_gripper=args.normalize_binary_gripper,
        gripper_convention=args.lerobot_gripper_convention,
        LOCAL_DINOV3_PATH=args.dinov3_path, rank=rank, world_size=world_size,
        chunk_size=args.chunk_size, split=args.dataset_split, shuffle_buffer=args.shuffle_buffer,
        step_mix_buffer_size=args.cts_mix_buffer_windows, seed=args.seed,
        local_files_only=not args.allow_hf_download, expected_image_size=args.expected_image_size,
        vision_encoder_type=args.vision_encoder_type, vision_image_size=args.vision_image_size,
        enable_dinov3_teacher=distillation_enabled, light_augmentation=args.libero_light_augmentation,
        window=args.cts_window, burn_in=args.cts_burn_in, stride=args.cts_stride,
        windows_per_episode=args.cts_windows_per_episode,
    )


# ------------------------------------------------------------------------- step

def window_batch(batch, device, burn_in):
    """Collated window -> model inputs (student pixels only) and loss-query targets."""
    per_query, present, loss_valid, steps = batch
    queries = [(instructions, {"pixel_values": samples["pixel_values"].to(device, non_blocking=True)},
                states.to(device, non_blocking=True))
               for samples, instructions, states, _, _ in per_query]
    loss_queries = per_query[burn_in:]
    targets = torch.stack([query[3] for query in loss_queries], dim=1).to(device, non_blocking=True)
    chunk_mask = torch.stack([query[4] for query in loss_queries], dim=1).to(device, non_blocking=True)
    valid = loss_valid[:, burn_in:].to(device)
    mask = chunk_mask * valid[..., None].to(chunk_mask.dtype)
    teacher_inputs = [
        (instructions, samples.get("teacher_pixel_values"), states)
        for samples, instructions, states, _, _ in loss_queries
    ]
    return queries, present.to(device), targets, mask, teacher_inputs


def window_losses(args, model, teacher, batch, device):
    queries, present, targets, mask, teacher_inputs = window_batch(batch, device, args.cts_burn_in)
    need_features = teacher is not None
    output = model(None, None, None, cts_window=dict(
        queries=queries, present=present, burn_in=args.cts_burn_in, state_dropout=args.cts_state_dropout,
    ), return_distillation_features=need_features)
    predictions, features = output if need_features else (output, None)
    if predictions.shape != targets.shape:
        raise ValueError(f"pred {tuple(predictions.shape)} vs target {tuple(targets.shape)}")
    action_loss = masked_l1_loss(predictions, targets, mask)
    zero = predictions.new_zeros(())
    feature_kd, action_kd, text_kd = zero, zero, zero
    if teacher is not None:
        teacher_actions = []
        for index, (instructions, teacher_pixels, states) in enumerate(teacher_inputs):
            if teacher_pixels is None:
                raise ValueError("Distillation requires teacher pixel values from the dataset")
            teacher_samples = {"pixel_values": teacher_pixels.to(device, non_blocking=True)}
            states = states.to(device, non_blocking=True)
            with torch.no_grad():
                if args.distill_text_weight > 0:
                    actions, teacher_features = teacher(
                        instructions, teacher_samples, states, return_distillation_features=True)
                    teacher_visual = teacher_features["visual"]
                else:
                    actions, teacher_visual = teacher(
                        instructions, teacher_samples, states, return_visual_tokens=True)
            student = features[index]
            if student["visual"].shape != teacher_visual.shape:
                raise ValueError("student/teacher visual token mismatch")
            feature_kd = feature_kd + F.mse_loss(student["visual"].float(), teacher_visual.float())
            if args.distill_text_weight > 0:
                if not torch.equal(student["text_valid"], teacher_features["text_valid"]):
                    raise ValueError("student/teacher token layouts differ")
                text_kd = text_kd + masked_text_alignment_loss(
                    student["text"], teacher_features["text"], teacher_features["text_valid"])
            teacher_actions.append(actions)
        count = len(teacher_inputs)
        feature_kd, text_kd = feature_kd / count, text_kd / count
        action_kd = masked_mse_loss(predictions.float(), torch.stack(teacher_actions, dim=1).float(), mask)
    loss = (action_loss + args.distill_feature_weight * feature_kd
            + args.distill_action_weight * action_kd + args.distill_text_weight * text_kd)
    per_query = [masked_l1_loss(predictions[:, k], targets[:, k], mask[:, k]).detach()
                 for k in range(predictions.shape[1])]
    return loss, {"act": action_loss.detach(), "feat_kd": feature_kd.detach(),
                  "act_kd": action_kd.detach(), "text_kd": text_kd.detach()}, per_query


def cts_parameter_report(model):
    """Current retention / memory statistics, logged next to the losses."""
    retention, rho, gain = [], [], []
    for name, param in model.named_parameters():
        if name.endswith(".retention_logit"):
            retention.append(param.detach().sigmoid().flatten())
        elif ".cts_memory" in name and name.endswith(".rho_logit"):
            rho.append(param.detach().sigmoid().flatten())
        elif ".cts_memory" in name and name.endswith(".gain"):
            gain.append(param.detach().abs().flatten())
    report = {}
    if retention:
        values = torch.cat(retention).float()
        report["retention_mean"] = float(values.mean())
        report["retention_p10"] = float(values.quantile(0.1))
    if rho:
        report["memory_tau_median"] = float((-1.0 / torch.cat(rho).float().clamp(1e-6, 1 - 1e-6).log()).median())
    if gain:
        report["memory_gain_mean_abs"] = float(torch.cat(gain).float().mean())
    return report


# ------------------------------------------------------------------------- main

def train_model():
    args = parse_args()
    is_distributed, rank, world_size, local_rank, device = setup_distributed()
    try:
        validate_args(args, world_size)
        distillation_enabled = max(args.distill_feature_weight, args.distill_action_weight,
                                   args.distill_text_weight) > 0
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
        torch.backends.cudnn.allow_tf32 = False
        set_seed(args.seed)

        model = trainer.build_model_architecture(args)  # pi0.5 patch sets the vision precision
        teacher = build_distillation_teacher(args) if distillation_enabled else None
        fe_report, init_report, joint_groups = None, None, None
        if args.cts_init_checkpoint is not None:
            init_report = load_policy_init(model, args)
        elif args.pretrained_init_ckpt is not None:
            if args.require_feature_enhancer_preload:
                fe_report = load_interaction_from_init_checkpoint(model, args.pretrained_init_ckpt)
            if args.load_text_projection_from_init:
                load_text_projection_from_init_checkpoint(model, args.pretrained_init_ckpt)
        elif not args.cts_smoke_test:
            raise ValueError("--pretrained_init_ckpt is required for a fresh CTS run")
        if args.text_projection_init and args.cts_init_checkpoint is None:
            from ..training.joint_scope import load_calibrated_projection
            load_calibrated_projection(model, args)
        if args.joint_scope_audit:
            from ..training.joint_scope import check_preload, joint_scope
            check_preload(model, fe_report)
            joint_groups = joint_scope(model, args)
        if args.freeze_backbones:
            trainer.freeze_backbones(model)

        cts_names = cts_parameter_names(model)
        if rank == 0:
            print("cts_config:", json.dumps(model.config.cts.__dict__), flush=True)
            print("cts_training:", json.dumps({k: v for k, v in vars(args).items()
                                              if k.startswith("cts_") and k not in CTSConfig.__dataclass_fields__}),
                  flush=True)
            print(f"cts_parameters: tensors={len(cts_names)}, "
                  f"elements={sum(p.numel() for n, p in model.named_parameters() if n in set(cts_names))}",
                  flush=True)
            if init_report is not None:
                print("pilot policy init:", json.dumps(init_report), flush=True)
            if fe_report is not None:
                print("interaction preload:", {k: fe_report[k] for k in ("mapped", "skipped_missing", "skipped_shape")})
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        if is_distributed:
            dist.barrier()

        global_step, checkpoint = 0, None
        latest = get_latest_checkpoint(args.checkpoint_dir, args.checkpoint_prefix)
        if latest is not None and args.resume_mode != "none":
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
            saved_cts = checkpoint.get("model_config", {}).get("cts")
            if saved_cts != model.config.to_dict()["cts"]:
                raise ValueError(f"Cannot resume a different CTS configuration: {saved_cts}")
            from ..training.wta_norms import check_wta_resume_config
            check_wta_resume_config(model.config, checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            global_step = int(checkpoint.get("global_step", 0)) if args.resume_mode == "all" else 0
            if rank == 0:
                print(f"resumed {latest} (mode={args.resume_mode}, step={global_step})", flush=True)
        elif args.resume_mode != "none":
            raise FileNotFoundError("Explicit resume requested but no matching checkpoint exists")
        elif latest is not None and not args.cts_smoke_test:
            raise RuntimeError(f"Refusing to overwrite an existing run: {latest}")

        model.to(device)
        if teacher is not None:
            teacher.to(device).eval()
        if is_distributed:
            kwargs = dict(device_ids=[local_rank], output_device=local_rank) if device.type == "cuda" else {}
            model = DDP(model, find_unused_parameters=True, **kwargs)
        policy = unwrap_model(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer, optimizer_summary = build_optimizer(policy, args)
        scheduler = build_scheduler(
            optimizer, max_steps=args.lr_schedule_steps, warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio, global_batch=args.cts_loss_queries_per_step,
            warmup_samples=args.lr_warmup_samples, sample_offset=args.lr_sample_offset,
            step_offset=args.lr_step_offset, reference_batch=args.lr_reference_batch,
        )
        if checkpoint is not None and args.resume_mode == "all":
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            optimizer.restore_ema(checkpoint)
        if args.joint_scope_audit:
            from ..training.joint_scope import joint_scope
            joint_groups = joint_scope(policy, args)

        dataset = build_window_dataset(args, rank, world_size, distillation_enabled)
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=window_collate,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            persistent_workers=args.num_workers > 0)
        data_iter = iter(loader)
        if rank == 0:
            print("optimizer param groups:", optimizer_summary, flush=True)
            print(f"windows/GPU={args.batch_size} x accum={args.grad_accum_steps} x world={world_size} "
                  f"x K={args.cts_window} -> loss queries/step={args.cts_loss_queries_per_step}; "
                  f"burn_in={args.cts_burn_in}, stride={args.cts_stride}", flush=True)

        model.train()
        progress = tqdm(total=args.max_steps, initial=global_step, desc="cts") if rank == 0 else None
        history = {}
        while global_step < args.max_steps:
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accum, per_query_accum = {}, None
            for micro_step in range(args.grad_accum_steps):
                batch = next(data_iter)
                sync = (model.no_sync() if is_distributed and micro_step + 1 < args.grad_accum_steps
                        else nullcontext())
                with sync:
                    loss, parts, per_query = window_losses(args, model, teacher, batch, device)
                    (loss / args.grad_accum_steps).backward()
                parts["loss"] = loss.detach()
                for key, value in parts.items():
                    accum[key] = accum.get(key, 0.0) + float(value) / args.grad_accum_steps
                values = [float(value) / args.grad_accum_steps for value in per_query]
                per_query_accum = values if per_query_accum is None else [
                    a + b for a, b in zip(per_query_accum, values)]

            if global_step == 0:
                missing = [n for n, p in policy.named_parameters()
                           if n in set(cts_names) and p.requires_grad and p.grad is None]
                if missing:
                    raise RuntimeError(f"CTS parameters without gradient at step 1: {missing[:8]}")
                if args.joint_scope_audit:
                    from ..training.joint_scope import gradient_report, representative_snapshots
                    first_gradients = gradient_report(joint_groups)
                    first_parameters = representative_snapshots(joint_groups)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if global_step == 0 and args.joint_scope_audit:
                from ..training.joint_scope import record_first_update
                record_first_update(policy, args, joint_groups, first_parameters, first_gradients, rank)
                del first_parameters
            global_step += 1

            logged = {key: reduce_mean(value, device, is_distributed, world_size) for key, value in accum.items()}
            for k, value in enumerate(per_query_accum):
                logged[f"act_q{k}"] = reduce_mean(value, device, is_distributed, world_size)
            logged["sec_per_step"] = time.perf_counter() - step_start  # rank 0 wall time, data included
            for key, value in logged.items():
                window = history.setdefault(key, [])
                window.append(value)
                del window[:-args.log_freq]
            if rank == 0:
                progress.update(1)
                if global_step % args.log_freq == 0 or global_step == 1:
                    averages = {key: sum(v) / len(v) for key, v in history.items()}
                    progress.set_postfix({key: f"{averages[key]:.4f}" for key in ("loss", "act", "act_kd")})
                    print(json.dumps({"event": "train", "step": global_step, **averages,
                                      "lr": optimizer.param_groups[0]["lr"],
                                      **cts_parameter_report(policy)}), flush=True)

            should_save = (global_step in args.extra_save_steps or global_step % args.save_steps == 0
                           or (args.save_final and global_step == args.max_steps))
            if rank == 0 and should_save:
                save_checkpoint(args, policy, optimizer, scheduler, global_step, logged["loss"],
                                world_size, device)
        if progress is not None:
            progress.close()
        if rank == 0:
            print(json.dumps({"event": "done", "step": global_step, **cts_parameter_report(policy)}), flush=True)
    finally:
        cleanup_distributed()


def save_checkpoint(args, policy, optimizer, scheduler, global_step, loss, world_size, device):
    from ..models.components.wta_operators import verify_running_norms
    from ..training.wta_norms import normalization_metadata

    verify_running_norms(policy)
    path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_{global_step}.pth")
    if os.path.exists(path):
        raise RuntimeError(f"Refusing to overwrite checkpoint: {path}")
    payload = {
        "training_scope": ("cts_smoke_test" if args.cts_smoke_test
                           else "cts_query_window" + ("_joint" if args.joint_scope_audit else "")),
        "global_step": global_step,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "args": vars(args),
        "numerics": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "model_config": policy.config.to_dict(),
        "wta_normalization": normalization_metadata(),
        "cts_training": {key: value for key, value in vars(args).items()
                         if key.startswith("cts_") and key not in {"cts_" + f for f in CTSConfig.__dataclass_fields__}},
        "resources": {"world_size": world_size,
                      "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
                      if device.type == "cuda" else None},
    }
    # torch.save is patched by turbovla.training.pi05: EMA weights and EMA norm metadata are added.
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)
    print(f"saved: {path}", flush=True)



if __name__ == "__main__":
    train_model()
