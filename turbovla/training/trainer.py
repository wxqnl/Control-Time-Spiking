import argparse
import glob
import json
import math
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.libero_rlds import (
    LiberoRLDSDataset,
    vla_collate_fn,
)
from ..models.turbovla import (
    build_turbovla,
)


class DummyArgs:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the complete TurboVLA model with online BERT and DINOv3."
    )

    parser.add_argument("--dataset_dir", type=str, default="./data/libero/libero_10_no_noops/1.0.0")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dinov3_path", type=str, required=True)
    parser.add_argument("--vision_encoder_type", type=str, default="dinov3", choices=["dinov3", "sdtv3_19m"])
    parser.add_argument("--vision_model_path", type=str, default=None)
    parser.add_argument("--vision_pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--vision_model_source_path", type=str, default=None)
    parser.add_argument("--vision_image_size", type=int, default=None)
    parser.add_argument("--vision_output_grid_size", type=int, default=14)
    parser.add_argument("--vision_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vision_checkpoint_scope", choices=["all", "early"], default="all")
    parser.add_argument("--projection_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument(
        "--text_encoder_type",
        type=str,
        default="bert",
        choices=["bert", "smoothspike_bert"],
    )
    parser.add_argument("--text_model_path", type=str, default=None)
    parser.add_argument("--text_projection_init", type=str, default=None)
    parser.add_argument("--joint_scope_audit", action="store_true")
    parser.add_argument("--text_timesteps", type=int, default=1)
    parser.add_argument(
        "--text_mask_version", choices=["legacy", "corrected"], default="corrected",
        help="Version both tokenizer sentence masks and batched text-attention head masks.",
    )
    parser.add_argument("--text_sentence_mask_version", choices=["legacy", "corrected"], default=None)
    parser.add_argument("--text_head_mask_version", choices=["legacy", "corrected"], default=None)
    parser.add_argument(
        "--teacher_text_head_mask_version", choices=["legacy", "corrected"], default="corrected",
        help="Use batch-invariant frozen-teacher targets; legacy is only for reproducing old runs.",
    )
    parser.add_argument(
        "--pretrained_init_ckpt",
        type=str,
        required=True,
    )
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--checkpoint_prefix", type=str, default="turbovla_step")
    parser.add_argument("--resume_mode", type=str, default="none", choices=["none", "model", "all"])

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--dinov3_lr", type=float, default=5e-5)
    parser.add_argument("--head_weight_decay", type=float, default=1e-10)
    parser.add_argument("--dinov3_weight_decay", type=float, default=1e-10)
    parser.add_argument("--vision_encoder_lr", type=float, default=None)
    parser.add_argument("--vision_encoder_weight_decay", type=float, default=None)
    parser.add_argument("--vision_precision", type=str, default=None, choices=["fp32", "bf16", "bf16_autocast"])
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--distill_feature_weight", type=float, default=0.0)
    parser.add_argument("--distill_action_weight", type=float, default=0.0)
    parser.add_argument("--spike_state_population", type=int, default=0)
    parser.add_argument("--spike_state_population_range", type=float, default=4.0)
    parser.add_argument("--spike_state_attention_gain", type=float, default=1.0)
    parser.add_argument("--distill_text_weight", type=float, default=0.0,
                        help="Relative masked MSE on projected text tokens; frozen ANN teacher is training-only.")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16_amp"])
    parser.add_argument(
        "--float32_matmul_precision", choices=["highest", "high"], default="highest",
        help="FP32 matrix multiplication policy, saved for inference. highest matches SmoothSpike calibration.",
    )
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument(
        "--lr_schedule_steps",
        type=int,
        default=None,
        help="Cosine LR horizon. Defaults to max_steps; set separately for multi-fidelity searches.",
    )
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--lr_warmup_samples", type=int, default=0)
    parser.add_argument("--lr_sample_offset", type=int, default=0)
    parser.add_argument("--lr_step_offset", type=int, default=0)
    parser.add_argument("--lr_reference_batch", type=int, default=128)
    parser.add_argument("--min_lr_ratio", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--extra_save_steps", type=int, nargs="*", default=[])
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.set_defaults(save_final=True)
    parser.add_argument("--save_final", dest="save_final", action="store_true")
    parser.add_argument("--no_save_final", dest="save_final", action="store_false")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer", type=int, default=512)
    parser.add_argument("--step_mix_buffer_size", type=int, default=64)
    parser.add_argument("--expected_image_size", type=int, default=256)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument(
        "--text_padding_length",
        type=int,
        default=21,
        help="Fixed token length. LIBERO uses 21 to match the released training features.",
    )
    parser.add_argument(
        "--text_layout_path",
        type=str,
        default="experiments/libero/configs/online_text_layout.json",
        help="Online tokenizer layout metadata for exact released-checkpoint compatibility.",
    )
    parser.add_argument("--vla_feature_enhancer_layers", type=int, default=6)
    parser.add_argument("--enhancer_inner_dim", type=int, default=1024)
    parser.add_argument("--cross_attention_type", choices=["ann", "spike_sdsa", "spike_wta"], default="ann")
    parser.add_argument("--fusion_backbone_type", choices=["hybrid", "spike_temporal", "spike_wta", "spike_text_only"], default="hybrid")
    parser.add_argument("--text_only_spike_projections", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--text_attention_routing", choices=["wta", "sdsa"], default="wta")
    parser.add_argument("--wta_text_topk", type=int, default=3)
    parser.add_argument("--wta_vision_topk", type=int, default=8)
    parser.add_argument("--wta_surrogate_temperature", type=float, default=1.0)
    parser.add_argument("--spike_norm_momentum", type=float, default=0.1)
    parser.add_argument("--spike_norm_eps", type=float, default=1e-5)
    parser.add_argument("--spike_norm_sync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spike_calibration_batches", type=int, default=8)
    parser.add_argument("--approve_spike_fusion_training", action="store_true",
                        help="Explicit approval after reviewing the v2 architecture; off by default")
    parser.add_argument("--cross_timesteps", type=int, default=4)
    parser.add_argument("--cross_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action_head_type", choices=[
        "ann", "spike_sdsa", "minimal_cross_wta", "progressive_full_wta", "hybrid_spike_ffn", "deep_spike_wta",
        "full_spike_state8_wta", "full_spike_state8_shiftmax", "full_spike_state8_shiftmax_continuous_inputs", "full_spike_state8_shiftmax_continuous_inputs_ann_ffn", "full_spike_state8_shiftmax_ann_ffn",
    ], default="ann")
    parser.add_argument("--spike_state_encoding", choices=["perdim_plif", "multithreshold", "project_then_plif", "perdim_ann"], default="perdim_plif")
    parser.add_argument("--continuous_state", action="store_true",
                        help="Use A continuous state MLP with a spike-body action head")
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--chunk_size", type=int, default=12)
    parser.add_argument("--state_dim", type=int, default=8)
    parser.add_argument("--num_state_tokens", type=int, default=2)
    parser.add_argument("--text_dropout", type=float, default=0.0)
    parser.add_argument("--fusion_dropout", type=float, default=0.0)
    parser.add_argument("--fusion_droppath", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.set_defaults(allow_hf_download=False)
    parser.add_argument("--allow_hf_download", dest="allow_hf_download", action="store_true")
    parser.add_argument("--no_allow_hf_download", dest="allow_hf_download", action="store_false")

    parser.set_defaults(shuffle_steps_within_episode=True)
    parser.add_argument("--shuffle_steps_within_episode", dest="shuffle_steps_within_episode", action="store_true")
    parser.add_argument("--no_shuffle_steps_within_episode", dest="shuffle_steps_within_episode", action="store_false")

    parser.set_defaults(freeze_backbones=False)
    parser.add_argument("--freeze_backbones", dest="freeze_backbones", action="store_true")
    parser.add_argument("--no_freeze_backbones", dest="freeze_backbones", action="store_false")

    parser.set_defaults(freeze_text_encoder=True)
    parser.add_argument("--freeze_text_encoder", dest="freeze_text_encoder", action="store_true")
    parser.add_argument("--train_text_encoder", dest="freeze_text_encoder", action="store_false")

    parser.set_defaults(require_feature_enhancer_preload=True)
    parser.add_argument(
        "--require_feature_enhancer_preload",
        dest="require_feature_enhancer_preload",
        action="store_true",
    )
    parser.add_argument(
        "--no_require_feature_enhancer_preload",
        dest="require_feature_enhancer_preload",
        action="store_false",
    )

    parser.set_defaults(load_text_projection_from_init=True)
    parser.add_argument(
        "--load_text_projection_from_init",
        dest="load_text_projection_from_init",
        action="store_true",
    )
    parser.add_argument(
        "--no_load_text_projection_from_init",
        dest="load_text_projection_from_init",
        action="store_false",
    )

    parser.set_defaults(require_text_proj_preload=True)
    parser.add_argument("--require_text_proj_preload", dest="require_text_proj_preload", action="store_true")
    parser.add_argument("--no_require_text_proj_preload", dest="require_text_proj_preload", action="store_false")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        use_cuda = torch.cuda.is_available()
        backend = "nccl" if use_cuda else "gloo"
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(backend=backend, device_id=device)
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend=backend)
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_distributed = False
    return is_distributed, rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_model_architecture(args):
    text_layout = {}
    if args.text_layout_path:
        with open(args.text_layout_path, "r", encoding="utf-8") as handle:
            text_layout = json.load(handle)
        configured_length = int(text_layout["output_padding_length"])
        if configured_length != args.text_padding_length:
            raise ValueError(
                f"text layout output length {configured_length} does not match "
                f"--text_padding_length={args.text_padding_length}"
            )
    model_args = DummyArgs()
    model_args.dinov3_path = args.dinov3_path
    model_args.vision_encoder_type = args.vision_encoder_type
    model_args.vision_model_path = args.vision_model_path or args.dinov3_path
    model_args.vision_pretrained_checkpoint = args.vision_pretrained_checkpoint
    model_args.vision_model_source_path = args.vision_model_source_path
    model_args.vision_output_grid_size = args.vision_output_grid_size
    model_args.vision_gradient_checkpointing = getattr(args, "vision_gradient_checkpointing", False)
    model_args.vision_checkpoint_scope = getattr(args, "vision_checkpoint_scope", "all")
    model_args.projection_gradient_checkpointing = getattr(args, "projection_gradient_checkpointing", False)
    model_args.bert_path = args.bert_path
    model_args.text_encoder_type = args.text_encoder_type
    model_args.text_model_path = args.text_model_path or args.bert_path
    model_args.text_timesteps = args.text_timesteps
    model_args.text_mask_version = getattr(args, "text_mask_version", "legacy")
    model_args.text_sentence_mask_version = getattr(args, "text_sentence_mask_version", None)
    model_args.text_head_mask_version = getattr(args, "text_head_mask_version", None)
    model_args.hidden_dim = args.hidden_dim
    model_args.nheads = args.nheads
    model_args.dim_feedforward = args.dim_feedforward
    model_args.max_text_len = args.max_text_len
    model_args.text_padding_length = args.text_padding_length
    model_args.text_padding_length_by_instruction = text_layout.get("padding_length_by_instruction", {})
    model_args.vla_feature_enhancer_layers = args.vla_feature_enhancer_layers
    model_args.enhancer_inner_dim = args.enhancer_inner_dim
    model_args.cross_attention_type = getattr(args, "cross_attention_type", "ann")
    model_args.fusion_backbone_type = getattr(args, "fusion_backbone_type", "hybrid")
    model_args.text_only_spike_projections = getattr(args, "text_only_spike_projections", False)
    for name in ("text_attention_routing", "wta_text_topk", "wta_vision_topk", "wta_surrogate_temperature",
                 "spike_norm_momentum", "spike_norm_eps", "spike_norm_sync"):
        if hasattr(args, name):
            setattr(model_args, name, getattr(args, name))
    model_args.cross_timesteps = getattr(args, "cross_timesteps", 4)
    model_args.cross_gradient_checkpointing = getattr(args, "cross_gradient_checkpointing", True)
    model_args.text_dropout = args.text_dropout
    model_args.fusion_dropout = args.fusion_dropout
    model_args.fusion_droppath = args.fusion_droppath
    model_args.action_head_type = args.action_head_type
    model_args.continuous_state = bool(getattr(args, "continuous_state", False))
    model_args.spike_state_encoding = getattr(args, "spike_state_encoding", "perdim_plif")
    model_args.spike_state_population = getattr(args, "spike_state_population", 0)
    model_args.spike_state_population_range = getattr(args, "spike_state_population_range", 4.0)
    model_args.spike_state_attention_gain = getattr(args, "spike_state_attention_gain", 1.0)
    model_args.act_dropout = 0.0 if args.action_head_type == "spike_sdsa" else 0.1
    model_args.action_dim = args.action_dim
    model_args.chunk_size = args.chunk_size
    model_args.state_dim = args.state_dim
    model_args.num_state_tokens = args.num_state_tokens
    model_args.local_files_only = not args.allow_hf_download
    model_args.freeze_vision_encoder = args.freeze_backbones
    model_args.freeze_text_encoder = args.freeze_text_encoder
    model_args.dinov3_precision = args.vision_precision or getattr(args, "dinov3_precision", "bf16_autocast")
    model_args.num_views = 2
    model_args.image_size = args.vision_image_size or args.expected_image_size
    model_args.position_embedding = "view"
    model_args.encode_views_separately = True
    model_args.padding_strategy = "key_padding_mask"
    for name, value in vars(args).items():
        if name.startswith("cts_"):
            setattr(model_args, name, value)
    return build_turbovla(model_args)


def build_distillation_teacher(args):
    if args.teacher_checkpoint is None:
        raise ValueError("--teacher_checkpoint is required when a distillation weight is non-zero")
    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if model_config is not None:
        from ..models.configuration import TurboVLAConfig

        config = TurboVLAConfig.from_mapping(model_config)
        config.vision.encoder_type = "dinov3"
        config.vision.model_name_or_path = args.dinov3_path
        config.vision.image_size = args.vision_image_size or 224
        config.vision.frozen = True
        config.vision.local_files_only = not args.allow_hf_download
        config.vision.compute_precision = "bf16_autocast"
        config.text.encoder_type = "bert"
        config.text.model_name_or_path = args.bert_path
        config.text.timesteps = 1
        config.text.frozen = True
        config.text.local_files_only = not args.allow_hf_download
        config.interaction.backbone_type = "hybrid"
        config.interaction.text_only_spike_projections = False
        config.interaction.cross_attention_type = "ann"
        config.action.head_type = "ann"
        teacher = build_turbovla(config)
    else:
        teacher_args = DummyArgs()
        for key, value in vars(args).items():
            setattr(teacher_args, key, value)
        teacher_args.vision_encoder_type = "dinov3"
        teacher_args.vision_model_path = args.dinov3_path
        teacher_args.vision_pretrained_checkpoint = None
        teacher_args.freeze_backbones = True
        teacher_args.text_encoder_type = "bert"
        teacher_args.text_model_path = args.bert_path
        teacher_args.text_timesteps = 1
        # Do not inherit the new student's mask semantics for an old teacher.
        teacher_args.text_mask_version = "legacy"
        teacher_args.text_sentence_mask_version = None
        teacher_args.text_head_mask_version = None
        teacher_args.cross_attention_type = "ann"
        teacher_args.fusion_backbone_type = "hybrid"
        teacher_args.text_attention_routing = "wta"
        teacher_args.action_head_type = "ann"
        teacher_args.continuous_state = False
        # The public teacher keeps its original two state tokens.
        teacher_args.num_state_tokens = 2
        teacher_args.text_only_spike_projections = False
        teacher_args.freeze_vision_encoder = True
        teacher = build_model_architecture(teacher_args)
    state = _extract_state_dict(checkpoint)
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"teacher checkpoint is not strictly compatible: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    # Head expansion is a batch-layout operation, not the checkpoint\'s sentence
    # protocol. Match the frozen teacher\'s per-sample inference without changing
    # its tokenization, sentence masks, position IDs, or checkpoint weights.
    teacher_head_version = getattr(args, "teacher_text_head_mask_version", None) or "corrected"
    if teacher_head_version != teacher.config.head_mask_version:
        teacher.config.text_head_mask_version = teacher_head_version
        for layer in teacher.vision_language_interaction.text_layers:
            layer.text_mask_version = teacher_head_version
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def freeze_backbones(model):
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder.backbone"):
            param.requires_grad = False


def get_latest_checkpoint(ckpt_dir, prefix):
    # EMA evaluation exports are not resumable training checkpoints.
    candidates = []
    for path in glob.glob(os.path.join(ckpt_dir, f"{prefix}_*.pth")):
        step = os.path.splitext(os.path.basename(path))[0][len(prefix) + 1:]
        if step.isdigit():
            candidates.append((int(step), path))
    return max(candidates)[1] if candidates else None


def masked_l1_loss(pred, target, mask):
    l1 = torch.abs(pred - target)
    mask = mask.unsqueeze(-1).float()
    l1 = l1 * mask
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return l1.sum() / denom


def masked_mse_loss(pred, target, mask):
    squared = (pred - target).square()
    mask = mask.unsqueeze(-1).float()
    squared = squared * mask
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return squared.sum() / denom


def masked_text_alignment_loss(pred, target, valid):
    if pred.shape != target.shape or pred.shape[:2] != valid.shape:
        raise ValueError("text alignment requires matching token positions and dimensions")
    weight = valid.unsqueeze(-1).float()
    squared_error = ((pred.float() - target.float()).square() * weight).sum()
    teacher_energy = (target.float().square() * weight).sum().detach()
    return squared_error / teacher_energy.clamp_min(1e-6)


def build_scheduler(optimizer, max_steps, warmup_steps, min_lr_ratio=0.1, *,
                    global_batch=0, warmup_samples=0, sample_offset=0,
                    step_offset=0, reference_batch=128):
    if warmup_samples > 0:
        if min_lr_ratio != 1 or min(global_batch, reference_batch) < 1 or min(sample_offset, step_offset) < 0:
            raise ValueError("Sample-based warmup requires a valid constant-after-warmup recipe")

        def sample_lr_lambda(step):
            # Preserve A's 128-example step+1 lookahead.
            samples = sample_offset + max(0, step - step_offset) * global_batch + reference_batch
            return 0.1 + 0.9 * min(samples / warmup_samples, 1.0)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sample_lr_lambda)

    def lr_lambda(step):
        if step < warmup_steps:
            warmup_scale = float(step + 1) / float(max(1, warmup_steps))
            return 0.1 + 0.9 * warmup_scale

        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _use_weight_decay(name, param):
    if param.ndim <= 1:
        return False
    lowered = name.lower()
    if lowered.endswith(".bias"):
        return False
    if "norm" in lowered or "layernorm" in lowered:
        return False
    return True


def build_param_group_optimizer(model, args):
    head_lr = args.lr if args.head_lr is None else args.head_lr
    head_wd = args.weight_decay if args.head_weight_decay is None else args.head_weight_decay
    vision_lr = args.dinov3_lr if args.vision_encoder_lr is None else args.vision_encoder_lr
    vision_wd = (
        args.dinov3_weight_decay
        if args.vision_encoder_weight_decay is None
        else args.vision_encoder_weight_decay
    )
    grouped = {
        ("vision_decay", vision_lr, vision_wd): [],
        ("vision_no_decay", vision_lr, 0.0): [],
        ("head_decay", head_lr, head_wd): [],
        ("head_no_decay", head_lr, 0.0): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_vision = name.startswith("vision_encoder.backbone")
        decay = _use_weight_decay(name, param)
        if is_vision and decay:
            key = ("vision_decay", vision_lr, vision_wd)
        elif is_vision:
            key = ("vision_no_decay", vision_lr, 0.0)
        elif decay:
            key = ("head_decay", head_lr, head_wd)
        else:
            key = ("head_no_decay", head_lr, 0.0)
        grouped[key].append(param)

    param_groups = []
    summary = []
    for (group_name, lr, weight_decay), params in grouped.items():
        if not params:
            continue
        count = sum(p.numel() for p in params)
        param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay, "name": group_name})
        summary.append({"name": group_name, "lr": lr, "weight_decay": weight_decay, "params": count})
    return AdamW(param_groups), summary


def reduce_mean(value, device, is_distributed, world_size):
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor.item()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["model_state_dict", "model", "state_dict"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("unsupported checkpoint format")


def load_interaction_from_init_checkpoint(model, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    source_state = _extract_state_dict(ckpt)
    source_state = {(k[7:] if k.startswith("module.") else k): v for k, v in source_state.items()}

    if model.config.interaction.backbone_type in {"spike_temporal", "spike_wta", "spike_text_only"}:
        from .temporal_fusion import load_common_interaction
        return load_common_interaction(model, source_state)

    target_state = model.state_dict()
    mapped = {}
    skipped_missing = 0
    skipped_shape = 0
    missing_keys = []
    shape_mismatch_keys = []

    prefix_pairs = [
        ("transformer.encoder.fusion_layers.", "vision_language_interaction.fusion_layers."),
        ("transformer.encoder.text_layers.", "vision_language_interaction.text_layers."),
    ]

    for src_key, tensor in source_state.items():
        tgt_key = None
        for src_prefix, tgt_prefix in prefix_pairs:
            if src_key.startswith(src_prefix):
                tgt_key = tgt_prefix + src_key[len(src_prefix):]
                break
        if tgt_key is None:
            continue
        if tgt_key not in target_state:
            skipped_missing += 1
            missing_keys.append((src_key, tgt_key))
            continue
        if target_state[tgt_key].shape != tensor.shape:
            skipped_shape += 1
            shape_mismatch_keys.append((src_key, tgt_key, tuple(tensor.shape), tuple(target_state[tgt_key].shape)))
            continue
        mapped[tgt_key] = tensor

    if not mapped:
        raise RuntimeError("no feature-enhancer parameters were mapped from checkpoint")

    target_state.update(mapped)
    missing, unexpected = model.load_state_dict(target_state, strict=False)
    return {
        "mapped": len(mapped),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "load_missing_after_update": len(missing),
        "load_unexpected_after_update": len(unexpected),
        "missing_keys": missing_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }


def load_text_projection_from_init_checkpoint(model, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    source_state = _extract_state_dict(ckpt)
    source_state = {(k[7:] if k.startswith("module.") else k): v for k, v in source_state.items()}

    target_state = model.state_dict()
    mapped = {}
    skipped_missing = 0
    skipped_shape = 0
    missing_keys = []
    shape_mismatch_keys = []

    for src_key, tensor in source_state.items():
        tgt_key = None
        if src_key.startswith("feat_map."):
            tgt_key = "text_encoder.text_projection." + src_key[len("feat_map."):]
        else:
            continue

        if tgt_key not in target_state:
            skipped_missing += 1
            missing_keys.append((src_key, tgt_key))
            continue
        if target_state[tgt_key].shape != tensor.shape:
            skipped_shape += 1
            shape_mismatch_keys.append((src_key, tgt_key, tuple(tensor.shape), tuple(target_state[tgt_key].shape)))
            continue
        mapped[tgt_key] = tensor

    if not mapped:
        raise RuntimeError("no text projection parameters were mapped from checkpoint")

    target_state.update(mapped)
    missing, unexpected = model.load_state_dict(target_state, strict=False)
    return {
        "mapped": len(mapped),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "load_missing_after_update": len(missing),
        "load_unexpected_after_update": len(unexpected),
        "missing_keys": missing_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }


def move_samples_to_device(samples, device):
    if isinstance(samples, dict):
        return {k: v.to(device, non_blocking=True) for k, v in samples.items()}
    return samples.to(device, non_blocking=True)


def train_model():
    args = parse_args()
    if getattr(args, "fusion_backbone_type", "hybrid") in {"spike_temporal", "spike_wta", "spike_text_only"}:
        if not getattr(args, "approve_spike_fusion_training", False):
            raise RuntimeError("New spike fusion training is held pending architecture review and explicit approval")
        if args.float32_matmul_precision != "highest" or args.precision != "fp32":
            raise ValueError("The new temporal fusion recipe requires highest/fp32 consistently")
        if args.fusion_backbone_type == "spike_temporal" and args.spike_calibration_batches < 1:
            raise ValueError("Use at least one real training-data calibration batch")
        if not args.require_feature_enhancer_preload or not args.text_projection_init:
            raise ValueError("Temporal fusion requires common GroundingDINO and calibrated text projection initialization")
    if args.head_lr is None:
        args.head_lr = args.lr
    if args.head_weight_decay is None:
        args.head_weight_decay = args.weight_decay
    if args.vision_encoder_lr is None:
        args.vision_encoder_lr = args.dinov3_lr
    if args.vision_encoder_weight_decay is None:
        args.vision_encoder_weight_decay = args.dinov3_weight_decay
    if args.vision_image_size is None:
        args.vision_image_size = 224 if args.vision_encoder_type == "sdtv3_19m" else args.expected_image_size
    if min(args.distill_feature_weight, args.distill_action_weight, args.distill_text_weight) < 0:
        raise ValueError("distillation weights must be non-negative")
    distillation_enabled = max(args.distill_feature_weight, args.distill_action_weight, args.distill_text_weight) > 0
    if args.precision == "bf16_amp":
        if not torch.cuda.is_available():
            raise ValueError("bf16_amp precision requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16_amp requested but current CUDA device does not report bf16 support")
    # Apply before construction/forward: frozen SmoothSpike caches a rotation
    # computed on its first forward under the active matrix multiplication policy.
    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    if args.fusion_backbone_type in {"spike_wta", "spike_text_only"}:
        torch.backends.cudnn.allow_tf32 = False
    print(f"float32_matmul_precision={torch.get_float32_matmul_precision()}", flush=True)

    if args.lr_schedule_steps is None:
        args.lr_schedule_steps = args.max_steps
    if args.lr_schedule_steps <= args.warmup_steps:
        raise ValueError(
            f"lr_schedule_steps={args.lr_schedule_steps} must be greater than warmup_steps={args.warmup_steps}"
        )
    set_seed(args.seed)

    is_distributed, rank, world_size, local_rank, device = setup_distributed()

    try:
        model = build_model_architecture(args)
        teacher = build_distillation_teacher(args) if distillation_enabled else None

        need_any_init_preload = (
            args.require_feature_enhancer_preload
            or args.require_text_proj_preload
            or args.load_text_projection_from_init
        )
        if args.pretrained_init_ckpt is None and need_any_init_preload:
            raise ValueError(
                "`--pretrained_init_ckpt` is required for requested preload options "
                "(feature-enhancer and/or text_proj)"
            )

        if args.pretrained_init_ckpt is not None:
            if args.require_feature_enhancer_preload:
                fe_report = load_interaction_from_init_checkpoint(model, args.pretrained_init_ckpt)
                if rank == 0:
                    print("vision-language interaction preload from init checkpoint:")
                    print(f"  ckpt: {args.pretrained_init_ckpt}")
                    print(f"  mapped={fe_report['mapped']}")
                    print(f"  skipped_missing={fe_report['skipped_missing']}")
                    print(f"  skipped_shape={fe_report['skipped_shape']}")
                    print("missing_keys:", fe_report["missing_keys"])
                    print("shape_mismatch_keys:", fe_report["shape_mismatch_keys"])

            if args.load_text_projection_from_init:
                text_proj_report = load_text_projection_from_init_checkpoint(model, args.pretrained_init_ckpt)
                if rank == 0:
                    print("text projection preload from init checkpoint:")
                    print(f"  ckpt: {args.pretrained_init_ckpt}")
                    print(f"  mapped={text_proj_report['mapped']}")
                    print(f"  skipped_missing={text_proj_report['skipped_missing']}")
                    print(f"  skipped_shape={text_proj_report['skipped_shape']}")
                    print("missing_keys:", text_proj_report["missing_keys"])
                    print("shape_mismatch_keys:", text_proj_report["shape_mismatch_keys"])
            elif args.require_text_proj_preload:
                raise ValueError(
                    "text projection preload is required, but `--no_load_text_projection_from_init` was set"
                )

        if args.text_projection_init:
            from .joint_scope import load_calibrated_projection
            projection_metadata = load_calibrated_projection(model, args)
            if rank == 0:
                print("common-pretraining calibrated projection:", json.dumps(projection_metadata), flush=True)
        if args.joint_scope_audit:
            from .joint_scope import check_preload, joint_scope
            if not args.require_feature_enhancer_preload or not args.load_text_projection_from_init:
                raise ValueError("Joint stage requires complete common GroundingDINO initialization")
            if args.text_encoder_type == "smoothspike_bert" and not args.text_projection_init:
                raise ValueError("SmoothSpike joint stage requires a calibrated projection")
            check_preload(model, fe_report)
            joint_groups = joint_scope(model, args)
            if (get_latest_checkpoint(args.checkpoint_dir, args.checkpoint_prefix) is not None
                    and args.resume_mode == "none"):
                raise RuntimeError("Refusing to restart or overwrite an existing joint run")
            if rank == 0:
                print("joint_trainable_parameter_groups:",
                      {name: sum(p.numel() for _, p in group) for name, group in joint_groups.items()},
                      flush=True)

        if args.freeze_backbones:
            freeze_backbones(model)

        if rank == 0:
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
            print(f"device={device}, distributed={is_distributed}, world_size={world_size}")
            print(f"vision_encoder_type={args.vision_encoder_type}")
            print(f"vision_model_path={args.vision_model_path or args.dinov3_path}")
            print(f"vision_pretrained_checkpoint={args.vision_pretrained_checkpoint}")
            print(f"teacher_checkpoint={args.teacher_checkpoint}")
            print(f"student_text_mask_version={model.config.text_mask_version}")
            print(f"student_sentence_mask={model.config.sentence_mask_version}, "
                  f"student_head_mask={model.config.head_mask_version}")
            print(f"student_cross_attention={model.config.interaction.cross_attention_type}, "
                  f"cross_timesteps={model.config.interaction.cross_timesteps}, "
                  f"cross_gradient_checkpointing={model.config.interaction.cross_gradient_checkpointing}")
            if teacher is not None:
                print(f"teacher_text_mask_version={teacher.config.text_mask_version}")
                print(f"teacher_sentence_mask={teacher.config.sentence_mask_version}, "
                      f"teacher_head_mask={teacher.config.head_mask_version}")
            print(
                f"distill_feature_weight={args.distill_feature_weight}, "
                f"distill_action_weight={args.distill_action_weight}, "
                f"distill_text_weight={args.distill_text_weight} (valid-token relative MSE)"
            )
            print(f"bert_path={args.bert_path}")
            print(
                f"text_encoder_type={args.text_encoder_type}, "
                f"text_model_path={args.text_model_path or args.bert_path}, "
                f"text_timesteps={args.text_timesteps}"
            )
            print(
                f"online_text_encoder=True, freeze_text_encoder={args.freeze_text_encoder}, "
                f"text_padding_length={args.text_padding_length}, text_layout_path={args.text_layout_path}"
            )
            print(f"pretrained_init_ckpt={args.pretrained_init_ckpt}")
            print(f"load_text_projection_from_init={args.load_text_projection_from_init}")
            print(f"max_steps={args.max_steps}, lr_schedule_steps={args.lr_schedule_steps}")
            print(
                f"precision={args.precision}, head_lr={args.head_lr}, vision_lr={args.vision_encoder_lr}, "
                f"head_wd={args.head_weight_decay}, vision_wd={args.vision_encoder_weight_decay}"
            )
            print(f"warmup_steps={args.warmup_steps}, min_lr_ratio={args.min_lr_ratio}")
            print(
                f"shuffle: episode_buffer={args.shuffle_buffer}, "
                f"within_episode={args.shuffle_steps_within_episode}"
            )
            print(f"trainable params: {len(trainable)}")
            print(f"frozen params: {len(frozen)}")
            print("first 30 trainable:", trainable[:30])

        if rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        if is_distributed:
            dist.barrier()

        global_step = 0
        ckpt = None
        latest_ckpt_path = get_latest_checkpoint(args.checkpoint_dir, args.checkpoint_prefix)
        if latest_ckpt_path is not None and args.resume_mode != "none":
            if rank == 0:
                print(f"resume from {latest_ckpt_path} with mode={args.resume_mode}")
            ckpt = torch.load(latest_ckpt_path, map_location="cpu")
            if model.config.interaction.backbone_type in {"spike_temporal", "spike_wta", "spike_text_only"}:
                saved = ckpt.get("model_config", {}).get("interaction", {}).get("backbone_type", "hybrid")
                if saved != model.config.interaction.backbone_type:
                    raise ValueError("Cannot resume a different fusion architecture")
                if saved in {"spike_wta", "spike_text_only"}:
                    from .wta_norms import check_wta_resume_config
                    check_wta_resume_config(model.config, ckpt["model_config"])
            missing, unexpected = model.load_state_dict(
                ckpt["model_state_dict"], strict=model.config.interaction.backbone_type in {"spike_temporal", "spike_wta", "spike_text_only"}
            )
            if rank == 0:
                print("resume missing keys:", len(missing))
                print("resume unexpected keys:", len(unexpected))
            if args.resume_mode == "all":
                global_step = int(ckpt.get("global_step", 0))
            else:
                global_step = 0
                if rank == 0:
                    print("optimizer/scheduler reset due to resume_mode=model")
        elif args.resume_mode != "none":
            raise FileNotFoundError("Explicit resume requested but no matching checkpoint exists")
        elif rank == 0:
            print("no checkpoint resumed, train from current initialization")

        model.to(device)
        if teacher is not None:
            teacher.to(device)
            teacher.eval()
        if is_distributed:
            if device.type == "cuda":
                model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            else:
                model = DDP(model, find_unused_parameters=True)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer, optimizer_summary = build_param_group_optimizer(unwrap_model(model), args)
        if rank == 0:
            print("optimizer param groups:", optimizer_summary)
        scheduler = build_scheduler(
            optimizer,
            max_steps=args.lr_schedule_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            global_batch=args.batch_size * args.grad_accum_steps * world_size,
            warmup_samples=args.lr_warmup_samples,
            sample_offset=args.lr_sample_offset,
            step_offset=args.lr_step_offset,
            reference_batch=args.lr_reference_batch,
        )

        if ckpt is not None and args.resume_mode == "all":
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if hasattr(optimizer, "restore_ema"):
                optimizer.restore_ema(ckpt)

        model_to_load = unwrap_model(model)
        if args.joint_scope_audit:
            joint_groups = joint_scope(model_to_load, args)

        dataset = LiberoRLDSDataset(
            dataset_dir=args.dataset_dir,
            LOCAL_DINOV3_PATH=args.dinov3_path,
            rank=rank,
            world_size=world_size,
            chunk_size=args.chunk_size,
            split=args.dataset_split,
            shuffle_buffer=args.shuffle_buffer,
            shuffle_steps_within_episode=args.shuffle_steps_within_episode,
            step_mix_buffer_size=args.step_mix_buffer_size,
            seed=args.seed,
            local_files_only=not args.allow_hf_download,
            expected_image_size=args.expected_image_size,
            vision_encoder_type=args.vision_encoder_type,
            vision_image_size=args.vision_image_size,
            enable_dinov3_teacher=distillation_enabled,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=None,
            collate_fn=vla_collate_fn,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        data_iter = iter(dataloader)

        if model_to_load.config.interaction.backbone_type == "spike_temporal":
            from .temporal_fusion import calibrate_training_iterator
            data_iter, calibration_report = calibrate_training_iterator(
                model_to_load, data_iter, device, args.spike_calibration_batches,
                already_calibrated=ckpt is not None,
            )
            if rank == 0:
                with open(os.path.join(args.checkpoint_dir, "spike_calibration.json"), "w", encoding="utf-8") as handle:
                    json.dump(calibration_report, handle, indent=2)

        model.train()
        if args.freeze_backbones:
            model_to_load.dinov3.eval()

        if rank == 0:
            per_gpu_effective_bs = args.batch_size * args.grad_accum_steps
            global_effective_bs = per_gpu_effective_bs * world_size
            print(f"batch_size(per_gpu)={args.batch_size}, grad_accum_steps={args.grad_accum_steps}")
            print(f"effective_batch_size(per_gpu)={per_gpu_effective_bs}")
            print(f"effective_batch_size(global)={global_effective_bs}")
            pbar = tqdm(total=args.max_steps, initial=global_step, desc="training")
        else:
            pbar = None

        loss_window = []
        action_loss_window = []
        feature_distill_window = []
        action_distill_window = []
        text_distill_window = []
        while global_step < args.max_steps:
            if args.action_head_type == "progressive_full_wta":
                head = unwrap_model(model).action_head
                alpha = head.set_progress(global_step, args.lr_schedule_steps)
                if rank == 0 and global_step % args.log_freq == 0:
                    print(f"transition step={global_step} alpha={alpha:.8f} head_T=8 upstream_T=4", flush=True)
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            action_loss_accum = 0.0
            feature_distill_accum = 0.0
            action_distill_accum = 0.0
            text_distill_accum = 0.0

            for micro_step in range(args.grad_accum_steps):
                try:
                    samples, instructions, states, gt_actions, action_chunk_masks = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    samples, instructions, states, gt_actions, action_chunk_masks = next(data_iter)

                samples = move_samples_to_device(samples, device)
                states = states.to(device, non_blocking=True)
                gt_actions = gt_actions.to(device, non_blocking=True)
                action_chunk_masks = action_chunk_masks.to(device, non_blocking=True)

                # DDP only needs to synchronize accumulated parameter gradients
                # on the last microbatch; custom synchronized BN stays active.
                sync_context = (model.no_sync() if is_distributed
                                and micro_step + 1 < args.grad_accum_steps else nullcontext())
                with sync_context:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=args.precision == "bf16_amp",
                    ):
                        if args.distill_text_weight > 0:
                            pred_actions, student_features = model(
                                instructions, samples, states, return_distillation_features=True
                            )
                            student_visual_tokens = student_features["visual"]
                        elif distillation_enabled:
                            pred_actions, student_visual_tokens = model(
                                instructions, samples, states, return_visual_tokens=True
                            )
                        else:
                            pred_actions = model(instructions, samples, states)
                        if pred_actions.shape != gt_actions.shape:
                            raise ValueError(
                                f"pred_actions.shape={pred_actions.shape}, gt_actions.shape={gt_actions.shape} mismatch"
                            )

                        action_loss = masked_l1_loss(pred_actions, gt_actions, action_chunk_masks)
                        feature_distill_loss = pred_actions.new_zeros(())
                        action_distill_loss = pred_actions.new_zeros(())
                        text_distill_loss = pred_actions.new_zeros(())
                        if teacher is not None:
                            teacher_samples = {"pixel_values": samples["teacher_pixel_values"]}
                            with torch.no_grad():
                                if args.distill_text_weight > 0:
                                    teacher_actions, teacher_features = teacher(
                                        instructions, teacher_samples, states,
                                        return_distillation_features=True,
                                    )
                                    teacher_visual_tokens = teacher_features["visual"]
                                else:
                                    teacher_actions, teacher_visual_tokens = teacher(
                                        instructions, teacher_samples, states, return_visual_tokens=True,
                                    )
                            if args.distill_text_weight > 0:
                                if not torch.equal(student_features["text_valid"], teacher_features["text_valid"]):
                                    raise ValueError("student/teacher token layouts differ")
                                text_distill_loss = masked_text_alignment_loss(
                                    student_features["text"], teacher_features["text"],
                                    teacher_features["text_valid"],
                                )
                            if student_visual_tokens.shape != teacher_visual_tokens.shape:
                                raise ValueError(
                                    "student/teacher visual token mismatch: "
                                    f"{student_visual_tokens.shape} vs {teacher_visual_tokens.shape}"
                                )
                            feature_distill_loss = F.mse_loss(
                                student_visual_tokens.float(), teacher_visual_tokens.float()
                            )
                            action_distill_loss = masked_mse_loss(
                                pred_actions.float(), teacher_actions.float(), action_chunk_masks
                            )
                        loss = (
                            action_loss
                            + args.distill_feature_weight * feature_distill_loss
                            + args.distill_action_weight * action_distill_loss
                            + args.distill_text_weight * text_distill_loss
                        )
                    (loss / args.grad_accum_steps).backward()
                loss_accum += loss.detach().item()
                action_loss_accum += action_loss.detach().item()
                feature_distill_accum += feature_distill_loss.detach().item()
                action_distill_accum += action_distill_loss.detach().item()
                text_distill_accum += text_distill_loss.detach().item()

            if args.joint_scope_audit and global_step == 0:
                from .joint_scope import gradient_report, representative_snapshots, record_first_update
                first_gradients = gradient_report(joint_groups)
                first_parameters = representative_snapshots(joint_groups)
            torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=args.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            if args.joint_scope_audit and global_step == 0:
                record_first_update(unwrap_model(model), args, joint_groups, first_parameters, first_gradients, rank)
                del first_parameters

            local_loss = loss_accum / args.grad_accum_steps
            global_loss = reduce_mean(local_loss, device, is_distributed, world_size)
            global_action_loss = reduce_mean(
                action_loss_accum / args.grad_accum_steps, device, is_distributed, world_size
            )
            global_feature_distill = reduce_mean(
                feature_distill_accum / args.grad_accum_steps, device, is_distributed, world_size
            )
            global_action_distill = reduce_mean(
                action_distill_accum / args.grad_accum_steps, device, is_distributed, world_size
            )

            global_text_distill = reduce_mean(
                text_distill_accum / args.grad_accum_steps, device, is_distributed, world_size
            )

            loss_window.append(global_loss)
            if len(loss_window) > args.log_freq:
                loss_window.pop(0)
            action_loss_window.append(global_action_loss)
            feature_distill_window.append(global_feature_distill)
            action_distill_window.append(global_action_distill)
            text_distill_window.append(global_text_distill)
            for window in (action_loss_window, feature_distill_window, action_distill_window, text_distill_window):
                if len(window) > args.log_freq:
                    window.pop(0)

            global_step += 1
            if args.action_head_type == "progressive_full_wta":
                unwrap_model(model).action_head.set_progress(global_step, args.lr_schedule_steps)

            if rank == 0:
                avg_window_loss = sum(loss_window) / len(loss_window)
                pbar.update(1)
                if global_step % args.log_freq == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{global_loss:.5f}",
                            "avg": f"{avg_window_loss:.5f}",
                            "act": f"{sum(action_loss_window) / len(action_loss_window):.5f}",
                            "feat_kd": f"{sum(feature_distill_window) / len(feature_distill_window):.5f}",
                            "act_kd": f"{sum(action_distill_window) / len(action_distill_window):.5f}",
                            "text_kd": f"{sum(text_distill_window) / len(text_distill_window):.5f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        }
                    )

            should_save = global_step in args.extra_save_steps or global_step % args.save_steps == 0 or (
                args.save_final and global_step == args.max_steps
            )
            if rank == 0 and should_save:
                wta_metadata = None
                if model_to_load.config.interaction.backbone_type in {"spike_wta", "spike_text_only"}:
                    from .wta_norms import normalization_metadata
                    from ..models.components.wta_operators import verify_running_norms
                    verify_running_norms(model_to_load)
                    wta_metadata = normalization_metadata()
                save_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_{global_step}.pth")
                if os.path.exists(save_path):
                    raise RuntimeError(f"Refusing to overwrite checkpoint: {save_path}")
                temporary_path = save_path + ".tmp"
                torch.save(
                    {
                        "training_scope": "joint_except_frozen_text_backbone" if args.joint_scope_audit else "standard",
                        "global_step": global_step,
                        "model_state_dict": unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": global_loss,
                        "args": vars(args),
                        "numerics": {
                            "float32_matmul_precision": torch.get_float32_matmul_precision(),
                            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        },
                        "model_config": unwrap_model(model).config.to_dict(),
                        "wta_normalization": wta_metadata,
                        "resources": {"world_size": world_size,
                                      "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
                                      if device.type == "cuda" else None},
                    },
                    temporary_path,
                )
                os.replace(temporary_path, save_path)
                print(f"saved: {save_path}", flush=True)

        if pbar is not None:
            pbar.close()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    train_model()
