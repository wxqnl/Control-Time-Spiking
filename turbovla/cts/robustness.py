"""Deployment-stress hooks for evaluation (E0b/E4). Never used in training.

* numerics "tf32": TF32 matmul/cuDNN after the fp32 checks, the setting that
  drops the base model from 90.9% to 60.2% on LIBERO.
* weight fake-quantization: symmetric per-output-channel int-k weights for
  every Linear/Conv of the policy (the frozen text backbone is left as is).
* observation perturbations for a window of control queries: blackout
  (zero image), Gaussian pixel noise or a brightness factor, on the primary,
  wrist or both views (start 0 and a large count perturb the whole episode).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def apply_numerics_stress(policy, mode: str) -> dict:
    if mode in {"", "none"}:
        return {"numerics_stress": "none"}
    if mode != "tf32":
        raise ValueError(f"Unknown numerics stress {mode!r}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    policy.float32_matmul_precision = "high"
    return {"numerics_stress": "tf32", "float32_matmul_precision": "high", "cudnn_allow_tf32": True}


@torch.no_grad()
def fake_quantize_weights(model: nn.Module, bits: int) -> dict:
    if bits <= 0:
        return {"weight_bits": None}
    if bits < 2 or bits > 16:
        raise ValueError("weight bits must be in [2, 16]")
    levels = 2 ** (bits - 1) - 1
    count, elements, error = 0, 0, 0.0
    for name, module in model.named_modules():
        if name.startswith("text_encoder.bert") or not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            continue
        weight = module.weight
        flat = weight.reshape(weight.shape[0], -1)
        scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / levels
        quantized = (flat / scale).round().clamp(-levels, levels) * scale
        error += float((quantized - flat).square().sum())
        elements += flat.numel()
        weight.copy_(quantized.reshape_as(weight))
        count += 1
    return {"weight_bits": bits, "quantized_modules": count,
            "weight_rmse": (error / max(elements, 1)) ** 0.5}


def perturb_view(image: np.ndarray, mode: str, cfg, rng: np.random.Generator) -> np.ndarray:
    if mode == "blackout":
        return np.zeros_like(image)
    if mode == "brightness":
        return np.clip(image.astype(np.float32) * cfg.perturb_brightness, 0, 255).astype(np.uint8)
    if mode == "noise":
        noisy = image.astype(np.float32) + rng.normal(0.0, cfg.perturb_noise_std, size=image.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown perturbation {mode!r}")


def perturb_observation(primary, wrist, query_index: int, cfg, rng):
    """Apply cfg.perturb_mode to the views for queries [start, start + count)."""
    mode = getattr(cfg, "perturb_mode", "")
    if not mode:
        return primary, wrist, False
    start, count = int(cfg.perturb_start_query), int(cfg.perturb_queries)
    if not start <= query_index < start + count:
        return primary, wrist, False
    views = cfg.perturb_views
    if views not in {"primary", "wrist", "both"}:
        raise ValueError("perturb_views must be primary, wrist or both")
    if views in {"primary", "both"}:
        primary = perturb_view(primary, mode, cfg, rng)
    if views in {"wrist", "both"}:
        wrist = perturb_view(wrist, mode, cfg, rng)
    return primary, wrist, True
