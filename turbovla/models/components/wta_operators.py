"""Exact event operators for WTA fusion (our adaptation, not an official model).

Method sources:
- WTA-SpikingTransformer: https://arxiv.org/abs/2604.11321, sections 3.1-3.3.
- Bipolar Self-attention, NeurIPS 2025, equations 9-11 (signed firing).
Forward uses unit signed events and binary routing, not NI-LIF integer training.
Softmax exists only in the explicitly declared WTA surrogate BACKWARD.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_reduce as differentiable_all_reduce

from .spike_fusion import FunctionalPLIF, _ATanSpike

_RECOMPUTING = ContextVar("wta_norm_recomputing", default=False)


@contextmanager
def no_running_stat_updates():
    token = _RECOMPUTING.set(True)
    try:
        yield
    finally:
        _RECOMPUTING.reset(token)


def checkpoint_contexts():
    return nullcontext(), no_running_stat_updates()


class BipolarPLIF(FunctionalPLIF):
    """Learned leaky integration, symmetric thresholds, unit signed firing.

    Uses the existing PLIF charging equation, not the paper's exact LIF
    discretization. A positive OR negative event hard-resets the membrane.
    State belongs to the caller and is never retained between observations.
    """

    def forward(self, current, membrane=None, threshold_scale=1.0, valid=None):
        if membrane is None:
            membrane = torch.zeros_like(current)
        voltage = membrane + self.decay_logit.sigmoid() * (current - membrane)
        threshold = self.log_threshold.exp() * threshold_scale
        normalized = voltage / threshold
        spike = _ATanSpike.apply(normalized - 1.0) - _ATanSpike.apply(-normalized - 1.0)
        if valid is not None:
            spike = spike.masked_fill(~valid, 0.0)
            voltage = voltage.masked_fill(~valid, 0.0)
        return spike, voltage * (1.0 - spike.detach().abs())


class _MultiWinner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, allowed, topk, temperature):
        # Stable ties are resolved by source-token order, independent of batch
        # layout. Invalid and silent sources cannot win, even if all scores tie.
        ranked = scores.masked_fill(~allowed, -torch.inf)
        indices = torch.argsort(ranked, dim=-1, descending=True, stable=True)[..., :topk]
        winners = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
        winners.masked_fill_(~allowed, 0.0)
        ctx.temperature = temperature
        ctx.save_for_backward(scores, allowed)
        return winners

    @staticmethod
    def backward(ctx, grad_output):
        scores, allowed = ctx.saved_tensors
        live_rows = allowed.any(-1, keepdim=True)
        logits = (scores / ctx.temperature).masked_fill(~allowed, -torch.inf)
        # No NaNs for fully masked rows. This is a surrogate Jacobian, NOT the
        # derivative of top-k; use the full softmax Jacobian (including cross
        # terms), not a diagonal or identity straight-through approximation.
        logits = torch.where(live_rows, logits, torch.zeros_like(logits))
        probability = logits.softmax(-1).masked_fill(~allowed, 0.0)
        centered = grad_output - (grad_output * probability).sum(-1, keepdim=True)
        return probability * centered / ctx.temperature, None, None, None


class MultiWinnerAttention(nn.Module):
    """Binary top-k routing over signed QK scores; Q, K and V stay events."""

    def __init__(self, topk, head_dim, surrogate_temperature=1.0):
        super().__init__()
        if topk < 1 or surrogate_temperature <= 0:
            raise ValueError("WTA requires positive top-k and surrogate temperature")
        self.topk = int(topk)
        # Positive scaling does not affect hard forward ranking. It only sets
        # the softmax-surrogate temperature during backward.
        self.temperature = math.sqrt(head_dim) * float(surrogate_temperature)

    def forward(self, scores, allowed, query_spikes, key_spikes):
        if allowed.dtype != torch.bool or scores.ndim != 4:
            raise ValueError("WTA requires [B,H,Nq,Nk] scores and a boolean mask")
        if allowed.ndim == 3:
            allowed = allowed[:, None]
        allowed = allowed.expand_as(scores)
        # At least one event in a query/key vector is required. This activity
        # guard is our cross-modal adaptation, not a soft attention fallback.
        allowed = allowed & query_spikes.detach().ne(0).any(-1, keepdim=True)
        allowed = allowed & key_spikes.detach().ne(0).any(-1).unsqueeze(-2)
        return _MultiWinner.apply(scores, allowed, min(self.topk, scores.shape[-1]), self.temperature)


class SpikeDotProductAttention(nn.Module):
    """Unscaled signed-spike QK routing without top-k or softmax.

    Preserves WTA's padding/sentence masks and silent-query/key guards.
    Autograd differentiates the dot products directly; the surrounding PLIF
    surrogate is unchanged. There is no WTA surrogate in this branch.
    """

    def forward(self, scores, allowed, query_spikes, key_spikes):
        if allowed.dtype != torch.bool or scores.ndim != 4:
            raise ValueError("Spike dot-product attention requires 4D scores and a boolean mask")
        if allowed.ndim == 3:
            allowed = allowed[:, None]
        allowed = allowed.expand_as(scores)
        allowed = allowed & query_spikes.detach().ne(0).any(-1, keepdim=True)
        allowed = allowed & key_spikes.detach().ne(0).any(-1).unsqueeze(-2)
        return scores.masked_fill(~allowed, 0.0)


class MaskedTemporalBatchNorm(nn.Module):
    """Mask-aware, time-indexed BN for directly trained spiking currents.

    Each SNN timestep has its own running mean/variance; learnable channel
    affine parameters are shared across T. Statistics include valid tokens
    from all ranks, with differentiable reduction for correct synchronized-BN
    gradients. Training output uses batch moments; eval uses saved moments.
    This is tdBN-inspired normalization, not a reproduction of PTBN/tdLN.

    Initial affine gain=1 pairs with PLIF threshold=0.5. Both are trainable.
    No dynamic LayerNorm and no calibration on benchmark/eval observations.
    """

    def __init__(self, channels, timesteps, momentum=0.1, eps=1e-5, sync=True):
        super().__init__()
        self.channels, self.timesteps = channels, timesteps
        self.momentum, self.eps, self.sync = float(momentum), float(eps), bool(sync)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(timesteps, channels))
        self.register_buffer("running_var", torch.ones(timesteps, channels))
        self.register_buffer("num_batches_tracked", torch.zeros(timesteps, dtype=torch.long))
        self.register_buffer("observed_tokens", torch.zeros(timesteps, dtype=torch.float64))
        self.calibrating = False
        # CTS ablation: one set of statistics for all neuron steps (index 0).
        self.time_shared = False

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()
        self.observed_tokens.zero_()

    @torch.no_grad()
    def _track(self, step, mean, variance, count):
        # Keep guards on device: item() here used to serialize every SNN step.
        enough = count >= 2
        first = self.num_batches_tracked[step] == 0
        unbiased = variance * count / (count - 1).clamp_min(1)
        next_mean = torch.where(first, mean, torch.lerp(self.running_mean[step], mean, self.momentum))
        next_var = torch.where(first, unbiased, torch.lerp(self.running_var[step], unbiased, self.momentum))
        if self.calibrating:
            previous_count = self.observed_tokens[step]
            # Count-weighted parallel variance merge, not an unweighted
            # average of variable-length text batches.
            total = (previous_count + count).clamp_min(2)
            delta = mean - self.running_mean[step]
            merged_mean = self.running_mean[step] + delta * (count / total)
            m2 = (self.running_var[step] * (previous_count - 1)
                  + variance * count + delta.square() * previous_count * count / total)
            next_mean = torch.where(previous_count > 0, merged_mean, next_mean)
            next_var = torch.where(previous_count > 0, m2 / (total - 1), next_var)
        self.running_mean[step].copy_(torch.where(enough, next_mean, self.running_mean[step]))
        self.running_var[step].copy_(torch.where(enough, next_var, self.running_var[step]))
        self.num_batches_tracked[step].add_(enough)
        self.observed_tokens[step].add_(torch.where(enough, count, 0))

    def forward(self, current, valid=None, time_index=0):
        if current.shape[-1] != self.channels or not 0 <= time_index < self.timesteps:
            raise ValueError("Invalid current shape or SNN timestep for normalization")
        if self.time_shared:
            time_index = 0
        x = current.float()
        if valid is None:
            valid = torch.ones((*x.shape[:-1], 1), device=x.device, dtype=torch.bool)
        if valid.dtype != torch.bool:
            raise ValueError("Normalization validity mask must be boolean")
        valid = valid.expand(*x.shape[:-1], 1)
        x = x.masked_fill(~valid, 0)
        if self.training or self.calibrating:
            dimensions = tuple(range(x.ndim - 1))
            count = valid.sum().to(dtype=x.dtype)
            stats = torch.cat((x.sum(dimensions), x.square().sum(dimensions), count.reshape(1)))
            if self.sync and dist.is_available() and dist.is_initialized():
                if stats.requires_grad:
                    stats = differentiable_all_reduce(stats)
                else:
                    dist.all_reduce(stats)
            count = stats[-1]
            # The reduced count is shared by all ranks. Invalid/tiny batches
            # still return zero with zero gradients and leave buffers unchanged.
            # A tensor guard avoids a CPU/GPU barrier per normalization call.
            denominator = count.clamp_min(2)
            mean = stats[:self.channels] / denominator
            variance = (stats[self.channels:2*self.channels] / denominator - mean.square()).clamp_min(0)
            valid = valid & (count >= 2)
            if not _RECOMPUTING.get():
                self._track(time_index, mean.detach(), variance.detach(), count.detach())
        else:
            if self.num_batches_tracked[time_index].item() == 0:
                raise RuntimeError("WTA BN has no training statistics; eval-data calibration is forbidden")
            mean = self.running_mean[time_index]
            variance = self.running_var[time_index]
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight + self.bias).masked_fill(~valid, 0)


def running_norms(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, MaskedTemporalBatchNorm)]


def verify_running_norms(model):
    norms = running_norms(model)
    if not norms:
        raise RuntimeError("No WTA running-normalization modules found")
    for name, module in norms:
        steps = slice(0, 1) if getattr(module, "time_shared", False) else slice(None)
        if (module.num_batches_tracked[steps] <= 0).any() or (module.observed_tokens[steps] < 2).any():
            raise RuntimeError(f"Missing training statistics at {name}")
        if (not torch.isfinite(module.running_mean).all()
                or not torch.isfinite(module.running_var).all()
                or (module.running_var < 0).any()):
            raise RuntimeError(f"Invalid running statistics at {name}")
    return {name: {"batches_per_timestep": module.num_batches_tracked.tolist(),
                   "tokens_per_timestep": module.observed_tokens.tolist()}
            for name, module in norms}
