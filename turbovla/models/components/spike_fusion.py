"""Bidirectional SDSA-style cross-attention; version cross_sdsa_plif_v1.

Method basis: Meta-SpikeFormer, https://arxiv.org/abs/2404.03663, Appendix A.
This is our cross-modal adaptation, not an official pretrained cross-attention.
All six projection inputs and Q/K/V are binary spikes. Softmax is absent.
Layer normalization, membrane dynamics, length-dependent thresholds, residuals,
and the temporal output readout remain continuous. Dense PyTorch simulation is
not a sparse hardware kernel and does not establish a measured energy saving.
"""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .fusion import BiMultiHeadAttention


class _ATanSpike(torch.autograd.Function):
    """Exact binary forward, alpha=2 arctangent surrogate in backward."""

    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (voltage,) = ctx.saved_tensors
        return grad_output / (1 + (math.pi * voltage).square())


class FunctionalPLIF(nn.Module):
    """Learned time constant and positive threshold; state is explicitly local.

    u[t] = u[t-1] + sigmoid(w) * (I[t] - u[t-1]).
    A spike hard-resets u to zero. Reset is detached; the other temporal
    gradient paths are retained. A caller creates fresh state for each input,
    so training, eval, exceptions and activation recomputation cannot leak state.
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.decay_logit = nn.Parameter(torch.zeros(()))  # tau = 2 initially
        self.log_threshold = nn.Parameter(torch.tensor(math.log(threshold)))

    def forward(self, current, membrane=None, threshold_scale=1.0, valid=None):
        if membrane is None:
            membrane = torch.zeros_like(current)
        voltage = membrane + self.decay_logit.sigmoid() * (current - membrane)
        threshold = self.log_threshold.exp() * threshold_scale
        spike = _ATanSpike.apply(voltage / threshold - 1.0)
        if valid is not None:
            spike = spike.masked_fill(~valid, 0.0)
            voltage = voltage.masked_fill(~valid, 0.0)
        return spike, voltage * (1.0 - spike.detach())


class SpikeBiMultiHeadAttention(BiMultiHeadAttention):
    def __init__(self, *args, timesteps=4, gradient_checkpointing=True, **kwargs):
        super().__init__(*args, **kwargs)
        if timesteps < 1 or self.dropout != 0:
            raise ValueError("spike cross-attention needs positive T and zero attention dropout")
        self.timesteps = timesteps
        self.gradient_checkpointing = gradient_checkpointing
        self.spike_nodes = nn.ModuleDict({
            name: FunctionalPLIF()
            for name in ("input_v", "input_l", "query_v", "key_l", "value_v", "value_l")
        })
        # Different modality lengths otherwise drive the reverse path into
        # saturation. This initialization is equivalent to a scale 64/(d*N),
        # absorbed into an independently learned threshold in each direction.
        self.spike_nodes["context_v"] = FunctionalPLIF(self.head_dim / 64.0)
        self.spike_nodes["context_l"] = FunctionalPLIF(self.head_dim / 64.0)

    @staticmethod
    def _valid(tokens, padding):
        if padding is None:
            return torch.ones((*tokens.shape[:2], 1), device=tokens.device, dtype=torch.bool)
        if padding.dtype != torch.bool or padding.shape != tokens.shape[:2]:
            raise ValueError("padding mask must be bool [B,N], where True means excluded")
        return ~padding.unsqueeze(-1)

    def _forward_spike(self, v, l, attention_mask_v=None, attention_mask_l=None,
                       initial_state=None, return_state=False):
        if v.ndim != 3 or l.ndim != 3 or v.shape[0] != l.shape[0]:
            raise ValueError("vision and text must be [B,N,C] with the same batch size")
        bsz, nv, _ = v.shape
        nl = l.shape[1]
        valid_v, valid_l = self._valid(v, attention_mask_v), self._valid(l, attention_mask_l)
        v, l = v.masked_fill(~valid_v, 0.0), l.masked_fill(~valid_l, 0.0)
        source_count_v = valid_v.sum(1, keepdim=True).clamp_min(1).to(v.dtype)
        source_count_l = valid_l.sum(1, keepdim=True).clamp_min(1).to(l.dtype)
        # No temporal state is kept in module attributes or checkpoint buffers.
        state = {name: None for name in self.spike_nodes}
        if initial_state is not None:
            # CTS: membranes carried from the previous control query.
            from turbovla.cts.state import wrap_carried
            state.update(wrap_carried(initial_state))
        output_v, output_l = None, None

        def fire(name, current, valid, scale=1.0):
            spikes, state[name] = self.spike_nodes[name](current, state[name], scale, valid)
            return spikes

        for _ in range(self.timesteps):
            input_v = fire("input_v", v, valid_v)
            input_l = fire("input_l", l, valid_l)
            # Token-local, non-affine normalization: no padding/batch running
            # statistics. These continuous operations are explicitly disclosed.
            q = fire("query_v", nn.functional.layer_norm(self.v_proj(input_v), (self.embed_dim,)), valid_v)
            k = fire("key_l", nn.functional.layer_norm(self.l_proj(input_l), (self.embed_dim,)), valid_l)
            vv = fire("value_v", nn.functional.layer_norm(self.values_v_proj(input_v), (self.embed_dim,)), valid_v)
            vl = fire("value_l", nn.functional.layer_norm(self.values_l_proj(input_l), (self.embed_dim,)), valid_l)
            q, k = self._shape(q, nv, bsz), self._shape(k, nl, bsz)
            vv, vl = self._shape(vv, nv, bsz), self._shape(vl, nl, bsz)
            # Shared binary Q/K interaction, preserving individual cross-modal
            # token routing. (QK^T)V is cheaper than Q(K^TV) for short text.
            counts = q @ k.transpose(-1, -2)
            context_v = (counts @ vl).transpose(1, 2).reshape(bsz, nv, self.embed_dim)
            context_l = (counts.transpose(-1, -2) @ vv).transpose(1, 2).reshape(bsz, nl, self.embed_dim)
            sv = fire("context_v", context_v, valid_v, source_count_l)
            sl = fire("context_l", context_l, valid_l, source_count_v)
            # Project each binary timestep BEFORE temporal averaging.
            dv, dl = self.out_v_proj(sv), self.out_l_proj(sl)
            output_v = dv if output_v is None else output_v + dv
            output_l = dl if output_l is None else output_l + dl
        keep_v = valid_v & valid_l.any(1, keepdim=True)
        keep_l = valid_l & valid_v.any(1, keepdim=True)
        outputs = (
            (output_v / self.timesteps).masked_fill(~keep_v, 0.0),
            (output_l / self.timesteps).masked_fill(~keep_l, 0.0),
        )
        return (*outputs, state) if return_state else outputs

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None,
                initial_state=None, return_state=False):
        if self.training and torch.is_grad_enabled() and self.gradient_checkpointing:
            return checkpoint(
                self._forward_spike, v, l, attention_mask_v, attention_mask_l,
                initial_state, return_state, use_reentrant=False,
            )
        return self._forward_spike(v, l, attention_mask_v, attention_mask_l, initial_state, return_state)
