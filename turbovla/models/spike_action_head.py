"""AA: T=4 spike-driven ACT action head, built on A's unchanged fusion output.

All learned projections consume binary/signed unit events. Self/cross attention
uses binary Q/K/V coincidence counts without softmax; FFNs and state/readout
projections use signed PLIF events. Masked temporal BN, membrane residuals and
the final tanh(mean_T(linear(events))) action readout remain continuous.
This is our directly trained adaptation, not an official pretrained action head
or a sparse hardware implementation. State is local to one observation.
Optional state-repair settings add a signed population-event residual and
post-coincidence state weighting. Defaults preserve the original AA path.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .action_head import StateProjection
from .components.spike_fusion import FunctionalPLIF, _ATanSpike
from .components.wta_operators import BipolarPLIF, MaskedTemporalBatchNorm, checkpoint_contexts


def _norm(channels, interaction):
    return MaskedTemporalBatchNorm(
        channels, interaction.cross_timesteps,
        momentum=interaction.spike_norm_momentum, eps=interaction.spike_norm_eps,
        sync=interaction.spike_norm_sync,
    )


class PhasePopulationSpike(nn.Module):
    """Fixed staggered thresholds; every emitted value is -1, 0, or +1.

    K thresholds per coordinate and timestep form T*K distinct boundaries.
    Subtracting the zero-state event code centers the signed events without
    feeding a continuous current into the learned population projection.
    This is an input event encoder, not an analog learned projection or an
    extra rollout timestep. The existing T-step PLIF head remains intact.
    """
    def __init__(self, bins, timesteps, limit):
        super().__init__()
        self.bins, self.limit = bins, float(limit)
        grid = (torch.arange(bins * timesteps, dtype=torch.float32) + 0.5)
        grid = grid * (2 * limit / (bins * timesteps)) - limit
        self.register_buffer("centers", grid.reshape(bins, timesteps).T.contiguous(), persistent=False)

    def forward(self, current, time_index):
        centers = self.centers[time_index]
        positive = _ATanSpike.apply(current[..., None] - centers)
        events = positive - (centers <= 0).to(current.dtype)
        return events.flatten(-2)


class SpikeStateProjection(nn.Module):
    def __init__(self, config, hidden_dim, interaction):
        super().__init__()
        self.num_tokens, self.hidden_dim = config.num_state_tokens, hidden_dim
        self.linear1 = nn.Linear(config.state_dim, config.state_hidden_dim)
        self.linear2 = nn.Linear(config.state_hidden_dim, config.num_state_tokens * hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, config.num_state_tokens, hidden_dim))
        self.norms = nn.ModuleDict({
            "input": _norm(config.state_dim, interaction),
            "hidden": _norm(config.state_hidden_dim, interaction),
            "output": _norm(hidden_dim, interaction),
        })
        self.spike_nodes = nn.ModuleDict({name: BipolarPLIF() for name in ("input", "hidden")})
        self.population_encoder = self.population_projection = None
        if config.spike_state_population:
            self.population_encoder = PhasePopulationSpike(
                config.spike_state_population, config.spike_timesteps,
                config.spike_state_population_range,
            )
            self.population_projection = nn.Linear(
                config.state_dim * config.spike_state_population,
                config.state_hidden_dim, bias=False,
            )
            nn.init.zeros_(self.population_projection.weight)

    @torch.no_grad()
    def initialize_population_residual(self, scale=0.25):
        if self.population_projection is None:
            raise ValueError("Population encoding is disabled")
        encoder = self.population_encoder
        # Start as a small higher-resolution approximation to the existing
        # first-layer current. Individual event weights remain independently
        # trainable; the coarse PLIF path and all its weights are preserved.
        self.population_projection.weight.copy_(
            self.linear1.weight.repeat_interleave(encoder.bins, dim=1)
            * (2 * encoder.limit / encoder.bins) * scale
        )

    def forward_step(self, current, state, t):
        x = current[:, None, :]
        for name, linear in (("input", self.linear1), ("hidden", self.linear2)):
            normalized = self.norms[name](x, time_index=t)
            spikes, state[name] = self.spike_nodes[name](
                normalized, state.get(name)
            )
            x = linear(spikes)
            if name == "input" and self.population_projection is not None:
                x = x + self.population_projection(self.population_encoder(normalized, t))
        x = x.reshape(current.shape[0], self.num_tokens, self.hidden_dim) + self.position
        return self.norms["output"](x, time_index=t)

    @torch.no_grad()
    def load_common(self, original):
        self.linear1.load_state_dict(original.net[1].state_dict())
        self.linear2.load_state_dict(original.net[4].state_dict())
        self.position.copy_(original.position)
        for name, old in (("input", original.net[0]), ("output", original.output_norm)):
            self.norms[name].weight.copy_(old.weight)
            self.norms[name].bias.copy_(old.bias)


class SpikeAttention(nn.Module):
    """Unidirectional SDSA-style routing with temporal-BN-normalized currents."""
    def __init__(self, hidden_dim, nheads, interaction, state_tokens=0, state_gain=1.0):
        super().__init__()
        self.nheads, self.head_dim = nheads, hidden_dim // nheads
        self.state_tokens, self.state_gain = int(state_tokens), float(state_gain)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        names = ("input_query", "input_memory", "query", "key", "value", "context")
        self.norms = nn.ModuleDict({name: _norm(hidden_dim, interaction) for name in names})
        self.spike_nodes = nn.ModuleDict({name: FunctionalPLIF() for name in names})

    def forward_step(self, query, memory, memory_valid, state, t):
        query_valid = torch.ones((*query.shape[:2], 1), dtype=torch.bool, device=query.device)

        def fire(name, current, valid):
            spikes, state[name] = self.spike_nodes[name](
                self.norms[name](current, valid, t), state.get(name), valid=valid
            )
            return spikes

        sq = fire("input_query", query, query_valid)
        sm = fire("input_memory", memory, memory_valid)
        q = fire("query", self.query(sq), query_valid)
        k = fire("key", self.key(sm), memory_valid)
        v = fire("value", self.value(sm), memory_valid)
        batch, nq, hidden_dim = q.shape
        nk = k.shape[1]
        q = q.reshape(batch, nq, self.nheads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, nk, self.nheads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, nk, self.nheads, self.head_dim).transpose(1, 2)
        # Both dot-product operands are exact 0/1 events; no softmax or ANN fallback.
        scores = q @ k.transpose(-1, -2)
        context = scores @ v
        if self.state_tokens and self.state_gain != 1.0:
            # Reweight the state contribution AFTER binary Q/K/V products.
            # Every learned projection still consumes exact unit events.
            state_context = scores[..., -self.state_tokens:] @ v[..., -self.state_tokens:, :]
            context = context + (self.state_gain - 1.0) * state_context
        context = context.transpose(1, 2).reshape(batch, nq, hidden_dim)
        live = query_valid & memory_valid.any(dim=1, keepdim=True)
        return self.output(fire("context", context, live)).masked_fill(~live, 0)

    @torch.no_grad()
    def load_common(self, original, query_norm):
        for i, linear in enumerate((self.query, self.key, self.value)):
            linear.weight.copy_(original.in_proj_weight.chunk(3, dim=0)[i])
            linear.bias.copy_(original.in_proj_bias.chunk(3, dim=0)[i])
        self.output.load_state_dict(original.out_proj.state_dict())
        self.norms["input_query"].weight.copy_(query_norm.weight)
        self.norms["input_query"].bias.copy_(query_norm.bias)


class SpikeDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, nheads, dim_feedforward, interaction, state_tokens=0, state_gain=1.0):
        super().__init__()
        self.self_attn = SpikeAttention(hidden_dim, nheads, interaction)
        self.cross_attn = SpikeAttention(hidden_dim, nheads, interaction, state_tokens, state_gain)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norms = nn.ModuleDict({
            "input": _norm(hidden_dim, interaction),
            "hidden": _norm(dim_feedforward, interaction),
        })
        self.spike_nodes = nn.ModuleDict({name: BipolarPLIF() for name in ("input", "hidden")})

    def forward_step(self, x, memory, memory_valid, state, t):
        query_valid = torch.ones((*x.shape[:2], 1), device=x.device, dtype=torch.bool)
        x = x + self.self_attn.forward_step(x, x, query_valid, state.setdefault("self", {}), t)
        x = x + self.cross_attn.forward_step(x, memory, memory_valid, state.setdefault("cross", {}), t)
        residual, ff_state = x, state.setdefault("ffn", {})
        for name, linear in (("input", self.linear1), ("hidden", self.linear2)):
            spikes, ff_state[name] = self.spike_nodes[name](
                self.norms[name](x, time_index=t), ff_state.get(name)
            )
            x = linear(spikes)
        return residual + x

    @torch.no_grad()
    def load_common(self, original):
        self.self_attn.load_common(original.self_attn, original.norm1)
        self.cross_attn.load_common(original.multihead_attn, original.norm2)
        self.linear1.load_state_dict(original.linear1.state_dict())
        self.linear2.load_state_dict(original.linear2.state_dict())
        self.norms["input"].weight.copy_(original.norm3.weight)
        self.norms["input"].bias.copy_(original.norm3.bias)


class SpikeActionReadout(nn.Module):
    def __init__(self, config, hidden_dim, interaction):
        super().__init__()
        dims = (hidden_dim, config.mlp_hidden_dim, config.mlp_hidden_dim, config.action_dim)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(_norm(dim, interaction) for dim in dims[:-1])
        self.spike_nodes = nn.ModuleList(BipolarPLIF() for _ in dims[:-1])

    def forward_step(self, x, state, t):
        for i, (norm, spike, linear) in enumerate(zip(self.norms, self.spike_nodes, self.layers)):
            events, state[i] = spike(norm(x, time_index=t), state.get(i))
            x = linear(events)
        return x


class SpikeACTDecoder(nn.Module):
    def __init__(self, config, hidden_dim, nheads, dim_feedforward, interaction):
        super().__init__()
        self.action_queries = nn.Embedding(config.horizon, hidden_dim)
        self.layers = nn.ModuleList(
            SpikeDecoderLayer(
                hidden_dim, nheads, dim_feedforward, interaction,
                config.num_state_tokens, config.spike_state_attention_gain,
            )
            for _ in range(config.num_layers)
        )
        self.action_projection = SpikeActionReadout(config, hidden_dim, interaction)

    def forward(self, memories, memory_valid, initial_state=None, return_state=False):
        carried = return_state or initial_state is not None
        if carried:
            # CTS: each layer and the readout resume from carried membranes.
            from turbovla.cts.state import wrap_carried
            initial_state = initial_state or {}
            layer_states = initial_state.get("layers") or [None] * len(self.layers)
        queries = self.action_queries.weight[None].expand(memories.shape[1], -1, -1)
        sequence = queries[None].expand(memories.shape[0], -1, -1, -1)
        final_layers = []
        for index, layer in enumerate(self.layers):
            state = wrap_carried(layer_states[index]) if carried else {}
            sequence = torch.stack([
                layer.forward_step(sequence[t], memories[t], memory_valid, state, t)
                for t in range(memories.shape[0])
            ])
            final_layers.append(state)
        state = wrap_carried(initial_state.get("readout")) if carried else {}
        currents = [
            self.action_projection.forward_step(sequence[t], state, t)
            for t in range(memories.shape[0])
        ]
        actions = torch.tanh(torch.stack(currents).mean(0))
        if return_state:
            return actions, {"layers": final_layers, "readout": state}
        return actions

    @torch.no_grad()
    def load_common(self, original):
        self.action_queries.load_state_dict(original.action_queries.state_dict())
        for layer, old in zip(self.layers, original.decoder.layers):
            layer.load_common(old)
        for linear, old in zip(self.action_projection.layers, original.action_projection.layers):
            linear.load_state_dict(old.state_dict())


class SpikeTurboVLAActionHead(nn.Module):
    def __init__(self, config, hidden_dim, nheads, dim_feedforward, interaction):
        super().__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.state_projection = (
            StateProjection(config, hidden_dim) if config.continuous_state
            else SpikeStateProjection(config, hidden_dim, interaction)
        )
        self.decoder = SpikeACTDecoder(config, hidden_dim, nheads, dim_feedforward, interaction)

    @torch.no_grad()
    def load_common(self, original):
        if self.config.continuous_state:
            self.state_projection.load_state_dict(original.state_projection.state_dict(), strict=True)
        else:
            self.state_projection.load_common(original.state_projection)
        self.decoder.load_common(original.decoder)

    def _forward(self, memory, state, memory_key_padding_mask, initial_state=None, return_state=False):
        state_state, memories = {}, []
        continuous_tokens = self.state_projection(state) if self.config.continuous_state else None
        # A already reads out its fusion timesteps. Integrate that unchanged
        # condition as a constant current through the COMPLETE new action head.
        for t in range(self.config.spike_timesteps):
            state_tokens = (continuous_tokens if continuous_tokens is not None
                            else self.state_projection.forward_step(state, state_state, t))
            memories.append(torch.cat((memory, state_tokens), dim=1))
        state_valid = torch.ones(
            (state.shape[0], self.config.num_state_tokens), device=memory.device, dtype=torch.bool
        )
        valid = torch.cat((~memory_key_padding_mask, state_valid), dim=1)[..., None]
        if return_state or initial_state is not None:
            return self.decoder(torch.stack(memories), valid, initial_state, return_state)
        return self.decoder(torch.stack(memories), valid)

    def forward(self, memory, state, memory_key_padding_mask=None, initial_state=None, return_state=False):
        if memory.dtype != torch.float32 or state.dtype != torch.float32:
            raise ValueError("The spike action head requires FP32 inputs")
        if state.ndim == 3:
            state = state[:, -1]
        if memory.ndim != 3 or state.shape != (memory.shape[0], self.state_dim):
            raise ValueError("Action memory/state must be [B,N,C] and [B,state_dim]")
        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.zeros(
                memory.shape[:2], device=memory.device, dtype=torch.bool
            )
        if (memory_key_padding_mask.dtype != torch.bool
                or memory_key_padding_mask.shape != memory.shape[:2]):
            raise ValueError("Action memory padding mask must be bool [B,N]")
        if self.training and torch.is_grad_enabled() and self.config.spike_gradient_checkpointing:
            return checkpoint(
                self._forward, memory, state, memory_key_padding_mask, initial_state, return_state,
                use_reentrant=False, context_fn=checkpoint_contexts,
            )
        return self._forward(memory, state, memory_key_padding_mask, initial_state, return_state)
