from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .action_head import TurboVLAActionHead
from .components.fusion import BiAttentionBlock
from .components.transformer import TransformerEncoderLayer
from .components.utils import _get_clones
from .configuration import (
    ActionHeadConfig,
    CTSConfig,
    InteractionConfig,
    TextEncoderConfig,
    TurboVLAConfig,
    VisionEncoderConfig,
)
from .text_encoder import TurboVLATextEncoder
from .vision_encoder import build_vision_encoder


class VisionProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.output_norm = nn.LayerNorm(out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.skip(tokens) + self.mlp(self.input_norm(tokens)))


class VisionLanguageInteraction(nn.Module):
    def __init__(self, config: InteractionConfig, text_mask_version: str = "corrected") -> None:
        super().__init__()
        text_layer = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=max(1, config.nheads // 2),
            dim_feedforward=config.enhancer_inner_dim,
            dropout=config.text_dropout,
            text_mask_version=text_mask_version,
        )
        fusion_layer = BiAttentionBlock(
            v_dim=config.hidden_dim,
            l_dim=config.hidden_dim,
            embed_dim=config.enhancer_inner_dim,
            num_heads=max(1, config.nheads // 2),
            dropout=config.fusion_dropout,
            drop_path=config.fusion_droppath,
            residual_style=config.residual_style,
            attention_backend=config.attention_backend,
            cross_attention_type="spike_sdsa" if config.backbone_type == "spike_wta" else config.cross_attention_type,
            cross_timesteps=config.cross_timesteps,
            cross_gradient_checkpointing=config.cross_gradient_checkpointing,
        )
        self.text_layers = _get_clones(text_layer, config.num_layers)
        self.fusion_layers = _get_clones(fusion_layer, config.num_layers)
        self.padding_strategy = config.padding_strategy

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        text_self_attention_masks: torch.Tensor | None,
        cts_state: list | None = None,
        return_state: bool = False,
    ):
        zero_fill = self.padding_strategy == "zero_fill"
        if zero_fill:
            text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)

        final_states = []
        for index, (fusion_layer, text_layer) in enumerate(zip(self.fusion_layers, self.text_layers)):
            layer_state = (cts_state[index] if cts_state is not None else None) or {}
            if return_state:
                visual_tokens, text_tokens, cross_state = fusion_layer(
                    v=visual_tokens,
                    l=text_tokens,
                    attention_mask_v=None,
                    attention_mask_l=text_key_padding_mask,
                    cts_state=layer_state.get("cross"),
                    return_state=True,
                )
            else:
                visual_tokens, text_tokens = fusion_layer(
                    v=visual_tokens,
                    l=text_tokens,
                    attention_mask_v=None,
                    attention_mask_l=text_key_padding_mask,
                )
            source_mask = None if text_self_attention_masks is None else ~text_self_attention_masks
            text_kwargs = dict(
                src=text_tokens.transpose(0, 1),
                src_mask=source_mask,
                src_key_padding_mask=None if zero_fill else text_key_padding_mask,
                pos=None,
            )
            if return_state:
                text_tokens, text_state = text_layer(
                    **text_kwargs, initial_state=layer_state.get("text"), return_state=True)
                final_states.append({"cross": cross_state, "text": text_state})
            else:
                text_tokens = text_layer(**text_kwargs)
            text_tokens = text_tokens.transpose(0, 1)
            if zero_fill:
                text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)
        if return_state:
            return visual_tokens, text_tokens, final_states
        return visual_tokens, text_tokens


class TurboVLA(nn.Module):
    """Shared TurboVLA architecture for LIBERO and RoboTwin."""

    def __init__(self, config: TurboVLAConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.interaction.hidden_dim
        self.action_dim = int(config.action.action_dim)
        self.chunk_size = int(config.action.horizon)
        self.state_dim = int(config.action.state_dim)
        self.num_views = int(config.vision.num_views)

        self.text_encoder = TurboVLATextEncoder(
            config.text, hidden_dim=hidden_dim, text_mask_version=config.sentence_mask_version
        )
        self.vision_encoder = build_vision_encoder(config.vision)
        self.vision_projection = VisionProjection(
            in_dim=self.vision_encoder.hidden_size,
            out_dim=hidden_dim,
            hidden_dim=max(hidden_dim * 4, self.vision_encoder.hidden_size // 2),
            dropout=config.vision.dropout,
        )

        if config.vision.position_embedding == "learned_patch":
            self.view_embedding = nn.Parameter(torch.zeros(1, self.num_views, 1, hidden_dim))
            self.patch_position_embedding = nn.Parameter(
                torch.zeros(1, self.num_views, self.vision_encoder.num_patches, hidden_dim)
            )
            self.patch_position_scale = nn.Parameter(
                torch.full((1, self.num_views, 1, 1), float(config.vision.position_scale_init))
            )
            nn.init.trunc_normal_(self.patch_position_embedding, std=config.vision.position_init_std)
        else:
            self.view_embedding = nn.Parameter(torch.zeros(1, self.num_views, hidden_dim))
            self.register_parameter("patch_position_embedding", None)
            self.register_parameter("patch_position_scale", None)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)

        self.vision_language_interaction = VisionLanguageInteraction(
            config.interaction, text_mask_version=config.head_mask_version
        )
        self.action_head = TurboVLAActionHead(
            config=config.action,
            hidden_dim=hidden_dim,
            nheads=config.interaction.nheads,
            dim_feedforward=config.interaction.dim_feedforward,
        )
        if config.interaction.backbone_type == "spike_text_only":
            from .components.spike_text_only import SpikeTextEncoderLayer
            # Keep the original Cross objects and their initialization.
            # Projection replacement, when requested, is separate below.
            with torch.random.fork_rng(devices=[]):
                self.vision_language_interaction.text_layers = nn.ModuleList(
                    [SpikeTextEncoderLayer(config.interaction)
                     for _ in range(config.interaction.num_layers)]
                )
        self.temporal_spike = config.interaction.backbone_type in {"spike_temporal", "spike_wta"}
        self.spike_projectors = self.temporal_spike or config.interaction.text_only_spike_projections
        if self.spike_projectors:
            from .components.temporal_fusion import (
                SpikeTextProjection, SpikeVisionProjection, TemporalSpikeInteraction,
            )
            projection_options = {}
            if config.interaction.backbone_type == "spike_wta" or config.interaction.text_only_spike_projections:
                from .components.wta_fusion import (
                    WTATextProjection as SpikeTextProjection,
                    WTAVisionProjection as SpikeVisionProjection,
                    WTASpikeInteraction as TemporalSpikeInteraction,
                )
                projection_options = {"config": config.interaction}
            # Preserve initialization of the unchanged encoders and action head.
            with torch.random.fork_rng(devices=[]):
                old = self.text_encoder.text_projection
                projection = SpikeTextProjection(old.in_features, old.out_features, **projection_options)
                projection.load_common_projection(old.state_dict())
                self.text_encoder.text_projection = projection
                old = self.vision_projection
                projection = SpikeVisionProjection(
                    old.skip.in_features, old.skip.out_features, old.mlp[0].out_features,
                    **projection_options,
                )
                projection.linear1.load_state_dict(old.mlp[0].state_dict())
                projection.linear2.load_state_dict(old.mlp[3].state_dict())
                projection.skip.load_state_dict(old.skip.state_dict())
                self.vision_projection = projection
                if self.temporal_spike:
                    self.vision_language_interaction = TemporalSpikeInteraction(config.interaction)
        if config.action.head_type == "progressive_full_wta":
            from .extensions.progressive_full_wta_head import ProgressiveFullWTAHead
            with torch.random.fork_rng(devices=[]):
                self.action_head = ProgressiveFullWTAHead(self.action_head, config.action, config.interaction)
        elif config.action.head_type == "minimal_cross_wta":
            from .extensions.minimal_cross_wta_head import MinimalCrossWTAActionHead
            with torch.random.fork_rng(devices=[]):
                self.action_head = MinimalCrossWTAActionHead(
                    config.action, hidden_dim, config.interaction.nheads,
                    config.interaction.dim_feedforward, config.interaction, self.action_head,
                )
        elif config.action.head_type == "spike_sdsa":
            from .spike_action_head import SpikeTurboVLAActionHead
            # Finish A's original constructor first. Replacing the head must not
            # change initialization/RNG consumption of any unchanged component.
            with torch.random.fork_rng(devices=[]):
                head = SpikeTurboVLAActionHead(
                    config.action, hidden_dim, config.interaction.nheads,
                    config.interaction.dim_feedforward, config.interaction,
                )
                head.load_common(self.action_head)
                self.action_head = head

        elif config.action.head_type == "hybrid_spike_ffn":
            from .extensions.hybrid_action_head import HybridSpikeFFNActionHead
            with torch.random.fork_rng(devices=[]):
                head = HybridSpikeFFNActionHead(
                    config.action, hidden_dim, config.interaction.nheads,
                    config.interaction.dim_feedforward,
                )
                head.load_common(self.action_head)
                self.action_head = head
        elif config.action.head_type in {"deep_spike_wta", "full_spike_state8_wta", "full_spike_state8_shiftmax", "full_spike_state8_shiftmax_continuous_inputs", "full_spike_state8_shiftmax_continuous_inputs_ann_ffn", "full_spike_state8_shiftmax_ann_ffn"}:
            if config.action.head_type in {"full_spike_state8_shiftmax", "full_spike_state8_shiftmax_continuous_inputs", "full_spike_state8_shiftmax_continuous_inputs_ann_ffn", "full_spike_state8_shiftmax_ann_ffn"}:
                from .shiftmax_action_head import FullStateShiftmaxActionHead as Head
            elif config.action.head_type == "full_spike_state8_wta":
                from .full_state_action_head import FullStateSpikeActionHead as Head
            else:
                from .deep_spike_action_head import DeepSpikeActionHead as Head
            with torch.random.fork_rng(devices=[]):
                head = Head(
                    config.action, hidden_dim, config.interaction.nheads,
                    config.interaction.dim_feedforward, config.interaction,
                )
                head.load_common(self.action_head)
                self.action_head = head
        if config.cts.enabled:
            from ..cts.apply import apply_cts
            # New CTS parameters are deterministic; unchanged modules keep
            # their constructed values and RNG consumption.
            with torch.random.fork_rng(devices=[]):
                apply_cts(self)

    def _normalize_samples(self, samples: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(samples, Mapping):
            key = "pixel_values" if "pixel_values" in samples else "dinov3"
            if key not in samples:
                raise ValueError("samples mapping must contain 'pixel_values' (or legacy 'dinov3')")
            pixel_values = samples[key]
        else:
            pixel_values = samples
        if pixel_values.ndim == 6:
            pixel_values = pixel_values[:, -1]
        if pixel_values.ndim != 5:
            raise ValueError(f"samples must be [B,V,3,H,W] or [B,T,V,3,H,W], got {tuple(pixel_values.shape)}")
        return pixel_values

    def _position_visual_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.config.vision.position_embedding == "learned_patch":
            if tokens.shape[2] != self.patch_position_embedding.shape[2]:
                raise ValueError(
                    f"configured patch position length {self.patch_position_embedding.shape[2]} "
                    f"does not match encoded length {tokens.shape[2]}"
                )
            position = self.patch_position_embedding.to(device=tokens.device, dtype=tokens.dtype)
            scale = self.patch_position_scale.to(device=tokens.device, dtype=tokens.dtype)
            view = self.view_embedding.to(device=tokens.device, dtype=tokens.dtype)
            return tokens + scale * position + view
        view = self.view_embedding[:, :, None, :].to(device=tokens.device, dtype=tokens.dtype)
        return tokens + view

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.vision_encoder(pixel_values)
        tokens = tokens.to(dtype=self.vision_projection.skip.weight.dtype)
        if self.spike_projectors:
            state = {}
            sequence = [
                self._position_visual_tokens(self.vision_projection.forward_step(tokens, state, time_index=t)).flatten(1, 2)
                for t in range(self.config.interaction.cross_timesteps)
            ]
            return torch.stack(sequence).mean(0)
        tokens = self.vision_projection(tokens)
        return self._position_visual_tokens(tokens).flatten(1, 2)

    def encode_condition(
        self,
        instructions: Sequence[str],
        samples: torch.Tensor | Mapping[str, torch.Tensor],
        return_visual_tokens: bool = False,
        return_distillation_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        pixel_values = self._normalize_samples(samples)
        device = pixel_values.device
        if (self.config.interaction.backbone_type == "spike_text_only" and not self.training
                and not self.vision_language_interaction.text_layers[0].norms["input"].calibrating
                and pixel_values.shape[0] > 1):
            if len(instructions) != pixel_values.shape[0]:
                raise ValueError("Instruction batch size does not match image batch size")
            # Fixed B=1 spike inference, as for the full-WTA variant. Training
            # and training-stat refresh still use the real requested batch.
            rows = [self.encode_condition(
                [instruction], pixel_values[i:i+1], return_visual_tokens,
                return_distillation_features) for i, instruction in enumerate(instructions)]
            if return_distillation_features:
                return torch.cat([row[0] for row in rows]), {
                    key: torch.cat([row[1][key] for row in rows]) for key in rows[0][1]}
            if return_visual_tokens:
                return tuple(torch.cat([row[index] for row in rows]) for index in (0, 1))
            return torch.cat(rows)
        if self.spike_projectors:
            if len(instructions) != pixel_values.shape[0]:
                raise ValueError("Instruction batch size does not match image batch size")
            if (self.config.interaction.backbone_type == "spike_wta" and not self.training
                    and not self.text_encoder.text_projection.input_norm.calibrating
                    and pixel_values.shape[0] > 1):
                # Hard thresholds can amplify batch-shape-dependent GEMM
                # roundoff. Keep the complete spike inference path at B=1,
                # matching online robot observations; training/BN refresh stay
                # truly batched. No rounding, smoothing or cached embeddings.
                rows = [self._encode_temporal_condition(
                    [instruction], pixel_values[i:i+1], return_visual_tokens,
                    return_distillation_features) for i, instruction in enumerate(instructions)]
                if return_distillation_features:
                    return torch.cat([row[0] for row in rows]), {
                        key: torch.cat([row[1][key] for row in rows]) for key in rows[0][1]}
                if return_visual_tokens:
                    return tuple(torch.cat([row[index] for row in rows]) for index in (0, 1))
                return torch.cat(rows)
            return self._encode_temporal_condition(
                instructions, pixel_values, return_visual_tokens, return_distillation_features
            )
        precision_context = nullcontext()
        if self.config.interaction.compute_precision == "bf16_autocast" and device.type == "cuda":
            precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with precision_context:
            text_tokens, text_key_padding_mask, text_self_attention_masks = self.text_encoder(
                instructions,
                device=device,
            )
            if text_tokens.shape[0] != pixel_values.shape[0]:
                raise ValueError("instruction batch size does not match image batch size")
            projected_text_tokens = text_tokens
            visual_tokens = self.encode_vision(pixel_values)
            projected_visual_tokens = visual_tokens
            visual_tokens, text_tokens = self.vision_language_interaction(
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_key_padding_mask=text_key_padding_mask,
                text_self_attention_masks=text_self_attention_masks,
            )
            condition = torch.cat([visual_tokens, text_tokens], dim=1)
            if return_distillation_features:
                return condition, {
                    "visual": projected_visual_tokens,
                    "text": projected_text_tokens,
                    "text_valid": ~text_key_padding_mask,
                }
            if return_visual_tokens:
                return condition, projected_visual_tokens
            return condition

    def _encode_temporal_condition(
        self, instructions, pixel_values, return_visual_tokens=False, return_distillation_features=False
    ):
        hidden, valid_tokens, sentence_allowed = self.text_encoder.encode_bert_hidden(
            instructions, pixel_values.device
        )
        if hidden.shape[0] != pixel_values.shape[0]:
            raise ValueError("Instruction batch size does not match image batch size")
        hidden = hidden.to(self.text_encoder.text_projection.weight.dtype)
        raw_visual = self.vision_encoder(pixel_values).to(self.vision_projection.skip.weight.dtype)
        valid = valid_tokens[..., None]
        def project(vision, text):
            v_state, l_state = {}, {}
            visual, language = [], []
            for t in range(self.config.interaction.cross_timesteps):
                current = self.vision_projection.forward_step(vision, v_state, time_index=t)
                visual.append(self._position_visual_tokens(current).flatten(1, 2))
                language.append(self.text_encoder.text_projection.forward_step(text, l_state, valid, time_index=t))
            return torch.stack(visual), torch.stack(language)

        if (self.training and torch.is_grad_enabled()
                and self.config.interaction.projection_gradient_checkpointing):
            from .components.wta_operators import checkpoint_contexts
            visual, language = checkpoint(
                project, raw_visual, hidden, use_reentrant=False, context_fn=checkpoint_contexts)
        else:
            visual, language = project(raw_visual, hidden)
        initial_visual, initial_text = visual, language
        if not self.temporal_spike:
            # Original Cross accepts [B,N,C] and owns its existing T=4 loop.
            # Read out the two spike projections once; do not wrap Cross in T.
            visual, language = visual.mean(0), language.mean(0)
        visual, language = self.vision_language_interaction(
            visual, language, ~valid_tokens, sentence_allowed
        )
        if self.temporal_spike:
            # Full temporal fusion keeps T until this final policy readout.
            visual, language = visual.mean(0), language.mean(0)
        condition = torch.cat([visual, language], dim=1)
        if return_distillation_features:
            return condition, {
                "visual": initial_visual.mean(0), "text": initial_text.mean(0),
                "text_valid": valid_tokens,
            }
        if return_visual_tokens:
            return condition, initial_visual.mean(0)
        return condition

    def forward_cts(
        self,
        instructions: Sequence[str],
        samples: torch.Tensor | Mapping[str, torch.Tensor],
        state: torch.Tensor,
        carry: dict | None = None,
        return_distillation_features: bool = False,
    ):
        """One control query of Control-Time Spiking.

        `carry` is the episode state returned by the previous query (None at
        episode start). Returns (actions, new_carry) or, with features,
        (actions, new_carry, features). The carry is never mutated in place.
        """
        cts = self.config.cts
        if not cts.enabled:
            raise RuntimeError("forward_cts requires config.cts.enabled")
        from ..cts.state import wrap_carried
        from .components.wta_operators import checkpoint_contexts

        carry = carry or {}
        new_carry: dict[str, Any] = {}
        pixel_values = self._normalize_samples(samples)
        device = pixel_values.device

        # The frozen text backbone sees a constant instruction; encode it once per episode.
        text_cache = carry.get("instruction")
        if text_cache is None:
            hidden, valid_tokens, sentence_allowed = self.text_encoder.encode_bert_hidden(instructions, device)
            if hidden.shape[0] != pixel_values.shape[0]:
                raise ValueError("Instruction batch size does not match image batch size")
            text_cache = (hidden.to(self.text_encoder.text_projection.weight.dtype), valid_tokens, sentence_allowed)
        new_carry["instruction"] = text_cache
        hidden, valid_tokens, sentence_allowed = text_cache
        valid = valid_tokens[..., None]
        raw_visual = self.vision_encoder(pixel_values).to(self.vision_projection.skip.weight.dtype)

        def project(vision, text, init):
            init = init or {}
            v_state, l_state = wrap_carried(init.get("vision")), wrap_carried(init.get("text"))
            visual, language = [], []
            for t in range(self.config.interaction.cross_timesteps):
                current = self.vision_projection.forward_step(vision, v_state, time_index=t)
                visual.append(self._position_visual_tokens(current).flatten(1, 2))
                language.append(self.text_encoder.text_projection.forward_step(text, l_state, valid, time_index=t))
            return torch.stack(visual), torch.stack(language), {"vision": v_state, "text": l_state}

        projection_in = carry.get("projection") if cts.carry_projection else None
        if (self.training and torch.is_grad_enabled()
                and self.config.interaction.projection_gradient_checkpointing):
            visual, language, projection_state = checkpoint(
                project, raw_visual, hidden, projection_in, use_reentrant=False, context_fn=checkpoint_contexts)
        else:
            visual, language, projection_state = project(raw_visual, hidden, projection_in)
        if cts.carry_projection:
            new_carry["projection"] = projection_state
        initial_visual, initial_text = visual.mean(0), language.mean(0)

        fusion_in = carry.get("fusion")
        visual, language, fusion_state = self.vision_language_interaction(
            initial_visual, initial_text, ~valid_tokens, sentence_allowed,
            cts_state=fusion_in, return_state=True,
        )
        if cts.carry_fusion or cts.memory or cts.memory_vision:
            new_carry["fusion"] = [
                {"cross": {"attn": layer["cross"]["attn"] if cts.carry_fusion else None,
                           "memory": layer["cross"]["memory"], "memory_v": layer["cross"]["memory_v"]},
                 "text": layer["text"] if cts.carry_fusion else None}
                for layer in fusion_state
            ]
        condition = torch.cat([visual, language], dim=1)

        action_dtype = self.action_head.decoder.action_queries.weight.dtype
        condition, state = condition.to(dtype=action_dtype), state.to(dtype=action_dtype)
        if self.config.action.head_type == "ann":
            actions = self.action_head(condition, state)
        else:
            visual_padding = torch.zeros(visual.shape[:2], dtype=torch.bool, device=condition.device)
            padding = torch.cat((visual_padding, ~valid_tokens), dim=1)
            memory = condition
            if cts.history_tokens:
                # E3 baseline: explicit previous-query visual tokens, no neuron state needed.
                previous = carry.get("history")
                previous_valid = carry.get("history_valid")
                if previous is None:
                    previous = torch.zeros_like(visual)
                    previous_valid = torch.zeros(visual.shape[0], dtype=torch.bool, device=visual.device)
                memory = torch.cat((condition, previous.to(action_dtype)), dim=1)
                padding = torch.cat((padding, (~previous_valid)[:, None].expand(-1, visual.shape[1])), dim=1)
                new_carry["history"] = visual
                new_carry["history_valid"] = torch.ones(visual.shape[0], dtype=torch.bool, device=visual.device)
            head_in = carry.get("head") if cts.carry_head else None
            actions, head_state = self.action_head(
                memory, state, memory_key_padding_mask=padding, initial_state=head_in, return_state=True)
            if cts.carry_head:
                new_carry["head"] = head_state
        new_carry["query_index"] = int(carry.get("query_index", -1)) + 1
        if return_distillation_features:
            features = {"visual": initial_visual, "text": initial_text, "text_valid": valid_tokens}
            return actions, new_carry, features
        return actions, new_carry

    def forward_window(
        self,
        queries: Sequence[tuple[Sequence[str], Any, torch.Tensor]],
        present: torch.Tensor,
        burn_in: int,
        state_dropout: float = 0.0,
        return_distillation_features: bool = False,
    ):
        """C3: run Q consecutive control queries of a batch of episodes.

        queries[j] = (instructions, samples, state). The first `burn_in` queries
        run without gradients and only warm up the carry. present[b, j] is False
        for burn-in queries before the episode start; the state is zeroed after
        them. Before every loss query the state of each sample is dropped with
        probability `state_dropout`. Returns actions [B, K, H, A] for the K loss
        queries (and their distillation features as a list).
        """
        from ..cts.state import detach_carry, reset_samples

        if not 0 <= burn_in < len(queries):
            raise ValueError("burn_in must leave at least one loss query")
        carry, actions, features = None, [], []
        for index, (instructions, samples, state) in enumerate(queries):
            loss_query = index >= burn_in
            with (nullcontext() if loss_query else torch.no_grad()):
                result = self.forward_cts(
                    instructions, samples, state, carry,
                    return_distillation_features=loss_query and return_distillation_features,
                )
            if loss_query:
                actions.append(result[0])
                if return_distillation_features:
                    features.append(result[2])
                carry = result[1]
            else:
                carry = detach_carry(result[1])
            if index + 1 < len(queries):
                keep = present[:, index].to(device=result[0].device)
                if index + 1 >= burn_in and state_dropout > 0 and self.training:
                    keep = keep & (torch.rand(keep.shape, device=keep.device) >= state_dropout)
                carry = reset_samples(carry, keep)
        stacked = torch.stack(actions, dim=1)
        return (stacked, features) if return_distillation_features else stacked

    def forward(
        self,
        instructions: Sequence[str] | None,
        samples: torch.Tensor | Mapping[str, torch.Tensor] | None,
        state: torch.Tensor | None,
        return_visual_tokens: bool = False,
        return_distillation_features: bool = False,
        cts_window: Mapping[str, Any] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        if cts_window is not None:
            # Routed through forward so DDP sees one forward per backward.
            return self.forward_window(**cts_window, return_distillation_features=return_distillation_features)
        if self.config.cts.enabled:
            # A stateless call is one cold-start control query.
            actions, _, features = self.forward_cts(
                instructions, samples, state, None, return_distillation_features=True)
            if return_distillation_features:
                return actions, features
            if return_visual_tokens:
                return actions, features["visual"]
            return actions
        spike_action_head = self.config.action.head_type in {
            "spike_sdsa", "deep_spike_wta", "full_spike_state8_wta", "full_spike_state8_shiftmax", "full_spike_state8_shiftmax_continuous_inputs", "full_spike_state8_shiftmax_continuous_inputs_ann_ffn", "full_spike_state8_shiftmax_ann_ffn",
        }
        need_features = return_distillation_features or spike_action_head
        encoded = self.encode_condition(
            instructions, samples, return_visual_tokens=return_visual_tokens,
            return_distillation_features=need_features,
        )
        if need_features:
            condition, features = encoded
        elif return_visual_tokens:
            condition, visual_tokens = encoded
        else:
            condition = encoded
        action_dtype = self.action_head.decoder.action_queries.weight.dtype
        condition, state = condition.to(dtype=action_dtype), state.to(dtype=action_dtype)
        if spike_action_head:
            text_valid = features["text_valid"]
            visual_padding = torch.zeros(
                (condition.shape[0], condition.shape[1] - text_valid.shape[1]),
                dtype=torch.bool, device=condition.device,
            )
            padding = torch.cat((visual_padding, ~text_valid), dim=1)
            actions = self.action_head(condition, state, memory_key_padding_mask=padding)
        else:
            actions = self.action_head(condition, state)
        if return_distillation_features:
            return actions, features
        if return_visual_tokens:
            return actions, features["visual"] if spike_action_head else visual_tokens
        return actions

    # Transitional read-only names used only by legacy checkpoint initialization.
    @property
    def dinov3(self):
        return self.vision_encoder.backbone

    @property
    def text_proj(self):
        return self.text_encoder.text_projection

    @property
    def vision_proj(self):
        return self.vision_projection

    @property
    def feature_enhancer(self):
        return self.vision_language_interaction

    @property
    def state_proj(self):
        return self.action_head.state_projection

    @property
    def action_policy(self):
        return self.action_head.decoder


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_turbovla(args: TurboVLAConfig | Mapping[str, Any] | Any) -> TurboVLA:
    if isinstance(args, TurboVLAConfig):
        config = args
    elif isinstance(args, Mapping):
        config = TurboVLAConfig.from_mapping(args)
    else:
        config = TurboVLAConfig(
            text_mask_version=str(_arg(args, "text_mask_version", "legacy")),
            text_sentence_mask_version=_arg(args, "text_sentence_mask_version", None),
            text_head_mask_version=_arg(args, "text_head_mask_version", None),
            text=TextEncoderConfig(
                encoder_type=str(_arg(args, "text_encoder_type", "bert")),
                model_name_or_path=(
                    _arg(args, "text_model_path", None)
                    or _arg(args, "bert_path", "bert-base-uncased")
                ),
                timesteps=int(_arg(args, "text_timesteps", 1)),
                max_length=int(_arg(args, "max_text_len", 256)),
                padding_length=_arg(args, "text_padding_length", None),
                padding_length_by_instruction=dict(_arg(args, "text_padding_length_by_instruction", {})),
                sub_sentence_present=bool(_arg(args, "sub_sentence_present", True)),
                frozen=bool(_arg(args, "freeze_text_encoder", True)),
                force_eval_when_frozen=True,
                zero_padded_tokens=bool(_arg(args, "zero_padded_text", False)),
                local_files_only=bool(_arg(args, "local_files_only", True)),
                attention_implementation=_arg(args, "text_attention_implementation", None),
            ),
            vision=VisionEncoderConfig(
                encoder_type=str(_arg(args, "vision_encoder_type", "dinov3")),
                model_name_or_path=_arg(
                    args, "vision_model_path", _arg(args, "dinov3_path", _arg(args, "LOCAL_DINOV3_PATH", ""))
                ),
                image_size=int(_arg(args, "image_size", _arg(args, "expected_image_size", 256))),
                num_views=int(_arg(args, "num_views", 2)),
                position_embedding=str(_arg(args, "position_embedding", "view")),
                encode_views_separately=bool(_arg(args, "encode_views_separately", True)),
                frozen=bool(_arg(args, "freeze_vision_encoder", False)),
                local_files_only=bool(_arg(args, "local_files_only", True)),
                attention_implementation=_arg(args, "vision_attention_implementation", None),
                compute_precision=str(_arg(args, "dinov3_precision", "bf16_autocast")),
                dropout=float(_arg(args, "vision_dropout", 0.1)),
                pretrained_checkpoint=_arg(args, "vision_pretrained_checkpoint", None),
                model_source_path=_arg(args, "vision_model_source_path", None),
                output_grid_size=int(_arg(args, "vision_output_grid_size", 14)),
                gradient_checkpointing=bool(_arg(args, "vision_gradient_checkpointing", False)),
                checkpoint_scope=str(_arg(args, "vision_checkpoint_scope", "all")),
            ),
            interaction=InteractionConfig(
                backbone_type=str(_arg(args, "fusion_backbone_type", "hybrid")),
                text_only_spike_projections=bool(_arg(args, "text_only_spike_projections", False)),
                hidden_dim=int(_arg(args, "hidden_dim", 256)),
                nheads=int(_arg(args, "nheads", 8)),
                num_layers=int(_arg(args, "vla_feature_enhancer_layers", 6)),
                dim_feedforward=int(_arg(args, "dim_feedforward", 2048)),
                enhancer_inner_dim=int(_arg(args, "enhancer_inner_dim", 1024)),
                text_dropout=float(_arg(args, "text_dropout", 0.0)),
                fusion_dropout=float(_arg(args, "fusion_dropout", 0.0)),
                fusion_droppath=float(_arg(args, "fusion_droppath", 0.1)),
                padding_strategy=str(_arg(args, "padding_strategy", "key_padding_mask")),
                residual_style=str(_arg(args, "residual_style", "normalized")),
                attention_backend=str(_arg(args, "attention_backend", "manual")),
                compute_precision=str(_arg(args, "interaction_precision", "fp32")),
                cross_attention_type=str(_arg(args, "cross_attention_type", "ann")),
                cross_timesteps=int(_arg(args, "cross_timesteps", 4)),
                cross_gradient_checkpointing=bool(_arg(args, "cross_gradient_checkpointing", True)),
                projection_gradient_checkpointing=bool(_arg(args, "projection_gradient_checkpointing", False)),
                text_attention_routing=str(_arg(args, "text_attention_routing", "wta")),
                wta_text_topk=int(_arg(args, "wta_text_topk", 3)),
                wta_vision_topk=int(_arg(args, "wta_vision_topk", 8)),
                wta_surrogate_temperature=float(_arg(args, "wta_surrogate_temperature", 1.0)),
                spike_norm_momentum=float(_arg(args, "spike_norm_momentum", 0.1)),
                spike_norm_eps=float(_arg(args, "spike_norm_eps", 1e-5)),
                spike_norm_sync=bool(_arg(args, "spike_norm_sync", True)),
            ),
            action=ActionHeadConfig(
                head_type=str(_arg(args, "action_head_type", "ann")),
                continuous_state=bool(_arg(args, "continuous_state", False)),
                spike_state_encoding=str(_arg(args, "spike_state_encoding", "perdim_plif")),
                spike_timesteps=8 if _arg(args, "action_head_type", "ann") == "progressive_full_wta" else int(_arg(args, "cross_timesteps", 4)),
                spike_gradient_checkpointing=bool(_arg(args, "action_head_gradient_checkpointing", True)),
                spike_state_population=int(_arg(args, "spike_state_population", 0)),
                spike_state_population_range=float(_arg(args, "spike_state_population_range", 4.0)),
                spike_state_attention_gain=float(_arg(args, "spike_state_attention_gain", 1.0)),
                action_dim=int(_arg(args, "action_dim", 7)),
                state_dim=int(_arg(args, "state_dim", 8)),
                horizon=int(_arg(args, "chunk_size", _arg(args, "action_horizon", 12))),
                num_state_tokens=int(_arg(args, "num_state_tokens", 2)),
                num_layers=int(_arg(args, "act_num_layers", 3)),
                mlp_hidden_dim=int(_arg(args, "act_mlp_hidden_dim", 512)),
                state_hidden_dim=int(_arg(args, "act_state_hidden_dim", 256)),
                dropout=float(_arg(args, "act_dropout", 0.1)),
            ),
            cts=cts_config_from_args(args),
        )
    return TurboVLA(config)


def cts_config_from_args(args: Any) -> CTSConfig:
    """Read `cts_*` attributes (CLI names) into a CTSConfig; absent means disabled."""
    defaults = CTSConfig()
    values = {}
    for name in CTSConfig.__dataclass_fields__:
        default = getattr(defaults, name)
        value = _arg(args, "cts_" + name, default)
        values[name] = type(default)(value) if value is not None else default
    return CTSConfig(**values)
