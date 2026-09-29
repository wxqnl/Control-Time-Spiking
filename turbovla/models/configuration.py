from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class TextEncoderConfig:
    encoder_type: str = "bert"
    model_name_or_path: str = "bert-base-uncased"
    timesteps: int = 1
    max_length: int = 256
    padding_length: int | None = None
    padding_length_by_instruction: dict[str, int] = field(default_factory=dict)
    sub_sentence_present: bool = True
    frozen: bool = True
    force_eval_when_frozen: bool = True
    zero_padded_tokens: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None


@dataclass
class VisionEncoderConfig:
    encoder_type: str = "dinov3"
    model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    image_size: int = 256
    num_views: int = 2
    position_embedding: str = "view"
    encode_views_separately: bool = True
    frozen: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None
    compute_precision: str = "bf16_autocast"
    position_init_std: float = 0.01
    position_scale_init: float = 0.01
    dropout: float = 0.1
    pretrained_checkpoint: str | None = None
    model_source_path: str | None = None
    output_grid_size: int = 14
    gradient_checkpointing: bool = False
    checkpoint_scope: str = "all"


@dataclass
class InteractionConfig:
    backbone_type: str = "hybrid"
    # Optional projection upgrade for the old-Cross / new-text variant.
    # False preserves previously saved text-only checkpoints.
    text_only_spike_projections: bool = False
    hidden_dim: int = 256
    nheads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    enhancer_inner_dim: int = 1024
    text_dropout: float = 0.0
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1
    # These three switches apply to hybrid only. spike_temporal has a fixed
    # membrane-residual, explicit spike-attention, padding-isolated contract.
    padding_strategy: str = "key_padding_mask"
    residual_style: str = "normalized"
    attention_backend: str = "manual"
    compute_precision: str = "fp32"
    cross_attention_type: str = "ann"
    cross_timesteps: int = 4
    cross_gradient_checkpointing: bool = True
    projection_gradient_checkpointing: bool = False
    text_attention_routing: str = "wta"
    wta_text_topk: int = 3
    wta_vision_topk: int = 8
    wta_surrogate_temperature: float = 1.0
    spike_norm_momentum: float = 0.1
    spike_norm_eps: float = 1e-5
    spike_norm_sync: bool = True


@dataclass
class ActionHeadConfig:
    head_type: str = "ann"
    ann_fraction: float = 0.8
    transition_fraction: float = 0.05
    continuous_state: bool = False
    spike_state_encoding: str = "perdim_plif"
    spike_timesteps: int = 4
    spike_gradient_checkpointing: bool = True
    spike_state_population: int = 0
    spike_state_population_range: float = 4.0
    spike_state_attention_gain: float = 1.0
    action_dim: int = 7
    state_dim: int = 8
    horizon: int = 12
    num_state_tokens: int = 2
    num_layers: int = 3
    mlp_hidden_dim: int = 512
    state_hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class CTSConfig:
    """Control-Time Spiking. Disabled by default; old checkpoints have no `cts` entry.

    neuron: <charge>_<reset>. charge "plif" is the repository's EMA form
    v = u + g(I - u); "if" accumulates v = u + g*I. reset "hard" zeroes the
    membrane after a spike, "soft" subtracts the threshold. plif_hard with all
    carry flags off and memory off reproduces the base model exactly.
    carry_*: keep the membrane of that scope across control queries of one
    episode, scaled by a learnable per-channel retention.
    memory: C2 persistent dendritic trace on the instruction-token stream;
    memory_vision adds the same trace on the visual-token stream (ablation).
    """

    enabled: bool = False
    neuron: str = "if_soft"
    residual_clamp: bool = True
    carry_projection: bool = True
    carry_fusion: bool = True
    carry_head: bool = True
    carry_retention_init: float = 0.9
    carry_retention_learnable: bool = True
    memory: bool = True
    memory_tau_min: float = 1.0
    memory_tau_max: float = 16.0
    memory_gain_init: float = 1e-4
    memory_vision: bool = False
    history_tokens: bool = False
    shared_time_norm: bool = False

    @property
    def any_carry(self) -> bool:
        return self.carry_projection or self.carry_fusion or self.carry_head

    @property
    def stateful(self) -> bool:
        return self.enabled and (self.any_carry or self.memory or self.memory_vision or self.history_tokens)


@dataclass
class TurboVLAConfig:
    name: str = "TurboVLA"
    text_mask_version: str = "corrected"
    text_sentence_mask_version: str | None = None
    text_head_mask_version: str | None = None
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    action: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    cts: CTSConfig = field(default_factory=CTSConfig)

    def __post_init__(self) -> None:
        if self.name != "TurboVLA":
            raise ValueError(f"model name must be 'TurboVLA', got {self.name!r}")
        if self.text_mask_version not in {"legacy", "corrected"}:
            raise ValueError("text_mask_version must be legacy or corrected")
        for name in ("text_sentence_mask_version", "text_head_mask_version"):
            if getattr(self, name) not in {None, "legacy", "corrected"}:
                raise ValueError(f"{name} must be legacy, corrected, or None")
        if self.vision.num_views < 1:
            raise ValueError("vision.num_views must be positive")
        if self.vision.encoder_type not in {"dinov3", "sdtv3_19m"}:
            raise ValueError("vision.encoder_type must be 'dinov3' or 'sdtv3_19m'")
        if self.text.encoder_type not in {"bert", "smoothspike_bert"}:
            raise ValueError("text.encoder_type must be 'bert' or 'smoothspike_bert'")
        if self.text.timesteps < 1:
            raise ValueError("text.timesteps must be positive")
        if self.text.encoder_type == "smoothspike_bert" and self.text.attention_implementation not in {None, "eager"}:
            raise ValueError("SmoothSpike text encoder requires eager spike-driven attention")
        if self.vision.position_embedding not in {"view", "learned_patch"}:
            raise ValueError("vision.position_embedding must be 'view' or 'learned_patch'")
        if self.vision.compute_precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError("vision.compute_precision must be fp32, bf16, or bf16_autocast")
        if self.text.padding_length is not None:
            if self.text.padding_length < 1:
                raise ValueError("text.padding_length must be positive")
            if self.text.padding_length > self.text.max_length:
                raise ValueError("text.padding_length cannot exceed text.max_length")
        for instruction, length in self.text.padding_length_by_instruction.items():
            if not instruction:
                raise ValueError("text.padding_length_by_instruction cannot contain an empty instruction")
            if length < 1 or length > self.text.max_length:
                raise ValueError(f"invalid text padding length {length} for instruction {instruction!r}")
        if self.interaction.hidden_dim % self.interaction.nheads != 0:
            raise ValueError("interaction.hidden_dim must be divisible by interaction.nheads")
        if self.interaction.padding_strategy not in {"key_padding_mask", "zero_fill"}:
            raise ValueError("interaction.padding_strategy must be key_padding_mask or zero_fill")
        if self.interaction.residual_style not in {"normalized", "pre_norm"}:
            raise ValueError("interaction.residual_style must be normalized or pre_norm")
        if self.interaction.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("interaction.attention_backend must be manual or sdpa")
        if self.interaction.cross_attention_type not in {"ann", "spike_sdsa", "spike_wta"}:
            raise ValueError("interaction.cross_attention_type must be ann, spike_sdsa, or spike_wta")
        if self.interaction.backbone_type not in {"hybrid", "spike_temporal", "spike_wta", "spike_text_only"}:
            raise ValueError("interaction.backbone_type must be hybrid, spike_temporal, spike_wta, or spike_text_only")
        if self.interaction.cross_attention_type == "spike_wta" and self.interaction.backbone_type != "spike_wta":
            raise ValueError("spike_wta attention requires the full spike_wta backbone")
        if self.interaction.text_only_spike_projections and self.interaction.backbone_type != "spike_text_only":
            raise ValueError("text_only_spike_projections applies only to spike_text_only")
        if self.interaction.text_attention_routing not in {"wta", "sdsa"}:
            raise ValueError("text_attention_routing must be wta or sdsa")
        if self.interaction.text_attention_routing != "wta" and self.interaction.backbone_type != "spike_text_only":
            raise ValueError("The text routing ablation applies only to spike_text_only")
        if self.action.head_type in {"minimal_cross_wta", "progressive_full_wta"}:
            if self.interaction.backbone_type != "spike_text_only" or self.action.num_state_tokens != 2:
                raise ValueError("A-shaped heads require spike_text_only and two state tokens")
            if self.action.spike_timesteps < 1 or self.vision.compute_precision != "fp32":
                raise ValueError("A-shaped spike heads require positive T and FP32 vision")
        if self.action.head_type == "progressive_full_wta":
            if not (0 <= self.action.ann_fraction < 1 and 0 < self.action.transition_fraction <= 1-self.action.ann_fraction):
                raise ValueError("Invalid ANN/transition fractions")
        if self.interaction.backbone_type == "spike_text_only":
            if self.interaction.cross_attention_type != "spike_sdsa":
                raise ValueError("spike_text_only preserves the original spike_sdsa Cross")
            if self.interaction.text_dropout != 0 or self.interaction.compute_precision != "fp32":
                raise ValueError("Spike text replacement requires zero text dropout and fp32")
            if self.interaction.padding_strategy != "key_padding_mask":
                raise ValueError("Spike text replacement requires explicit padding isolation")
            # Cross DropPath and residual style keep their original values;
            # optional spike projections do not imply the full-WTA Cross.
        if self.interaction.backbone_type in {"spike_temporal", "spike_wta"}:
            expected = "spike_wta" if self.interaction.backbone_type == "spike_wta" else "spike_sdsa"
            if self.interaction.cross_attention_type != expected:
                raise ValueError(f"{self.interaction.backbone_type} requires cross_attention_type={expected}")
            if self.interaction.text_dropout != 0 or self.interaction.fusion_dropout != 0:
                raise ValueError("Temporal spike fusion requires zero attention/text dropout")
            if self.interaction.fusion_droppath != 0:
                raise ValueError("Temporal spike fusion uses membrane residuals without DropPath; set fusion_droppath=0")
            if self.interaction.compute_precision != "fp32":
                raise ValueError("Temporal spike fusion initially requires fp32")
        if min(self.interaction.wta_text_topk, self.interaction.wta_vision_topk) < 1:
            raise ValueError("WTA budgets must be positive")
        if self.interaction.wta_surrogate_temperature <= 0 or self.interaction.spike_norm_eps <= 0:
            raise ValueError("WTA temperature and normalization epsilon must be positive")
        if not 0 < self.interaction.spike_norm_momentum <= 1:
            raise ValueError("spike_norm_momentum must be in (0,1]")
        if self.interaction.cross_timesteps < 1:
            raise ValueError("interaction.cross_timesteps must be positive")
        if self.interaction.cross_attention_type == "spike_sdsa" and self.interaction.fusion_dropout != 0:
            raise ValueError("spike_sdsa requires fusion_dropout=0 to preserve binary spike operands")
        if self.interaction.compute_precision not in {"fp32", "bf16_autocast"}:
            raise ValueError("interaction.compute_precision must be fp32 or bf16_autocast")
        if self.action.action_dim < 1 or self.action.state_dim < 1 or self.action.horizon < 1:
            raise ValueError("action dimensions and horizon must be positive")
        if self.action.head_type not in {"ann", "spike_sdsa", "minimal_cross_wta", "progressive_full_wta", "hybrid_spike_ffn", "deep_spike_wta", "full_spike_state8_wta", "full_spike_state8_shiftmax", "full_spike_state8_shiftmax_continuous_inputs", "full_spike_state8_shiftmax_continuous_inputs_ann_ffn", "full_spike_state8_shiftmax_ann_ffn"}:
            raise ValueError("Unknown action head type")
        coordinate_heads = {"full_spike_state8_wta", "full_spike_state8_shiftmax", "full_spike_state8_shiftmax_continuous_inputs", "full_spike_state8_shiftmax_continuous_inputs_ann_ffn", "full_spike_state8_shiftmax_ann_ffn"}
        coordinate_ann = self.action.head_type in coordinate_heads and self.action.spike_state_encoding == "perdim_ann"
        if self.action.continuous_state and self.action.head_type not in {"spike_sdsa", "deep_spike_wta"} and not coordinate_ann:
            raise ValueError("continuous_state applies to the spike-body action heads")
        if self.action.head_type == "spike_sdsa":
            if self.action.spike_state_population < 0:
                raise ValueError("spike_state_population must be nonnegative")
            if self.action.spike_state_population_range <= 0 or self.action.spike_state_attention_gain <= 0:
                raise ValueError("Population range and state attention gain must be positive")
            if self.interaction.backbone_type != "spike_text_only":
                raise ValueError("AA preserves A's spike_text_only fusion")
            if self.action.spike_timesteps != self.interaction.cross_timesteps:
                raise ValueError("AA action and temporal BN must use the same T")
            if self.action.dropout != 0 or self.vision.compute_precision != "fp32":
                raise ValueError("AA requires zero action dropout and FP32 vision")
        if self.action.head_type == "hybrid_spike_ffn":
            if self.interaction.backbone_type != "spike_text_only":
                raise ValueError("Hybrid FFN preserves A's spike_text_only fusion")
            if self.action.spike_timesteps < 1 or self.vision.compute_precision != "fp32":
                raise ValueError("Hybrid FFN requires positive neuron time and FP32 vision")
        if self.action.head_type == "deep_spike_wta":
            if self.interaction.backbone_type != "spike_text_only" or not self.action.continuous_state:
                raise ValueError("Deep WTA preserves A's upstream fusion and continuous state")
            if self.action.spike_timesteps < 1 or self.vision.compute_precision != "fp32":
                raise ValueError("Deep WTA requires positive neuron time and FP32 vision")
        if self.action.head_type in coordinate_heads:
            if self.interaction.backbone_type != "spike_text_only":
                raise ValueError("Coordinate state heads preserve A's spike_text_only fusion")
            if self.action.continuous_state != coordinate_ann:
                raise ValueError("perdim_ann requires continuous_state; event encodings must leave it disabled")
            if self.action.num_state_tokens != self.action.state_dim:
                raise ValueError("Full state WTA requires one token per state coordinate")
            if self.action.spike_state_encoding not in {"perdim_plif", "multithreshold", "project_then_plif", "perdim_ann"}:
                raise ValueError("Unknown coordinate event encoding")
            if self.action.spike_state_population < 1 or self.action.spike_state_population_range <= 0:
                raise ValueError("Population bin count and range must be positive")
            if self.action.spike_timesteps < 1 or self.vision.compute_precision != "fp32":
                raise ValueError("Full state WTA requires positive neuron time and FP32 vision")
        if self.cts.enabled:
            if self.cts.neuron not in {"plif_hard", "plif_soft", "if_hard", "if_soft"}:
                raise ValueError("cts.neuron must be plif_hard, plif_soft, if_hard or if_soft")
            if (self.interaction.backbone_type != "spike_text_only"
                    or not self.interaction.text_only_spike_projections):
                raise ValueError("CTS is defined on spike_text_only with spike projections (the base upstream)")
            cts_heads = {"ann", "full_spike_state8_shiftmax", "full_spike_state8_shiftmax_continuous_inputs",
                         "full_spike_state8_shiftmax_continuous_inputs_ann_ffn", "full_spike_state8_shiftmax_ann_ffn"}
            if self.action.head_type not in cts_heads:
                raise ValueError(f"CTS supports action heads {sorted(cts_heads)}")
            if self.action.head_type == "ann" and self.cts.carry_head:
                raise ValueError("An ANN action head has no neuron state; set cts.carry_head=false")
            if self.action.head_type == "ann" and self.cts.history_tokens:
                raise ValueError("History tokens are implemented for the spike action heads")
            if not 0.0 < self.cts.carry_retention_init < 1.0:
                raise ValueError("cts.carry_retention_init must be in (0, 1)")
            if not 0.0 < self.cts.memory_tau_min <= self.cts.memory_tau_max:
                raise ValueError("cts memory timescales must satisfy 0 < tau_min <= tau_max")

    @property
    def sentence_mask_version(self) -> str:
        return self.text_sentence_mask_version or self.text_mask_version

    @property
    def head_mask_version(self) -> str:
        return self.text_head_mask_version or self.text_mask_version

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TurboVLAConfig":
        data = dict(payload)
        return cls(
            name=str(data.get("name", "TurboVLA")),
            # Unversioned checkpoints were trained with the original mask behavior.
            text_mask_version=str(data.get("text_mask_version", "legacy")),
            text_sentence_mask_version=data.get("text_sentence_mask_version"),
            text_head_mask_version=data.get("text_head_mask_version"),
            text=TextEncoderConfig(**dict(data.get("text", {}))),
            vision=VisionEncoderConfig(**dict(data.get("vision", {}))),
            interaction=InteractionConfig(**dict(data.get("interaction", {}))),
            action=ActionHeadConfig(**dict(data.get("action", {}))),
            cts=CTSConfig(**dict(data.get("cts") or {})),
        )
