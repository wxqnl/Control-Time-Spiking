"""Install CTS neurons and memory into a constructed TurboVLA.

Called at the end of TurboVLA.__init__ when config.cts.enabled. Only neuron
modules inside the named scopes are swapped; every other parameter keeps its
constructed value, and no initialization RNG is consumed.
"""
from __future__ import annotations

from torch import nn

from ..models.components.spike_fusion import FunctionalPLIF
from ..models.components.wta_operators import BipolarPLIF, MaskedTemporalBatchNorm
from .neuron import CTSNeuron, DendriticMemory

SCOPES = ("projection", "fusion", "head")


def _norm_channels(norm) -> int:
    if isinstance(norm, MaskedTemporalBatchNorm):
        return norm.channels
    if isinstance(norm, nn.LayerNorm):
        return int(norm.normalized_shape[-1])
    raise TypeError(f"Cannot infer neuron width from {type(norm)}")


def _swap(container, key, channels, config, carry):
    old = container[key]
    if not isinstance(old, FunctionalPLIF):
        raise TypeError(f"Expected a PLIF neuron at {key}, found {type(old)}")
    charge, reset = config.neuron.split("_")
    container[key] = CTSNeuron(
        channels, bipolar=isinstance(old, BipolarPLIF), charge=charge, reset=reset,
        residual_clamp=config.residual_clamp, carry=carry,
        retention_init=config.carry_retention_init,
        retention_learnable=config.carry_retention_learnable,
        decay_logit=float(old.decay_logit.detach()), log_threshold=float(old.log_threshold.detach()),
    ).to(device=old.decay_logit.device)


def neuron_sites(model):
    """Yield (scope, container, key, channels) for every spiking neuron CTS manages."""
    sites = []
    vision = model.vision_projection
    sites += [("projection", vision.spike_nodes, "input", vision.input_norm.channels),
              ("projection", vision.spike_nodes, "hidden", vision.hidden_norm.channels)]
    text = model.text_encoder.text_projection
    sites.append(("projection", text.spike_nodes, "input", text.input_norm.channels))
    interaction = model.vision_language_interaction
    for layer in interaction.fusion_layers:
        attn = layer.attn
        dims = {"input_v": attn.v_proj.in_features, "input_l": attn.l_proj.in_features}
        for name in attn.spike_nodes.keys():
            sites.append(("fusion", attn.spike_nodes, name, dims.get(name, attn.embed_dim)))
    for layer in interaction.text_layers:
        for name in layer.self_attn.spike_nodes.keys():
            sites.append(("fusion", layer.self_attn.spike_nodes, name, _norm_channels(layer.self_attn.norms[name])))
        for name in layer.spike_nodes.keys():
            sites.append(("fusion", layer.spike_nodes, name, _norm_channels(layer.norms[name])))
    if model.config.action.head_type != "ann":
        decoder = model.action_head.decoder
        for layer in decoder.layers:
            for attention in (layer.self_attn, layer.cross_attn):
                for name in attention.spike_nodes.keys():
                    sites.append(("head", attention.spike_nodes, name, _norm_channels(attention.norms[name])))
            if hasattr(layer, "spike_nodes"):
                for name in layer.spike_nodes.keys():
                    sites.append(("head", layer.spike_nodes, name, _norm_channels(layer.norms[name])))
        readout = decoder.action_projection
        for index in range(len(readout.spike_nodes)):
            sites.append(("head", readout.spike_nodes, index, _norm_channels(readout.norms[index])))
        if getattr(model.action_head.state_projection, "encoding", "perdim_ann") != "perdim_ann":
            raise ValueError("CTS keeps the base continuous (perdim_ann) state encoder")
    return sites


def apply_cts(model):
    config = model.config.cts
    carry = {"projection": config.carry_projection, "fusion": config.carry_fusion, "head": config.carry_head}
    for scope, container, key, channels in neuron_sites(model):
        _swap(container, key, channels, config, carry[scope])
    hidden = model.config.interaction.hidden_dim
    for layer in model.vision_language_interaction.fusion_layers:
        if config.memory:
            layer.cts_memory = DendriticMemory(hidden, config.memory_tau_min, config.memory_tau_max,
                                               config.memory_gain_init)
        if config.memory_vision:
            layer.cts_memory_v = DendriticMemory(hidden, config.memory_tau_min, config.memory_tau_max,
                                                 config.memory_gain_init)
    if config.shared_time_norm:
        for module in model.modules():
            if isinstance(module, MaskedTemporalBatchNorm):
                module.time_shared = True
    return model


def cts_parameter_names(model):
    return [name for name, _ in model.named_parameters()
            if ".retention_logit" in name or ".cts_memory" in name]
