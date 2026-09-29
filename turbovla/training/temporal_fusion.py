"""Strict common initialization and buffer-only calibration for fusion v2."""
from itertools import chain

import torch

from turbovla.models.components.temporal_fusion import begin_calibration, finish_calibration


def common_pretrained_names(model):
    interaction = model.vision_language_interaction
    if model.config.interaction.backbone_type == "spike_text_only":
        return {name for name, _ in interaction.named_parameters()
                if ".spike_nodes." not in name and ".norms." not in name and ".cts_memory" not in name}
    return interaction.common_pretrained_parameter_names()


def load_common_interaction(model, source):
    interaction = model.vision_language_interaction
    wanted = common_pretrained_names(model)
    text_only = model.config.interaction.backbone_type == "spike_text_only"
    target = interaction.state_dict()
    mapped, omitted = {}, []
    for key, tensor in source.items():
        prefix = "transformer.encoder."
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        if not name.startswith(("fusion_layers.", "text_layers.")):
            continue
        replaced_norms = (".norm1.", ".norm2.") if text_only else (
            ".layer_norm_v.", ".layer_norm_l.", ".norm1.", ".norm2.")
        if any(part in name for part in replaced_norms):
            omitted.append(key)
            continue
        name = name.replace(".self_attn.in_proj_weight", ".self_attn.qkv.weight")
        name = name.replace(".self_attn.in_proj_bias", ".self_attn.qkv.bias")
        if name not in wanted or target[name].shape != tensor.shape:
            raise RuntimeError(f"Unexpected or incompatible common fusion tensor: {key}")
        mapped[name] = tensor
    if set(mapped) != wanted:
        raise RuntimeError(f"Missing common fusion tensors: {sorted(wanted-set(mapped))}")
    if len(omitted) != (4 if text_only else 8) * model.config.interaction.num_layers:
        raise RuntimeError(f"Unexpected count of replaced ANN norm tensors: {len(omitted)}")
    target.update(mapped)
    interaction.load_state_dict(target, strict=True)
    return {"mapped": len(mapped), "skipped_missing": 0, "skipped_shape": 0,
            "load_missing_after_update": 0, "load_unexpected_after_update": 0,
            "missing_keys": [], "shape_mismatch_keys": [],
            "intentionally_replaced_ann_norms": omitted,
            "initialization_kind": "common_groundingdino_into_spike_text_only" if text_only
            else "common_groundingdino_into_temporal_spike_fusion"}


def calibrate_training_iterator(model, iterator, device, batches, already_calibrated=False):
    if already_calibrated:
        return iterator, {"resumed": True, "layers": finish_calibration(model)}
    prior_training = model.training
    model.eval()
    cached = []
    begin_calibration(model)
    try:
        with torch.no_grad():
            for _ in range(batches):
                batch = next(iterator)
                cached.append(batch)
                images, instructions, states, _, _ = batch
                if isinstance(images, dict):
                    images = {key: value.to(device) for key, value in images.items()}
                else:
                    images = images.to(device)
                predictions = model(instructions, images, states.to(device))
                if not torch.isfinite(predictions).all():
                    raise RuntimeError("Non-finite output during training-data spike calibration")
        layers = finish_calibration(model)
    finally:
        from turbovla.models.components.temporal_fusion import calibration_modules
        for _, module in calibration_modules(model):
            module.calibrating = False
        model.train(prior_training)
    # Calibration does not shift training sample order or count as optimizer updates.
    return iter(chain(cached, iterator)), {
        "calibration_kind": "training_data_static_channel_affine",
        "optimizer_steps": 0, "batches": batches, "layers": layers,
    }
