"""Neuron-level diagnostics shared by experiments/cts/diagnose.py.

* NeuronRecorder: forward hooks on every spiking neuron (base PLIF or CTS);
  per control query it measures firing rate, the fraction of valid units that
  never fire in the query (silent), and how many distinct per-unit spike
  counts occur (the realized code resolution).
* constant_input_resolution: drive one neuron with constant currents for
  T steps per query and Q queries (retention applied between queries) and
  report distinct spike counts, dead zone and best affine-decode RMSE: the
  analytic probe of the research plan, on the learned parameters.
* SpikeTape: records every spike tensor of a forward to count flips between
  a clean and a perturbed run, layer by layer.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from ..models.components.spike_fusion import FunctionalPLIF
from ..models.components.wta_operators import BipolarPLIF
from .neuron import Carried, CTSNeuron

NEURON_TYPES = (FunctionalPLIF, CTSNeuron)


def neuron_modules(model):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, NEURON_TYPES)]


def neuron_form(module) -> str:
    if isinstance(module, CTSNeuron):
        return f"{module.charge}_{module.reset}"
    return "plif_hard"


def is_bipolar(module) -> bool:
    return module.bipolar if isinstance(module, CTSNeuron) else isinstance(module, BipolarPLIF)


def neuron_parameters(module) -> dict:
    gain = float(module.decay_logit.detach().sigmoid())
    threshold = float(module.log_threshold.detach().exp())
    result = {"form": neuron_form(module), "g": gain, "theta": threshold,
              "bipolar": is_bipolar(module)}
    if isinstance(module, CTSNeuron) and module.retention_logit is not None:
        retention = module.retention_logit.detach().float().sigmoid()
        result.update(retention_mean=float(retention.mean()), retention_min=float(retention.min()),
                      retention_max=float(retention.max()))
    return result


def _group(name: str) -> str:
    for prefix, group in (("vision_projection", "projection/vision"),
                          ("text_encoder.text_projection", "projection/text"),
                          ("vision_language_interaction.fusion_layers", "fusion/cross"),
                          ("vision_language_interaction.text_layers", "fusion/text"),
                          ("action_head", "head")):
        if name.startswith(prefix):
            return group
    return "other"


class NeuronRecorder:
    def __init__(self, model):
        self.modules = OrderedDict(neuron_modules(model))
        self.current: dict[str, dict] = {}
        self.queries: dict[str, list[dict]] = {name: [] for name in self.modules}
        self.handles = [module.register_forward_hook(self._hook(name), with_kwargs=True)
                        for name, module in self.modules.items()]

    def _hook(self, name):
        def hook(module, args, kwargs, output):
            spike = output[0].detach().float()
            current = args[0].detach().float()
            valid = kwargs.get("valid")
            valid = (torch.ones_like(spike, dtype=torch.bool) if valid is None
                     else valid.expand_as(spike))
            entry = self.current.get(name)
            if entry is None:
                entry = self.current[name] = {"count": torch.zeros_like(spike), "fired": torch.zeros_like(spike),
                                              "valid": valid.clone(), "steps": 0, "current_abs": 0.0}
            entry["count"] += spike
            entry["fired"] += spike.abs()
            entry["steps"] += 1
            entry["current_abs"] += float(current.abs()[valid].mean()) if valid.any() else 0.0
        return hook

    def end_query(self):
        for name, entry in self.current.items():
            valid = entry["valid"]
            if not valid.any():
                continue
            fired, count = entry["fired"][valid], entry["count"][valid]
            self.queries[name].append({
                "steps": entry["steps"],
                "rate": float(fired.mean() / entry["steps"]),
                "silent": float((fired == 0).float().mean()),
                "levels": int(torch.unique(count).numel()),
                "current_abs": entry["current_abs"] / entry["steps"],
            })
        self.current = {}

    def close(self):
        for handle in self.handles:
            handle.remove()

    def report(self) -> dict:
        layers = OrderedDict()
        for name, module in self.modules.items():
            records = self.queries[name]
            if not records:
                continue
            mean = lambda key: sum(r[key] for r in records) / len(records)  # noqa: E731
            layers[name] = {"group": _group(name), **neuron_parameters(module),
                            "steps_per_query": records[0]["steps"], "rate": mean("rate"),
                            "silent": mean("silent"), "levels": mean("levels"),
                            "current_abs": mean("current_abs"), "queries": len(records)}
        groups = OrderedDict()
        for layer in layers.values():
            group = groups.setdefault(layer["group"], {"layers": 0, "rate": 0.0, "silent": 0.0, "levels": 0.0})
            group["layers"] += 1
            for key in ("rate", "silent", "levels"):
                group[key] += layer[key]
        for group in groups.values():
            for key in ("rate", "silent", "levels"):
                group[key] /= group["layers"]
        return {"groups": groups, "layers": layers}


@torch.no_grad()
def constant_input_resolution(module, steps: int, queries: int = 1, grid: int = 4001, span: float = 2.0) -> dict:
    """Spike-count code of one neuron for constant currents (signed range if bipolar).

    Every query lasts `steps` timesteps; between queries a CTS neuron with carry
    keeps its membrane (scaled by its retention), any other neuron restarts.
    Reports distinct spike counts, the dead zone (largest |I| that never fires),
    and the RMSE of the best affine decode I ~ a * count + b, averaged over channels.
    """
    low = -span if is_bipolar(module) else 0.0
    currents = torch.linspace(low, span, grid).unsqueeze(1)
    if isinstance(module, CTSNeuron):
        currents = currents.expand(grid, module.channels).contiguous()
    membrane, total = None, torch.zeros_like(currents)
    for query in range(queries):
        if query > 0:
            membrane = Carried(membrane) if isinstance(module, CTSNeuron) and module.carry else None
        for _ in range(steps):
            spike, membrane = module(currents, membrane)
            total += spike
    results = []
    for column in range(total.shape[1]):
        count, current = total[:, column].double(), currents[:, column].double()
        design = torch.stack([count, torch.ones_like(count)], dim=1)
        solution = torch.linalg.lstsq(design, current.unsqueeze(1)).solution
        rmse = float((design @ solution - current.unsqueeze(1)).square().mean().sqrt())
        silent = count == 0
        results.append({"levels": int(torch.unique(count).numel()),
                        "dead_zone": float(current[silent].abs().max()) if silent.any() else 0.0,
                        "rmse": rmse, "silent_fraction": float(silent.double().mean())})
    summary = {key: sum(r[key] for r in results) / len(results) for key in results[0]}
    return {**summary, "steps": steps, "queries": queries, "range": [low, span]}


class SpikeTape:
    """Record the spike tensors of every neuron call, in call order."""

    def __init__(self, model):
        self.names = {module: name for name, module in neuron_modules(model)}
        self.tape: list[tuple[str, torch.Tensor, torch.Tensor | None]] = []
        self.handles = [module.register_forward_hook(self._hook, with_kwargs=True) for module in self.names]

    def _hook(self, module, args, kwargs, output):
        valid = kwargs.get("valid")
        self.tape.append((self.names[module], output[0].detach().to(torch.int8).cpu(),
                          None if valid is None else valid.detach().cpu()))

    def take(self):
        tape, self.tape = self.tape, []
        return tape

    def close(self):
        for handle in self.handles:
            handle.remove()


def flip_profile(clean, perturbed) -> OrderedDict:
    """Per neuron site: fraction of valid spike entries that differ, over all calls."""
    if len(clean) != len(perturbed):
        raise ValueError("Clean and perturbed runs made different neuron calls")
    profile = OrderedDict()
    for (name, a, valid), (other, b, _) in zip(clean, perturbed):
        if name != other or a.shape != b.shape:
            raise ValueError(f"Call order differs at {name}/{other}")
        mask = torch.ones_like(a, dtype=torch.bool) if valid is None else valid.expand_as(a)
        flips = int((a != b)[mask].sum())
        total = int(mask.sum())
        entry = profile.setdefault(name, {"group": _group(name), "flips": 0, "entries": 0,
                                          "spikes": 0})
        entry["flips"] += flips
        entry["entries"] += total
        entry["spikes"] += int(a[mask].abs().sum())
    for entry in profile.values():
        entry["flip_rate"] = entry["flips"] / max(entry["entries"], 1)
        entry["flips_per_spike"] = entry["flips"] / max(entry["spikes"], 1)
    return profile
