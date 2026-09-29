"""CTS efficiency (E5): per-query compute, theoretical energy and B=1 latency.

For every NAME=checkpoint (base, CTS or ANN; refreshed EMA exports or
training checkpoints) on consecutive queries of real LIBERO episodes:

* spiking steps per query, firing rate and silent fraction per neuron group
  (turbovla.cts.diagnostics.NeuronRecorder) and neuron updates per query;
* operations of every matmul / convolution (torch flop formulas, counted at
  the aten level, so attention products are included), per model part. An op
  with a spike-valued operand (all values integer multiples k*q of one
  quantum, k <= --max_level: binary, ternary or SDT-V3 integer spikes) counts
  as accumulates, AC = MAC x mean k of that operand (the sparser one if both
  are spike-valued); every other op counts as MACs;
* energy = 4.6 pJ x MAC + 0.9 pJ x AC (45 nm, 32-bit FP, Horowitz 2014). A
  theoretical estimate: it ignores memory traffic and neuron updates;
* the frozen text backbone runs once per episode for every model (cached for
  base and CTS alike); its cost is reported separately and excluded from the
  per-query numbers;
* the CTS carry that persists between queries (elements, fp32/bf16 bytes);
* B=1 latency per query on --device after warm-up (median / p90 / mean) at
  the evaluation numerics (fp32, highest matmul precision), and peak memory.

Example
  python experiments/cts/efficiency.py base=eval/E0-s42/ema_refreshed.pth \
      cts=eval/E1-s42/ema_refreshed.pth --device cuda:0 \
      --out results/efficiency.json --markdown results/efficiency.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils.flop_counter import flop_registry  # noqa: E402

from experiments.cts.diagnose import load_model, windows  # noqa: E402

E_MAC_PJ, E_AC_PJ = 4.6, 0.9
PARTS = (("text_encoder.bert", "text_backbone"), ("text_encoder.text_projection", "projection/text"),
         ("vision_encoder", "vision_encoder"), ("vision_projection", "projection/vision"),
         ("vision_language_interaction", "fusion"), ("action_head", "head"))
BIAS_FIRST = (torch.ops.aten.addmm, torch.ops.aten.baddbmm)


def part_of(name: str) -> str:
    for prefix, part in PARTS:
        if name == prefix or name.startswith(prefix + "."):
            return part
    return "other"


def spike_level(tensor, max_level: int):
    """Mean events per element when the values are integer multiples of one quantum, else None."""
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or tensor.numel() == 0:
        return None
    magnitude = tensor.detach().abs().float()
    nonzero = magnitude[magnitude > 0]
    if nonzero.numel() == 0:
        return 0.0
    ratio = nonzero / nonzero.min()
    rounded = ratio.round()
    if float(rounded.max()) > max_level or float((ratio - rounded).abs().max()) > 1e-3:
        return None
    return float(rounded.sum()) / tensor.numel()


class OpCounter(TorchDispatchMode):
    """MAC/AC counts of every flop-bearing aten op, attributed to the innermost model part."""

    def __init__(self, model, max_level: int):
        super().__init__()
        self.max_level = max_level
        self.stack: list[str] = []
        self.parts = defaultdict(lambda: {"mac": 0, "ac": 0.0, "spike_op_macs": 0, "neuron_updates": 0})
        self.handles, self.patched = [], []
        from turbovla.cts.diagnostics import NEURON_TYPES
        for name, module in model.named_modules():
            if not name:
                continue
            part = part_of(name)
            self.handles.append(module.register_forward_pre_hook(lambda *_, p=part: self.stack.append(p)))
            self.handles.append(module.register_forward_hook(self._leave, always_call=True))
            if isinstance(module, NEURON_TYPES):
                self.handles.append(module.register_forward_hook(self._neuron(part)))
            # Projections and heads step through time with forward_step, which bypasses module hooks.
            if hasattr(module, "forward_step"):
                self.patched.append(module)
                module.forward_step = self._scoped(module.forward_step, part)

    def _leave(self, *_):
        self.stack.pop()

    def _scoped(self, method, part):
        def scoped(*args, **kwargs):
            self.stack.append(part)
            try:
                return method(*args, **kwargs)
            finally:
                self.stack.pop()
        return scoped

    def _neuron(self, part):
        def hook(module, args, output):
            self.parts[part]["neuron_updates"] += output[0].numel()
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        for module in self.patched:
            del module.forward_step

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        packet = func._overloadpacket
        formula = flop_registry.get(packet)
        if formula is not None:
            macs = formula(*args, **kwargs, out_val=out) // 2
            operands = [a for a in args[1 if packet in BIAS_FIRST else 0:] if isinstance(a, torch.Tensor)][:2]
            levels = [level for level in (spike_level(a, self.max_level) for a in operands) if level is not None]
            entry = self.parts[self.stack[-1] if self.stack else "other"]
            if levels:
                entry["ac"] += macs * min(levels)
                entry["spike_op_macs"] += macs
            else:
                entry["mac"] += macs
        return out


def energy_mj(mac, ac) -> float:
    return (E_MAC_PJ * mac + E_AC_PJ * ac) * 1e-9


def summarize(parts: dict, divisor: int) -> dict:
    result = {}
    for part, entry in sorted(parts.items()):
        mac, ac = entry["mac"] / divisor, entry["ac"] / divisor
        dense = (entry["mac"] + entry["spike_op_macs"]) / divisor
        result[part] = {"mac": mac, "ac": ac, "dense_equivalent_mac": dense,
                        "spike_op_share": entry["spike_op_macs"] / divisor / dense if dense else 0.0,
                        "neuron_updates": entry["neuron_updates"] / divisor, "energy_mj": energy_mj(mac, ac)}
    total = {key: sum(value[key] for value in result.values())
             for key in ("mac", "ac", "dense_equivalent_mac", "neuron_updates", "energy_mj")}
    return {"total": total, "parts": result}


def cache_text(model):
    """Memoize the frozen text backbone per instruction batch (the evaluation policy's text cache)."""
    encoder = model.text_encoder
    original = encoder.encode_bert_hidden
    memo = {}

    def cached(instructions, device):
        # Per instruction: the base model encodes eval batches row by row (fixed B=1 spike inference).
        rows = []
        for item in instructions:
            key = (str(item), str(device))
            if key not in memo:
                memo[key] = original([item], device)
            rows.append(memo[key])
        return tuple(torch.cat(parts) for parts in zip(*rows))
    encoder.encode_bert_hidden = cached
    return original


def episode(model, batch, device, on_query=None, timer=None):
    """Run the consecutive queries of one window batch; CTS carries state, base restarts every query."""
    from turbovla.cts.state import reset_samples

    per_query, present, _, _ = batch
    stateful = model.config.cts.enabled
    carry = None
    for index, (samples, instructions, states, _, _) in enumerate(per_query):
        pixels = {"pixel_values": samples["pixel_values"].to(device)}
        states = states.to(device)
        model.text_encoder.encode_bert_hidden(instructions, device)  # cached from here on
        start = timer() if timer else None
        if stateful:
            _, carry = model.forward_cts(instructions, pixels, states, carry)
        else:
            model(instructions, pixels, states)
        if timer:
            timer(start)
        if stateful and not bool(present[:, index].all()):
            carry = reset_samples(carry, present[:, index].to(device))
        if on_query:
            on_query(carry)
    return carry


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure(name: str, path: str, options) -> dict:
    from turbovla.cts.diagnostics import NeuronRecorder
    from turbovla.cts.state import CONSTANT_KEYS, carry_numel

    device = torch.device(options.device)
    model, args, weights = load_model(path, options.weights, device)
    bert = cache_text(model)
    steps = model.config.interaction.cross_timesteps

    # Pass 1 at B=1 (base eval runs row by row): operation counts, neuron statistics, carry size.
    recorder = NeuronRecorder(model)
    counter = OpCounter(model, options.max_level)
    carry_parts = {}
    queries = 0
    iterator = windows(args, options.window, 1, options.num_workers, options.seed)
    for _ in range(options.count_episodes):
        batch = next(iterator)
        model.text_encoder.encode_bert_hidden(batch[0][0][1], device)  # text cache, outside the count
        with counter:
            carry = episode(model, batch, device, on_query=lambda _carry: recorder.end_query())
        queries += len(batch[0])
        for key, value in (carry or {}).items():
            if key not in CONSTANT_KEYS:
                carry_parts[key] = max(carry_parts.get(key, 0), carry_numel({key: value}))
    recorder.close()
    counter.close()
    if counter.parts.get("text_backbone", {}).get("mac"):
        raise RuntimeError("text backbone ran inside a query; the text cache is not in effect")
    per_query = summarize(counter.parts, queries)
    neurons = recorder.report()
    group_steps = defaultdict(list)
    for layer in neurons["layers"].values():
        group_steps[layer["group"]].append(layer["steps_per_query"])
    for group, values in neurons["groups"].items():
        values["steps_per_query"] = statistics.median(group_steps[group])

    # The text backbone, once per episode (B=1).
    instructions = next(iterator)[0][0][1][:1]
    text_counter = OpCounter(model, options.max_level)
    with text_counter:
        bert(instructions, device)
    text_counter.close()
    text = summarize(text_counter.parts, 1)["total"]
    text_times = []
    for _ in range(options.text_repeats):
        synchronize(device)
        start = time.perf_counter()
        bert(instructions, device)
        synchronize(device)
        text_times.append(1e3 * (time.perf_counter() - start))
    text["latency_ms"] = statistics.median(text_times)

    # Pass 2: B=1 latency on consecutive queries of real episodes, no hooks.
    times = []

    def begin():
        synchronize(device)
        return time.perf_counter()

    def end(start):
        synchronize(device)
        times.append(1e3 * (time.perf_counter() - start))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator = windows(args, options.window, 1, options.num_workers, options.seed + 1)
    while len(times) < options.warmup + options.latency_queries:
        episode(model, next(iterator), device, timer=lambda start=None: begin() if start is None else end(start))
    timed = times[options.warmup:options.warmup + options.latency_queries]
    timed_sorted = sorted(timed)
    latency = {"median": statistics.median(timed), "p90": timed_sorted[int(0.9 * (len(timed) - 1))],
               "mean": statistics.fmean(timed), "queries": len(timed), "warmup": options.warmup}
    element = sum(carry_parts.values())
    return {
        "name": name, "checkpoint": path, "weights": weights, "cross_timesteps": steps,
        "cts": model.config.cts.__dict__ if model.config.cts.enabled else None,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "per_query": per_query, "neurons": {"groups": neurons["groups"]},
        "text_backbone_per_episode": text,
        "carry": {"elements": element, "bytes_fp32": 4 * element, "bytes_bf16": 2 * element,
                  "elements_by_part": carry_parts},
        "latency_ms": latency,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None,
        "counted_queries": queries,
    }


def markdown(results: list[dict]) -> str:
    lines = ["| model | T_q | fusion steps | fusion rate | head rate | MAC (G) | AC (G) | energy (mJ, theor.) "
             "| carry (MB fp32) | latency ms (median / p90) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for result in results:
        total, groups = result["per_query"]["total"], result["neurons"]["groups"]
        fusion, head = groups.get("fusion/cross", {}), groups.get("head", {})
        rate = lambda group: f"{group['rate']:.3f}" if group else "-"  # noqa: E731
        lines.append(
            f"| {result['name']} | {result['cross_timesteps']} | {fusion.get('steps_per_query', '-')} "
            f"| {rate(fusion)} | {rate(head)} | {total['mac'] / 1e9:.3f} | {total['ac'] / 1e9:.3f} "
            f"| {total['energy_mj']:.3f} | {result['carry']['bytes_fp32'] / 2 ** 20:.1f} "
            f"| {result['latency_ms']['median']:.2f} / {result['latency_ms']['p90']:.2f} |")
    text = results[0]["text_backbone_per_episode"]
    lines += ["", f"Per query, B=1, {results[0]['device_name']}, fp32 highest precision; text backbone cached "
              f"and excluded (once per episode: {text['mac'] / 1e9:.2f} G MAC, {text['energy_mj']:.2f} mJ, "
              f"{text['latency_ms']:.1f} ms). Energy = {E_MAC_PJ} pJ x MAC + {E_AC_PJ} pJ x AC (45 nm), "
              "a theoretical estimate."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="NAME=checkpoint")
    parser.add_argument("--weights", choices=["auto", "ema", "raw"], default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", type=int, default=8, help="consecutive queries per episode sample")
    parser.add_argument("--count_episodes", type=int, default=16, help="B=1 episode samples for the counts")
    parser.add_argument("--latency_queries", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--text_repeats", type=int, default=10)
    parser.add_argument("--max_level", type=int, default=8, help="largest spike count treated as events")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--markdown")
    options = parser.parse_args()
    results = []
    for item in options.models:
        name, _, path = item.partition("=")
        if not path:
            parser.error(f"expected NAME=checkpoint, got {item!r}")
        results.append(measure(name, path, options))
        print(json.dumps({k: results[-1][k] for k in ("name", "cross_timesteps", "latency_ms")}), flush=True)
    table = markdown(results)
    print(table)
    for path, payload in ((options.out, json.dumps(results, indent=2) + "\n"), (options.markdown, table)):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(payload)


if __name__ == "__main__":
    main()
