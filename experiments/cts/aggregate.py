"""Aggregate LIBERO evaluations: per-suite rates, Wilson CIs, seed mean±std, paired tests.

Input: evaluation directories written by experiments/cts/evaluate.py, grouped
by method as NAME=dir1,dir2,... (one directory per training seed).

    python experiments/cts/aggregate.py base=eval/E0-s42,eval/E0-s43 cts=eval/E1-s42,eval/E1-s43 \
        --reference base --markdown results.md

Paired test: every evaluation uses the same initial state for episode i of
task t, so two methods are compared with an exact two-sided McNemar test on
the discordant episodes (pooled over seeds).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SHORT = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}


def wilson(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (A only) and c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(directory: str) -> dict:
    summary = json.loads((Path(directory) / "summary.json").read_text())
    return summary["suites"]


def method_table(runs: list[dict]) -> dict:
    table = {}
    for suite in SUITES:
        present = [run[suite] for run in runs if suite in run]
        if not present:
            continue
        rates = [item["successes"] / item["episodes"] for item in present]
        successes = sum(item["successes"] for item in present)
        episodes = sum(item["episodes"] for item in present)
        table[suite] = {"mean": statistics.mean(rates),
                        "std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
                        "pooled_ci": wilson(successes, episodes), "seeds": len(rates)}
    if len(table) == len(SUITES):
        averages = [statistics.mean(run[suite]["successes"] / run[suite]["episodes"] for suite in SUITES)
                    for run in runs if all(suite in run for suite in SUITES)]
        table["average"] = {"mean": statistics.mean(averages),
                            "std": statistics.stdev(averages) if len(averages) > 1 else 0.0,
                            "seeds": len(averages)}
    return table


def paired(runs_a: list[dict], runs_b: list[dict]) -> dict:
    """Discordant episodes of A vs B, matched by seed index, suite, task and episode."""
    result = {}
    for suite in SUITES:
        b = c = 0
        for run_a, run_b in zip(runs_a, runs_b):
            if suite not in run_a or suite not in run_b:
                continue
            for task, episodes_a in run_a[suite]["tasks"].items():
                episodes_b = run_b[suite]["tasks"].get(task, [])
                for success_a, success_b in zip(episodes_a, episodes_b):
                    b += success_a and not success_b
                    c += success_b and not success_a
        result[suite] = {"a_only": b, "b_only": c, "p_value": mcnemar_exact(b, c)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("methods", nargs="+", help="NAME=dir1,dir2,...")
    parser.add_argument("--reference", help="method used as A in paired tests")
    parser.add_argument("--markdown", help="write the table to this file")
    parser.add_argument("--json", help="write all numbers to this file")
    options = parser.parse_args()
    methods = {}
    for item in options.methods:
        name, _, directories = item.partition("=")
        methods[name] = [load(directory) for directory in directories.split(",") if directory]
    tables = {name: method_table(runs) for name, runs in methods.items()}

    lines = ["| Method | seeds | " + " | ".join(SHORT[s] for s in SUITES) + " | Avg |",
             "|---|---|" + "---|" * (len(SUITES) + 1)]
    for name, table in tables.items():
        cells = []
        for suite in (*SUITES, "average"):
            entry = table.get(suite)
            if entry is None:
                cells.append("–")
                continue
            text = f"{100 * entry['mean']:.1f}"
            if entry["seeds"] > 1:
                text += f" ± {100 * entry['std']:.1f}"
            cells.append(text)
        seeds = max(entry["seeds"] for entry in table.values()) if table else 0
        lines.append(f"| {name} | {seeds} | " + " | ".join(cells) + " |")
    output = {"tables": tables}
    if options.reference:
        reference = methods[options.reference]
        lines += ["", f"Paired exact McNemar vs {options.reference} "
                  "(discordant episodes: ref-only / method-only, p):", ""]
        output["paired"] = {}
        for name, runs in methods.items():
            if name == options.reference:
                continue
            test = paired(reference, runs)
            output["paired"][name] = test
            lines.append(f"- {name}: " + "; ".join(
                f"{SHORT[s]} {t['a_only']}/{t['b_only']} p={t['p_value']:.3g}" for s, t in test.items()))
    text = "\n".join(lines)
    print(text)
    if options.markdown:
        Path(options.markdown).write_text(text + "\n")
    if options.json:
        Path(options.json).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
