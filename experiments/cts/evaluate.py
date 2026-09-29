"""Evaluate a CTS (or base) checkpoint on the four LIBERO suites, standard protocol.

1. If the checkpoint is a training checkpoint, export EMA weights with training
   statistics first (refresh_norms.py for CTS, refresh_wta_norms.py for base;
   ANN / hybrid checkpoints have no WTA norms and are evaluated as saved);
   --weights raw instead evaluates the raw weights with their training norms.
2. Run the four suites in parallel over the given GPUs with the protocol of the
   base pipeline: EMA weights, 50 trials/task, seed 7, 10 wait steps, 12-step
   open loop, 256px, fp32/highest.
3. Write summary.json (per suite and overall) for aggregate.py.

Extra rollout options (perturbations, stress, traces, CTS test-time resets)
pass through after `--`, e.g.
    python experiments/cts/evaluate.py --checkpoint ckpt.pth --out_dir eval/E4-blackout \
        --gpus 0,1,2,3 -- --perturb_mode blackout --perturb_start_query 4 --perturb_queries 2
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def environment(gpu: str | None) -> dict:
    env = os.environ.copy()
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    env.update(PYTHONPATH=f"{CODE}:{CODE}/third_party/vla_adapter:{CODE}/third_party/starvla_runtime",
               PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0",
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def run(command: list[str], log: Path, env: dict, timeout: int):
    if log.exists():
        raise FileExistsError(f"Refusing to rerun into an existing log: {log}")
    print(json.dumps({"event": "start", "command": shlex.join(command), "log": str(log)}), flush=True)
    with log.open("x") as stream:
        stream.write("command: " + shlex.join(command) + "\n")
        stream.flush()
        result = subprocess.run(["timeout", "--signal=TERM", "--kill-after=10s", str(timeout), *command],
                                cwd=CODE, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {log}")


def inspect(path: Path) -> tuple[bool, bool, bool]:
    """(refreshed EMA export, CTS model, WTA norms that need a training-data refresh)."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    status = checkpoint.get("wta_normalization", {}).get("ema", {}).get("status")
    config = checkpoint.get("model_config", {})
    cts = bool((config.get("cts") or {}).get("enabled"))
    backbone = config.get("interaction", {}).get("backbone_type", "hybrid")
    return status == "refreshed_with_training_data", cts, cts or backbone in {"spike_wta", "spike_text_only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--refresh_batches", type=int, default=64)
    parser.add_argument("--refresh_window", type=int, default=4)
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema",
                        help="raw: training weights with their training running norms, no refresh (E2-f)")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--timeout_hours", type=float, default=12.0)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    extra = [item for item in options.extra if item != "--"]

    from experiments.libero.settings import BERT, DINO, LIBERO_CONFIG, LIBERO_ROOT

    out = Path(options.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    gpus = [item.strip() for item in options.gpus.split(",") if item.strip()]
    suites = [item.strip() for item in options.suites.split(",") if item.strip()]
    source = Path(options.checkpoint).expanduser().resolve()
    exported, cts, needs_refresh = inspect(source)
    checkpoint = source
    if options.weights == "raw" and exported:
        raise ValueError("An EMA export has no raw weights; pass the training checkpoint")
    # ANN / hybrid models (E0e) have no WTA norms: their EMA weights are evaluated as saved.
    if options.weights == "ema" and not exported and needs_refresh:
        checkpoint = out / "ema_refreshed.pth"
        script = "experiments/cts/refresh_norms.py" if cts else "experiments/libero/refresh_wta_norms.py"
        command = [sys.executable, script, "--checkpoint", str(source), "--output", str(checkpoint),
                   "--batches", str(options.refresh_batches)]
        if cts:
            command += ["--window", str(options.refresh_window)]
        run(command, out / "refresh.log", environment(gpus[0]), 7200)

    def suite_job(gpu: str, suite: str):
        command = [sys.executable, "experiments/libero/evaluate.py",
                   "--ckpt_path", str(checkpoint), "--checkpoint_weights", options.weights,
                   "--libero_root", LIBERO_ROOT, "--dinov3_path", DINO, "--bert_path", BERT,
                   "--stats_path", str(CODE / "experiments/libero/configs/libero_all4_stats.json"),
                   "--stats_key", "libero_all4_no_noops", "--normalize_binary_gripper", "false",
                   "--task_suite_name", suite, "--num_trials_per_task", str(options.trials),
                   "--num_steps_wait", "10", "--num_open_loop_steps", "12", "--chunk_size", "12",
                   "--env_img_res", "256", "--seed", str(options.seed), "--control_mode", "relative",
                   "--precision", "fp32", "--float32_matmul_precision", "auto",
                   "--mujoco_gl", "osmesa", "--pyopengl_platform", "osmesa",
                   "--save_video", "true" if options.save_video else "false",
                   "--video_out_path", str(out / "videos"),
                   "--result_json_path", str(out / f"{suite}.result.json"),
                   "--log_path", str(out / f"{suite}.events.log"), *extra]
        env = environment(gpu)
        env["LIBERO_CONFIG_PATH"] = LIBERO_CONFIG
        run(command, out / f"{suite}.stdout.log", env, int(options.timeout_hours * 3600))

    # One worker per GPU; each takes the next suite in the queue.
    queue = list(suites)
    def worker(gpu):
        while queue:
            suite_job(gpu, queue.pop(0))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for job in [pool.submit(worker, gpu) for gpu in gpus]:
            job.result()

    summary = {"checkpoint": str(source), "evaluated": str(checkpoint), "weights": options.weights,
               "seed": options.seed,
               "trials_per_task": options.trials, "extra_rollout_args": extra, "suites": {}}
    successes = episodes = 0
    for suite in suites:
        result = json.loads((out / f"{suite}.result.json").read_text())
        if result["total_episodes"] != 10 * options.trials:
            raise RuntimeError(f"Incomplete evaluation for {suite}")
        summary["suites"][suite] = {
            "successes": result["total_successes"], "episodes": result["total_episodes"],
            "success_rate": result["overall_success_rate"],
            "tasks": {task["task_id"]: task["episode_successes"] for task in result["tasks"]},
        }
        summary.setdefault("numerics", result["numerics"])
        successes += result["total_successes"]
        episodes += result["total_episodes"]
    summary["overall_success_rate"] = successes / max(episodes, 1)
    summary["suite_mean_success_rate"] = sum(
        item["success_rate"] for item in summary["suites"].values()) / max(len(summary["suites"]), 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "done", "overall": summary["overall_success_rate"],
                      **{suite: item["success_rate"] for suite, item in summary["suites"].items()}}), flush=True)


if __name__ == "__main__":
    main()
