"""CTS experiment launcher: named recipes on top of the spike-driven-vla base recipe.

The base argument list comes from experiments/libero/train_action_head.py
(--variant spike-driven-vla), so paths, optimizer, KD, precision and
augmentation are identical to the base. A recipe then appends:

* its rank layout: world x windows_per_gpu x grad_accum x K = 256 loss queries;
* base overrides (e.g. --cross_timesteps 2);
* CTS flags (entry "cts" runs turbovla.cts.train; entry "base" the base trainer;
  entry "ann" the base trainer with the repository's ANN preset modules).

Examples
    python experiments/cts/train.py --list
    python experiments/cts/train.py --recipe E1-cts --run_dir /runs/E1-cts-s42 \
        --vision_checkpoint ... --text_projection_init ... --dry-run --world-size 8
    torchrun --standalone --nproc_per_node 8 experiments/cts/train.py --recipe E1-cts \
        --run_dir /runs/E1-cts-s42 --vision_checkpoint ... --text_projection_init ... --approve_training
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
RECIPES = json.loads((Path(__file__).with_name("configs") / "recipes.json").read_text())
TRAINING_KEYS = {"window", "burn_in", "stride", "state_dropout", "lr_scale"}


def cts_flags(settings: dict) -> list[str]:
    flags = []
    for name, value in settings.items():
        option = f"cts_{name}"
        if isinstance(value, bool):
            flags.append(f"--{option}" if value else f"--no-{option}")
        else:
            flags += [f"--{option}", str(value)]
    return flags


def resolve(recipe_name: str, options) -> tuple[str, list[str], dict]:
    from experiments.libero import train_action_head as base_launcher

    recipe = RECIPES["recipes"][recipe_name]
    if recipe["entry"] == "ann":
        return ann_arguments(recipe_name, options)
    budget = RECIPES["budget"]
    base_recipe = base_launcher.RECIPES["variants"][RECIPES["base_variant"]]
    entry = recipe["entry"]
    settings = {**RECIPES["cts_defaults"], **recipe.get("cts", {})} if entry == "cts" else {}
    window = int(settings.get("window", 1))
    per_gpu = options.windows_per_gpu or recipe.get("windows_per_gpu") or (
        base_recipe["batch_per_gpu"] // window if entry == "cts" else base_recipe["batch_per_gpu"])
    world = options.effective_world_size
    per_step = world * per_gpu * window
    if budget["global_loss_queries_per_step"] % per_step:
        raise ValueError(f"world {world} x windows/GPU {per_gpu} x K {window} does not divide "
                         f"{budget['global_loss_queries_per_step']}; choose --windows_per_gpu")
    accum = budget["global_loss_queries_per_step"] // per_step
    max_steps = int(recipe.get("max_steps", budget["max_steps"]))

    # Base arguments exactly as the base launcher builds them for its own layout.
    base = copy.copy(options)
    base.variant = RECIPES["base_variant"]
    base.effective_world_size = base_recipe["world_size"]
    base.batch_size = base_recipe["batch_per_gpu"]
    base.grad_accum_steps = base_launcher.RECIPES["common_training"]["gradient_accumulation"]
    base.max_steps = base_recipe["max_steps"]
    base.resume_mode = "none"
    base.dry_run = True  # only skips the base launcher's run-directory existence check
    arguments = base_launcher.action_head_arguments(base)

    overrides = {
        "--batch_size": str(per_gpu), "--grad_accum_steps": str(accum),
        "--max_steps": str(max_steps), "--lr_schedule_steps": str(max_steps),
        "--seed": str(options.seed), "--checkpoint_prefix": recipe_name.replace("-", "_").lower(),
        "--resume_mode": options.resume_mode,
        **recipe.get("base_overrides", {}),
    }
    if recipe.get("requires_init_checkpoint"):
        if not options.init_checkpoint:
            raise ValueError(f"{recipe_name} needs --init_checkpoint (trained spike-driven-vla checkpoint)")
        # The joint-scope audit certifies fresh training only.
        arguments = [argument for argument in arguments if argument != "--joint_scope_audit"]
    for flag, value in overrides.items():
        arguments += [flag, value]
    if entry == "cts":
        arguments += cts_flags({key: value for key, value in settings.items() if key not in TRAINING_KEYS})
        arguments += ["--cts_window", str(settings["window"]), "--cts_burn_in", str(settings["burn_in"]),
                      "--cts_stride", str(settings["stride"]), "--cts_state_dropout", str(settings["state_dropout"]),
                      "--cts_lr_scale", str(settings["lr_scale"])]
        if options.init_checkpoint:
            arguments += ["--cts_init_checkpoint", options.init_checkpoint]
    elif options.init_checkpoint:
        raise ValueError("--init_checkpoint only applies to CTS recipes")
    arguments += options.extra
    layout = {"world_size": world, "windows_per_gpu": per_gpu, "grad_accum": accum, "K": window,
              "loss_queries_per_step": world * per_gpu * accum * window, "max_steps": max_steps}
    return entry, arguments, layout


def ann_arguments(recipe_name: str, options) -> tuple[str, list[str], dict]:
    """E0e: the repository's --preset ann modules on the spike-driven-vla budget and recipe."""
    from experiments.libero import run_cross_joint_stage as pipeline
    from experiments.libero import train_action_head as base_launcher

    budget = RECIPES["budget"]
    base_recipe = base_launcher.RECIPES["variants"][RECIPES["base_variant"]]
    per_gpu = options.windows_per_gpu or base_recipe["batch_per_gpu"]
    world = options.effective_world_size
    if budget["global_loss_queries_per_step"] % (world * per_gpu):
        raise ValueError("world x batch/GPU must divide 256")
    accum = budget["global_loss_queries_per_step"] // (world * per_gpu)
    saved = pipeline.VISION_TYPE, pipeline.TEXT_TYPE, pipeline.CROSS_TYPE
    pipeline.VISION_TYPE, pipeline.TEXT_TYPE, pipeline.CROSS_TYPE = "dinov3", "bert", "ann"
    try:
        arguments = pipeline.training_arguments(CODE, Path(options.run_dir))
    finally:
        pipeline.VISION_TYPE, pipeline.TEXT_TYPE, pipeline.CROSS_TYPE = saved
    common = base_launcher.RECIPES["common_training"]
    arguments += [*common["trainer_arguments"],
                  "--batch_size", str(per_gpu), "--grad_accum_steps", str(accum),
                  "--max_steps", str(budget["max_steps"]), "--lr_schedule_steps", str(budget["max_steps"]),
                  "--warmup_steps", str(base_recipe["warmup_steps"]),
                  "--lr_warmup_samples", str(base_recipe["lr_warmup_samples"]),
                  "--save_steps", str(common["save_steps"]), "--extra_save_steps", "20",
                  "--seed", str(options.seed), "--checkpoint_prefix", recipe_name.replace("-", "_").lower(),
                  "--resume_mode", options.resume_mode, "--libero_light_augmentation",
                  "--num_workers", str(options.num_workers), *options.extra]
    layout = {"world_size": world, "batch_per_gpu": per_gpu, "grad_accum": accum,
              "loss_queries_per_step": world * per_gpu * accum, "max_steps": budget["max_steps"]}
    return "base", arguments, layout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List recipes and exit")
    parser.add_argument("--recipe", choices=sorted(RECIPES["recipes"]))
    parser.add_argument("--run_dir")
    parser.add_argument("--vision_checkpoint", help="SDT-V3 19M checkpoint (V3_19.0M_1x4.pth)")
    parser.add_argument("--text_projection_init", help="calibrated SmoothSpike projection (text_projection.pth)")
    parser.add_argument("--init_checkpoint", help="pilot recipes: trained base checkpoint to start from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows_per_gpu", type=int, help="override the per-GPU batch (windows)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume_mode", choices=["none", "all"], default="none")
    parser.add_argument("--approve_training", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--world-size", "--world_size", dest="world_size", type=int,
                        help="dry-run only; live runs use torchrun WORLD_SIZE")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="after --: raw trainer flags appended last")
    options = parser.parse_args()
    if options.list:
        for name, recipe in RECIPES["recipes"].items():
            print(f"{name:20s} [{recipe['entry']:4s}] {recipe['label']}")
        return
    if not (options.recipe and options.run_dir and options.vision_checkpoint and options.text_projection_init):
        parser.error("--recipe, --run_dir, --vision_checkpoint and --text_projection_init are required")
    options.extra = [item for item in options.extra if item != "--"]
    if options.world_size is not None and not options.dry_run:
        raise ValueError("--world-size is a dry-run preview option; use torchrun for live training")
    options.effective_world_size = options.world_size or int(os.environ.get("WORLD_SIZE", "1"))
    run = Path(options.run_dir).expanduser().resolve()
    options.run_dir = str(run)
    if not options.dry_run:
        if not options.approve_training:
            raise RuntimeError("Review the resolved recipe (--dry-run) and pass --approve_training")
        if options.resume_mode == "none" and any((run / "checkpoints").glob("*.pth")):
            raise FileExistsError(f"Fresh runs need an unused run directory: {run}")
    # Base launcher fields that train_action_head.action_head_arguments reads.
    options.lr_schedule_steps = options.max_steps = RECIPES["budget"]["max_steps"]
    options.warmup_steps, options.log_freq, options.save_steps = 0, 20, 1000
    options.extra_save_steps = [20]
    options.vision_gradient_checkpointing = False
    options.vision_checkpoint_scope = "all"
    options.projection_gradient_checkpointing = False
    entry, arguments, layout = resolve(options.recipe, options)
    if options.dry_run:
        print(json.dumps({"recipe": options.recipe, "entry": entry,
                          "label": RECIPES["recipes"][options.recipe]["label"], "layout": layout,
                          "training_arguments": arguments,
                          "note": "Preview only; no trainer import, model load or GPU use."}, indent=2))
        return
    run.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        (run / "launcher.json").write_text(json.dumps({"recipe": options.recipe, "layout": layout,
                                                       "arguments": arguments}, indent=2) + "\n")
    sys.argv = [sys.argv[0], *arguments]
    if entry == "cts":
        from turbovla.cts.train import train_model
        train_model()
    else:
        from turbovla.training import train_mixed
        train_mixed.main()


if __name__ == "__main__":
    main()
