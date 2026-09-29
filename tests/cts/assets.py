"""Stand-in assets for CPU tests: no downloads, no trained weights.

* a randomly initialized SDT-V3 19M checkpoint with the official key layout;
* a DINO-style image-processor config (the teacher branch of the dataset
  normalizes with it; only its shape contract matters for plumbing tests);
* the locally cached bert-base-uncased, if present.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[2]
SDT_SOURCE = CODE / "third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py"


def bert_path() -> str | None:
    explicit = os.environ.get("CTS_TEST_BERT")
    if explicit:
        return explicit
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    found = sorted(glob.glob(str(cache / "models--bert-base-uncased/snapshots/*/")))
    return found[0] if found else None


def libero_root() -> Path | None:
    root = os.environ.get("CTS_TEST_LIBERO_DATA")
    if root and Path(root).is_dir():
        return Path(root)
    return None


def random_sdtv3(directory: Path) -> Path:
    import importlib.util

    path = directory / "sdtv3_random.pth"
    if path.is_file():
        return path
    spec = importlib.util.spec_from_file_location("cts_test_sdtv3", SDT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    backbone = module.Efficient_Spiking_Transformer_l()
    state = {k: v for k, v in backbone.state_dict().items() if not k.startswith("head.")}
    torch.save({"model": state, "args": {"model": "Efficient_Spiking_Transformer_l"}}, path)
    return path


def processor_dir(directory: Path, size: int = 224) -> Path:
    path = directory / "dino_processor_standin"
    path.mkdir(parents=True, exist_ok=True)
    (path / "preprocessor_config.json").write_text(json.dumps({
        "image_processor_type": "ViTImageProcessor",
        "do_resize": False, "do_rescale": True, "rescale_factor": 1 / 255,
        "do_normalize": True, "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225],
        "size": {"height": size, "width": size},
    }))
    return path


def smoke_args(sdt: Path, bert: str, timesteps: int = 2, **cts):
    """Small-T spike-driven-vla model args (random SDT-V3, cached BERT as the text model)."""
    from types import SimpleNamespace

    layout = json.loads((CODE / "experiments/libero/configs/online_text_layout.json").read_text())
    base = dict(
        dinov3_path=bert, vision_encoder_type="sdtv3_19m", vision_model_path=str(sdt),
        vision_pretrained_checkpoint=str(sdt), vision_model_source_path=str(SDT_SOURCE), image_size=224,
        bert_path=bert, text_encoder_type="bert", text_model_path=bert, text_timesteps=1,
        text_mask_version="legacy", text_sentence_mask_version="legacy", text_head_mask_version="corrected",
        text_padding_length=21, text_padding_length_by_instruction=layout["padding_length_by_instruction"],
        fusion_backbone_type="spike_text_only", text_only_spike_projections=True,
        cross_attention_type="spike_sdsa", cross_timesteps=timesteps, fusion_droppath=0.1,
        action_head_type="full_spike_state8_shiftmax_continuous_inputs", continuous_state=True,
        spike_state_encoding="perdim_ann", num_state_tokens=8, spike_state_population=32,
        chunk_size=12, dinov3_precision="fp32", num_views=2, local_files_only=True,
    )
    base.update({"cts_" + key: value for key, value in cts.items()})
    return SimpleNamespace(**base)


def smoke_batch(batch: int = 2, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    images = {"pixel_values": torch.randn(batch, 2, 3, 224, 224, generator=generator)}
    state = torch.randn(batch, 8, generator=generator)
    instructions = ["put both the alphabet soup and the tomato sauce in the basket",
                    "pick up the black bowl on the ramekin and place it on the plate"][:batch]
    return instructions, images, state
