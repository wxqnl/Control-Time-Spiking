"""Query-window samples for Control-Time Spiking (C3).

A window is Q = burn_in + K consecutive control queries of one demonstration,
spaced `stride` frames apart (the deployed open-loop chunk, 12 in LIBERO):

    steps = s + (j - burn_in) * stride,  j = 0..Q-1

The last K queries carry the loss; the first burn_in queries only warm up the
episode state without gradients (as in R2D2). A burn-in step before the
episode start is "absent": the trainer zeroes the state after it, so a window
near the episode start sees exactly the deployed cold start. A loss step past
the episode end is clamped to the last frame and excluded from the loss.

Light augmentation draws one shift/brightness/contrast per camera per window,
so the image perturbation is constant over the window like a real camera.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageEnhance

from ..data.libero_lerobot import LiberoMixedLeRobotDataset
from ..data.libero_rlds import vla_collate_fn


def sample_augmentation(rng: np.random.Generator):
    dx, dy = rng.integers(-4, 5, size=2)
    brightness, contrast = rng.uniform(0.9, 1.1, size=2)
    return int(dx), int(dy), float(brightness), float(contrast)


def apply_augmentation(image: np.ndarray, params) -> np.ndarray:
    """Same operations and ranges as turbovla.data.light_augmentation, fixed parameters."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected real uint8 HWC RGB frame")
    dx, dy, brightness, contrast = params
    h, w = image.shape[:2]
    padded = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode="edge")
    shifted = padded[4 + dy:4 + dy + h, 4 + dx:4 + dx + w]
    out = ImageEnhance.Brightness(Image.fromarray(shifted)).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    return np.array(out, dtype=np.uint8, copy=True)


def window_steps(start: int, length: int, burn_in: int, window: int, stride: int):
    raw = [start + (j - burn_in) * stride for j in range(burn_in + window)]
    present = [step >= 0 for step in raw]
    in_episode = [step < length for step in raw]
    clamped = [min(max(step, 0), length - 1) for step in raw]
    return clamped, present, in_episode


class LiberoQueryWindowDataset(LiberoMixedLeRobotDataset):
    def __init__(self, *args, window: int = 2, burn_in: int = 2, stride: int = 12,
                 windows_per_episode: str = "length_over_window", **kwargs):
        super().__init__(*args, **kwargs)
        if window < 1 or burn_in < 0 or stride < 1:
            raise ValueError("window >= 1, burn_in >= 0 and stride >= 1 are required")
        if windows_per_episode not in {"length_over_window", "length"}:
            raise ValueError("windows_per_episode must be length_over_window or length")
        self.window, self.burn_in, self.stride = int(window), int(burn_in), int(stride)
        self.windows_per_episode = windows_per_episode
        self._window_augmentation = None
        self._view_cursor = 0

    def _process_image(self, image):
        # Replaces the base per-frame random augmentation: parameters are drawn
        # once per window and camera (primary first, then wrist, as in the base).
        if self._window_augmentation is None:
            return self._process_image_pair(image)
        params = self._window_augmentation[self._view_cursor % 2]
        self._view_cursor += 1
        return self._process_image_pair(apply_augmentation(image, params))

    def _window_sample(self, episode_data: dict, start: int, rng: np.random.Generator):
        length = len(episode_data["actions"])
        steps, present, in_episode = window_steps(start, length, self.burn_in, self.window, self.stride)
        self._window_augmentation = (
            (sample_augmentation(rng), sample_augmentation(rng)) if self.light_augmentation else None)
        queries = []
        try:
            for step in steps:
                self._view_cursor = 0
                queries.append(self._build_step_sample(episode_data, step))
        finally:
            self._window_augmentation = None
        loss_valid = torch.tensor(
            [in_episode[j] and j >= self.burn_in for j in range(len(steps))], dtype=torch.bool)
        present = torch.tensor(present, dtype=torch.bool)
        return queries, present, loss_valid, torch.tensor(steps, dtype=torch.long)

    def __iter__(self):
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        shard_count = self.world_size * num_workers
        shard_index = self.rank + self.world_size * worker_id
        local_episodes = self.episodes[shard_index::shard_count]
        if not local_episodes:
            raise RuntimeError(f"Empty LeRobot shard {shard_index}/{shard_count}")
        base_seed = self.seed + 1009 * self.rank + 9176 * worker_id
        epoch = 0
        while True:
            rng = np.random.default_rng(base_seed + epoch)
            buffer = []
            for episode in self._buffer_shuffle(local_episodes, self.shuffle_buffer, rng):
                episode_data = self._load_episode(episode)
                length = episode.expected_length
                count = length if self.windows_per_episode == "length" else max(1, round(length / self.window))
                starts = rng.integers(0, length, size=count)
                for start in starts:
                    sample = self._window_sample(episode_data, int(start), rng)
                    if self.step_mix_buffer_size > 0:
                        buffer.append(sample)
                        if len(buffer) >= self.step_mix_buffer_size:
                            yield buffer.pop(int(rng.integers(0, len(buffer))))
                    else:
                        yield sample
            while buffer:
                yield buffer.pop(int(rng.integers(0, len(buffer))))
            epoch += 1


def window_collate(batch):
    """-> list over queries of vla_collate_fn outputs, present [B,Q], loss_valid [B,Q], steps [B,Q]."""
    if not batch:
        raise ValueError("empty batch cannot be collated")
    queries = len(batch[0][0])
    per_query = [vla_collate_fn([sample[0][j] for sample in batch]) for j in range(queries)]
    present = torch.stack([sample[1] for sample in batch])
    loss_valid = torch.stack([sample[2] for sample in batch])
    steps = torch.stack([sample[3] for sample in batch])
    return per_query, present, loss_valid, steps
