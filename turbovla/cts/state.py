"""Episode state ("carry") utilities.

A carry is a plain nested structure of dicts/lists/tuples with tensor leaves,
created fresh for every control query and never mutated afterwards. That keeps
non-reentrant activation checkpointing exact: a recomputed forward sees the
same carry objects as the original forward.
"""
from __future__ import annotations

from typing import Any

import torch

from .neuron import Carried

# Top-level carry entries that are episode constants (the encoded instruction)
# or bookkeeping; they are not reset by state dropout. Only the top level is
# special: nested "text" keys are text-stream neuron membranes.
CONSTANT_KEYS = ("instruction", "query_index")


def wrap_carried(state: Any):
    """Copy a stored neuron-state tree, marking every tensor leaf as Carried.

    The copy is new at every level, so modules may mutate it in place while the
    stored carry stays untouched. A missing state (None) becomes a fresh dict.
    """
    if state is None:
        return {}
    return _wrap(state)


def _wrap(state):
    if isinstance(state, dict):
        return {key: _wrap(value) for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(_wrap(value) for value in state)
    if torch.is_tensor(state):
        return Carried(state)
    if state is None:
        return None
    raise TypeError(f"Unsupported carry leaf {type(state)}")


def map_tensors(tree: Any, fn, skip_keys=()):
    """Apply fn to every tensor leaf; `skip_keys` exempts top-level entries only."""
    if isinstance(tree, dict):
        return {key: (value if key in skip_keys else map_tensors(value, fn))
                for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(map_tensors(value, fn, skip_keys) for value in tree)
    if torch.is_tensor(tree):
        return fn(tree)
    return tree


def detach_carry(carry):
    return None if carry is None else map_tensors(carry, lambda t: t.detach())


def reset_samples(carry, keep: torch.Tensor):
    """Zero the episode state of samples where keep[b] is False (state dropout,
    absent burn-in queries). The cached instruction encoding is kept."""
    if carry is None:
        return None
    keep = keep.bool()
    if keep.all():
        return carry

    def apply(tensor):
        if tensor.dim() == 0 or tensor.shape[0] != keep.shape[0]:
            raise ValueError(f"Carry tensor without leading batch dim: {tuple(tensor.shape)}")
        mask = keep.to(device=tensor.device).view(-1, *([1] * (tensor.dim() - 1)))
        return tensor * mask.to(tensor.dtype) if tensor.is_floating_point() else tensor & mask
    return map_tensors(carry, apply, skip_keys=CONSTANT_KEYS)


def carry_numel(carry) -> int:
    total = 0

    def count(tensor):
        nonlocal total
        total += tensor.numel()
        return tensor
    map_tensors(carry, count, skip_keys=CONSTANT_KEYS)
    return total
