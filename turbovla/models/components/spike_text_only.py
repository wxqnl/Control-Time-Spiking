"""Text-only spike replacement; original projections and SDSA Cross are unchanged."""
import torch
from torch.utils.checkpoint import checkpoint

from .wta_fusion import WTATextBlock
from .wta_operators import checkpoint_contexts


class SpikeTextEncoderLayer(WTATextBlock):
    """Drop-in [N,B,C] text-layer interface for the original interaction loop.

    The old Cross already reads out its own T steps. This text replacement
    integrates a constant input current over T steps and reads out after the
    complete self-attention + FFN block. It does not wrap/reimplement Cross or
    pretend that the old Cross keeps a temporal axis between fusion layers.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None,
                initial_state=None, return_state=False):
        if src.ndim != 3 or pos is not None:
            raise ValueError("Spike text layer requires [N,B,C] and no extra positional input")
        current = src.transpose(0, 1)
        batch, length = current.shape[:2]
        if src_key_padding_mask is None:
            valid = torch.ones((batch, length, 1), device=src.device, dtype=torch.bool)
        else:
            if src_key_padding_mask.dtype != torch.bool or src_key_padding_mask.shape != (batch, length):
                raise ValueError("Text padding mask must be boolean [B,N]")
            valid = ~src_key_padding_mask[..., None]
        allowed = valid & valid.transpose(-1, -2)
        if src_mask is not None:
            if src_mask.dtype != torch.bool or src_mask.shape != (batch, length, length):
                raise ValueError("Text sentence mask must be boolean [B,N,N], True means blocked")
            allowed = allowed & ~src_mask

        if return_state:
            return self._forward_with_state(current, valid, allowed, initial_state)

        def run(x):
            state, sequence = {}, []
            for t in range(self.config.cross_timesteps):
                sequence.append(self.forward_step(x, valid, allowed, state, t))
            return torch.stack(sequence).mean(0)

        if self.training and torch.is_grad_enabled() and self.config.cross_gradient_checkpointing:
            result = checkpoint(run, current, use_reentrant=False, context_fn=checkpoint_contexts)
        else:
            result = run(current)
        return result.transpose(0, 1)

    def _forward_with_state(self, current, valid, allowed, initial_state):
        """CTS path: start from the carried membranes and return the final ones."""
        from turbovla.cts.state import wrap_carried

        def run(x, init):
            state, sequence = wrap_carried(init), []
            for t in range(self.config.cross_timesteps):
                sequence.append(self.forward_step(x, valid, allowed, state, t))
            return torch.stack(sequence).mean(0), state

        if self.training and torch.is_grad_enabled() and self.config.cross_gradient_checkpointing:
            result, state = checkpoint(run, current, initial_state, use_reentrant=False,
                                       context_fn=checkpoint_contexts)
        else:
            result, state = run(current, initial_state)
        return result.transpose(0, 1), state
