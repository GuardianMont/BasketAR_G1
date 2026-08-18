"""
Implementations of Temporal Neck modules for sequence modeling.
"""
from typing import Optional
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import BaseTemporalNeck


class IdentityNeck(BaseTemporalNeck):
    """Neutral Temporal Neck implementation. Used for Baseline experiment."""
    def __init__(self, in_dim: int):
        super().__init__(in_dim=in_dim, out_dim=in_dim)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x
