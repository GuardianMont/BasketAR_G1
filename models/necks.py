"""
Implementations of Temporal Neck modules for sequence modeling.
"""
from typing import Optional, List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

from .base import BaseTemporalNeck


class IdentityNeck(BaseTemporalNeck):
    """Neutral Temporal Neck implementation. Used for Baseline experiment."""
    def __init__(self, in_dim: int):
        super().__init__(in_dim=in_dim, out_dim=in_dim)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x

class BiGRUNeck(BaseTemporalNeck):
    """
    Bidirectional GRU Temporal Neck for sequence-level dynamics modeling.

    Role:
        Processes feature sequences extracted by the backbone using a multi-layer 
        Bidirectional Gated Recurrent Unit (Bi-GRU).

    Padding & Variable-Length Sequence Handling:
        When `lengths` is provided during the forward pass, padded time-steps 
        are ignored in both forward and backward directions, ensuring that 
        unpadded sequences yield exact recurrent representations.

    Tensor Shapes:
        - Input `x`:       Feature sequence of shape (B, T, D)
        - Input `lengths`: Optional 1D Tensor of shape (B,) with valid frame counts
        - Output:          Contextualized feature sequence of shape (B, T, 2 * hidden_dim)

    Args:
        in_dim (int): Dimensionality of input feature vectors D (from backbone).
        hidden_dim (int): Number of features in the hidden state per GRU direction.
        num_layers (int, optional): Number of stacked recurrent layers. Default: 2.
        dropout (float, optional): Dropout probability applied between stacked GRU 
            layers (active only when `num_layers > 1`). Default: 0
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0):
        out_dim = hidden_dim * 2
        super().__init__(in_dim=in_dim, out_dim=out_dim)

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            lengths_cpu = lengths.cpu().to(torch.int64)
            x_packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.gru(x_packed)
            out, _ = pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, _ = self.gru(x)

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.dropout2 = nn.Dropout(dropout)

        # Generally in_channels = out_channels, but for other experiments this fixes shape issues:
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        out = F.relu(self.conv2(out))
        out = self.dropout2(out)
        
        return F.relu(out + res)


class TCNNeck(BaseTemporalNeck):
    """
    Non-Causal Temporal Convolutional Network for Video Sequence Modeling.
    Inspired by this paper: https://arxiv.org/pdf/1803.01271
    (An Empirical Evaluation of Generic Convolutional and Recurrent Networks
      for Sequence Modeling)
    The main difference is that here we don't use causal convolutions 
    because we have the entire clip.
    
    Tensor Shapes:
        - Input `x`:       Feature sequence of shape (B, T, D)
        - Input `lengths`: Optional 1D Tensor of shape (B,) with valid frame counts
        - Output:          Contextualized feature sequence of shape (B, T, num_channels)
    """
    def __init__(
        self, 
        in_dim: int, 
        num_channels: List[int] = [128, 256], 
        kernel_size: int = 3, 
        dropout: float = 0.2
    ):
        out_dim = num_channels[-1]
        super().__init__(in_dim=in_dim, out_dim=out_dim)

        self.input_norm = nn.LayerNorm(in_dim)

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = in_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalResidualBlock1D(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.network = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            # Make a padding mask: True = Sample, False = Padding
            # `None` index adds a single dimension. Used for broadcasting operations
            lengths_device = lengths.to(x.device)
            mask = (
                torch.arange(max_len, device=x.device)[None, :]
                < lengths_device[:, None]
            ).unsqueeze(1)
            # Mask before LayerNorm and again after it: LayerNorm has a
            # learnable bias that can otherwise make padded steps non-zero.
            x = x * mask.transpose(1, 2)

        x = self.input_norm(x)
        if mask is not None:
            x = x * mask.transpose(1, 2)

        # PyTorch's Conv1D requires (B, Channels, Time)
        x_t = x.transpose(1, 2)

        for block in self.network:
            x_t = block(x_t)
            if mask is not None:
                x_t = x_t * mask

        # Back to original shape (B, T, D_neck) for Classifier Head
        return x_t.transpose(1, 2)


class TemporalTransformerNeck(BaseTemporalNeck):
    """Temporal Transformer Neck for modeling long-range dependencies.

    Tensor Shapes:
        - Input `x`: Feature sequence of shape (B, T, D)
        - Input `lengths`: Optional 1D tensor of shape (B,) containing the number of valid frames for each sequence
        - Output: Contextualized sequence of shape (B, T, embed_dim)

    Args:
        in_dim (int): Dimension of the input vectors from the backbone (D).
        embed_dim (int, optional): Dimension of the Transformer's latent space. If None, uses in_dim.
        num_heads (int, optional): Number of attention heads (default: 8).
        num_layers (int, optional): Number of stacked TransformerEncoder layers (default: 2).
        dim_feedforward (int, optional): Dimension of the internal MLP (default: 2048).
        dropout (float, optional): Dropout probability (default: 0.1).
        max_len (int, optional): Maximum video sequence length (default: 500).
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: Optional[int] = None,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__(in_dim=in_dim, out_dim=out_dim)
        out_dim = embed_dim if embed_dim is not None else in_dim

        self.proj = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(out_dim, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

        # PyTorch Transformer encoder layer with batch-first input format.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        x = self.proj(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # Build the padding mask for variable-length sequences.
        key_padding_mask = None
        if lengths is not None:
            max_len = x.size(1)
            lengths_device = lengths.to(x.device)
            key_padding_mask = (
                torch.arange(max_len, device=x.device)[None, :]
                >= lengths_device[:, None]
            )

        out = self.transformer_encoder(
            x, src_key_padding_mask=key_padding_mask
        )
        out = self.norm(out)

        # Zero out padded frame representations to prevent downstream noise.
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return out

    
class PositionalEncoding(nn.Module):
    """Adds temporal position information to frame representations."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, D)
        return x + self.pe[:, : x.size(1), :]
