"""
Implementations of classification heads for action recognition.
"""
import torch
import torch.nn as nn
from typing import Optional

from .base import BaseClassifierHead


class MeanPoolingClassifierHead(BaseClassifierHead):
    """Classification head using Global Temporal Average Pooling."""
    def __init__(self, in_dim: int, num_classes: int = 9, dropout: float = 0.3):
        super().__init__(in_dim=in_dim, num_classes=num_classes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if lengths is None:
            x_pooled = torch.mean(x, dim=1)
        else:
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1)
            mask = (
                torch.arange(x.size(1), device=x.device)[None, :]
                < lengths[:, None]
            ).unsqueeze(-1)
            x_pooled = (x * mask).sum(dim=1) / lengths.unsqueeze(1)
        logits = self.fc(self.dropout(x_pooled))
        return logits

# In models/heads.py:
class AttentionClassifierHead(nn.Module):
    """
    Applica un'attenzione temporale sui 16 step prima del classificatore.
    Impara a pesare maggiormente i frame decisivi (es. il rilascio del tiro).
    """
    def __init__(self, in_dim: int, num_classes: int = 9, dropout: float = 0.2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (B, T, D)
        weights = self.attention(x)  # (B, T, 1)
        if lengths is not None:
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1)
            valid_mask = (
                torch.arange(x.size(1), device=x.device)[None, :]
                < lengths[:, None]
            ).unsqueeze(-1)
            weights = weights.masked_fill(~valid_mask, torch.finfo(weights.dtype).min)
        weights = torch.softmax(weights, dim=1)
        
        # Somma pesata lungo l'asse temporale T
        context = torch.sum(x * weights, dim=1)  # (B, D)
        
        out = self.dropout(context)
        return self.fc(out)
