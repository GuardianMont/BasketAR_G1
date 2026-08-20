"""
Implementations of classification heads for action recognition.
"""
import torch
import torch.nn as nn

from .base import BaseClassifierHead


class MeanPoolingClassifierHead(BaseClassifierHead):
    """Classification head using Global Temporal Average Pooling."""
    def __init__(self, in_dim: int, num_classes: int = 9, dropout: float = 0.3):
        super().__init__(in_dim=in_dim, num_classes=num_classes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pooled = torch.mean(x, dim=1)
        logits = self.fc(self.dropout(x_pooled))
        return logits