"""
Implementations of classification heads for action recognition.
"""
import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from .base import BaseClassifierHead, BaseTypeEnum


class HeadType(BaseTypeEnum):
    """Enumeration of supported classification head types."""
    MEAN_POOLING = "MeanPooling"
    ATTENTION_POOLING = "AttentionPooling"

import torch
import torch.nn as nn
from typing import Optional, Union, List


class MeanPoolingClassifierHead(BaseClassifierHead):
    """
    Classification head using Global Temporal Average Pooling.
    """
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 9,
        dropout: float = 0.3,
        hidden_dims: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__(in_dim=in_dim, num_classes=num_classes)

        # Normalize hidden_dims parameter into a list of integers
        if hidden_dims is None:
            hidden_dims_list = []
        elif isinstance(hidden_dims, int):
            hidden_dims_list = [hidden_dims] if hidden_dims > 0 else []
        else:
            hidden_dims_list = list(hidden_dims)

        layers = []
        curr_dim = in_dim

        if not hidden_dims_list:
            # Base case: Dropout -> Linear(in_dim, num_classes)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(curr_dim, num_classes))
        else:
            # MLP case: (Linear -> ReLU -> Dropout) per hidden layer
            for h_dim in hidden_dims_list:
                layers.append(nn.Linear(curr_dim, h_dim))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                curr_dim = h_dim
            # Final projection layer
            layers.append(nn.Linear(curr_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

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

        logits = self.classifier(x_pooled)
        return logits

class AttentionClassifierHead(nn.Module):
    """
    Implements a classification head with temporal attention pooling.
    This head learns to focus on the most informative temporal segments of the input sequence.
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


class HeadFactory:
    """Static method to create a classification head based on the specified type and configuration."""
    
    @staticmethod
    def create(head_type: str | HeadType, kwargs: Dict[str, Any], in_dim: int, num_classes: int) -> nn.Module:
        head_type = HeadType.from_str(head_type)
        if head_type == HeadType.MEAN_POOLING:
            return MeanPoolingClassifierHead(
                in_dim=in_dim,
                num_classes=num_classes,
                **kwargs
            )
        elif head_type == HeadType.ATTENTION_POOLING:
            return AttentionClassifierHead(
                in_dim=in_dim,
                num_classes=num_classes,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported: {head_type}")