"""
Exposes loss functions for model training and ablation studies.
"""
from torch.nn import CrossEntropyLoss
from .focal_loss import FocalLoss

__all__ = [
    "CrossEntropyLoss",
    "FocalLoss",
]