from torch import Tensor, nn
from typing import Dict, Any
from .focal_loss import FocalLoss
from models.base import BaseTypeEnum


class LossType(BaseTypeEnum):
    """Enumeration of supported Loss types."""
    CROSS_ENTROPY = "CrossEntropy"
    FOCAL = "FocalLoss"

class LossFactory:
    """Factory class to create loss functions based on configuration."""
    
    @staticmethod
    def create(loss_type: str | LossType, weights: Tensor, kwargs: Dict[str, Any]) -> nn.Module:

        loss_type = LossType.from_str(loss_type)

        if loss_type == LossType.CROSS_ENTROPY:
            return nn.CrossEntropyLoss(weight=weights, **kwargs)
        elif loss_type == LossType.FOCAL:
            return FocalLoss(alpha=weights, **kwargs)
        else:
            raise ValueError(f"Unsupported: {loss_type}")