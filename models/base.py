"""
Contains abstract base interfaces for all modular components of the architecture.
"""
from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Type
from enum import Enum
import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """Abstract Base Class for spatial or spatio-temporal feature extractors."""
    def __init__(self, out_dim: int, pretrained: bool = True, freeze_net: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.pretrained = pretrained
        self.freeze_net = freeze_net

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass

    def freeze(self) -> None:
        """Freezes the backbone parameters to prevent gradient updates."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        """Unfreezes the backbone parameters to allow gradient updates."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()


class BaseTemporalNeck(nn.Module, ABC):
    """Abstract Base Class for temporal dynamics modeling blocks."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


class BaseClassifierHead(nn.Module, ABC):
    """Abstract Base Class for video classification heads."""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


T = TypeVar("T", bound="BaseTypeEnum")

class BaseTypeEnum(str, Enum):
    """Provides a method to instantiate enum members from string labels.
    Type enums should inherit from this class to support case-insensitive string conversion.
    Used for NeckType, HeadType, and LossType enums."""

    @classmethod
    def from_str(cls: Type[T], label: str) -> T:
        """Converts a string label to the corresponding enum member."""
        if isinstance(label, cls):
            return label
        
        if not isinstance(label, str):
            raise TypeError(f"Required a string or {cls.__name__}, got {type(label)}")

        clean_label = label.strip().lower()

        for member in cls:
            if member.value.lower() == clean_label or member.name.lower() == clean_label:
                return member

        valid_values = [m.value for m in cls]
        raise ValueError(
            f"Type '{label}' is not valid for {cls.__name__}. Supported values: {valid_values}"
        )

    def __str__(self) -> str:
        return self.value
