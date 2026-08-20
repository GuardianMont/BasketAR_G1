"""
Contains abstract base interfaces for all modular components of the architecture.
"""
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """Abstract Base Class for spatial or spatio-temporal feature extractors."""
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


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