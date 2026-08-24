"""
Implementations of spatial and spatio-temporal backbone modules.
"""
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseBackbone


class X3DSBackbone(BaseBackbone):
    """
    X3D-S spatio-temporal feature extractor.

    The pretrained X3D classification head is removed. The remaining X3D trunk
    produces a 5D feature map which is pooled only over the spatial dimensions,
    preserving the temporal axis for the downstream Temporal Neck.

    Tensor Shapes:
        - Input `x`:  RGB video tensor of shape (B, C, T, H, W)
        - Output:     Feature sequence of shape (B, T, out_dim)

    Notes:
        - PyTorchVideo X3D-S has a native final trunk width of 192 channels.
        - The project proposal uses D=512 for X3D-S features, therefore a small
          linear projection is used by default to map 192 -> 512.
        - X3D-S does not downsample the temporal axis in its stem/residual stages,
          so the output temporal length is preserved (T_out = T_in).
        - `freeze_x3d=True` freezes only the pretrained X3D trunk. The optional
          output projection remains trainable, which allows adapting the native
          X3D feature width to the project feature dimension.

    Args:
        out_dim (int, optional): Output feature dimension. Default: 512.
        pretrained (bool, optional): Load Kinetics-pretrained X3D-S weights.
            Default: True.
        freeze_x3d (bool, optional): Freeze the X3D trunk parameters and keep its
            BatchNorm layers in evaluation mode. Default: True.
        progress (bool, optional): Show the checkpoint download progress bar when
            pretrained weights are requested. Default: True.
        x3d_model (nn.Module, optional): Pre-built X3D-compatible model. Intended
            for dependency injection in tests or custom checkpoint loading. The
            module must expose a `blocks` attribute whose final block is the
            original classification head.
    """

    NATIVE_OUT_DIM = 192

    def __init__(
        self,
        out_dim: int = 512,
        pretrained: bool = True,
        freeze_x3d: bool = True,
        progress: bool = True,
        x3d_model: Optional[nn.Module] = None,
    ):
        if out_dim <= 0:
            raise ValueError(f"out_dim must be > 0, got {out_dim}.")

        super().__init__(out_dim=out_dim)

        model = x3d_model or self._load_x3d_s(pretrained=pretrained, progress=progress)
        self._validate_x3d_model(model)

        # PyTorchVideo X3D is structured as:
        # Stem -> Stage1 -> Stage2 -> Stage3 -> Stage4 -> Classification Head.
        # The head performs spatio-temporal pooling and classification, therefore
        # it must be removed to preserve the temporal feature sequence.
        self.trunk = nn.Sequential(*list(model.blocks[:-1]))

        self.projection = (
            nn.Identity()
            if out_dim == self.NATIVE_OUT_DIM
            else nn.Linear(self.NATIVE_OUT_DIM, out_dim)
        )

        self._x3d_frozen = False
        if freeze_x3d:
            self.freeze_x3d()

    @staticmethod
    def _load_x3d_s(pretrained: bool, progress: bool) -> nn.Module:
        try:
            from pytorchvideo.models.hub import x3d_s
        except ImportError as exc:
            raise ImportError(
                "X3DSBackbone requires PyTorchVideo. Install it on the training VM "
                "with `pip install pytorchvideo`."
            ) from exc

        return x3d_s(pretrained=pretrained, progress=progress)

    @staticmethod
    def _validate_x3d_model(model: nn.Module) -> None:
        if not hasattr(model, "blocks"):
            raise TypeError("The supplied X3D model must expose a `blocks` attribute.")
        if len(model.blocks) < 2:
            raise ValueError(
                "The supplied X3D model must contain at least one feature block "
                "and one classification head."
            )

    def freeze_x3d(self) -> None:
        """Freeze the pretrained X3D trunk while leaving the projection trainable."""
        for parameter in self.trunk.parameters():
            parameter.requires_grad = False
        self.trunk.eval()
        self._x3d_frozen = True

    def unfreeze_x3d(self) -> None:
        """Unfreeze the X3D trunk for optional end-to-end fine-tuning."""
        for parameter in self.trunk.parameters():
            parameter.requires_grad = True
        self._x3d_frozen = False
        self.trunk.train(self.training)

    def freeze(self) -> None:
        """Freeze the entire backbone, including the output projection."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        self._x3d_frozen = True
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze the entire backbone for end-to-end fine-tuning."""
        for parameter in self.parameters():
            parameter.requires_grad = True
        self._x3d_frozen = False
        self.train()

    def train(self, mode: bool = True):
        """
        Set train/eval mode while keeping a frozen X3D trunk in evaluation mode.

        Freezing parameters alone is not enough for BatchNorm-based backbones:
        their running statistics would otherwise continue to change when the full
        pipeline is switched to training mode.
        """
        super().train(mode)
        if self._x3d_frozen:
            self.trunk.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                "X3DSBackbone expects a 5D video tensor with shape "
                f"(B, C, T, H, W), got {tuple(x.shape)}."
            )
        if x.size(1) != 3:
            raise ValueError(
                f"X3DSBackbone expects RGB input with C=3, got C={x.size(1)}."
            )

        features = self.trunk(x)  # (B, 192, T, H', W')

        if features.dim() != 5 or features.size(1) != self.NATIVE_OUT_DIM:
            raise RuntimeError(
                "Unexpected X3D-S trunk output shape. Expected channel dimension "
                f"{self.NATIVE_OUT_DIM}, got {tuple(features.shape)}."
            )

        # Spatial pooling only: preserve the temporal sequence for the neck.
        features = features.mean(dim=(-2, -1))  # (B, 192, T)
        features = features.transpose(1, 2).contiguous()  # (B, T, 192)

        return self.projection(features)  # (B, T, out_dim)