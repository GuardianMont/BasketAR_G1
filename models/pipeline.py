"""
Composite container model assembling Backbone, Neck, and Head.
"""
from typing import Optional
import torch
import torch.nn as nn

from .base import BaseBackbone, BaseTemporalNeck, BaseClassifierHead


class ARPipeline(nn.Module):
    """Action Recognition Pipeline (Composite Model)."""
    def __init__(
        self, 
        backbone: Optional[BaseBackbone], 
        neck: BaseTemporalNeck, 
        head: BaseClassifierHead
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 5:
            if self.backbone is None:
                raise ValueError("Received 5D video tensor, but no Backbone was configured.")
            features = self.backbone(x)
        else:
            features = x

        temporal_features = self.neck(features, lengths)
        logits = self.head(temporal_features, lengths=lengths)

        return logits


if __name__ == "__main__":

    from .necks import IdentityNeck
    from .heads import MeanPoolingClassifierHead

    B, T, D, K = 8, 16, 512, 9

    # Baseline (EXP-01)
    neck = IdentityNeck(in_dim=D)
    head = MeanPoolingClassifierHead(in_dim=D, num_classes=K)
    pipeline = ARPipeline(backbone=None, neck=neck, head=head)

    # Mock: offline features from .pt file
    dummy_features = torch.randn(B, T, D)
    logits = pipeline(dummy_features)

    assert logits.shape == (B, K), f"Error: expected ({B}, {K}), got {logits.shape}"
    print(f"ARPipeline successfully created!")
    print(f"Input Feature Shape: {dummy_features.shape}")
    print(f"Output Logits Shape: {logits.shape}")
