import pytest
import torch
import torch.nn as nn
from typing import Optional, List

from models.base import BaseBackbone
from models.heads import MeanPoolingClassifierHead
from models.necks import BiGRUNeck, TCNNeck, IdentityNeck
from models.pipeline import ARPipeline


# Real but minimal Mock Backbone to test 5D tensor flow without external dependencies
class MockBackbone(BaseBackbone):
    """Minimal real backbone mapping a 5D video tensor (B, C, T, H, W) -> 3D feature sequence (B, T, D)."""
    def __init__(self, out_dim: int = 512):
        super().__init__(out_dim=out_dim)
        self.conv = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        out = self.conv(x)  # (B, 16, T, H, W)
        out = out.mean(dim=[-2, -1]).transpose(1, 2)  # Global Spatial Pool -> (B, T, 16)
        return self.fc(out)  # (B, T, D)


def test_exp01_baseline_identity():
    """EXP-01: Baseline using IdentityNeck and pre-extracted features (3D)."""
    B, T, D, K = 4, 16, 512, 9
    
    neck = IdentityNeck(in_dim=D)
    head = MeanPoolingClassifierHead(in_dim=D, num_classes=K, dropout=0.3)
    pipeline = ARPipeline(backbone=None, neck=neck, head=head)

    x = torch.randn(B, T, D)
    logits = pipeline(x)

    # Shape and validity checks
    assert logits.shape == (B, K), f"[EXP-01] Incorrect Logits shape: expected ({B}, {K}), got {logits.shape}"
    assert not torch.isnan(logits).any(), "[EXP-01] Detected NaN values in logits!"
    print("✓ [PASSED] EXP-01: Baseline (IdentityNeck)")


@pytest.mark.parametrize("with_lengths", [False, True])
def test_exp02_bigru_neck(with_lengths: bool):
    """EXP-02: BiGRUNeck with support for variable-length clips and padding handling."""
    B, T, D, K = 4, 16, 512, 9
    hidden_dim = 256
    
    neck = BiGRUNeck(in_dim=D, hidden_dim=hidden_dim, num_layers=2, dropout=0.2)
    head = MeanPoolingClassifierHead(in_dim=neck.out_dim, num_classes=K)
    pipeline = ARPipeline(backbone=None, neck=neck, head=head)

    x = torch.randn(B, T, D)
    lengths = torch.tensor([16, 12, 8, 4]) if with_lengths else None

    # Intermediate shape assertion
    temp_features = neck(x, lengths)
    assert temp_features.shape == (B, T, hidden_dim * 2), (
        f"[EXP-02] Incorrect BiGRU output shape: expected ({B}, {T}, {hidden_dim * 2}), got {temp_features.shape}"
    )

    logits = pipeline(x, lengths=lengths)
    assert logits.shape == (B, K), f"[EXP-02] Incorrect Logits shape: expected ({B}, {K}), got {logits.shape}"
    print(f"✓ [PASSED] EXP-02: BiGRUNeck (lengths={with_lengths})")


@pytest.mark.parametrize("with_lengths", [False, True])
def test_exp03_tcn_neck(with_lengths: bool):
    """EXP-03: Generic TCNNeck with sequence padding masking."""
    B, T, D, K = 4, 16, 512, 9
    num_channels = [128, 256]
    
    neck = TCNNeck(in_dim=D, num_channels=num_channels, kernel_size=3, dropout=0.2)
    head = MeanPoolingClassifierHead(in_dim=neck.out_dim, num_classes=K)
    pipeline = ARPipeline(backbone=None, neck=neck, head=head)

    x = torch.randn(B, T, D)
    lengths = torch.tensor([16, 14, 6, 4]) if with_lengths else None

    # Intermediate shape assertion
    temp_features = neck(x, lengths)
    assert temp_features.shape == (B, T, num_channels[-1]), (
        f"[EXP-03] Incorrect TCN output shape: expected ({B}, {T}, {num_channels[-1]}), got {temp_features.shape}"
    )

    # Verify that the binary mask correctly zeroes out padded time steps
    if with_lengths:
        # For the clip with length 4, feature representations from frame index 4 onward must be strictly zero
        assert torch.all(temp_features[3, 4:, :] == 0), "[EXP-03] Masking failed: detected non-zero activations in padded frames!"

    logits = pipeline(x, lengths=lengths)
    assert logits.shape == (B, K), f"[EXP-03] Incorrect Logits shape: expected ({B}, {K}), got {logits.shape}"
    print(f"✓ [PASSED] EXP-03: Generic TCNNeck (lengths={with_lengths}, Masking Verified)")


def test_full_pipeline_with_backbone():
    """End-to-End test with real backbone (5D Video Input -> Logits)."""
    B, C, T, H, W, K = 2, 3, 8, 112, 112, 9
    D_backbone = 256

    backbone = MockBackbone(out_dim=D_backbone)
    neck = TCNNeck(in_dim=D_backbone, num_channels=[128, 128])
    head = MeanPoolingClassifierHead(in_dim=128, num_classes=K)
    
    pipeline = ARPipeline(backbone=backbone, neck=neck, head=head)

    x_5d = torch.randn(B, C, T, H, W)
    logits = pipeline(x_5d)

    assert logits.shape == (B, K), f"[E2E] Incorrect Logits shape: expected ({B}, {K}), got {logits.shape}"
    
    # Verify backward gradient flow across all three pipeline stages (Backbone -> Neck -> Head)
    logits.sum().backward()
    assert backbone.conv.weight.grad is not None, "[E2E] Gradient failed to propagate back to the Backbone!"
    print("✓ [PASSED] Full Pipeline E2E (5D Input + Backward Pass)")


if __name__ == "__main__":
    print("\n=== RUNNING UNIT TEST SUITE FOR FVAB ARCHITECTURE ===\n")
    test_exp01_baseline_identity()
    test_exp02_bigru_neck(with_lengths=False)
    test_exp02_bigru_neck(with_lengths=True)
    test_exp03_tcn_neck(with_lengths=False)
    test_exp03_tcn_neck(with_lengths=True)
    test_full_pipeline_with_backbone()
    print("\nAll component tests executed and passed successfully!")