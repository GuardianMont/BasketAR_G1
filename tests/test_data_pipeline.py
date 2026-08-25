import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any

from data.dataset import ActionRecognitionDataset
from data.samplers import VideoBatchSampler, BucketSampler
from data.utils import pad_and_collate_video_batch, compute_class_weights
from data.transforms import RandomTemporalShift
from models.base import BaseBackbone
from models.heads import MeanPoolingClassifierHead
from models.necks import BiGRUNeck, TCNNeck
from models.pipeline import ARPipeline


class MockBackbone(BaseBackbone):
    """Mock backbone 5D (B, C, T, H, W) -> 3D (B, T, D) per i test end-to-end."""
    def __init__(self, out_dim: int = 512):
        super().__init__(out_dim=out_dim)
        self.conv = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out.mean(dim=[-2, -1]).transpose(1, 2)
        return self.fc(out)


@pytest.fixture
def mock_feature_dataset_samples():
    """Genera campioni mock di feature 3D (T, D) con durate variabili (4-50 frame)."""
    samples = []
    lengths_list = [4, 6, 12, 16, 20, 32, 45, 50]
    labels_list = [0, 0, 0, 1, 2, 3, 4, 8]  # Distribuzione sbilanciata
    D_feature = 512

    for i in range(len(lengths_list)):
        T_i = lengths_list[i]
        samples.append({
            "id": i,
            "data": torch.randn(T_i, D_feature),
            "label": labels_list[i],
            "num_frames": T_i
        })
    return samples, lengths_list, labels_list


def test_video_batch_sampler_classic(mock_feature_dataset_samples):
    """Verifica il comportamento del VideoBatchSampler classico."""
    samples, lengths, _ = mock_feature_dataset_samples
    dataset = ActionRecognitionDataset(samples=samples, is_feature_dataset=True)
    
    batch_size = 3
    sampler = VideoBatchSampler(num_samples=len(dataset), batch_size=batch_size, shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=pad_and_collate_video_batch)

    batches = list(loader)
    assert len(batches) == 3, f"Expected 3 batches, got {len(batches)}"
    assert batches[0]["x"].shape[0] == 3
    assert batches[2]["x"].shape[0] == 2  # Last batch residual (8 % 3 = 2)
    print("✓ [PASSED] VideoBatchSampler (Classic Batching)")


def test_bucket_sampler(mock_feature_dataset_samples):
    samples, lengths, _ = mock_feature_dataset_samples
    dataset = ActionRecognitionDataset(samples=samples, is_feature_dataset=True)

    batch_size = 4
    sampler = BucketSampler(
        sequence_lengths=lengths,
        batch_size=batch_size,
        shuffle=False,
        bucket_boundaries=[10, 25, 60]
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=pad_and_collate_video_batch)

    batches = list(loader)
    for batch in batches:
        lengths_in_batch = batch["lengths"]
        length_diff = (lengths_in_batch.max() - lengths_in_batch.min()).item()
        assert length_diff <= 35, f"Bucket batching allowed a high delta: {length_diff}"

    print("✓ [PASSED] BucketSampler (Padding Minimization)")


def test_collate_fn_masking(mock_feature_dataset_samples):
    """Verifica che pad_and_collate_video_batch generi correttamente Attention Mask e Lengths."""
    samples, lengths, _ = mock_feature_dataset_samples
    dataset = ActionRecognitionDataset(samples=samples, is_feature_dataset=True)
    
    sampler = VideoBatchSampler(num_samples=len(dataset), batch_size=len(dataset), shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=pad_and_collate_video_batch)
    
    batch = next(iter(loader))
    x, mask, batch_lengths = batch["x"], batch["attention_mask"], batch["lengths"]

    expected_max_t = max(batch_lengths).item() if isinstance(batch_lengths, torch.Tensor) else max(batch_lengths)
    
    assert x.shape == (len(samples), expected_max_t, 512), f"Expected shape ({len(samples)}, {expected_max_t}, 512), got {x.shape}"
    assert mask.shape == (len(samples), expected_max_t)

    # Verifica la correttezza della maschera binaria e del padding
    for i, seq_len in enumerate(batch_lengths):
        assert torch.all(mask[i, :seq_len]), f"Unmarked valid frames in batch: {i}"
        if seq_len < expected_max_t:
            assert torch.all(~mask[i, seq_len:]), f"non zeroed padding in the sample mask for batch {i}"
            assert torch.all(x[i, seq_len:, :] == 0), f"non zero padding for when it should have been in batch {i}"

    print("✓ [PASSED] Collate Function (Padding, Lengths & Attention Mask Verification)")

def test_full_pipeline_with_data_loader(mock_feature_dataset_samples):
    """End-to-End integration test: DataLoader -> Neck (TCN/BiGRU) -> Head -> Backward Pass."""
    samples, lengths, labels = mock_feature_dataset_samples
    dataset = ActionRecognitionDataset(samples=samples, is_feature_dataset=True)

    sampler = BucketSampler(sequence_lengths=lengths, batch_size=4, shuffle=True)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=pad_and_collate_video_batch)

    neck = TCNNeck(in_dim=512, num_channels=[128, 128], kernel_size=3)
    head = MeanPoolingClassifierHead(in_dim=128, num_classes=9)
    pipeline = ARPipeline(backbone=None, neck=neck, head=head)

    class_weights = compute_class_weights(labels, num_classes=9)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for batch in loader:
        x, target_labels, batch_lengths = batch["x"], batch["labels"], batch["lengths"]

        logits = pipeline(x, lengths=batch_lengths)
        assert logits.shape == (x.shape[0], 9)

        loss = criterion(logits, target_labels)
        loss.backward()

        assert next(head.parameters()).grad is not None, "Gradient didn't reach the Classifier head"
        break

    print("✓ [PASSED] Full Data-Pipeline to ARPipeline E2E Integration")


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-s", __file__]))