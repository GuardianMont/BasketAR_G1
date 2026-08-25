import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader

from data.utils import pad_and_collate_video_batch
from scripts.extract_features import extract_features_offline


class MockFeatureBackbone(nn.Module):
    """
    Lightweight backbone used only for offline-extraction tests.

    Input:
        (B, C, T, H, W)

    Output:
        (B, T, D)
    """

    def __init__(self, out_dim: int = 6):
        super().__init__()
        self.out_dim = out_dim

        # Used to verify that padded T_max is NOT passed
        # blindly through the backbone.
        self.seen_temporal_lengths = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_temporal_lengths.append(x.shape[2])

        # Average C, H and W while preserving T:
        # (B, C, T, H, W) -> (B, T)
        features = x.mean(dim=(1, 3, 4))

        # Expand to arbitrary feature dimension:
        # (B, T) -> (B, T, D)
        return features.unsqueeze(-1).expand(
            -1,
            -1,
            self.out_dim,
        )


class TrainableMockBackbone(nn.Module):
    """Used to verify that extraction rejects an unfrozen model."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Forward should never be reached.")


def _build_samples():
    """
    Create clips with different temporal lengths.

    Values are constant inside each clip so that the
    extracted features can also be checked numerically.
    """

    lengths = [10, 4, 7]

    samples = []

    for i, length in enumerate(lengths):
        value = float(i + 1)

        samples.append(
            {
                "data": torch.full(
                    (3, length, 8, 8),
                    fill_value=value,
                    dtype=torch.float32,
                ),
                "label": i,
                "length": length,
                "sample_id": f"clip_{i}",
            }
        )

    return samples


def test_offline_feature_extraction(tmp_path):
    samples = _build_samples()

    loader = DataLoader(
        samples,
        batch_size=3,
        shuffle=False,
        collate_fn=pad_and_collate_video_batch,
    )

    backbone = MockFeatureBackbone(
        out_dim=6,
    )

    output_path = tmp_path / "train.pt"

    payload = extract_features_offline(
        backbone=backbone,
        dataloader=loader,
        output_path=output_path,
        device="cpu",
    )

    # --------------------------------------------------
    # Output file / metadata
    # --------------------------------------------------

    assert output_path.exists()

    assert payload["format_version"] == 1
    assert payload["backbone"] == "MockFeatureBackbone"
    assert payload["feature_dim"] == 6
    assert payload["num_samples"] == 3

    # --------------------------------------------------
    # Verify padding was not processed by the backbone.
    #
    # The collated batch has T_max=10, but the extractor
    # must send clips according to their VALID lengths.
    # --------------------------------------------------

    assert backbone.seen_temporal_lengths == [4, 7, 10]

    # --------------------------------------------------
    # Verify sample order, metadata and feature shapes.
    # --------------------------------------------------

    extracted = payload["samples"]

    assert len(extracted) == len(samples)

    for i, item in enumerate(extracted):
        expected_length = samples[i]["length"]
        expected_value = float(i + 1)

        assert item["id"] == f"clip_{i}"
        assert item["label"] == i
        assert item["num_frames"] == expected_length

        assert item["data"].shape == (
            expected_length,
            6,
        )

        assert torch.isfinite(
            item["data"]
        ).all()

        # Since every raw clip contains one constant value,
        # the mock backbone should preserve that value.
        assert torch.allclose(
            item["data"],
            torch.full(
                (expected_length, 6),
                expected_value,
            ),
        )

    # --------------------------------------------------
    # Verify saved file can be loaded again.
    # --------------------------------------------------

    saved_payload = torch.load(
        output_path,
        map_location="cpu",
        weights_only=False,
    )

    assert saved_payload["num_samples"] == 3
    assert saved_payload["feature_dim"] == 6

    assert [
        sample["id"]
        for sample in saved_payload["samples"]
    ] == [
        "clip_0",
        "clip_1",
        "clip_2",
    ]


def test_extractor_requires_frozen_backbone(tmp_path):
    backbone = TrainableMockBackbone()

    loader = DataLoader(
        _build_samples(),
        batch_size=2,
        shuffle=False,
        collate_fn=pad_and_collate_video_batch,
    )

    with pytest.raises(
        RuntimeError,
        match="fully frozen backbone",
    ):
        extract_features_offline(
            backbone=backbone,
            dataloader=loader,
            output_path=tmp_path / "features.pt",
            device="cpu",
        )


def test_extractor_does_not_overwrite_existing_file(tmp_path):
    samples = _build_samples()

    loader = DataLoader(
        samples,
        batch_size=3,
        shuffle=False,
        collate_fn=pad_and_collate_video_batch,
    )

    backbone = MockFeatureBackbone()

    output_path = tmp_path / "train.pt"

    extract_features_offline(
        backbone=backbone,
        dataloader=loader,
        output_path=output_path,
        device="cpu",
    )

    with pytest.raises(FileExistsError):
        extract_features_offline(
            backbone=backbone,
            dataloader=loader,
            output_path=output_path,
            device="cpu",
        )
