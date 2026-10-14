from argparse import Namespace

import pytest
import torch

from final_test import (
    build_model_from_checkpoint,
    collate_dual_clip_batch,
    compute_features_by_length,
    resolve_backbone_config,
    resolve_evaluation_preprocessing,
)
from models.heads import MeanPoolingClassifierHead
from models.necks import BiGRUNeck
from models.pipeline import ARPipeline


def _reference_args(path):
    return Namespace(feature_reference=str(path))


def _save_reference(tmp_path, experiment_config, metadata_extra=None):
    metadata = {"experiment_config": experiment_config}
    if metadata_extra:
        metadata.update(metadata_extra)
    path = tmp_path / "features.pt"
    torch.save({"metadata": metadata, "samples": []}, path)
    return path


def _experiment_config():
    return {
        "sampler": {"name": "edge_dense", "num_frames": 16, "edge_strength": 0.6},
        "spatial_mode": "stretch",
        "spatial_size": 192,
        "color": "grayscale",
        "backbone": "x3d_s",
        "tracker": "none",
    }


def test_collate_keeps_variable_length_clips_as_lists():
    samples = [
        {
            "x_norm": torch.ones(3, length, 2, 2),
            "x_raw": torch.zeros(3, length, 2, 2),
            "length": length,
            "label": index,
            "sample_id": str(index),
        }
        for index, length in enumerate((16, 10))
    ]

    batch = collate_dual_clip_batch(samples)

    assert isinstance(batch["x_norm"], list)
    assert isinstance(batch["x_raw"], list)
    assert batch["lengths"].tolist() == [16, 10]
    assert batch["labels"].tolist() == [0, 1]
    assert batch["sample_ids"] == ["0", "1"]


def test_compute_features_by_length_groups_then_pads_without_reordering():
    lengths = (16, 10, 16, 10)
    clips_norm = [
        torch.full((3, length, 2, 2), float(index + 1))
        for index, length in enumerate(lengths)
    ]
    clips_raw = [clip + 10 for clip in clips_norm]
    calls = []

    def fake_backbone(x_norm, x_raw):
        calls.append((x_norm.shape[0], x_norm.shape[2], x_raw.shape[2]))
        # (B, C, T, H, W) -> (B, T, C), with a small deterministic D=3.
        return x_norm.mean(dim=(-1, -2)).transpose(1, 2)

    features, valid_lengths = compute_features_by_length(
        clips_norm, clips_raw, fake_backbone
    )

    assert features.shape == (4, 16, 3)
    assert valid_lengths.tolist() == [16, 10, 16, 10]
    assert calls == [(2, 16, 16), (2, 10, 10)]
    assert features[:, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert torch.count_nonzero(features[1, 10:]) == 0
    assert torch.count_nonzero(features[3, 10:]) == 0


def test_reference_metadata_resolves_sampling_preprocessing_and_backbone(tmp_path):
    config = _experiment_config()
    path = _save_reference(tmp_path, config, {"experiment_id": "exp-123"})
    args = _reference_args(path)

    sampler, spatial_mode, spatial_size, color, resolved = (
        resolve_evaluation_preprocessing(args)
    )
    backbone = resolve_backbone_config(args)

    assert sampler.config() == config["sampler"]
    assert (spatial_mode, spatial_size, color) == ("stretch", 192, "grayscale")
    assert resolved["source"] == str(path)
    assert backbone["backbone_type"] == "X3D-S"
    assert backbone["feature_stream"] == "3d"
    assert backbone["source"] == str(path)


def test_reference_without_metadata_raises_value_error(tmp_path):
    path = tmp_path / "no_metadata.pt"
    torch.save({"samples": []}, path)
    args = _reference_args(path)

    with pytest.raises(ValueError, match="metadata"):
        resolve_evaluation_preprocessing(args)
    with pytest.raises(ValueError, match="metadata"):
        resolve_backbone_config(args)


@pytest.mark.parametrize(
    "resolver,missing_field",
    [
        (resolve_evaluation_preprocessing, "color"),
        (resolve_backbone_config, "tracker"),
        (resolve_backbone_config, "backbone"),
    ],
)
def test_reference_missing_required_config_field_raises(
    tmp_path, resolver, missing_field
):
    config = _experiment_config()
    del config[missing_field]
    path = _save_reference(tmp_path, config)

    with pytest.raises(ValueError, match=missing_field):
        resolver(_reference_args(path))


def test_checkpoint_input_dimension_is_checked(tmp_path):
    neck = BiGRUNeck(in_dim=432, hidden_dim=4, num_layers=1, dropout=0.0)
    head = MeanPoolingClassifierHead(in_dim=8, num_classes=3, dropout=0.0)
    model = ARPipeline(backbone=None, neck=neck, head=head)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    with pytest.raises(ValueError, match=r"D=432.*D=437"):
        build_model_from_checkpoint(
            str(checkpoint_path), torch.device("cpu"), in_dim=437
        )

    loaded = build_model_from_checkpoint(
        str(checkpoint_path), torch.device("cpu"), in_dim=432
    )
    assert loaded.neck.gru.input_size == 432
