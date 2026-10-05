import pytest
import torch

from data import VideoHorizontalFlip
from scripts.extract_dataset_features import validate_tracker_preprocessing
from utils.feature_loader import load_experiment_feature_pair


@pytest.mark.parametrize("feature_stream", ["all", "3d_tracker"])
@pytest.mark.parametrize(
    "spatial_mode,color",
    [("stretch", "rgb"), ("letterbox", "grayscale")],
)
def test_tracker_guard_rejects_incompatible_preprocessing(
    feature_stream, spatial_mode, color
):
    with pytest.raises(ValueError, match="--feature-stream 3d"):
        validate_tracker_preprocessing(spatial_mode, color, feature_stream)


@pytest.mark.parametrize(
    "spatial_mode,color",
    [("stretch", "rgb"), ("letterbox", "grayscale")],
)
def test_tracker_guard_allows_non_tracker_stream(spatial_mode, color):
    validate_tracker_preprocessing(spatial_mode, color, "3d")


def test_video_horizontal_flip_is_deterministic_for_20_calls():
    clip = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    flip = VideoHorizontalFlip()
    expected = clip.flip(-1)

    for _ in range(20):
        torch.testing.assert_close(flip(clip), expected)


def _save_features(path, experiment_id=None):
    metadata = {"experiment_config": {"color": "rgb"}}
    if experiment_id is not None:
        metadata["experiment_id"] = experiment_id
    torch.save(
        {
            "backbone": "X3D-L",
            "feature_dim": 3,
            "samples": [{"data": torch.ones(4, 3), "label": 1}],
            "metadata": metadata,
        },
        path,
    )


def test_experiment_id_is_resolved_from_train_metadata(tmp_path):
    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    _save_features(train_path, "exp-a")
    _save_features(val_path, "exp-a")

    train_payload, val_payload, experiment_id, experiment_config = load_experiment_feature_pair(
        str(train_path), str(val_path)
    )

    assert experiment_id == "exp-a"
    assert train_payload["feature_dim"] == val_payload["feature_dim"] == 3
    assert experiment_config == {"color": "rgb"}


def test_experiment_resolution_rejects_different_validation_id(tmp_path):
    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    _save_features(train_path, "exp-a")
    _save_features(val_path, "exp-b")

    with pytest.raises(ValueError, match="Experiment mismatch"):
        load_experiment_feature_pair(str(train_path), str(val_path))


@pytest.mark.parametrize("missing_split", ["train", "val"])
def test_experiment_resolution_rejects_missing_id(tmp_path, missing_split):
    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    _save_features(train_path, None if missing_split == "train" else "exp-a")
    _save_features(val_path, None if missing_split == "val" else "exp-a")

    with pytest.raises(ValueError, match="experiment_id"):
        load_experiment_feature_pair(str(train_path), str(val_path))