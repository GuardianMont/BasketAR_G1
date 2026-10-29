from __future__ import annotations

import json

import pytest
import torch

from scripts.extract_tracking_features import FEATURE_NAMES
from scripts.merge_tracking_into_features import (
    merge_features,
    feature_names_for_group,
)
from utils.feature_loader import load_experiment_feature_pair


BINARY = {
    "ball_visible", "rim_visible", "ball_velocity_valid", "ball_acceleration_valid",
    "ball_rim_valid", "ball_above_rim", "ball_below_rim",
    "ball_center_inside_expanded_rim_box", "ball_rim_motion_valid",
    "rim_crossing_valid", "ball_crossed_rim_down", "ball_window_motion_valid",
    "ball_rim_window_valid", "ball_crossing_window_or", "ball_seen_before",
    "ball_window_position_valid", "ball_window_speed_valid",
}


def _make_payloads(tmp_path, split):
    swin_path = tmp_path / f"{split}_swin.pt"
    track_path = tmp_path / f"{split}_tracking.pt"
    experiment_id = "videoswin-exp"
    tracker_id = "tracker-exp"
    source_samples = [
        {"id": "clip-a", "data": torch.ones(3, 768), "label": 0, "num_frames": 3},
        {"id": "clip-b", "data": torch.full((2, 768), 2.0), "label": 1, "num_frames": 2},
    ]
    track_by_id = {}
    for sample_id, t, x_value in (("clip-a", 3, 0.2), ("clip-b", 2, 0.4)):
        values = torch.zeros(t, len(FEATURE_NAMES))
        for flag in ("ball_visible", "rim_visible", "ball_velocity_valid", "ball_window_position_valid"):
            values[:, FEATURE_NAMES.index(flag)] = 1.0
        values[:, FEATURE_NAMES.index("ball_x_center_norm")] = x_value
        values[:, FEATURE_NAMES.index("rim_x_center_norm")] = x_value + 0.1
        track_by_id[sample_id] = {"id": sample_id, "data": values}
    torch.save(
        {
            "format_version": 1,
            "backbone": "CompositeBackbone",
            "feature_dim": 768,
            "num_samples": 2,
            "metadata": {
                "split": split,
                "experiment_id": experiment_id,
                "experiment_config": {
                    "sampler": {"name": "edge_dense", "num_frames": 32, "edge_strength": 1.0},
                    "spatial_mode": "stretch",
                    "spatial_size": 224,
                    "color": "rgb",
                    "backbone": "video_swin",
                    "extra": {"transform": "none"},
                },
            },
            "samples": source_samples,
        },
        swin_path,
    )
    torch.save(
        {
            "metadata": {
                "split": split,
                "experiment_id": experiment_id,
                "tracker_id": tracker_id,
                "detector_id": "detector-id",
                "frames_per_step": 2,
                "feature_names": list(FEATURE_NAMES),
            },
            # Deliberately reverse IDs to verify merge-by-ID, not list position.
            "samples": [track_by_id["clip-b"], track_by_id["clip-a"]],
        },
        track_path,
    )
    return swin_path, track_path, tracker_id


@pytest.mark.parametrize(("group", "tracking_dim"), [("rim_only", 6), ("ball_only", 25), ("all", 45)])
def test_feature_group_dimensions(group, tracking_dim):
    assert len(feature_names_for_group(group)) == tracking_dim


def test_merge_fits_train_scaler_and_reuses_it_for_val(tmp_path):
    train_swin, train_track, tracker_id = _make_payloads(tmp_path, "train")
    train_out, scaler_path = merge_features(
        train_swin, train_track, tmp_path / "merged", "all"
    )
    train_payload = torch.load(train_out, map_location="cpu", weights_only=False)
    assert train_payload["feature_dim"] == 768 + 45
    assert train_payload["samples"][0]["id"] == "clip-a"
    assert train_payload["samples"][0]["data"].shape == (3, 813)
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    assert scaler["fit_split"] == "train"
    assert scaler["tracker_id"] == tracker_id
    assert scaler["stats"]["ball_x_center_norm"]["mean"] == pytest.approx(0.28)
    assert "ball_visible" not in scaler["stats"]

    val_swin, val_track, _ = _make_payloads(tmp_path, "val")
    val_out, same_scaler = merge_features(
        val_swin, val_track, tmp_path / "merged", "all", scaler_path=scaler_path
    )
    assert same_scaler == scaler_path
    val_payload = torch.load(val_out, map_location="cpu", weights_only=False)
    assert val_payload["metadata"]["experiment_id"] == train_payload["metadata"]["experiment_id"]
    load_experiment_feature_pair(str(train_out), str(val_out))
