from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.temporal_sampling import build_temporal_sampler
from scripts.extract_tracking_features import (
    FEATURE_NAMES,
    _group_step_end_indices,
    _hash_tracker,
    extract_tracking_features,
)


@pytest.mark.parametrize(
    ("sampled_count", "expected_ends"),
    [
        (32, list(range(1, 32, 2))),
        (6, [1, 3, 5]),
        (5, [1, 3, 4]),
        (1, [0]),
    ],
)
def test_frames_per_step_groups_even_odd_and_short_clips(sampled_count, expected_ends):
    sampled_indices = list(range(sampled_count))
    assert _group_step_end_indices(sampled_indices, 2) == expected_ends
    assert len(expected_ends) == (sampled_count + 1) // 2


def test_frames_per_step_is_part_of_tracker_identity():
    common = ("detector", 0.3, 1.95, 0.2, 1.5)
    assert _hash_tracker(*common, 1) != _hash_tracker(*common, 2)


@pytest.mark.parametrize("sampled_count", [32, 5, 1])
def test_tracking_extraction_matches_swin_temporal_steps(tmp_path, sampled_count):
    split = "train"
    video = tmp_path / "clip.mp4"
    video.touch()
    (tmp_path / "manifest.csv").write_text(
        f"split,clip_id,path,label\n{split},clip_a,clip.mp4,idle\n", encoding="utf-8"
    )
    sampler_config = {"name": "edge_dense", "num_frames": 32, "edge_strength": 1.0}
    sampler = build_temporal_sampler(sampler_config)
    sampled = sampler.sample_indices(sampled_count, fps=25.0)
    expected_steps = (len(sampled) + 1) // 2
    reference = tmp_path / "swin.pt"
    torch.save(
        {
            "metadata": {
                "split": split,
                "experiment_id": "swin-exp",
                "experiment_config": {
                    "sampler": sampler_config,
                    "extra": {"transform": "none"},
                },
            },
            "samples": [{"id": "clip_a", "data": torch.zeros(expected_steps, 768)}],
        },
        reference,
    )

    detector_id = "detector-id"
    cache_dir = tmp_path / "detections" / detector_id / split
    cache_dir.mkdir(parents=True)
    cache_metadata = {"detector_id": detector_id}
    np.savez(
        cache_dir / "clip_a.npz",
        candidates=np.zeros((sampled_count, 2, 5, 5), dtype=np.float32),
        timestamps_sec=np.arange(sampled_count, dtype=np.float64) / 25.0,
        source_wh=np.tile(np.array([[100, 100]], dtype=np.int32), (sampled_count, 1)),
        fps=np.float64(25.0),
        num_decoded_frames=np.int64(sampled_count),
        stream_frames=np.int64(sampled_count),
        metadata_json=np.asarray(json.dumps(cache_metadata)),
    )

    args = Namespace(
        dataset_root=str(tmp_path), manifest=None, split=split,
        feature_reference=str(reference), expected_experiment_id="swin-exp",
        detector_id=detector_id, detections_root=None, confidence_threshold=0.3,
        max_speed=1.95, max_gap_s=0.2, rim_expansion=1.5, frames_per_step=2,
        output_dir=str(tmp_path / "tracking"), limit=None, overwrite=False,
        save_previews=False, show_rejected=False,
    )
    extract_tracking_features(args)
    sidecar = next((tmp_path / "tracking").glob("train__*.pt"))
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    assert payload["metadata"]["frames_per_step"] == 2
    assert payload["metadata"]["tracker_id"] in sidecar.name
    assert payload["samples"][0]["data"].shape == (expected_steps, len(FEATURE_NAMES))
