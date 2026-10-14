"""Serializable experiment configuration and collision-resistant feature naming."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any, Dict, Optional


def _slug(value: str) -> str:
    value = value.lower().replace(".", "p")
    return re.sub(r"[^a-z0-9]+", "", value)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one experiment.

    The feature identifier contains only settings that change extracted features;
    neck and head are recorded for traceability but do not duplicate feature files.
    """

    sampler: Dict[str, Any]
    spatial_mode: str = "letterbox"
    spatial_size: int = 312
    color: str = "rgb"
    backbone: str = "x3d_l"
    tracker: str = "none"
    neck: Optional[str] = None
    head: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.spatial_mode not in {"letterbox", "stretch"}:
            raise ValueError(f"Unsupported spatial mode: {self.spatial_mode}")
        if self.spatial_size <= 0:
            raise ValueError("spatial_size must be positive")
        if self.color not in {"rgb", "grayscale"}:
            raise ValueError(f"Unsupported color mode: {self.color}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def feature_id(self) -> str:
        sampler_name = self.sampler.get("name", "unknown")
        if sampler_name in {"legacy_uniform", "uniform"}:
            sampler_label = f"{sampler_name}{self.sampler.get('num_frames', 16)}"
        elif sampler_name == "first_last":
            k = self.sampler.get("frames_per_side", 8)
            sampler_label = f"firstlast{k}x2"
        elif sampler_name == "fixed_fps":
            sampler_label = f"fixedfps{self.sampler.get('target_fps', 'unknown')}"
        elif sampler_name == "edge_dense":
            num_frames = self.sampler.get("num_frames", 16)
            edge_strength = float(self.sampler.get("edge_strength", 1.0))
            strength_percent = round(edge_strength * 100)
            sampler_label = f"edgedense{num_frames}s{strength_percent}"
        else:
            sampler_label = sampler_name

        spatial_label = f"{self.spatial_mode}{self.spatial_size}"
        readable = "__".join(
            _slug(value)
            for value in (
                self.backbone,
                f"tracker-{self.tracker}",
                sampler_label,
                spatial_label,
                self.color,
            )
        )
        canonical = json.dumps(
            {
                "sampler": self.sampler,
                "spatial_mode": self.spatial_mode,
                "spatial_size": self.spatial_size,
                "color": self.color,
                "backbone": self.backbone,
                "tracker": self.tracker,
                "extra": self.extra,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
        return f"{readable}__{digest}"

    def feature_filename(self, split: str) -> str:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported dataset split: {split}")
        return f"{split}__{self.feature_id()}.pt"
