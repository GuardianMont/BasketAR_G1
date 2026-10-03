from typing import Any, Dict, List, Optional, Protocol


class TemporalSampler(Protocol):
    def sample_indices(self, total_frames: int, fps: Optional[float] = None) -> List[int]:
        """Return ordered frame indices for a clip."""

    def config(self) -> Dict[str, Any]:
        """Return a serializable description of this sampling policy."""


def _validate_length(total_frames: int) -> None:
    if total_frames < 1:
        raise ValueError("The clip must contain at least one frame.")


class LegacyUniformSampler:
    """Match the previous fixed-N uniform sampling, including edge repetition."""

    def __init__(self, num_frames: int = 16):
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        self.num_frames = num_frames

    def sample_indices(self, total_frames: int, fps: Optional[float] = None) -> List[int]:
        _validate_length(total_frames)
        if total_frames >= self.num_frames:
            stride = total_frames / float(self.num_frames)
            return [min(int(i * stride), total_frames - 1) for i in range(self.num_frames)]
        indices = list(range(total_frames))
        indices.extend([total_frames - 1] * (self.num_frames - total_frames))
        return indices

    def config(self) -> Dict[str, Any]:
        return {"name": "legacy_uniform", "num_frames": self.num_frames}


class UniformFixedSampler:
    """Select up to N uniformly spaced, unique frames; never repeat short clips."""

    def __init__(self, num_frames: int):
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        self.num_frames = num_frames

    def sample_indices(self, total_frames: int, fps: Optional[float] = None) -> List[int]:
        _validate_length(total_frames)
        if total_frames <= self.num_frames:
            return list(range(total_frames))
        if self.num_frames == 1:
            return [total_frames // 2]
        return [round(i * (total_frames - 1) / (self.num_frames - 1)) for i in range(self.num_frames)]

    def config(self) -> Dict[str, Any]:
        return {"name": "uniform", "num_frames": self.num_frames}


class PreserveAllSampler:
    """Preserve every temporal step, especially pre-extracted features."""

    def sample_indices(self, total_frames: int, fps: Optional[float] = None) -> List[int]:
        _validate_length(total_frames)
        return list(range(total_frames))

    def config(self) -> Dict[str, Any]:
        return {"name": "preserve_all"}


class FixedFPSSampler:
    """Subsample at an approximate target FPS while preserving variable length."""

    def __init__(self, target_fps: float):
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self.target_fps = target_fps

    def sample_indices(self, total_frames: int, fps: Optional[float] = None) -> List[int]:
        _validate_length(total_frames)
        if fps is None or fps <= 0:
            raise ValueError("FixedFPSSampler requires the source video's FPS")
        step = max(1, round(fps / self.target_fps))
        return list(range(0, total_frames, step))

    def config(self) -> Dict[str, Any]:
        return {"name": "fixed_fps", "target_fps": self.target_fps}


class FirstLastSampler:
    """Select the first K and last K unique frames, in temporal order."""

    def __init__(self, frames_per_side: int = 8):
        if frames_per_side < 1:
            raise ValueError("frames_per_side must be >= 1")
        self.frames_per_side = frames_per_side

    def sample_indices(self, total_frames: int, fps: Optional[float] = None) -> List[int]:
        _validate_length(total_frames)
        k = self.frames_per_side
        if total_frames <= 2 * k:
            return list(range(total_frames))
        return list(range(k)) + list(range(total_frames - k, total_frames))

    def config(self) -> Dict[str, Any]:
        return {"name": "first_last", "frames_per_side": self.frames_per_side}


def build_temporal_sampler(config: Dict[str, Any]) -> TemporalSampler:
    """Build a sampler from its serializable config representation."""
    if not isinstance(config, dict) or "name" not in config:
        raise ValueError("Sampler config must be a dictionary with a 'name'.")

    name = config["name"]
    if name == "legacy_uniform":
        return LegacyUniformSampler(config.get("num_frames", 16))
    if name == "uniform":
        if "num_frames" not in config:
            raise ValueError("The uniform sampler config requires 'num_frames'.")
        return UniformFixedSampler(config["num_frames"])
    if name == "preserve_all":
        return PreserveAllSampler()
    if name == "fixed_fps":
        if "target_fps" not in config:
            raise ValueError("The fixed_fps sampler config requires 'target_fps'.")
        return FixedFPSSampler(config["target_fps"])
    if name == "first_last":
        return FirstLastSampler(config.get("frames_per_side", 8))
    raise ValueError(f"Unsupported temporal sampler: {name}")
