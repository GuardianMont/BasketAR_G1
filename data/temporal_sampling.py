import math
from abc import ABC, abstractmethod
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar


class TemporalSampler(ABC):
    """Abstract interface for temporal frame-selection policies."""

    @abstractmethod
    def sample_indices(self, total_frames: int, fps: Optional[float] = None) -> List[int]:
        """Return ordered frame indices for a clip."""

    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Return a serializable description of this sampling policy."""


SamplerType = TypeVar("SamplerType", bound=Type[TemporalSampler])
SAMPLER_REGISTRY: Dict[str, Type[TemporalSampler]] = {}


def register_sampler(name: str) -> Callable[[SamplerType], SamplerType]:
    """Register a concrete sampler class under a stable config name."""
    if not name or not name.strip():
        raise ValueError("Sampler registry names cannot be empty.")

    def decorator(sampler_class: SamplerType) -> SamplerType:
        if not issubclass(sampler_class, TemporalSampler):
            raise TypeError("Registered samplers must inherit from TemporalSampler.")
        if name in SAMPLER_REGISTRY:
            raise ValueError(f"A sampler named '{name}' is already registered.")
        SAMPLER_REGISTRY[name] = sampler_class
        return sampler_class

    return decorator


def _validate_length(total_frames: int) -> None:
    if total_frames < 1:
        raise ValueError("The clip must contain at least one frame.")


@register_sampler("legacy_uniform")
class LegacyUniformSampler(TemporalSampler):
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


@register_sampler("uniform")
class UniformFixedSampler(TemporalSampler):
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


@register_sampler("preserve_all")
class PreserveAllSampler(TemporalSampler):
    """Preserve every temporal step, especially pre-extracted features."""

    def sample_indices(self, total_frames: int, fps: Optional[float] = None) -> List[int]:
        _validate_length(total_frames)
        return list(range(total_frames))

    def config(self) -> Dict[str, Any]:
        return {"name": "preserve_all"}


@register_sampler("fixed_fps")
class FixedFPSSampler(TemporalSampler):
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


@register_sampler("first_last")
class FirstLastSampler(TemporalSampler):
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


@register_sampler("edge_dense")
class EdgeDenseSampler(TemporalSampler):
    """num_frames campioni, piu' fitti ai bordi e radi al centro.
    edge_strength: 0 = uniforme, 1 = massima densita' ai bordi."""

    def __init__(self, num_frames: int = 16, edge_strength: float = 1.0):
        if num_frames < 2:
            raise ValueError("num_frames must be >= 2")
        if not 0.0 <= edge_strength <= 1.0:
            raise ValueError("edge_strength must be in [0, 1]")
        self.num_frames, self.edge_strength = num_frames, edge_strength

    def sample_indices(self, total_frames, fps=None):
        _validate_length(total_frames)
        n = self.num_frames
        if total_frames <= n:
            return list(range(total_frames))
        a = self.edge_strength
        idx = []
        for i in range(n):
            u = i / (n - 1)
            w = (1 - a) * u + a * (1 - math.cos(math.pi * u)) / 2
            idx.append(round(w * (total_frames - 1)))
        for i in range(1, n):                 # strettamente crescente
            idx[i] = max(idx[i], idx[i - 1] + 1)
        idx[-1] = total_frames - 1
        for i in range(n - 2, -1, -1):        # dentro i limiti
            idx[i] = min(idx[i], idx[i + 1] - 1)
        return idx

    def config(self):
        return {
            "name": "edge_dense",
            "num_frames": self.num_frames,
            "edge_strength": self.edge_strength,
        }


def build_temporal_sampler(config: Dict[str, Any]) -> TemporalSampler:
    """Instantiate a registered sampler from its serializable config."""
    if not isinstance(config, dict) or "name" not in config:
        raise ValueError("Sampler config must be a dictionary with a 'name'.")

    name = config["name"]
    if not isinstance(name, str) or name not in SAMPLER_REGISTRY:
        available = ", ".join(sorted(SAMPLER_REGISTRY))
        raise ValueError(f"Unknown temporal sampler '{name}'. Available: {available}.")

    sampler_class = SAMPLER_REGISTRY[name]
    params = {key: value for key, value in config.items() if key != "name"}
    init_parameters = signature(sampler_class.__init__).parameters
    accepted = {
        key
        for key, parameter in init_parameters.items()
        if key != "self"
        and parameter.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
    }
    required = {
        key
        for key, parameter in init_parameters.items()
        if key != "self"
        and parameter.default is Parameter.empty
        and parameter.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
    }
    missing = sorted(required - params.keys())
    unknown = sorted(params.keys() - accepted)
    problems = []
    if missing:
        problems.append(f"missing required parameter(s): {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown parameter(s): {', '.join(unknown)}")
    if problems:
        accepted_text = ", ".join(sorted(accepted)) or "none"
        raise ValueError(
            f"Invalid config for sampler '{name}': {'; '.join(problems)}. "
            f"Accepted parameters: {accepted_text}."
        )

    try:
        return sampler_class(**params)
    except TypeError as exc:
        raise ValueError(f"Invalid parameters for sampler '{name}': {exc}") from exc
