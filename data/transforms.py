import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import List, Tuple, Optional


class VideoRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return torch.flip(clip, dims=[-1])
        return clip


# Deterministic flip used by the feature extraction script.
class VideoHorizontalFlip:
    """Deterministically flip every frame horizontally."""

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return torch.flip(clip, dims=[-1])


class VideoColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p: float = 0.5):
        self.p = p
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (C, T, H, W)
        if random.random() < self.p:
            C, T_len, H, W = clip.shape
            frames = clip.permute(1, 0, 2, 3) # (T, C, H, W)
            frames = self.jitter(frames)
            return frames.permute(1, 0, 2, 3)
        return clip


class RandomTemporalShift:
    """
    Applies a random temporal offset during frame selection only if the
    total number of frames eccedes a minimum threshold.
    """
    def __init__(self, min_frames_threshold: int = 16, shift_range: int = 4):
        self.min_frames_threshold = min_frames_threshold
        self.shift_range = shift_range

    def __call__(self, indices: List[int], total_frames: int) -> List[int]:
        if total_frames < self.min_frames_threshold or len(indices) == 0:
            return indices
        
        shift = random.randint(-self.shift_range, self.shift_range)
        shifted_indices = [max(0, min(total_frames - 1, idx + shift)) for idx in indices]
        if len(set(shifted_indices)) != len(shifted_indices):
            return indices
        return shifted_indices
