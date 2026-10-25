import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import List, Tuple, Optional


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

from typing import List, Optional, Set
import torch
import torchvision.transforms.v2 as v2

class ClassAwareVideoTransform:
    """Gestisce trasformazioni multiple ed esegue l'augmentation 

    usando esclusivamente torchvision.transforms.v2.
    """

    def __init__(
        self,
        transform_names: List[str],
        minority_classes: Optional[Set[int]] = None,
        temporal_shift_range: int = 3,
        flip_probability: float = 0.5,
    ):
        self.transform_names = [t for t in transform_names if t != "none"]
        self.minority_classes = minority_classes
        self.temporal_shift_range = temporal_shift_range
        self.flip_probability = flip_probability

        self.has_horizontal_flip = "horizontal_flip" in self.transform_names
        self.has_temporal_shift = "temporal_shift" in self.transform_names

        v2_ops = []
        for name in self.transform_names:
            if name == "color_jitter":
                v2_ops.append(
                    v2.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                    )
                )
            elif name == "gaussian_blur":
                v2_ops.append(
                    v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
                )
            elif name == "random_crop":
                v2_ops.append(
                    v2.RandomResizedCrop(
                        size=(224, 224), scale=(0.85, 1.0), antialias=True
                    )
                )

        self.v2_pipeline = v2.Compose(v2_ops) if v2_ops else None

    def should_apply(self, label_id: int) -> bool:
        if not self.transform_names:
            return False
        if self.minority_classes is None:
            return True
        return label_id in self.minority_classes

    def apply_temporal_shift(self, indices: List[int], total_frames: int) -> List[int]:
        """Applica uno shift temporale casuale agli indici campionati."""
        shift = random.randint(-self.temporal_shift_range, self.temporal_shift_range)
        return [max(0, min(total_frames - 1, idx + shift)) for idx in indices]

    def apply_spatial_transforms(
        self, clip_norm: torch.Tensor, clip_raw: torch.Tensor, label_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Applica flip orizzontale e augmentation v2 ai tensori (C, T, H, W)."""
        is_flipped = False

        if not self.should_apply(label_id):
            return clip_norm, clip_raw, is_flipped

        # Horizontal Flip nativo (funziona sia su C,T,H,W che T,C,H,W poiché agisce su W)
        if self.has_horizontal_flip and random.random() < self.flip_probability:
            clip_norm = v2.functional.horizontal_flip(clip_norm)
            clip_raw = v2.functional.horizontal_flip(clip_raw)
            is_flipped = True

        # Altre trasformazioni spaziali/cromatiche v2
        if self.v2_pipeline is not None:
            # (C, T, H, W) -> (T, C, H, W) affinché torchvision legga C=3 al dim -3
            clip_norm_t = clip_norm.permute(1, 0, 2, 3)
            clip_norm_t = self.v2_pipeline(clip_norm_t)
            # Ripristina (T, C, H, W) -> (C, T, H, W)
            clip_norm = clip_norm_t.permute(1, 0, 2, 3)

        return clip_norm, clip_raw, is_flipped