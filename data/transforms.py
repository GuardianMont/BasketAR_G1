import random
from typing import List, Optional, Set, Tuple

import torch
import torchvision.transforms.v2 as v2


class RandomTemporalShift:
    """
    Applies a random temporal offset during frame selection only if the
    total number of frames exceeds a minimum threshold.
    """

    def __init__(self, min_frames_threshold: int = 16, shift_range: int = 4):
        self.min_frames_threshold = min_frames_threshold
        self.shift_range = shift_range

    def __call__(self, indices: List[int], total_frames: int) -> List[int]:
        if total_frames < self.min_frames_threshold or len(indices) == 0:
            return indices

        shift = random.randint(-self.shift_range, self.shift_range)
        shifted_indices = [
            max(0, min(total_frames - 1, idx + shift)) for idx in indices
        ]
        if len(set(shifted_indices)) != len(shifted_indices):
            return indices
        return shifted_indices


class ClassAwareVideoTransform:
    """Gestisce trasformazioni multiple ed esegue l'augmentation 

    usando esclusivamente torchvision.transforms.v2 su tensori non normalizzati [0, 1].
    """

    def __init__(
        self,
        transform_names: List[str],
        minority_classes: Optional[Set[int]] = None,
        temporal_shift_range: int = 3,
        flip_probability: float = 0.5,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
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
        self.normalizer = v2.Normalize(mean=mean, std=std)

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
        self, clip_unnorm: torch.Tensor, clip_raw: torch.Tensor, label_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Applica flip orizzontale e augmentation v2 al tensore NON normalizzato (C, T, H, W) [0, 1].
        Applica infine la normalizzazione con media e deviazione standard prima di restituire.
        """
        is_flipped = False

        # Applica augmentation solo se la classe è target
        if self.should_apply(label_id):
            if self.has_horizontal_flip and random.random() < self.flip_probability:
                clip_unnorm = v2.functional.horizontal_flip(clip_unnorm)
                clip_raw = v2.functional.horizontal_flip(clip_raw)
                is_flipped = True

            if self.v2_pipeline is not None:
                # Permuta in (T, C, H, W) affinché v2 legga C=3 in dim -3
                clip_t = clip_unnorm.permute(1, 0, 2, 3)
                clip_t = self.v2_pipeline(clip_t)  # Sicuro: operazione eseguita in range [0, 1]
                clip_unnorm = clip_t.permute(1, 0, 2, 3)

        # NORMALIZZAZIONE FINALE (Sia per campioni aumentati che non)
        clip_t = clip_unnorm.permute(1, 0, 2, 3)
        clip_norm = self.normalizer(clip_t).permute(1, 0, 2, 3)

        return clip_norm, clip_raw, is_flipped