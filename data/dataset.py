import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Callable
from .temporal_sampling import (
    FixedFPSSampler,
    LegacyUniformSampler,
    PreserveAllSampler,
    TemporalSampler,
)


phase_1_labels = {}

class ActionRecognitionDataset(Dataset):
    """Generic dataset for Action Recognition classification on video clips or extracted features.

    Supports both 5D Tensors for raw video input (C, T, H, W) and 3D Tensors for
    pre-extracted temporal features (T, D).

    Args:
        samples (List[Dict[str, Any]]): List of dictionary items containing sample metadata.
            Each dictionary must include at least:
                - "data": Tensor of raw video (C, T, H, W) or features (T, D), or path string.
                - "label": Integer class target.
                - "num_frames" (Optional): Total frame count of the clip.
        sampling_strategy (str, optional): Backward-compatible selector for
            'uniform' or 'fixed_fps'. If omitted, raw video tensors use the
            legacy uniform sampler and pre-extracted features are preserved.
        num_sampled_frames (int, optional): Fixed number of frames to sample when using 
            'uniform' strategy. Defaults to 16.
        target_fps (float, optional): Desired frame rate when using 'fixed_fps' strategy. 
            Defaults to 15.0.
        original_fps (float, optional): Original frame rate of input videos. Defaults to 30.0.
        spatial_transform (Optional[Callable], optional): Transformation function for 
            spatial data augmentation on raw video frames (C, T, H, W). Defaults to None.
        temporal_shift (Optional[Callable], optional): Transformation function for 
            temporal data augmentation/shifting on sampled frame indices. Defaults to None.
        is_feature_dataset (bool, optional): Flag indicating whether the dataset holds
            3D pre-extracted feature tensors (True) or raw video frames (False).
            Defaults to False.
        temporal_sampler (TemporalSampler, optional): Explicit temporal sampling
            policy. If omitted for pre-extracted features, all stored steps are kept.

    Example:
        >>> # Usage 1: Raw Video Dataset (5D Tensors)
        >>> video_samples = [
        ...     {"id": 0, "data": torch.randn(3, 60, 224, 224), "label": 1, "num_frames": 60}
        ... ]
        >>> raw_dataset = ActionRecognitionDataset(
        ...     samples=video_samples,
        ...     sampling_strategy="uniform",
        ...     num_sampled_frames=16,
        ...     is_feature_dataset=False
        ... )
        >>> sample = raw_dataset[0]
        >>> print(sample["data"].shape)
        torch.Size([3, 16, 224, 224])

        >>> # Usage 2: Pre-extracted Features Dataset (3D Tensors)
        >>> feature_samples = [
        ...     {"id": 0, "data": torch.randn(45, 512), "label": 3, "num_frames": 45}
        ... ]
        >>> feat_dataset = ActionRecognitionDataset(
        ...     samples=feature_samples,
        ...     sampling_strategy="fixed_fps",
        ...     target_fps=15.0,
        ...     original_fps=30.0,
        ...     is_feature_dataset=True
        ... )
        >>> feat_sample = feat_dataset[0]
        >>> print(feat_sample["data"].shape)
        torch.Size([23, 512])
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        sampling_strategy: Optional[str] = None,
        num_sampled_frames: int = 16,
        target_fps: float = 15.0,
        original_fps: float = 30.0,
        spatial_transform: Optional[Callable] = None,
        temporal_shift: Optional[Callable] = None,
        is_feature_dataset: bool = False,
        temporal_sampler: Optional[TemporalSampler] = None,
    ):
        self.samples = samples
        self.original_fps = original_fps
        self.spatial_transform = spatial_transform
        self.temporal_shift = temporal_shift
        self.is_feature_dataset = is_feature_dataset

        if temporal_sampler is not None:
            self.temporal_sampler = temporal_sampler
        elif sampling_strategy == "fixed_fps":
            self.temporal_sampler = FixedFPSSampler(target_fps)
        elif sampling_strategy in (None, "uniform"):
            self.temporal_sampler = (
                PreserveAllSampler()
                if is_feature_dataset and sampling_strategy is None
                else LegacyUniformSampler(num_sampled_frames)
            )
        else:
            raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")

    def __len__(self) -> int:
        return len(self.samples)

    def _get_sampled_indices(self, total_frames: int) -> List[int]:
        """Use the injected temporal policy, or preserve all feature steps."""
        indices = self.temporal_sampler.sample_indices(
            total_frames,
            fps=self.original_fps,
        )

        if self.temporal_shift is not None:
            indices = self.temporal_shift(indices, total_frames)

        return indices

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        orig_label = item["label"]  # 0..8
        
        # Lable mapping (-1 is for easy filtering)
        # Phase 1: Action (0 = passaggio, 1 = tiro, 2 = idle/non-gioco)
        if orig_label == 0:
            label_p1 = 0
        elif orig_label in [1, 2, 3, 4, 5, 6]:
            label_p1 = 1
        else:
            label_p1 = 2

        # Phase 2: Shot's type (0 = tiro da 2, 1 = tiro da 3, 2 = tiro libero, -1 = N/A)
        if orig_label in [1, 2]:    label_p2 = 0
        elif orig_label in [3, 4]:  label_p2 = 1
        elif orig_label in [5, 6]:  label_p2 = 2
        else:                       label_p2 = -1

        # Phase 3: Shot's outcome (0 = sbagliato, 1 = segnato, -1 = N/A)
        if orig_label in [1, 3, 5]:   label_p3 = 0
        elif orig_label in [2, 4, 6]: label_p3 = 1 
        else:                         label_p3 = -1 

        data_source = item["data"]
        # The tensor shape is authoritative; payload metadata may be stale.
        total_frames = (
            data_source.shape[0]
            if self.is_feature_dataset
            else data_source.shape[1]
        )

        indices = self._get_sampled_indices(total_frames)
        valid_length = len(indices)

        x = data_source[indices] if self.is_feature_dataset else data_source[:, indices, :, :]

        return {
            "data": x,
            "label_orig": orig_label,
            "label_p1": label_p1,     
            "label_p2": label_p2,    
            "label_p3": label_p3,    
            "length": valid_length,
            "sample_id": item.get("id", idx)
        }
