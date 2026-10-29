"""Shared spatial and color preprocessing for X3D video inputs."""

from __future__ import annotations

import torch
import torchvision.transforms.functional as F



def preprocess_frame(
    frame,
    mean_norm_vector: tuple[float],
    std_norm_vector: tuple[float],
    spatial_mode: str = "letterbox",
    spatial_size: int = 312,
    color_mode: str = "rgb",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a decoded PyAV frame to normalized and raw ``(C,H,W)`` tensors.

    The raw tensor is in [0, 1], before X3D normalization. Grayscale is
    replicated to three channels so the pretrained X3D input shape is kept.
    """
    if spatial_mode not in {"letterbox", "stretch"}:
        raise ValueError(f"Unsupported spatial mode: {spatial_mode}")
    if spatial_size <= 0:
        raise ValueError("spatial_size must be positive")
    if color_mode not in {"rgb", "grayscale"}:
        raise ValueError(f"Unsupported color mode: {color_mode}")

    array = frame.to_ndarray(format="rgb24")
    x = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        / 255.0
    )

    _, height, width = x.shape
    if spatial_mode == "stretch":
        x_raw = F.resize(x, [spatial_size, spatial_size], antialias=True)
    else:
        if height > width:
            new_height = spatial_size
            new_width = int(width * (spatial_size / height))
        else:
            new_height = int(height * (spatial_size / width))
            new_width = spatial_size

        resized = F.resize(x, [new_height, new_width], antialias=True)
        pad_height = spatial_size - new_height
        pad_width = spatial_size - new_width
        padding = [
            pad_width // 2,
            pad_height // 2,
            pad_width - pad_width // 2,
            pad_height - pad_height // 2,
        ]
        x_raw = F.pad(resized, padding, fill=0)

    if color_mode == "grayscale":
        x_raw = F.rgb_to_grayscale(x_raw, num_output_channels=3)

    x_norm = F.normalize(x_raw, mean=mean_norm_vector, std=std_norm_vector)
    return x_norm, x_raw
