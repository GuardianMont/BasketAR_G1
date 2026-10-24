"""
Offline feature extraction utilities.

Converts raw video batches into pre-extracted temporal features that can be
reused to train Temporal Neck and Action Head without recomputing the backbone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


_REQUIRED_BATCH_KEYS = {
    "labels",
    "lengths",
    "sample_ids",
}


def _validate_batch(batch: Dict[str, Any]) -> None:
    missing = _REQUIRED_BATCH_KEYS.difference(batch)

    if missing:
        raise KeyError(
            "Feature extraction requires batch keys "
            f"{sorted(_REQUIRED_BATCH_KEYS)}. "
            f"Missing: {sorted(missing)}."
        )

    if "x" in batch:
        x = batch["x"]
    elif "x_norm" in batch and "x_raw" in batch:
        x = batch["x_norm"]
    else:
        raise KeyError("Batch must contain either 'x' or both 'x_norm' and 'x_raw'.")

    if not isinstance(x, torch.Tensor) or x.dim() != 5:
        raise ValueError(
            "Expected raw video batch with shape "
            f"(B, C, T, H, W), got {getattr(x, 'shape', None)}."
        )

    batch_size = x.shape[0]

    if len(batch["labels"]) != batch_size:
        raise ValueError("labels length does not match batch size.")

    if len(batch["lengths"]) != batch_size:
        raise ValueError("lengths length does not match batch size.")

    if len(batch["sample_ids"]) != batch_size:
        raise ValueError("sample_ids length does not match batch size.")


def _resolve_device(
    device: Optional[torch.device | str],
) -> torch.device:

    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def extract_features_offline(
    backbone: nn.Module,
    dataloader: DataLoader,
    output_path: str | Path,
    device: Optional[torch.device | str] = None,
    overwrite: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract temporal features for every clip in a DataLoader.

    Expected batch:
        x:
            (B, C, T_max, H, W)

        labels:
            (B,)

        lengths:
            valid temporal length for every clip

        sample_ids:
            identifier of every clip

    Output:
        A .pt file containing:

        {
            "format_version": 1,
            "backbone": "...",
            "feature_dim": D,
            "num_samples": N,
            "samples": [
                {
                    "id": ...,
                    "data": Tensor[T, D],
                    "label": int,
                    "num_frames": T,
                },
                ...
            ]
        }

    Important:
        The backbone must already be fully frozen.

        For X3DSBackbone(out_dim=512), the 192 -> 512 projection
        should contain the desired trained weights before calling
        backbone.freeze() and performing offline extraction.
    """

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use overwrite=True to replace it."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Offline extraction should use a fixed feature extractor.
    if any(
        parameter.requires_grad
        for parameter in backbone.parameters()
    ):
        raise RuntimeError(
            "Offline extraction requires a fully frozen backbone. "
            "Load the desired checkpoint first and then call "
            "backbone.freeze()."
        )

    device = _resolve_device(device)

    backbone = backbone.to(device)
    backbone.eval()

    extracted_samples: List[Dict[str, Any]] = []

    for batch_index, batch in enumerate(dataloader):

        _validate_batch(batch)

        is_dual = "x_norm" in batch and "x_raw" in batch
        videos = batch["x"] if not is_dual else batch["x_norm"]
        labels = batch["labels"]

        lengths = torch.as_tensor(
            batch["lengths"],
            dtype=torch.long,
        )

        sample_ids = batch["sample_ids"]

        if torch.any(lengths <= 0):
            raise ValueError(
                "All samples must have a positive temporal length."
            )

        if torch.any(lengths > videos.shape[2]):
            raise ValueError(
                "A sample length exceeds the padded batch length."
            )

        features_by_sample: List[
            Optional[torch.Tensor]
        ] = [None] * videos.shape[0]

        #
        # Important:
        #
        # X3D is not attention-mask aware.
        #
        # Therefore we do NOT send padded frames through X3D.
        # Samples having the same valid length are grouped together
        # and truncated before entering the backbone.
        #
        for valid_length in torch.unique(lengths, sorted=True).tolist():
            sample_indices = torch.nonzero(lengths == valid_length, as_tuple=False).flatten()
            if is_dual:
                clip_group_norm = batch["x_norm"][sample_indices,:,:valid_length,:,:].to(
                    device,
                    non_blocking=True,
                )
                clip_group_raw = batch["x_raw"][sample_indices,:,:valid_length,:,:].to(
                    device,
                    non_blocking=True,
                )
                if hasattr(backbone, 'forward_dual'):
                    group_features = backbone.forward_dual(clip_group_norm, clip_group_raw)
                else:
                    raise RuntimeError("Dual inputs provided but backbone lacks forward_dual()")
            else:
                clip_group = videos[sample_indices,:,:valid_length,:,:].to(
                    device,
                    non_blocking=True,
                )
    
                group_features = backbone(
                    clip_group
                )

            if group_features.dim() != 3:
                raise RuntimeError(
                    "Backbone must return features with shape (B, T, D), got "
                    f"{tuple(group_features.shape)}."
                )

            if group_features.shape[0] != len(sample_indices):
                raise RuntimeError("Backbone unexpectedly changed the batch dimension.")

            #
            # X3DSBackbone currently preserves T.
            #
            """
            if group_features.shape[1] != valid_length:
                raise RuntimeError(
                    "Current extractor expects a backbone "
                    "that preserves temporal length. "
                    f"Input T={valid_length}, "
                    f"output T={group_features.shape[1]}."
                )"""

            group_features = (group_features.detach().cpu())

            for group_position, original_index in enumerate(sample_indices.tolist()):
                features_by_sample[original_index] = (group_features[group_position].contiguous())

        #
        # Restore original DataLoader sample order.
        #
        for sample_index, features in enumerate(features_by_sample):
            if features is None:
                raise RuntimeError("Feature extraction failed for one or more samples.")

            extracted_samples.append(
                {
                    "id": sample_ids[sample_index],
                    "data": features,
                    "label": int(labels[sample_index]),
                    "num_frames": int(features.shape[0]),
                }
            )

        try:
            total_batches = len(dataloader)
            progress = (f"{batch_index + 1}/{total_batches}")

        except TypeError:
            progress = str(batch_index + 1)

        print(f"\rExtracted batch {progress}", end="", flush=True)

    print()

    feature_dim = None

    if extracted_samples:
        feature_dim = int(extracted_samples[0]["data"].shape[1])

    payload = {
        "format_version": 1,
        "backbone": backbone.__class__.__name__,
        "feature_dim": feature_dim,
        "metadata": metadata or {},
        "num_samples": len(
            extracted_samples
        ),
        "samples": extracted_samples,
    }

    #
    # Save atomically: avoid leaving a corrupted final file
    # if the process stops during torch.save().
    #
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    torch.save(payload, temporary_path)

    temporary_path.replace(output_path)

    return payload
