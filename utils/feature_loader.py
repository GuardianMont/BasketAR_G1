import os
from typing import Any

import torch


class FeatureLoader:
    """Load and validate pre-extracted feature payloads."""

    @staticmethod
    def load_payload(
        pt_file_path: str,
        expected_dim: int | None = None,
        expected_experiment_id: str | None = None,
    ) -> dict[str, Any]:
        if not os.path.isfile(pt_file_path):
            raise FileNotFoundError(
                f"Feature file not found: '{pt_file_path}'. "
                "Extract features for the intended experiment before training."
            )

        payload = torch.load(pt_file_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid feature payload in '{pt_file_path}': expected a dictionary.")

        backbone = payload.get("backbone")
        if backbone == "MockBackbone":
            raise ValueError(f"Invalid feature payload in '{pt_file_path}': MockBackbone features are not usable.")

        feature_dim = payload.get("feature_dim")
        if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim <= 0:
            raise ValueError(
                f"Invalid feature payload in '{pt_file_path}': feature_dim must be a positive integer."
            )
        if expected_dim is not None and feature_dim != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch in '{pt_file_path}': payload has D={feature_dim}, "
                f"expected D={expected_dim}."
            )

        samples = payload.get("samples")
        if not isinstance(samples, (list, tuple)) or not samples:
            raise ValueError(f"Invalid feature payload in '{pt_file_path}': 'samples' must be non-empty.")

        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"Invalid sample {index} in '{pt_file_path}': expected a dictionary.")
            data = sample.get("data")
            if not isinstance(data, torch.Tensor):
                raise ValueError(f"Invalid sample {index} in '{pt_file_path}': 'data' must be a tensor.")
            if data.ndim != 2 or data.shape[0] <= 0 or data.shape[1] != feature_dim:
                raise ValueError(
                    f"Invalid sample {index} in '{pt_file_path}': expected data with shape (T, {feature_dim}) "
                    f"and T>0, got {tuple(data.shape)}."
                )

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        experiment_id = metadata.get("experiment_id")
        if expected_experiment_id is not None and experiment_id != expected_experiment_id:
            raise ValueError(
                f"Experiment mismatch in '{pt_file_path}': metadata experiment_id is "
                f"{experiment_id!r}, expected {expected_experiment_id!r}."
            )

        experiment_config = metadata.get("experiment_config", {})
        if not isinstance(experiment_config, dict):
            experiment_config = {}
        sampling = experiment_config.get("sampler", experiment_config.get("temporal_sampler", "unavailable"))
        spatial_mode = experiment_config.get("spatial_mode", "unavailable")
        spatial_size = experiment_config.get("spatial_size", "unavailable")
        color = experiment_config.get("color", "unavailable")
        print(
            f"[INFO] Loaded {len(samples)} samples (D={feature_dim}, backbone={backbone!r}) "
            f"from '{pt_file_path}'."
        )
        print(
            "[INFO] Feature preprocessing: "
            f"sampler={sampling}, spatial_mode={spatial_mode}, spatial_size={spatial_size}, color={color}, "
            f"experiment_id={experiment_id or 'unavailable'}"
        )
        return payload

    @staticmethod
    def load(
        pt_file_path: str,
        expected_dim: int | None = None,
        expected_experiment_id: str | None = None,
    ) -> list[dict[str, Any]]:
        payload = FeatureLoader.load_payload(
            pt_file_path,
            expected_dim=expected_dim,
            expected_experiment_id=expected_experiment_id,
        )
        return list(payload["samples"])


def load_experiment_feature_pair(
    train_path: str,
    val_path: str | None = None,
    default_experiment_id: str | None = None,
    expected_experiment_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str, dict[str, Any] | None]:
    """Load a train/validation pair and resolve its experiment from train metadata."""
    train_payload = FeatureLoader.load_payload(train_path)
    train_metadata = train_payload.get("metadata")
    train_metadata = train_metadata if isinstance(train_metadata, dict) else {}
    train_experiment_id = train_metadata.get("experiment_id")

    resolved_id = expected_experiment_id or train_experiment_id or default_experiment_id
    if resolved_id is None:
        raise ValueError(
            f"Feature file '{train_path}' has no metadata['experiment_id']; "
            "pass --expected-experiment-id only when the intended ID is known."
        )
    if train_experiment_id is None:
        raise ValueError(
            f"Feature file '{train_path}' has no metadata['experiment_id']; "
            "cannot verify the selected experiment."
        )
    if train_experiment_id != resolved_id:
        raise ValueError(
            f"Experiment mismatch in '{train_path}': metadata experiment_id is "
            f"{train_experiment_id!r}, expected {resolved_id!r}."
        )

    val_payload = None
    if val_path is not None:
        val_payload = FeatureLoader.load_payload(
            val_path,
            expected_dim=train_payload["feature_dim"],
            expected_experiment_id=resolved_id,
        )

    experiment_config = train_metadata.get("experiment_config")
    if not isinstance(experiment_config, dict):
        experiment_config = None
    return train_payload, val_payload, resolved_id, experiment_config
