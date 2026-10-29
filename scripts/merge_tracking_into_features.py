from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.extract_tracking_features import FEATURE_NAMES


GROUPS = ("rim_only", "ball_only", "all")
RIM_FEATURES = {
    "rim_visible",
    "rim_confidence",
    "rim_x_center_norm",
    "rim_y_center_norm",
    "rim_width_norm",
    "rim_height_norm",
}
BALL_RIM_RELATION_FEATURES = {
    "ball_rim_valid",
    "ball_rim_dx_in_rim_widths",
    "ball_rim_dy_in_rim_heights",
    "ball_rim_distance_in_rim_units",
    "ball_above_rim",
    "ball_below_rim",
    "ball_center_inside_expanded_rim_box",
    "ball_rim_motion_valid",
    "ball_rim_approach_speed_per_s",
    "rim_crossing_valid",
    "ball_crossed_rim_down",
    "ball_rim_min_distance_window",
    "ball_rim_window_valid",
    "ball_crossing_window_or",
}
BINARY_FEATURES = {
    "ball_visible",
    "rim_visible",
    "ball_velocity_valid",
    "ball_acceleration_valid",
    "ball_rim_valid",
    "ball_above_rim",
    "ball_below_rim",
    "ball_center_inside_expanded_rim_box",
    "ball_rim_motion_valid",
    "rim_crossing_valid",
    "ball_crossed_rim_down",
    "ball_window_motion_valid",
    "ball_rim_window_valid",
    "ball_crossing_window_or",
    "ball_seen_before",
    "ball_window_position_valid",
    "ball_window_speed_valid",
}
VALIDITY_MASKS = {
    "ball_confidence": "ball_visible",
    "ball_x_center_norm": "ball_visible",
    "ball_y_center_norm": "ball_visible",
    "ball_width_norm": "ball_visible",
    "ball_height_norm": "ball_visible",
    "rim_confidence": "rim_visible",
    "rim_x_center_norm": "rim_visible",
    "rim_y_center_norm": "rim_visible",
    "rim_width_norm": "rim_visible",
    "rim_height_norm": "rim_visible",
    "ball_vx_norm_per_s": "ball_velocity_valid",
    "ball_vy_norm_per_s": "ball_velocity_valid",
    "ball_speed_norm_per_s": "ball_velocity_valid",
    "ball_ax_norm_per_s2": "ball_acceleration_valid",
    "ball_ay_norm_per_s2": "ball_acceleration_valid",
    "ball_acceleration_norm_per_s2": "ball_acceleration_valid",
    "ball_rim_dx_in_rim_widths": "ball_rim_valid",
    "ball_rim_dy_in_rim_heights": "ball_rim_valid",
    "ball_rim_distance_in_rim_units": "ball_rim_valid",
    "ball_rim_approach_speed_per_s": "ball_rim_motion_valid",
    "ball_last_x_center_norm_window": "ball_window_position_valid",
    "ball_last_y_center_norm_window": "ball_window_position_valid",
    "ball_window_displacement_x_per_s": "ball_window_motion_valid",
    "ball_window_displacement_y_per_s": "ball_window_motion_valid",
    "ball_window_max_speed_norm_per_s": "ball_window_speed_valid",
    "ball_rim_min_distance_window": "ball_rim_window_valid",
    "ball_time_since_detection_s": "ball_seen_before",
}


def feature_names_for_group(group: str) -> list[str]:
    if group not in GROUPS:
        raise ValueError(f"Gruppo non supportato: {group!r}")
    if group == "rim_only":
        names = [name for name in FEATURE_NAMES if name in RIM_FEATURES]
    elif group == "ball_only":
        names = [
            name for name in FEATURE_NAMES
            if name not in RIM_FEATURES and name not in BALL_RIM_RELATION_FEATURES
        ]
    else:
        names = list(FEATURE_NAMES)
    if not names:
        raise ValueError(f"Il gruppo {group!r} non seleziona feature.")
    return names


def _load_payload(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} non trovato: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"Payload {description} non valido: metadata mancante.")
    if not isinstance(payload.get("samples"), list) or not payload["samples"]:
        raise ValueError(f"Payload {description} senza samples.")
    return payload


def _index_samples(payload: dict[str, Any], description: str) -> dict[str, dict[str, Any]]:
    result = {}
    for sample in payload["samples"]:
        if not isinstance(sample, dict) or sample.get("id") is None:
            raise ValueError(f"Sample senza ID nel payload {description}.")
        sample_id = str(sample["id"])
        if sample_id in result:
            raise ValueError(f"ID duplicato {sample_id!r} nel payload {description}.")
        data = sample.get("data")
        if not isinstance(data, torch.Tensor) or data.ndim != 2 or data.shape[0] <= 0:
            raise ValueError(f"Tensor data non valido per {sample_id!r} in {description}.")
        if not torch.isfinite(data).all():
            raise ValueError(f"Feature non finite per {sample_id!r} in {description}.")
        result[sample_id] = sample
    return result


def _valid_values(data: torch.Tensor, feature_index: int, names: list[str]) -> torch.Tensor:
    name = names[feature_index]
    mask_name = VALIDITY_MASKS.get(name)
    values = data[:, feature_index]
    if mask_name is None:
        return values
    mask_index = names.index(mask_name)
    return values[data[:, mask_index] > 0.5]


def fit_scaler(tracking_samples: list[dict[str, Any]]) -> dict[str, Any]:
    continuous_names = [name for name in FEATURE_NAMES if name not in BINARY_FEATURES]
    stats: dict[str, dict[str, float]] = {}
    for name in continuous_names:
        index = FEATURE_NAMES.index(name)
        chunks = [_valid_values(sample["data"], index, list(FEATURE_NAMES)) for sample in tracking_samples]
        chunks = [chunk.float() for chunk in chunks if chunk.numel()]
        if not chunks:
            stats[name] = {"mean": 0.0, "std": 1.0}
            continue
        values = torch.cat(chunks)
        if not torch.isfinite(values).all():
            raise ValueError(f"Valori non finiti nella colonna continua {name!r} del train.")
        mean = float(values.mean().item())
        std = float(values.std(unbiased=False).item())
        stats[name] = {"mean": mean, "std": std if std > 1e-8 else 1.0}
    canonical = json.dumps(stats, sort_keys=True, separators=(",", ":"))
    scaler_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return {
        "schema_version": 1,
        "fit_split": "train",
        "feature_names": list(FEATURE_NAMES),
        "binary_features": sorted(BINARY_FEATURES),
        "validity_mask_policy": "Fit on valid train values; invalid/missing values remain zero after scaling.",
        "stats": stats,
        "scaler_id": scaler_id,
    }


def apply_scaler(data: torch.Tensor, selected_names: list[str], scaler: dict[str, Any]) -> torch.Tensor:
    name_to_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    normalized = []
    for name in selected_names:
        index = name_to_index[name]
        values = data[:, index].float().clone()
        if name in BINARY_FEATURES:
            normalized.append(values)
            continue
        stats = scaler["stats"].get(name)
        if not isinstance(stats, dict):
            raise ValueError(f"Scaler senza statistiche per {name!r}.")
        mask_name = VALIDITY_MASKS.get(name)
        if mask_name is not None:
            valid = data[:, name_to_index[mask_name]] > 0.5
            values[valid] = (values[valid] - float(stats["mean"])) / float(stats["std"])
            values[~valid] = 0.0
        else:
            values = (values - float(stats["mean"])) / float(stats["std"])
        normalized.append(values)
    return torch.stack(normalized, dim=1)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def merge_features(
    swin_path: Path,
    tracking_path: Path,
    output_dir: Path,
    group: str,
    scaler_path: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    selected_names = feature_names_for_group(group)
    swin = _load_payload(swin_path, "Feature Swin")
    tracking = _load_payload(tracking_path, "Sidecar tracking")
    swin_meta, track_meta = swin["metadata"], tracking["metadata"]
    split = swin_meta.get("split")
    if split not in {"train", "val", "test"} or track_meta.get("split") != split:
        raise ValueError("Split Swin e tracking non corrispondenti o non valido.")
    swin_experiment_id = swin_meta.get("experiment_id")
    if not isinstance(swin_experiment_id, str) or not swin_experiment_id:
        raise ValueError("Feature Swin senza experiment_id.")
    if track_meta.get("experiment_id") != swin_experiment_id:
        raise ValueError("Il sidecar tracking non fa riferimento allo stesso esperimento Swin.")
    experiment_config = swin_meta.get("experiment_config", {})
    configured_backbone = str(experiment_config.get("backbone", "")).lower().replace("-", "_")
    if swin.get("feature_dim") != 768 or configured_backbone != "video_swin":
        raise ValueError(
            "Il payload deve contenere feature Video Swin (experiment_config.backbone='video_swin') "
            "con D=768."
        )
    if track_meta.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Schema delle 45 feature tracking non corrispondente.")
    tracker_id = track_meta.get("tracker_id")
    detector_id = track_meta.get("detector_id")
    if not tracker_id or not detector_id:
        raise ValueError("Metadati sidecar senza detector_id o tracker_id.")
    if track_meta.get("frames_per_step") != 2:
        raise ValueError("Il sidecar per Video Swin deve avere frames_per_step=2.")

    swin_by_id = _index_samples(swin, "feature Swin")
    track_by_id = _index_samples(tracking, "sidecar tracking")
    if swin_by_id.keys() != track_by_id.keys():
        missing = sorted(swin_by_id.keys() - track_by_id.keys())
        extra = sorted(track_by_id.keys() - swin_by_id.keys())
        raise ValueError(f"ID clip non coincidenti; mancanti nel tracking={missing[:5]}, extra={extra[:5]}.")

    for sample_id in swin_by_id:
        sw, tr = swin_by_id[sample_id]["data"], track_by_id[sample_id]["data"]
        if sw.shape[0] != tr.shape[0]:
            raise ValueError(f"T diverso per {sample_id!r}: Swin={sw.shape[0]}, tracking={tr.shape[0]}.")
        if tr.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"D tracking errata per {sample_id!r}: {tr.shape[1]} != {len(FEATURE_NAMES)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if scaler_path is None:
        scaler_path = output_dir / f"tracking_scaler__{swin_experiment_id}__{tracker_id}.json"
    if split == "train":
        if scaler_path.is_file():
            scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
            if scaler.get("fit_split") != "train" or scaler.get("swin_experiment_id") != swin_experiment_id or scaler.get("tracker_id") != tracker_id:
                raise ValueError("Scaler esistente non corrisponde a questo train/tracker.")
        else:
            train_tracking_samples = [track_by_id[sample["id"]] for sample in swin["samples"]]
            scaler = fit_scaler(train_tracking_samples)
            scaler["swin_experiment_id"] = swin_experiment_id
            scaler["tracker_id"] = tracker_id
            scaler["detector_id"] = detector_id
            _atomic_json(scaler_path, scaler)
    else:
        if not scaler_path.is_file():
            raise FileNotFoundError(f"Scaler train richiesto per {split}: {scaler_path}")
        scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
        if scaler.get("fit_split") != "train" or scaler.get("swin_experiment_id") != swin_experiment_id or scaler.get("tracker_id") != tracker_id:
            raise ValueError("Scaler non calcolato sul train corrispondente.")

    merge_identity = {
        "swin_experiment_id": swin_experiment_id,
        "tracker_id": tracker_id,
        "feature_group": group,
        "scaler_id": scaler["scaler_id"],
        "merge_schema_version": 1,
    }
    digest = hashlib.sha256(json.dumps(merge_identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    experiment_id = f"merge_{digest}"
    output_path = output_dir / f"{split}__{group}__{experiment_id}.pt"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output già presente: {output_path}; usa --overwrite per sostituirlo.")

    merged_samples = []
    for swin_sample in swin["samples"]:
        sample_id = str(swin_sample["id"])
        tracking_data = track_by_id[sample_id]["data"]
        normalized = apply_scaler(tracking_data, selected_names, scaler)
        swin_data = swin_sample["data"].float()
        merged = torch.cat((swin_data, normalized), dim=1)
        merged_sample = dict(swin_sample)
        merged_sample["data"] = merged.contiguous()
        merged_sample["num_frames"] = int(merged.shape[0])
        merged_samples.append(merged_sample)

    experiment_config = dict(swin_meta.get("experiment_config", {}))
    extra = dict(experiment_config.get("extra", {}))
    extra["tracking_merge"] = {
        "feature_group": group,
        "tracker_id": tracker_id,
        "detector_id": detector_id,
        "frames_per_step": track_meta["frames_per_step"],
        "scaler_id": scaler["scaler_id"],
    }
    experiment_config["extra"] = extra
    experiment_config["tracker"] = f"cached_ball_rim_{group}"
    metadata = {
        "split": split,
        "experiment_id": experiment_id,
        "experiment_config": experiment_config,
        "source_swin_experiment_id": swin_experiment_id,
        "detector_id": detector_id,
        "tracker_id": tracker_id,
        "frames_per_step": track_meta["frames_per_step"],
        "feature_group": group,
        "swin_feature_dim": int(swin["feature_dim"]),
        "tracking_feature_names": selected_names,
        "tracking_feature_dim": len(selected_names),
        "feature_dim": int(swin["feature_dim"]) + len(selected_names),
        "feature_names": [f"swin_{i:03d}" for i in range(int(swin["feature_dim"]))] + selected_names,
        "normalization": {
            "method": "train-only z-score on continuous tracking features",
            "scaler_id": scaler["scaler_id"],
            "scaler_path": str(scaler_path),
            "binary_features_unchanged": True,
            "invalid_masked_values_remain_zero": True,
        },
        "merge_identity": merge_identity,
    }
    payload = {
        "format_version": 1,
        "backbone": "VideoSwinBackbone+BallRimTracking",
        "feature_dim": metadata["feature_dim"],
        "num_samples": len(merged_samples),
        "metadata": metadata,
        "samples": merged_samples,
    }
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, output_path)
    return output_path, scaler_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unisce le feature Video Swin con le feature tracking allineate.")
    parser.add_argument("--swin-features", required=True, help="Payload feature .pt di uno split Video Swin.")
    parser.add_argument("--tracking-sidecar", required=True, help="Sidecar .pt tracking dello stesso split.")
    parser.add_argument("--output-dir", default="features/merged")
    parser.add_argument("--feature-groups", required=True, choices=GROUPS)
    parser.add_argument("--scaler-path", help="Scaler train da riutilizzare; se omesso viene derivato da output-dir.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path, scaler_path = merge_features(
        Path(args.swin_features).resolve(),
        Path(args.tracking_sidecar).resolve(),
        Path(args.output_dir).resolve(),
        args.feature_groups,
        Path(args.scaler_path).resolve() if args.scaler_path else None,
        args.overwrite,
    )
    print(f"Merged feature file: {output_path}")
    print(f"Train scaler: {scaler_path}")


if __name__ == "__main__":
    main()
