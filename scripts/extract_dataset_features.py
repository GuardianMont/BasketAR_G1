from __future__ import annotations

import argparse
from inspect import Parameter, signature
from pathlib import Path
from typing import Optional

import av
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.experiment_config import ExperimentConfig
from data.spatial import preprocess_frame
from data.utils import pad_and_collate_video_batch
from data.transforms import ClassAwareVideoTransform
from data.temporal_sampling import (
    LegacyUniformSampler,
    SAMPLER_REGISTRY,
    TemporalSampler,
    build_temporal_sampler,
)
from models import (
    BallTrackerBackbone,
    CompositeBackbone,
    X3DLBackbone,
)
from models.backbone import BackboneType, BackboneFactory
from scripts.extract_features import extract_features_offline

import torchvision.transforms.v2 as v2


LABEL_TO_ID = {
    "passaggio": 0,
    "tiroDaDue0": 1,
    "tiroDaDue1": 2,
    "tiroDaTre0": 3,
    "tiroDaTre1": 4,
    "tiroLibero0": 5,
    "tiroLibero1": 6,
    "idle": 7,
    "non-gioco": 8,
}

X3D_MEAN = (0.45, 0.45, 0.45)
X3D_STD = (0.225, 0.225, 0.225)

IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)


def validate_tracker_preprocessing(
    spatial_mode: str, color: str, feature_stream: str
) -> None:
    """Tracker preprocessing is supported only on RGB letterboxed frames."""
    if (
        feature_stream in {"3d_tracker", "all"}
        and (spatial_mode != "letterbox" or color != "rgb")
    ):
        raise ValueError(
            "Il tracker richiede preprocessing letterbox e RGB; per i confronti "
            "su spatial_mode/colore usa --feature-stream 3d."
        )


class ManifestVideoDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        num_frames=16,
        target_size=312,
        limit=None,
        transform: Optional[ClassAwareVideoTransform] = None,
        temporal_sampler: TemporalSampler | None = None,
        spatial_mode="letterbox",
        color_mode="rgb",
        normalization_vector=(X3D_MEAN, X3D_STD),
        oversample_factor=1
    ):
        self.root = Path(root)
        self.transform = transform
        self.num_frames = num_frames
        self.target_size = target_size
        self.temporal_sampler = temporal_sampler or LegacyUniformSampler(num_frames)
        
        sampler_name = self.temporal_sampler.config().get("name")
        has_temporal_shift = (
            isinstance(transform, ClassAwareVideoTransform)
            and transform.has_temporal_shift
        )

        if has_temporal_shift and sampler_name in {"first_last", "edge_dense"}:
            raise ValueError(
                "RandomTemporalShift cannot be combined with the "
                f"'{sampler_name}' sampler because it moves the selected endpoints."
            )

        df = pd.read_csv(self.root / "manifest.csv")
        df = df[df["split"] == split].reset_index(drop=True)

        if limit is not None:
            df = df.iloc[:limit].copy()

        if len(df) == 0:
            raise ValueError(f"No samples for split={split}")

        unknown = sorted(set(df["label"]) - set(LABEL_TO_ID))
        if unknown:
            raise ValueError(f"Unknown labels: {unknown}")

        raw_rows = df.to_dict("records")
        self.rows = []

        for row in raw_rows:
            label_id = LABEL_TO_ID[row["label"]]
            
            if (
                isinstance(transform, ClassAwareVideoTransform)
                and transform.should_apply(label_id)
                and oversample_factor > 1
            ):
                for aug_idx in range(oversample_factor):
                    augmented_row = row.copy()
                    # Unique id, avoids collisions
                    augmented_row["clip_id"] = f"{row['clip_id']}_aug{aug_idx}"
                    self.rows.append(augmented_row)
            else:
                self.rows.append(row)
        
        if spatial_mode not in {"letterbox", "stretch"}:
            raise ValueError(f"Unsupported spatial mode: {spatial_mode}")
        if color_mode not in {"rgb", "grayscale"}:
            raise ValueError(f"Unsupported color mode: {color_mode}")

        self.spatial_mode = spatial_mode
        self.color_mode = color_mode
        self.normalization_vector = normalization_vector

    def __len__(self):
        return len(self.rows)

    def _preprocess_frame(self, frame):
        return preprocess_frame(
            frame,
            spatial_mode=self.spatial_mode,
            spatial_size=self.target_size,
            color_mode=self.color_mode,
            mean_norm_vector=self.normalization_vector[0],
            std_norm_vector=self.normalization_vector[1],
        )

    def _decode_clip(self, path, label_id: int):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            total_frames = int(stream.frames or 0)
            fps = float(stream.average_rate) if stream.average_rate else None

        if total_frames <= 0:
            with av.open(str(path)) as container:
                total_frames = sum(1 for _ in container.decode(video=0))

        indices = self.temporal_sampler.sample_indices(total_frames, fps=fps)

        if (
            isinstance(self.transform, ClassAwareVideoTransform)
            and self.transform.has_temporal_shift
            and self.transform.should_apply(label_id)
        ):
            indices = self.transform.apply_temporal_shift(indices, total_frames)

        wanted = set(indices)
        decoded_norm = {}
        decoded_raw = {}

        with av.open(str(path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index not in wanted:
                    continue

                x_n, x_r = self._preprocess_frame(frame)
                decoded_norm[frame_index] = x_n
                decoded_raw[frame_index] = x_r

                if len(decoded_norm) == len(wanted):
                    break

        missing = sorted(wanted - set(decoded_norm))
        if missing:
            raise RuntimeError(f"Could not decode frames {missing} from {path}")

        frames_norm = [decoded_norm[i] for i in indices]
        frames_raw = [decoded_raw[i] for i in indices]
        clip_norm = torch.stack(frames_norm, dim=1)  # (C, T, H, W)
        clip_raw = torch.stack(frames_raw, dim=1)    # (C, T, H, W)

        frames_unnorm = [decoded_raw[i] for i in indices]  # Tensori [0, 1]
        clip_unnorm = torch.stack(frames_unnorm, dim=1)    # (C, T, H, W)
        clip_raw = clip_unnorm.clone()

        is_flipped = False

        if isinstance(self.transform, ClassAwareVideoTransform):
            clip_norm, clip_raw, is_flipped = self.transform.apply_spatial_transforms(
                clip_unnorm, clip_raw, label_id
            )
        else:
            # Fallback 
            normalizer = v2.Normalize(
                mean=self.normalization_vector[0], std=self.normalization_vector[1]
            )
            clip_t = clip_unnorm.permute(1, 0, 2, 3)
            clip_norm = normalizer(clip_t).permute(1, 0, 2, 3)

        return clip_norm, clip_raw, is_flipped

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / row["path"]
        label_id = LABEL_TO_ID[row["label"]]

        if not path.exists():
            raise FileNotFoundError(path)

        clip_norm, clip_raw, is_flipped = self._decode_clip(path, label_id)

        return {
            "x_norm": clip_norm,
            "x_raw": clip_raw,
            "label": label_id,
            "length": clip_norm.shape[1],
            "sample_id": row["clip_id"],
            "is_flipped": is_flipped,
        }


def build_backbone_and_filename(
    config_type: str,
    backbone_type: str = BackboneType.X3D_L.value,
    dino_type: Optional[str] = None,
    tracker_model_path: str = "yolov8n.pt",
) -> tuple[nn.Module, str]:
    backbone_list = []
    filename_parts = ["3d"]

    factory = BackboneFactory()
    if dino_type:
        backbone = factory.create(backbone_type, model_name=dino_type)
    else:
        backbone = factory.create(backbone_type)

    backbone_list.append(backbone)
    print(f"[DEBUG DIRECT] Created {backbone_type} with out_dim={backbone_list[-1].out_dim}")

    if "tracker" in config_type:
        backbone_list.append(BallTrackerBackbone(model_path=tracker_model_path))
        filename_parts.append("tracker")

    if isinstance(backbone, X3DLBackbone):
        filename_parts.append("L")

    output_filename = f"{'_'.join(filename_parts)}.pt"
    composite = CompositeBackbone(backbone_list)
    composite.freeze()

    print(f"[DEBUG COMPOSITE] Total composite.out_dim = {composite.out_dim}")
    return composite, output_filename


def extract_all_configurations(args):
    validate_tracker_preprocessing(
        args.spatial_mode, args.color, args.feature_stream
    )

    if args.backbone_type != BackboneType.DINO and args.dino_type:
        raise ValueError(
            "Can't set `--dino_type` argument when `--backbone_type` is not DINO"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.dino_type:
        target_size = BackboneFactory().create(args.backbone_type).out_dim
    else:
        target_size = BackboneFactory().create(
            args.backbone_type, model_name=args.dino_type
        ).out_dim

    print(
        f"\n{'='*70}\n[RUNNER] Inizio estrazione per SPLIT:"
        f" {args.split.upper()} | Model: {args.backbone_type} | Target Size:"
        f" {target_size}\n{'='*70}"
    )

    transform_list = [t for t in args.transforms if t != "none"]
    minority_set = set(args.minority_classes) if args.minority_classes else None

    if transform_list:
        transform = ClassAwareVideoTransform(
            transform_names=transform_list,
            minority_classes=minority_set,
            mean=IMAGE_NET_MEAN if args.backbone_type is BackboneType.DINO else X3D_MEAN,
            std=IMAGE_NET_STD if args.backbone_type is BackboneType.DINO else X3D_STD
        )
    else:
        transform = None

    sampler_class = SAMPLER_REGISTRY.get(args.temporal_sampler)
    if sampler_class is None:
        available = ", ".join(sorted(SAMPLER_REGISTRY))
        raise ValueError(
            f"Unknown temporal sampler '{args.temporal_sampler}'. Available: {available}."
        )
        
    sampler_parameters = signature(sampler_class.__init__).parameters
    sampler_config = {"name": args.temporal_sampler}
    sampler_config.update(
        {
            name: getattr(args, name)
            for name, parameter in sampler_parameters.items()
            if name != "self"
            and parameter.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
            and hasattr(args, name)
        }
    )
    temporal_sampler = build_temporal_sampler(sampler_config)

    if args.backbone_type == BackboneType.DINO:
        nv = (IMAGE_NET_MEAN, IMAGE_NET_STD)
    else:
        nv = (X3D_MEAN, X3D_STD)

    dataset = ManifestVideoDataset(
        root=args.dataset_root,
        split=args.split,
        num_frames=args.num_frames,
        target_size=target_size,
        limit=args.limit,
        transform=transform,
        temporal_sampler=temporal_sampler,
        spatial_mode=args.spatial_mode,
        color_mode=args.color,
        normalization_vector=nv,
        oversample_factor=args.oversample_factor
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        collate_fn=pad_and_collate_video_batch,
    )

    if args.feature_stream == "all":
        configs_to_run = ["3d", "3d_tracker"]
    elif args.feature_stream == "3d":
        configs_to_run = ["3d"]
    else:
        configs_to_run = ["3d_tracker"]

    for idx, cfg in enumerate(configs_to_run, 1):
        backbone, _ = build_backbone_and_filename(
            config_type=cfg,
            backbone_type=args.backbone_type,
            dino_type=args.dino_type,
            tracker_model_path=args.tracker_model_path,
        )

        experiment_config = ExperimentConfig(
            sampler=temporal_sampler.config(),
            spatial_mode=args.spatial_mode,
            spatial_size=target_size,
            color=args.color,
            backbone=args.backbone_type.lower().replace("-", "_"),
            dino_type=args.dino_type,
            tracker="yolov8n_coco" if "tracker" in cfg else "none",
            extra={
                "transforms": transform_list,
                "minority_classes": list(minority_set) if minority_set else None,
                **({"limit": args.limit} if args.limit is not None else {}),
            },
        )
        output_path = output_dir / experiment_config.feature_filename(args.split)

        print(
            f"\n[{idx}/{len(configs_to_run)}] Estrazione Configurazione:"
            f" '{cfg}' -> Output: '{output_path.name}'"
        )

        payload = extract_features_offline(
            backbone=backbone,
            dataloader=loader,
            output_path=str(output_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            overwrite=args.overwrite,
            metadata={
                "split": args.split,
                "experiment_config": experiment_config.to_dict(),
                "experiment_id": experiment_config.feature_id(),
            },
        )

        print(
            f"[DONE] File salvato: '{output_path.name}' | Campioni:"
            f" {payload['num_samples']} | Dimensioni D: {payload['feature_dim']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Offline Feature Extraction Suite per Basket Action Recognition."
    )

    parser.add_argument("--dataset-root", default="dataset_hd")
    parser.add_argument(
        "--split", choices=["train", "val", "test"], required=True
    )
    parser.add_argument(
        "--output-dir",
        default="features",
        help="Where the .pt file should be saved",
    )
    parser.add_argument(
        "--backbone-type",
        choices=BackboneType.list_names(),
        default=BackboneType.X3D_L.value,
    )
    parser.add_argument("--dino_type", type=str, default=None)
    parser.add_argument(
        "--tracker-model-path",
        default="yolov8n.pt",
    )
    parser.add_argument(
        "--transforms",
        "--transform",
        nargs="+",
        choices=[
            "none",
            "horizontal_flip",
            "temporal_shift",
            "color_jitter",
            "gaussian_blur",
            "random_crop",
        ],
        default=["none"],
        help="transform list to be applied",
    )
    parser.add_argument(
        "--minority-classes",
        nargs="+",
        type=int,
        default=None,
        help="Class'IDs where the transforms should be applied",
    )

    parser.add_argument(
        "--oversample-factor",
        type=int,
        default=1,
        help="Number of samples to be produced for every `minority class` clip",
    )

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--temporal-sampler",
        choices=[
            "legacy_uniform",
            "uniform",
            "preserve_all",
            "fixed_fps",
            "first_last",
            "edge_dense",
        ],
        default="legacy_uniform",
    )
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--frames-per-side", type=int, default=8)
    parser.add_argument("--edge-strength", type=float, default=1.0)
    parser.add_argument(
        "--spatial-mode", choices=["letterbox", "stretch"], default="letterbox"
    )
    parser.add_argument("--color", choices=["rgb", "grayscale"], default="rgb")
    parser.add_argument(
        "--feature-stream",
        choices=["3d", "3d_tracker", "all"],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    extract_all_configurations(args)


if __name__ == "__main__":
    main()