from __future__ import annotations

import argparse
from pathlib import Path

import av
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset

from data import RandomTemporalShift, VideoHorizontalFlip
from data.utils import pad_and_collate_video_batch
from models import (
    BallTrackerBackbone,
    CompositeBackbone,
    ConvNeXt2DBackbone,
    X3DLBackbone,
    X3DSBackbone,
)
from scripts.extract_features import extract_features_offline

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


class ManifestVideoDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        num_frames=16,
        target_size=312,
        limit=None,
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform

        df = pd.read_csv(self.root / "manifest.csv")
        df = df[df["split"] == split].reset_index(drop=True)

        if limit is not None:
            df = df.iloc[:limit].copy()

        if len(df) == 0:
            raise ValueError(f"No samples for split={split}")

        unknown = sorted(set(df["label"]) - set(LABEL_TO_ID))
        if unknown:
            raise ValueError(f"Unknown labels: {unknown}")

        self.rows = df.to_dict("records")
        self.num_frames = num_frames
        self.target_size = target_size

    def __len__(self):
        return len(self.rows)

    def _uniform_indices(self, total_frames):
        if total_frames <= 0:
            raise ValueError("Video has no frames")

        if total_frames >= self.num_frames:
            stride = total_frames / float(self.num_frames)
            return [
                min(int(i * stride), total_frames - 1)
                for i in range(self.num_frames)
            ]

        indices = list(range(total_frames))
        indices.extend([total_frames - 1] * (self.num_frames - total_frames))

        if isinstance(self.transform, RandomTemporalShift):
            indices = self.transform(indices, total_frames)

        return indices

    def _preprocess_frame(self, frame):
        array = frame.to_ndarray(format="rgb24")
        x = (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            / 255.0
        )

        _, h, w = x.shape
        if h > w:
            new_h, new_w = self.target_size, int(w * (self.target_size / h))
        else:
            new_h, new_w = int(h * (self.target_size / w)), self.target_size

        x = F.resize(x, [new_h, new_w], antialias=True)

        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        padding = [
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        ]
        x_raw = F.pad(x, padding, fill=0)

        x_norm = F.normalize(x_raw, mean=X3D_MEAN, std=X3D_STD)
        return x_norm, x_raw

    def _decode_clip(self, path):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            total_frames = int(stream.frames or 0)

        if total_frames <= 0:
            with av.open(str(path)) as container:
                total_frames = sum(1 for _ in container.decode(video=0))

        indices = self._uniform_indices(total_frames)
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
            raise RuntimeError(
                f"Could not decode frames {missing} from {path}"
            )

        frames_norm = [decoded_norm[i] for i in indices]
        frames_raw = [decoded_raw[i] for i in indices]
        clip_norm = torch.stack(frames_norm, dim=1)
        clip_raw = torch.stack(frames_raw, dim=1)

        is_flipped = False
        if isinstance(self.transform, VideoHorizontalFlip):
            clip_norm = self.transform(clip_norm)
            clip_raw = self.transform(clip_raw)
            is_flipped = True

        return clip_norm, clip_raw, is_flipped

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / row["path"]

        if not path.exists():
            raise FileNotFoundError(path)

        clip_norm, clip_raw, is_flipped = self._decode_clip(path)

        return {
            "x_norm": clip_norm,
            "x_raw": clip_raw,
            "label": LABEL_TO_ID[row["label"]],
            "length": clip_norm.shape[1],
            "sample_id": row["clip_id"],
            "is_flipped": is_flipped,
        }


def build_backbone_and_filename(
    config_type: str,
    backbone_type: str = "X3D-L",
    tracker_model_path: str = "yolov8n.pt",
) -> tuple[nn.Module, str]:
    """Costruisce la CompositeBackbone dagli import ufficiali di models e

    definisce il nome del file.
    """
    backbone_list = []
    filename_parts = ["3d"]

    is_large = backbone_type.upper() == "X3D-L"
    if is_large:
        # X3D-L nativa estrae D=432
        backbone_list.append(
            X3DLBackbone(out_dim=432, pretrained=True, freeze_x3d=True)
        )
        print(f"[DEBUG DIRECT] Created X3DLBackbone with out_dim={backbone_list[-1].out_dim}")
    else:
        # X3D-S nativa estrae D=192
        backbone_list.append(
            X3DSBackbone(out_dim=192, pretrained=True, freeze_x3d=True)
        )
        print(f"[DEBUG DIRECT] Created X3DSBackbone with out_dim={backbone_list[-1].out_dim}")

    if "2d" in config_type:
        backbone_list.append(ConvNeXt2DBackbone(pretrained=True))
        filename_parts.append("2d")

    if "tracker" in config_type:
        backbone_list.append(
            BallTrackerBackbone(model_path=tracker_model_path)
        )
        filename_parts.append("tracker")

    if is_large:
        filename_parts.append("L")

    output_filename = f"{'_'.join(filename_parts)}.pt"
    composite = CompositeBackbone(backbone_list)
    composite.freeze()

    print(f"[DEBUG COMPOSITE] Total composite.out_dim = {composite.out_dim}")
    return composite, output_filename


def extract_all_configurations(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_size = 312 if args.backbone_type.upper() == "X3D-L" else 192

    print(
        f"\n{'='*70}\n[RUNNER] Inizio estrazione per SPLIT:"
        f" {args.split.upper()} | Model: {args.backbone_type} | Target Size:"
        f" {target_size}\n{'='*70}"
    )

    if args.transform == "horizontal_flip":
        transform = VideoHorizontalFlip()
    elif args.transform == "temporal_shift":
        transform = RandomTemporalShift(shift_range=3)
    else:
        transform = None

    dataset = ManifestVideoDataset(
        root=args.dataset_root,
        split=args.split,
        num_frames=args.num_frames,
        target_size=target_size,
        limit=args.limit,
        transform=transform,
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

    configs_to_run = [
        "3d",
        "3d_tracker",
    ]

    for idx, cfg in enumerate(configs_to_run, 1):
        backbone, filename = build_backbone_and_filename(
            config_type=cfg,
            backbone_type=args.backbone_type,
            tracker_model_path=args.tracker_model_path,
        )

        output_path = output_dir / f"{args.split}_{filename}"

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
        )

        print(
            f"[DONE] File salvato: '{output_path.name}' | Campioni:"
            f" {payload['num_samples']} | Dimensioni D: {payload['feature_dim']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Offline Feature Extraction Suite per Basket Action Recognition."
    )

    parser.add_argument("--dataset-root", default="dataset/dataset_hd")
    parser.add_argument(
        "--split", choices=["train", "val", "test"], required=True
    )
    parser.add_argument(
        "--output-dir",
        default="features",
        help="Directory di destinazione per salvataggio file .pt",
    )
    parser.add_argument(
        "--backbone-type",
        choices=["X3D-S", "X3D-L"],
        default="X3D-L",
    )
    parser.add_argument(
        "--tracker-model-path",
        default="yolov8n.pt",
    )
    parser.add_argument(
        "--transform",
        choices=["none", "horizontal_flip", "temporal_shift"],
        default="none",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    extract_all_configurations(args)


if __name__ == "__main__":
    main()