from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import av
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from data.utils import pad_and_collate_video_batch
from models import X3DSBackbone
from scripts.extract_features import extract_features_offline

from models import X3DSBackbone, ConvNeXt2DBackbone, CompositeBackbone, BallTrackerBackbone
from data import RandomTemporalShift, VideoHorizontalFlip


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
        target_size=192,
        limit=None,
        transform=None
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
        indices.extend(
            [total_frames - 1] *
            (self.num_frames - total_frames)
        )

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
        )  # (C, H, W)

        # Shrink the longest side to target and the shortest accordingly (mantaining the aspect ratio)
        _, h, w = x.shape
        if h > w:
            new_h, new_w = self.target_size, int(w * (self.target_size / h))
        else:
            new_h, new_w = int(h * (self.target_size / w)), self.target_size

        x = F.resize(x, [new_h, new_w], antialias=True)

        # Apply paddding to have target x target resolution
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        padding = [
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        ]  # [left, top, right, bottom]
        x = F.pad(x, padding, fill=0)

        x = F.normalize(x, mean=X3D_MEAN, std=X3D_STD)

        return x

    def _decode_clip(self, path):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            total_frames = int(stream.frames or 0)

        if total_frames <= 0:
            with av.open(str(path)) as container:
                total_frames = sum(
                    1 for _ in container.decode(video=0)
                )

        indices = self._uniform_indices(total_frames)
        wanted = set(indices)
        decoded = {}

        with av.open(str(path)) as container:
            for frame_index, frame in enumerate(
                container.decode(video=0)
            ):
                if frame_index not in wanted:
                    continue

                decoded[frame_index] = (
                    self._preprocess_frame(frame)
                )

                if len(decoded) == len(wanted):
                    break

        missing = sorted(wanted - set(decoded))

        if missing:
            raise RuntimeError(
                f"Could not decode frames {missing} "
                f"from {path}"
            )

        frames = [decoded[i] for i in indices]
        clip = torch.stack(frames, dim=1)  # (C, T, H, W)

        # Se il transform e' VideoHorizontalFlip, lo applichiamo alla clip
        is_flipped = False
        if isinstance(self.transform, VideoHorizontalFlip):
            clip = self.transform(clip)
            is_flipped = True

        return clip, is_flipped

    def __getitem__(self, index):
        row = self.rows[index]

        path = self.root / row["path"]

        if not path.exists():
            raise FileNotFoundError(path)

        clip, is_flipped = self._decode_clip(path)

        return {
            "data": clip,
            "label": LABEL_TO_ID[row["label"]],
            "length": clip.shape[1],
            "sample_id": row["clip_id"],
            "is_flipped": is_flipped
        }


def split_and_save_hybrid_payload(hybrid_payload: Dict[str, Any], base_output_path: Path):
    """
    Prende il payload estratto dall'HybridBackbone (D=960) e genera due file separati:
    1. <base_output_path>_3d.pt (D=192)
    2. <base_output_path>_2d.pt (D=768)
    """
    samples_3d = []
    samples_2d = []

    for s in hybrid_payload["samples"]:
        data_hybrid = s["data"]  # (T, 960)
        
        data_3d = data_hybrid[:, :192].contiguous()
        data_2d = data_hybrid[:, 192:].contiguous()

        samples_3d.append({
            "id": s["id"],
            "data": data_3d,
            "label": s["label"],
            "num_frames": s["num_frames"]
        })
        
        samples_2d.append({
            "id": s["id"],
            "data": data_2d,
            "label": s["label"],
            "num_frames": s["num_frames"]
        })

    payload_3d = {
        "format_version": 1,
        "backbone": "X3DSBackbone",
        "feature_dim": 192,
        "num_samples": len(samples_3d),
        "samples": samples_3d
    }

    payload_2d = {
        "format_version": 1,
        "backbone": "ConvNeXt2DBackbone",
        "feature_dim": 768,
        "num_samples": len(samples_2d),
        "samples": samples_2d
    }

    path_3d = base_output_path.parent / f"{base_output_path.stem}_3d.pt"
    path_2d = base_output_path.parent / f"{base_output_path.stem}_2d.pt"

    torch.save(payload_3d, path_3d)
    torch.save(payload_2d, path_2d)

    print(f"[SPLIT] Separate 3D features saved to: '{path_3d}' (D=192)")
    print(f"[SPLIT] Separate 2D features saved to: '{path_2d}' (D=768)")


def main():
    parser = argparse.ArgumentParser(description="Extract offline features using 3D, 2D, or Hybrid Backbones.")

    parser.add_argument(
        "--dataset-root",
        default="dataset/dataset_hd",
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        required=True,
    )

    parser.add_argument(
        "--transform",
        choices=["none", "horizontal_flip", "temporal_shift"],
        default="none",
        help="Tipo di augmentation deterministica da applicare durante l'estrazione.",
    )

    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--target-size", type=int, default=192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

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
        target_size=args.target_size,
        limit=args.limit,
        transform=transform
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

    print(f"[INFO] split: {args.split}")
    print(f"[INFO] samples: {len(dataset)}")
    print(f"[INFO] CUDA: {torch.cuda.is_available()}")
    print(f"[INFO] output target: {args.output}")
    print(f"[INFO] augmentation: {args.transform}")

    backbone_list = [
        X3DSBackbone(out_dim=192, pretrained=True, freeze_x3d=True),  # 3D (D=192)
        # ConvNeXt2DBackbone(pretrained=True),                         # 2D (D=768)
        BallTrackerBackbone(model_path="yolov8n.pt")                 # Tracking (D=5)
    ]
    # Totale D = 192 + 768 + 5 = 965

    backbone = CompositeBackbone(backbone_list)
    backbone.freeze()

    # Estrazione principale
    payload = extract_features_offline(
        backbone=backbone,
        dataloader=loader,
        output_path=args.output,
        device="cuda" if torch.cuda.is_available() else "cpu",
        overwrite=args.overwrite,
    )

    print(
        f"[DONE] Main payload saved to '{args.output}': "
        f"{payload['num_samples']} samples, "
        f"D={payload['feature_dim']}"
    )



if __name__ == "__main__":
    main()