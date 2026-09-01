import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import av
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Project imports
from models import (
    ARPipeline,
    BallTrackerBackbone,
    CompositeBackbone,
    X3DLBackbone,
)
from models.heads import AttentionClassifierHead, MeanPoolingClassifierHead
from models.necks import BiGRUNeck, TCNNeck,IdentityNeck
from utils.plotting import save_confusion_matrix_plot

# =========================================================================
# 1. CONFIGURAZIONE HARDCODED DEI MIGLIORI CHECKPOINT E ARCHITETTURE
# =========================================================================
BEST_CHECKPOINTS = {
    "p1": "experiments/ORCHESTRATOR_RUN_20260901_121348/ARCH-02_BiGRU_Mean/p1/checkpoints/ARCH-02_BiGRU_Mean_run1_p1_best.pt",
    "p2": "experiments/ORCHESTRATOR_RUN_20260901_121348/ARCH-04_Hybrid_Attention/p2/checkpoints/ARCH-04_Hybrid_Attention_run1_p2_best.pt",
    "p3": "experiments/ORCHESTRATOR_RUN_20260901_121348/ARCH-02_BiGRU_Mean/p3/checkpoints/ARCH-02_BiGRU_Mean_run4_p3_best.pt",
}

LABEL_TO_ID = {
    "passaggio": 0,
    "tiroDaDue0": 1,
    "tiroDaDue1": 2,
    "tiroDaTre0": 3,
    "tiroDaTre1": 4,
    "tiroLibero0": 5,
    "tiroLibero1": 6,
    "idle": 7,
    "non-gioco": 7,
    "altro": 7,
}

CLASS_NAMES_AGGREGATED = [
    "passaggio",
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
    "altro",
]

X3D_MEAN = (0.45, 0.45, 0.45)
X3D_STD = (0.225, 0.225, 0.225)


class TestFolderDataset(Dataset):
    """Dataset per caricare i file video .mp4/.avi organizzati per sottocartelle."""

    def __init__(self, test_root: str, num_frames: int = 16, crop_size: int = 312):
        self.test_root = Path(test_root)
        self.num_frames = num_frames
        self.crop_size = crop_size

        self.samples = []
        for class_dir in sorted(self.test_root.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in LABEL_TO_ID:
                    continue
                label_id = LABEL_TO_ID[class_name]
                for video_path in list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi")):
                    self.samples.append({
                        "path": video_path,
                        "label": label_id,
                        "clip_id": f"{class_name}/{video_path.name}",
                    })

        print(f"[INFO] Trovati {len(self.samples)} video di test in '{test_root}'.")

    def __len__(self):
        return len(self.samples)

    def _uniform_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            raise ValueError("Il video non contiene frame validi.")

        if total_frames >= self.num_frames:
            stride = total_frames / float(self.num_frames)
            return [min(int(i * stride), total_frames - 1) for i in range(self.num_frames)]

        indices = list(range(total_frames))
        indices.extend([total_frames - 1] * (self.num_frames - total_frames))
        return indices

    def _preprocess_frame(self, frame) -> Tuple[torch.Tensor, torch.Tensor]:
        array = frame.to_ndarray(format="rgb24")
        x_raw = torch.from_numpy(array).permute(2, 0, 1).contiguous().float() / 255.0

        _, h, w = x_raw.shape
        if h > w:
            new_h, new_w = self.crop_size, int(w * (self.crop_size / h))
        else:
            new_h, new_w = int(h * (self.crop_size / w)), self.crop_size

        x_raw = TF.resize(x_raw, [new_h, new_w], antialias=True)

        pad_h, pad_w = self.crop_size - new_h, self.crop_size - new_w
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
        x_raw = TF.pad(x_raw, padding, fill=0)

        x_norm = TF.normalize(x_raw, mean=X3D_MEAN, std=X3D_STD)
        return x_norm, x_raw

    def _decode_clip(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        with av.open(str(path)) as container:
            total_frames = sum(1 for _ in container.decode(video=0))

        indices = self._uniform_indices(total_frames)
        wanted = set(indices)
        decoded_norm, decoded_raw = {}, {}

        with av.open(str(path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index not in wanted:
                    continue
                norm_f, raw_f = self._preprocess_frame(frame)
                decoded_norm[frame_index] = norm_f
                decoded_raw[frame_index] = raw_f
                if len(decoded_norm) == len(wanted):
                    break

        frames_norm = [decoded_norm[i] for i in indices]
        frames_raw = [decoded_raw[i] for i in indices]
        return torch.stack(frames_norm, dim=1), torch.stack(frames_raw, dim=1)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        clip_norm, clip_raw = self._decode_clip(sample["path"])
        return {
            "x_norm": clip_norm,
            "x_raw": clip_raw,
            "label": sample["label"],
            "length": clip_norm.shape[1],
            "sample_id": sample["clip_id"],
        }


def build_model_from_checkpoint(
    ckpt_path: str, device: torch.device, default_in_dim: int = 437
) -> nn.Module:
    """Ispeziona lo state_dict del checkpoint e ricostruisce automaticamente

    l'architettura esatta (Neck + Head) per evitare qualsiasi mismatch di forma.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: '{ckpt_path}'")

    raw_data = torch.load(ckpt_path, map_location=device)
    state_dict = (
        raw_data["model_state_dict"]
        if isinstance(raw_data, dict) and "model_state_dict" in raw_data
        else raw_data
    )

    # --- 1. RICOSTRUZIONE AUTOMATICA DEL NECK ---
    has_gru = any("neck.gru" in k for k in state_dict.keys())
    has_tcn = any("neck.network" in k for k in state_dict.keys())

    if has_gru:
        # Dimensione nascosta GRU (3 * hidden_dim = shape[0])
        hidden_dim = state_dict["neck.gru.weight_ih_l0"].shape[0] // 3
        # Conteggio dei layer
        layer_indices = {
            int(k.split("_l")[1].split("_")[0])
            for k in state_dict.keys()
            if "neck.gru.weight_ih_l" in k
        }
        num_layers = len(layer_indices)
        neck = BiGRUNeck(
            in_dim=default_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
        )

    elif has_tcn:
        # Kernel size TCN
        kernel_size = state_dict["neck.network.0.conv1.weight_v"].shape[2]
        # Canali di ogni blocco TCN
        block_indices = {
            int(k.split(".")[2])
            for k in state_dict.keys()
            if k.startswith("neck.network.")
        }
        num_blocks = len(block_indices)
        num_channels = [
            state_dict[f"neck.network.{i}.conv2.weight_v"].shape[0]
            for i in range(num_blocks)
        ]
        neck = TCNNeck(
            in_dim=default_in_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=0.0,
        )

    else:
        neck = IdentityNeck()

    # --- 2. RICOSTRUZIONE AUTOMATICA DELLA HEAD ---
    fc_weight = state_dict["head.fc.weight"]
    num_classes, head_in_dim = fc_weight.shape[0], fc_weight.shape[1]

    # Riconosce se la head usa Attention
    is_attention = any(
        "attn" in k or "attention" in k
        for k in state_dict.keys()
        if k.startswith("head.")
    )

    if is_attention:
        head = AttentionClassifierHead(
            in_dim=head_in_dim, num_classes=num_classes, dropout=0.0
        )
    else:
        head = MeanPoolingClassifierHead(
            in_dim=head_in_dim, num_classes=num_classes, dropout=0.0
        )

    # Costruzione e caricamento del modello
    model = ARPipeline(backbone=None, neck=neck, head=head).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_specialized_models(
    device: torch.device, check_points: Dict[str, str]
) -> Dict[str, nn.Module]:
    """Inizializza ed auto-configura i 3 modelli dai checkpoint specificati."""
    models = {}
    for phase, ckpt_path in check_points.items():
        models[phase] = build_model_from_checkpoint(ckpt_path, device)
        print(
            f"[MODEL] Fase {phase.upper()} caricata e auto-allineata da:"
            f" '{ckpt_path}'"
        )
    return models


def map_ground_truth_to_hierarchical_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p1_gt = np.where(labels == 0, 0, np.where((labels >= 1) & (labels <= 6), 1, 2))

    p2_gt = np.full_like(labels, -1)
    p2_gt[np.isin(labels, [1, 2])] = 0
    p2_gt[np.isin(labels, [3, 4])] = 1
    p2_gt[np.isin(labels, [5, 6])] = 2

    p3_gt = np.full_like(labels, -1)
    p3_gt[np.isin(labels, [1, 3, 5])] = 0
    p3_gt[np.isin(labels, [2, 4, 6])] = 1

    return p1_gt, p2_gt, p3_gt


def reconstruct_full_class_id(p1_pred: int, p2_pred: int, p3_pred: int) -> int:
    if p1_pred == 0:
        return 0
    elif p1_pred == 2:
        return 7

    if p2_pred == 0:
        return 1 if p3_pred == 0 else 2
    elif p2_pred == 1:
        return 3 if p3_pred == 0 else 4
    elif p2_pred == 2:
        return 5 if p3_pred == 0 else 6

    return 7


def collate_dual_clip_batch(batch: List[Dict]) -> Dict:
    return {
        "x_norm": torch.stack([item["x_norm"] for item in batch], dim=0),
        "x_raw": torch.stack([item["x_raw"] for item in batch], dim=0),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "lengths": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "sample_ids": [item["sample_id"] for item in batch],
    }


def main():
    parser = argparse.ArgumentParser(description="Valutazione End-to-End della pipeline a cascata.")
    parser.add_argument("--test-dir", default="dataset/dataset_hd/test", help="Path directory del test set.")
    parser.add_argument("--output-dir", default="experiments/EVALUATION_CASCADE", help="Cartella dei risultati.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device in uso: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Istanziazione Modelli con pesi hardcodati
    models = build_specialized_models(device, BEST_CHECKPOINTS)

    # 2. Backbone Composite Online (X3D-L + BallTracker)
    composite_backbone = CompositeBackbone([
        X3DLBackbone(out_dim=432, pretrained=True, freeze_x3d=True),
        BallTrackerBackbone(model_path="yolov8n.pt"),
    ]).to(device)
    composite_backbone.eval()
    composite_backbone.freeze()

    # 3. Dataloader Test
    test_dataset = TestFolderDataset(test_root=args.test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_dual_clip_batch,
    )

    all_raw_labels = []
    p1_preds, p2_preds, p3_preds = [], [], []
    final_cascade_preds = []

    print(f"\n{'='*70}\n   INIZIO VALUTAZIONE A CASCATA SUL TEST SET\n{'='*70}")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[VALUTAZIONE CASCATA]"):
            clips_norm = batch["x_norm"].to(device)
            clips_raw = batch["x_raw"].to(device)
            labels = batch["labels"].cpu().numpy()
            all_raw_labels.extend(labels)

            feat_3d = composite_backbone.backbones[0](clips_norm)
            feat_tracker = composite_backbone.backbones[1](clips_raw)
            features = torch.cat([feat_3d, feat_tracker], dim=-1)
            lengths = torch.tensor([features.shape[1]] * features.shape[0], device=device)

            logits_p1 = models["p1"](features, lengths=lengths)
            pred_p1 = torch.argmax(logits_p1, dim=1).cpu().numpy()
            p1_preds.extend(pred_p1)

            for i in range(len(labels)):
                curr_p1 = pred_p1[i]
                if curr_p1 == 1:
                    feat_single = features[i : i + 1]
                    len_single = lengths[i : i + 1]

                    logits_p2 = models["p2"](feat_single, lengths=len_single)
                    curr_p2 = torch.argmax(logits_p2, dim=1).item()

                    logits_p3 = models["p3"](feat_single, lengths=len_single)
                    curr_p3 = torch.argmax(logits_p3, dim=1).item()
                else:
                    curr_p2 = -1
                    curr_p3 = -1

                p2_preds.append(curr_p2)
                p3_preds.append(curr_p3)
                final_cascade_preds.append(reconstruct_full_class_id(curr_p1, curr_p2, curr_p3))

    # --- CALCOLO METRICHE ---
    all_raw_labels = np.array(all_raw_labels)
    final_cascade_preds = np.array(final_cascade_preds)
    p1_gt, p2_gt, p3_gt = map_ground_truth_to_hierarchical_labels(all_raw_labels)

    acc_p1 = accuracy_score(p1_gt, p1_preds)
    f1_p1 = f1_score(p1_gt, p1_preds, average="macro", zero_division=0)

    p2_preds_arr = np.array(p2_preds)
    valid_p2_mask = (p2_gt != -1) & (p2_preds_arr != -1)
    acc_p2 = accuracy_score(p2_gt[valid_p2_mask], p2_preds_arr[valid_p2_mask]) if np.any(valid_p2_mask) else 0.0
    f1_p2 = f1_score(p2_gt[valid_p2_mask], p2_preds_arr[valid_p2_mask], average="macro", zero_division=0) if np.any(valid_p2_mask) else 0.0

    p3_preds_arr = np.array(p3_preds)
    valid_p3_mask = (p3_gt != -1) & (p3_preds_arr != -1)
    acc_p3 = accuracy_score(p3_gt[valid_p3_mask], p3_preds_arr[valid_p3_mask]) if np.any(valid_p3_mask) else 0.0
    f1_p3 = f1_score(p3_gt[valid_p3_mask], p3_preds_arr[valid_p3_mask], average="macro", zero_division=0) if np.any(valid_p3_mask) else 0.0

    global_acc = accuracy_score(all_raw_labels, final_cascade_preds)
    global_f1 = f1_score(all_raw_labels, final_cascade_preds, average="macro", zero_division=0)

    print(f"\n{'='*70}\n   RISULTATI DELLA VALUTAZIONE A CASCATA (TEST SET)\n{'='*70}")
    print(f"  - Fase p1 (Macro-Azione): Acc = {acc_p1:.4f} | Macro-F1 = {f1_p1:.4f}")
    print(f"  - Fase p2 (Tipo Tiro):   Acc = {acc_p2:.4f} | Macro-F1 = {f1_p2:.4f}  (su {np.sum(valid_p2_mask)} campioni)")
    print(f"  - Fase p3 (Esito Tiro):  Acc = {acc_p3:.4f} | Macro-F1 = {f1_p3:.4f}  (su {np.sum(valid_p3_mask)} campioni)")
    print(f"  -------------------------------------------------------------")
    print(f"  [GLOBAL END-TO-END] Accuracy  = {global_acc:.4f}")
    print(f"  [GLOBAL END-TO-END] Macro-F1  = {global_f1:.4f}")
    print(f"{'='*70}\n")

    report_path = os.path.join(args.output_dir, "cascade_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("CASCADE PIPELINE END-TO-END REPORT\n")
        f.write(f"Global Accuracy: {global_acc:.4f}\nGlobal Macro-F1: {global_f1:.4f}\n\n")
        f.write(classification_report(all_raw_labels, final_cascade_preds, target_names=CLASS_NAMES_AGGREGATED, zero_division=0))

    cm_plot_path = os.path.join(args.output_dir, "global_cascade_confusion_matrix.png")
    save_confusion_matrix_plot(all_raw_labels, final_cascade_preds, cm_plot_path, class_names=CLASS_NAMES_AGGREGATED)

    print(f"[OUTPUT] Report salvato in: '{report_path}'")
    print(f"[OUTPUT] Matrice di confusione salvata in: '{cm_plot_path}'")


if __name__ == "__main__":
    main()