import argparse
import json
import os
from inspect import Parameter, signature
from pathlib import Path
from typing import Dict, List, Tuple

import av
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Project imports
from data.spatial import preprocess_frame
from data.temporal_sampling import (
    SAMPLER_REGISTRY,
    TemporalSampler,
    build_temporal_sampler,
)
from models import ARPipeline
from models.heads import AttentionClassifierHead, MeanPoolingClassifierHead
from models.necks import BiGRUNeck, TCNNeck, IdentityNeck
from scripts.extract_dataset_features import build_backbone_and_filename
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

class TestFolderDataset(Dataset):
    """Dataset per caricare i file video .mp4/.avi organizzati per sottocartelle."""

    def __init__(
        self,
        test_root: str,
        temporal_sampler: TemporalSampler,
        spatial_mode: str = "letterbox",
        spatial_size: int = 312,
        color_mode: str = "rgb",
    ):
        self.test_root = Path(test_root)
        self.temporal_sampler = temporal_sampler
        self.spatial_mode = spatial_mode
        self.spatial_size = spatial_size
        self.color_mode = color_mode

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

    def _preprocess_frame(self, frame) -> Tuple[torch.Tensor, torch.Tensor]:
        return preprocess_frame(
            frame,
            spatial_mode=self.spatial_mode,
            spatial_size=self.spatial_size,
            color_mode=self.color_mode,
        )

    def _decode_clip(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            total_frames = int(stream.frames or 0)
            fps = float(stream.average_rate) if stream.average_rate else None

        if total_frames <= 0:
            with av.open(str(path)) as container:
                total_frames = sum(1 for _ in container.decode(video=0))

        indices = self.temporal_sampler.sample_indices(total_frames, fps=fps)
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

        missing = sorted(wanted - set(decoded_norm))
        if missing:
            raise RuntimeError(f"Could not decode frames {missing} from {path}")

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
    ckpt_path: str,
    device: torch.device,
    in_dim: int,
    expected_experiment_id: str | None = None,
) -> nn.Module:
    """Ispeziona lo state_dict del checkpoint e ricostruisce automaticamente

    l'architettura esatta (Neck + Head) per evitare qualsiasi mismatch di forma.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: '{ckpt_path}'")

    raw_data = torch.load(ckpt_path, map_location=device)
    if expected_experiment_id is not None:
        checkpoint_experiment_id = (
            raw_data.get("experiment_id") if isinstance(raw_data, dict) else None
        )
        if checkpoint_experiment_id is None:
            print(
                f"[WARNING] Checkpoint '{ckpt_path}' senza experiment_id: "
                "compatibilita' non verificabile."
            )
        elif checkpoint_experiment_id != expected_experiment_id:
            raise ValueError(
                f"Experiment ID incompatibile nel checkpoint '{ckpt_path}': "
                f"{checkpoint_experiment_id!r} != {expected_experiment_id!r}."
            )

    state_dict = (
        raw_data["model_state_dict"]
        if isinstance(raw_data, dict) and "model_state_dict" in raw_data
        else raw_data
    )

    # --- 1. RICOSTRUZIONE AUTOMATICA DEL NECK ---
    has_gru = any("neck.gru" in k for k in state_dict.keys())
    has_tcn = any("neck.network" in k for k in state_dict.keys())

    head_weight = state_dict["head.fc.weight"]
    if has_gru:
        checkpoint_in_dim = state_dict["neck.gru.weight_ih_l0"].shape[1]
    elif has_tcn:
        checkpoint_in_dim = state_dict["neck.network.0.conv1.weight_v"].shape[1]
    else:
        checkpoint_in_dim = head_weight.shape[1]
    if checkpoint_in_dim != in_dim:
        raise ValueError(
            "Input feature dimension mismatch: checkpoint expects "
            f"D={checkpoint_in_dim}, but the configured backbone produces D={in_dim}. "
            "Check --feature-stream / --feature-reference."
        )

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
            in_dim=in_dim,
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
            in_dim=in_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=0.0,
        )

    else:
        neck = IdentityNeck(in_dim=in_dim)

    # --- 2. RICOSTRUZIONE AUTOMATICA DELLA HEAD ---
    fc_weight = head_weight
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
    device: torch.device,
    check_points: Dict[str, str],
    in_dim: int,
    expected_experiment_id: str | None = None,
) -> Dict[str, nn.Module]:
    """Inizializza ed auto-configura i 3 modelli dai checkpoint specificati."""
    models = {}
    for phase, ckpt_path in check_points.items():
        models[phase] = build_model_from_checkpoint(
            ckpt_path,
            device,
            in_dim=in_dim,
            expected_experiment_id=expected_experiment_id,
        )
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
    x_norm = [item["x_norm"] for item in batch]
    x_raw = [item["x_raw"] for item in batch]
    lengths = [int(clip.shape[1]) for clip in x_norm]
    if any(int(item["length"]) != length for item, length in zip(batch, lengths)):
        raise ValueError("Sample length does not match its decoded clip tensor.")
    if any(raw.shape[1] != length for raw, length in zip(x_raw, lengths)):
        raise ValueError("Normalized and raw clip lengths do not match.")
    return {
        "x_norm": x_norm,
        "x_raw": x_raw,
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "sample_ids": [item["sample_id"] for item in batch],
    }


def compute_features_by_length(x_norm_list, x_raw_list, backbone_fn):
    """Run the video backbone on equal-length groups, then pad only features."""
    if len(x_norm_list) != len(x_raw_list) or not x_norm_list:
        raise ValueError("Expected equally sized, non-empty normalized and raw clip lists.")

    groups = {}
    lengths = []
    for index, (x_norm, x_raw) in enumerate(zip(x_norm_list, x_raw_list)):
        if x_norm.ndim != 4 or x_raw.ndim != 4:
            raise ValueError("Each clip must have shape (C, T, H, W).")
        length = int(x_norm.shape[1])
        if length <= 0 or int(x_raw.shape[1]) != length:
            raise ValueError("Normalized and raw clips must have the same positive T.")
        lengths.append(length)
        groups.setdefault(length, []).append(index)

    features_by_sample = [None] * len(x_norm_list)
    for length, indices in groups.items():
        group_norm = torch.stack([x_norm_list[index] for index in indices], dim=0)
        group_raw = torch.stack([x_raw_list[index] for index in indices], dim=0)
        group_features = backbone_fn(group_norm, group_raw)
        if group_features.ndim != 3 or group_features.shape[0] != len(indices):
            raise ValueError(
                "backbone_fn must return features with shape (group_size, T, D)."
            )
        if group_features.shape[1] != length:
            raise ValueError(
                f"backbone_fn changed temporal length from {length} "
                f"to {group_features.shape[1]}."
            )
        for group_position, sample_index in enumerate(indices):
            features_by_sample[sample_index] = group_features[group_position]

    max_length = max(lengths)
    feature_dim = features_by_sample[0].shape[-1]
    features = features_by_sample[0].new_zeros(
        (len(features_by_sample), max_length, feature_dim)
    )
    for index, (sample_features, length) in enumerate(zip(features_by_sample, lengths)):
        features[index, :length] = sample_features

    valid_lengths = torch.tensor(lengths, dtype=torch.long, device=features.device)
    return features, valid_lengths


def load_feature_reference(feature_reference: str):
    reference_path = Path(feature_reference)
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"File di riferimento delle feature non trovato: {reference_path}"
        )
    payload = torch.load(
        reference_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(
            "Il file di riferimento non contiene un dizionario metadata valido."
        )
    experiment_config = payload["metadata"].get("experiment_config")
    if not isinstance(experiment_config, dict):
        raise ValueError(
            "Il file di riferimento non contiene "
            "metadata['experiment_config'] valido."
        )
    return payload, experiment_config


def resolve_evaluation_preprocessing(args, experiment_config=None):
    """Resolve sampling and image preprocessing from a feature artifact or CLI."""
    if args.feature_reference:
        if experiment_config is None:
            _, experiment_config = load_feature_reference(args.feature_reference)
        if not isinstance(experiment_config, dict):
            raise ValueError(
                "Il file di riferimento non contiene "
                "metadata['experiment_config']."
            )
        required_fields = {"sampler", "spatial_mode", "spatial_size", "color"}
        missing_fields = sorted(required_fields - experiment_config.keys())
        if missing_fields:
            raise ValueError(
                "Configurazione esperimento incompleta nel file di riferimento; "
                f"mancano: {', '.join(missing_fields)}."
            )
        sampler_config = experiment_config["sampler"]
        spatial_mode = experiment_config["spatial_mode"]
        spatial_size = experiment_config["spatial_size"]
        color_mode = experiment_config["color"]
        config_source = str(args.feature_reference)
    else:
        sampler_class = SAMPLER_REGISTRY.get(args.temporal_sampler)
        if sampler_class is None:
            available = ", ".join(sorted(SAMPLER_REGISTRY))
            raise ValueError(
                f"Unknown temporal sampler '{args.temporal_sampler}'. "
                f"Available: {available}."
            )
        sampler_parameters = signature(sampler_class.__init__).parameters
        sampler_config = {"name": args.temporal_sampler}
        sampler_config.update(
            {
                name: getattr(args, name)
                for name, parameter in sampler_parameters.items()
                if name != "self"
                and parameter.kind
                not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
                and hasattr(args, name)
            }
        )
        spatial_mode = args.spatial_mode
        spatial_size = (
            args.spatial_size
            if args.spatial_size is not None
            else (312 if args.backbone_type == "X3D-L" else 192)
        )
        color_mode = args.color
        config_source = "argomenti CLI"

    if not isinstance(sampler_config, dict):
        raise ValueError("La configurazione del sampler deve essere un dizionario.")
    if sampler_config.get("name") == "preserve_all":
        raise ValueError(
            "preserve_all non e' consentito in final_test sui video grezzi: "
            "decodificherebbe tutti i frame."
        )
    temporal_sampler = build_temporal_sampler(sampler_config)
    if spatial_mode not in {"letterbox", "stretch"}:
        raise ValueError(f"Unsupported spatial mode: {spatial_mode}")
    if color_mode not in {"rgb", "grayscale"}:
        raise ValueError(f"Unsupported color mode: {color_mode}")
    if not isinstance(spatial_size, int) or spatial_size <= 0:
        raise ValueError(f"spatial_size non valido: {spatial_size!r}")

    resolved = {
        "source": config_source,
        "sampler": temporal_sampler.config(),
        "spatial_mode": spatial_mode,
        "spatial_size": spatial_size,
        "color": color_mode,
    }
    return temporal_sampler, spatial_mode, spatial_size, color_mode, resolved


def resolve_backbone_config(args, experiment_config=None):
    """Resolve the video backbone from experiment metadata or CLI settings."""
    if args.feature_reference:
        if experiment_config is None:
            _, experiment_config = load_feature_reference(args.feature_reference)
        if not isinstance(experiment_config, dict):
            raise ValueError(
                "Il file di riferimento non contiene "
                "metadata['experiment_config'] valido."
            )
        required_fields = {"backbone", "tracker"}
        missing_fields = sorted(required_fields - experiment_config.keys())
        if missing_fields:
            raise ValueError(
                "Configurazione backbone incompleta nel file di riferimento; "
                f"mancano: {', '.join(missing_fields)}."
            )
        backbone_value = str(experiment_config["backbone"]).lower().replace("-", "_")
        backbone_types = {"x3d_l": "X3D-L", "x3d_s": "X3D-S"}
        if backbone_value not in backbone_types:
            raise ValueError(
                f"Backbone non supportato nel file di riferimento: "
                f"{experiment_config['backbone']!r}."
            )
        tracker = experiment_config["tracker"]
        if tracker != "none":
            raise ValueError(
                f"Il tracker integrato nel backbone è stato rimosso; valore nel "
                f"file di riferimento: {tracker!r}. Le nuove feature ball/rim "
                "si estraggono offline e non sono ancora integrate in final_test.py."
            )
        backbone_type = backbone_types[backbone_value]
        feature_stream = "3d"
        config_source = str(args.feature_reference)
    else:
        backbone_type = args.backbone_type
        feature_stream = args.feature_stream
        config_source = "argomenti CLI"
        if backbone_type not in {"X3D-L", "X3D-S"}:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        if feature_stream != "3d":
            raise ValueError(f"Unsupported feature stream: {feature_stream}")

    return {
        "backbone_type": backbone_type,
        "feature_stream": feature_stream,
        "config_type": feature_stream,
        "source": config_source,
    }


def main():
    parser = argparse.ArgumentParser(description="Valutazione End-to-End della pipeline a cascata.")
    parser.add_argument("--test-dir", default="dataset/dataset_hd/test", help="Path directory del test set.")
    parser.add_argument("--output-dir", default="experiments/EVALUATION_CASCADE", help="Cartella dei risultati.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--feature-reference",
        default=None,
        help="File .pt delle feature estratte: usa metadata['experiment_config'].",
    )
    parser.add_argument(
        "--temporal-sampler",
        choices=sorted(set(SAMPLER_REGISTRY) - {"preserve_all"}),
        default="legacy_uniform",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--frames-per-side", type=int, default=8)
    parser.add_argument("--edge-strength", type=float, default=1.0)
    parser.add_argument(
        "--spatial-mode", choices=["letterbox", "stretch"], default="letterbox"
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=None,
        help="Default: 312 con X3D-L, 192 con X3D-S.",
    )
    parser.add_argument("--color", choices=["rgb", "grayscale"], default="rgb")
    parser.add_argument("--backbone-type", choices=["X3D-L", "X3D-S"], default="X3D-L")
    parser.add_argument(
        "--feature-stream",
        choices=["3d"],
        default="3d",
    )
    parser.add_argument("--checkpoint-p1", default=BEST_CHECKPOINTS["p1"])
    parser.add_argument("--checkpoint-p2", default=BEST_CHECKPOINTS["p2"])
    parser.add_argument("--checkpoint-p3", default=BEST_CHECKPOINTS["p3"])

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device in uso: {device}")

    reference_payload = None
    experiment_config = None
    expected_experiment_id = None
    if args.feature_reference:
        reference_payload, experiment_config = load_feature_reference(
            args.feature_reference
        )
        expected_experiment_id = reference_payload["metadata"].get("experiment_id")
        reference_payload = None  # Release the feature tensors after reading metadata.
    else:
        print(
            "[WARNING] Nessun --feature-reference: sampler, preprocessing "
            "spaziale, colore e backbone vengono dai flag/default CLI e "
            "potrebbero non coincidere con quelli usati nel training."
        )

    temporal_sampler, spatial_mode, spatial_size, color_mode, preprocessing_config = (
        resolve_evaluation_preprocessing(args, experiment_config)
    )
    backbone_config = resolve_backbone_config(args, experiment_config)
    preprocessing_config["backbone"] = backbone_config["backbone_type"]
    preprocessing_config["feature_stream"] = backbone_config["feature_stream"]
    print(
        "[CONFIG] Preprocessing e campionamento: "
        f"{json.dumps(preprocessing_config, ensure_ascii=False, sort_keys=True)}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Backbone online: usa la stessa factory dell'estrattore offline.
    composite_backbone, _ = build_backbone_and_filename(
        config_type=backbone_config["config_type"],
        backbone_type=backbone_config["backbone_type"],
    )
    composite_backbone = composite_backbone.to(device)
    composite_backbone.eval()
    composite_backbone.freeze()

    # 2. Modelli della cascata: l'input dimension deriva dal backbone scelto.
    checkpoint_paths = {
        "p1": args.checkpoint_p1,
        "p2": args.checkpoint_p2,
        "p3": args.checkpoint_p3,
    }
    models = build_specialized_models(
        device,
        checkpoint_paths,
        in_dim=composite_backbone.out_dim,
        expected_experiment_id=expected_experiment_id,
    )

    # 3. Dataloader Test
    test_dataset = TestFolderDataset(
        test_root=args.test_dir,
        temporal_sampler=temporal_sampler,
        spatial_mode=spatial_mode,
        spatial_size=spatial_size,
        color_mode=color_mode,
    )
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
            labels = batch["labels"].cpu().numpy()
            all_raw_labels.extend(labels)

            clips_norm = [clip.to(device) for clip in batch["x_norm"]]
            clips_raw = [clip.to(device) for clip in batch["x_raw"]]
            features, lengths = compute_features_by_length(
                clips_norm,
                clips_raw,
                composite_backbone.forward_dual,
            )

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
