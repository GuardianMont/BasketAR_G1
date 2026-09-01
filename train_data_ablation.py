import argparse
import itertools
import os
import random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Module imports
from data.dataset import ActionRecognitionDataset
from data.samplers import BucketSampler
from data.utils import compute_class_weights, pad_and_collate_video_batch
from losses import FocalLoss
from models.heads import AttentionClassifierHead, MeanPoolingClassifierHead
from models.necks import BiGRUNeck, IdentityNeck, TCNNeck
from models.pipeline import ARPipeline
from utils.logger import ExperimentLogger
from utils.plotting import (
    CLASS_NAMES,
    PHASE_CLASS_NAMES,
    get_confusion_matrix_tensor,
    save_confusion_matrix_plot,
)

# Set seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Configuration for hierarchical phases
PHASE_CONFIGS = {
    "p1": {
        "name": "p1 (Action Type)",
        "num_classes": 3,
        "filter_shots_only": False,
    },
    "p2": {
        "name": "p2 (Shot type)",
        "num_classes": 3,
        "filter_shots_only": True,
    },
    "p3": {
        "name": "p3 (Shot's outcome)",
        "num_classes": 2,
        "filter_shots_only": True,
    },
}

# Network components configuration
DEFAULT_COMPONENTS = {
    "neck": lambda params, in_dim: TCNNeck(
        in_dim=in_dim,
        num_channels=[256, 512],
        kernel_size=3,
        dropout=0.2,
    ),
    "head": lambda params, in_dim, num_classes: AttentionClassifierHead(
        in_dim=512, num_classes=num_classes, dropout=0.2
    ),
    "loss": lambda params, weights: nn.CrossEntropyLoss(weight=weights),
}

DEFAULT_GRID = {"lr": [1e-3]}

# Dataset descriptions
DATA_CONFIGS = {
    "DATA-01_Solo3D": (
        "3D Spatio-Temporal Features (X3D-L)",
        "features/train_3d_L.pt",
        "features/val_3d_L.pt",
        432,  # ex 192
    ),
    "DATA-02_3D_Tracker": (
        "3D Features (X3D-L) + Ball Tracking Geometry",
        "features/train_3d_tracker_L.pt",
        "features/val_3d_tracker_L.pt",
        437,  # ex 197
    ),
}

EXPERIMENTS = {
    key: {
        "description": desc,
        "train_path": train_path,
        "val_path": val_path,
        "feature_dim": dim,
        "components": DEFAULT_COMPONENTS,
        "grid": DEFAULT_GRID,
    }
    for key, (desc, train_path, val_path, dim) in DATA_CONFIGS.items()
}


def load_preextracted_features(pt_file_path: str, expected_dim: int):
    """Loads feature file or generates mock tensor data if missing."""
    if not os.path.exists(pt_file_path):
        print(
            f"[WARNING] Feature file '{pt_file_path}' not found. Generating mock"
            " features for testing..."
        )
        os.makedirs(os.path.dirname(pt_file_path), exist_ok=True)

        num_samples = 12 if "train" in pt_file_path else 6
        mock_samples = []

        for i in range(num_samples):
            T_i = int(torch.randint(8, 41, (1,)).item())
            mock_samples.append({
                "id": f"clip_{i:06d}",
                "data": torch.randn(T_i, expected_dim),
                "label": i % 9,
                "num_frames": T_i,
            })

        mock_payload = {
            "format_version": 1,
            "backbone": "MockBackbone",
            "feature_dim": expected_dim,
            "num_samples": num_samples,
            "samples": mock_samples,
        }

        torch.save(mock_payload, pt_file_path)
        print(
            f"[INFO] Created mock dataset at '{pt_file_path}' with"
            f" {num_samples} samples (D={expected_dim})."
        )

    payload = torch.load(pt_file_path, map_location="cpu", weights_only=False)
    print(
        f"[INFO] Loaded {payload['num_samples']} samples (Dimension"
        f" D={payload['feature_dim']}) from '{pt_file_path}'"
    )

    return payload["samples"]


def build_dataloaders_for_phase(train_samples, val_samples, phase: str):
    """Filters dataset samples by phase constraints and creates DataLoaders."""
    p_cfg = PHASE_CONFIGS[phase]

    if p_cfg["filter_shots_only"]:
        train_filtered = [
            s for s in train_samples if s["label"] in [1, 2, 3, 4, 5, 6]
        ]
        val_filtered = [
            s for s in val_samples if s["label"] in [1, 2, 3, 4, 5, 6]
        ]
    else:
        train_filtered = train_samples
        val_filtered = val_samples

    train_dataset = ActionRecognitionDataset(
        samples=train_filtered, is_feature_dataset=True
    )
    val_dataset = ActionRecognitionDataset(
        samples=val_filtered, is_feature_dataset=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BucketSampler(
            sequence_lengths=[s["num_frames"] for s in train_filtered],
            batch_size=8,
        ),
        collate_fn=pad_and_collate_video_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=pad_and_collate_video_batch,
    )

    target_key = f"label_{phase}"
    train_phase_labels = [
        train_dataset[i][target_key] for i in range(len(train_dataset))
    ]
    class_weights = compute_class_weights(
        train_phase_labels,
        num_classes=p_cfg["num_classes"],
        method="inverse",
    )

    return train_loader, val_loader, class_weights


def train_single_phase(
    phase: str,
    experiment_name: str,
    exp_dir: str,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    patience: int,
    num_runs: int = 5,
    base_seed: int = 42,
):
    p_cfg = PHASE_CONFIGS[phase]
    num_classes = p_cfg["num_classes"]
    feature_dim = config["feature_dim"]

    phase_dir = os.path.join(exp_dir, phase)
    plots_dir = os.path.join(phase_dir, "plots")
    checkpoints_dir = os.path.join(phase_dir, "checkpoints")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(
        f"   STARTING PHASE: {p_cfg['name']} | Data Model: {experiment_name}"
        f" (D={feature_dim}) [{num_runs} Runs]"
    )
    print(f"{'='*70}")

    keys, values = zip(*config["grid"].items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    phase_results = []

    for run_idx in range(1, num_runs + 1):
        seed = base_seed + run_idx
        set_seed(seed)

        for combo_idx, params in enumerate(combinations):
            print(
                f"\n--> [{phase.upper()} | Run {run_idx}/{num_runs} | Seed {seed}]"
                f" Config: {params}"
            )

            tb_logger = ExperimentLogger(
                log_dir=os.path.join(phase_dir, "tb_runs"),
                experiment_name=f"{experiment_name}_run{run_idx}",
                run_id=combo_idx + 1,
            )

            neck = config["components"]["neck"](params, in_dim=feature_dim)
            head = config["components"]["head"](
                params,
                in_dim=neck.out_dim if hasattr(neck, "out_dim") else feature_dim,
                num_classes=num_classes,
            )
            criterion = config["components"]["loss"](
                params, class_weights.to(device)
            )

            model = ARPipeline(backbone=None, neck=neck, head=head).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=params.get("lr", 1e-3)
            )

            best_run_loss = float("inf")
            best_run_f1 = 0.0
            best_run_acc = 0.0
            best_run_epoch = 0
            epochs_without_improvement = 0

            best_preds, best_labels = [], []
            label_key = f"label_{phase}"

            epoch_pbar = tqdm(
                range(1, epochs + 1),
                desc=f"[{phase.upper()} | Run {run_idx}/{num_runs}]",
                unit="ep",
                leave=True,
            )

            for epoch in epoch_pbar:
                # --- TRAINING ---
                model.train()
                train_loss = 0.0

                for batch in train_loader:
                    x = batch["x"].to(device)
                    labels = batch[label_key].to(device)
                    lengths = batch["lengths"]

                    optimizer.zero_grad()
                    logits = model(x, lengths=lengths)
                    loss = criterion(logits, labels)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                # --- VALIDATION ---
                model.eval()
                all_preds, all_labels = [], []
                val_loss = 0.0

                with torch.no_grad():
                    for batch in val_loader:
                        x = batch["x"].to(device)
                        labels = batch[label_key].to(device)
                        lengths = batch["lengths"]

                        logits = model(x, lengths=lengths)
                        loss = criterion(logits, labels)
                        val_loss += loss.item()

                        preds = torch.argmax(logits, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                acc = accuracy_score(all_labels, all_preds)
                macro_f1 = f1_score(
                    all_labels, all_preds, average="macro", zero_division=0
                )
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)

                tb_logger.log_epoch(
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    val_acc=acc,
                    val_f1=macro_f1,
                )

                epoch_pbar.set_postfix({
                    "TrLoss": f"{avg_train_loss:.4f}",
                    "ValLoss": f"{avg_val_loss:.4f}",
                    "ValAcc": f"{acc:.4f}",
                    "ValF1": f"{macro_f1:.4f}",
                    "BestF1": f"{best_run_f1:.4f}" if best_run_f1 > 0 else "N/A",
                    "BestLoss": (
                        f"{best_run_loss:.4f}"
                        if best_run_loss < float("inf")
                        else "N/A"
                    ),
                })

                # Early Stopping basato sulla minimizzazione della Validation Loss
                if avg_val_loss < best_run_loss:
                    best_run_loss = avg_val_loss
                    best_run_f1 = macro_f1
                    best_run_acc = acc
                    best_run_epoch = epoch
                    epochs_without_improvement = 0

                    best_preds = all_preds
                    best_labels = all_labels

                    weight_path = os.path.join(
                        checkpoints_dir,
                        f"{experiment_name}_run{run_idx}_{phase}_best.pt",
                    )
                    torch.save(model.state_dict(), weight_path)
                else:
                    epochs_without_improvement += 1

                    if epochs_without_improvement >= patience:
                        epoch_pbar.write(
                            f"    [EARLY STOPPING - {phase.upper()}] Best Ep:"
                            f" {best_run_epoch} | Best Val Loss:"
                            f" {best_run_loss:.4f} (Val F1: {best_run_f1:.4f})"
                        )
                        break

            if best_preds and best_labels:
                plot_path = os.path.join(
                    plots_dir, f"{experiment_name}_run{run_idx}_{phase}_cm.png"
                )
                phase_class_names = PHASE_CLASS_NAMES.get(phase, CLASS_NAMES)

                save_confusion_matrix_plot(
                    best_labels,
                    best_preds,
                    plot_path,
                    class_names=phase_class_names,
                )
                cm_tensor = get_confusion_matrix_tensor(
                    best_labels, best_preds, class_names=phase_class_names
                )
                tb_logger.log_image(
                    f"Confusion_Matrix/{phase.upper()}_Best",
                    cm_tensor,
                    step=best_run_epoch,
                )

            tb_logger.close()

            run_record = {
                "phase": phase,
                "experiment_name": experiment_name,
                "feature_dim": feature_dim,
                "run_id": run_idx,
                "seed": seed,
                "hyperparameters": str(params),
                "best_epoch": best_run_epoch,
                "val_accuracy": round(best_run_acc, 4),
                "val_macro_f1": round(best_run_f1, 4),
                "checkpoint_path": os.path.join(
                    checkpoints_dir,
                    f"{experiment_name}_run{run_idx}_{phase}_best.pt",
                ),
                "confusion_matrix_path": os.path.join(
                    plots_dir, f"{experiment_name}_run{run_idx}_{phase}_cm.png"
                ),
            }
            phase_results.append(run_record)

            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return phase_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Data Representation Ablation Study with multi-run statistical evaluation."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum epochs for each configuration.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience on validation Macro-F1.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs with different random seeds for statistical evaluation.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device detected and in use: {device}")
    print(f"[INFO] Running multi-seed evaluation: {args.num_runs} runs per experiment")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = os.path.join("experiments", f"DATA_ABLATION_{timestamp}")
    os.makedirs(suite_dir, exist_ok=True)

    suite_all_results = []
    summary_stats = []

    for exp_name, config in EXPERIMENTS.items():
        print(f"\n\n{'#'*80}")
        print(
            f"   RUNNING DATA ABLATION EXPERIMENT: {exp_name} -"
            f" {config['description']}"
        )
        print(f"{'#'*80}")

        exp_dir = os.path.join(suite_dir, exp_name)
        csv_dir = os.path.join(exp_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)

        raw_train_samples = load_preextracted_features(
            config["train_path"], expected_dim=config["feature_dim"]
        )
        raw_val_samples = load_preextracted_features(
            config["val_path"], expected_dim=config["feature_dim"]
        )

        exp_hierarchical_results = []

        for phase in ["p1", "p2", "p3"]:
            train_loader, val_loader, class_weights = (
                build_dataloaders_for_phase(
                    raw_train_samples, raw_val_samples, phase=phase
                )
            )

            phase_results = train_single_phase(
                phase=phase,
                experiment_name=exp_name,
                exp_dir=exp_dir,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                num_runs=args.num_runs,
            )

            # Save per-phase detailed runs CSV
            df_phase = pd.DataFrame(phase_results)
            phase_csv_path = os.path.join(csv_dir, f"results_{phase}_all_runs.csv")
            df_phase.to_csv(phase_csv_path, index=False)

            # Compute statistical summary (mean +/- std)
            acc_mean = round(float(df_phase["val_accuracy"].mean()), 4)
            acc_std = round(float(df_phase["val_accuracy"].std()), 4)
            f1_mean = round(float(df_phase["val_macro_f1"].mean()), 4)
            f1_std = round(float(df_phase["val_macro_f1"].std()), 4)

            summary_stats.append({
                "experiment_name": exp_name,
                "phase": phase,
                "feature_dim": config["feature_dim"],
                "num_runs": args.num_runs,
                "mean_val_accuracy": acc_mean,
                "std_val_accuracy": acc_std,
                "mean_val_macro_f1": f1_mean,
                "std_val_macro_f1": f1_std,
                "accuracy_formatted": f"{acc_mean:.4f} +/- {acc_std:.4f}",
                "macro_f1_formatted": f"{f1_mean:.4f} +/- {f1_std:.4f}",
            })

            exp_hierarchical_results.extend(phase_results)

        # Save experiment overall CSV
        exp_overall_csv = os.path.join(exp_dir, "overall_results.csv")
        pd.DataFrame(exp_hierarchical_results).to_csv(
            exp_overall_csv, index=False
        )

        suite_all_results.extend(exp_hierarchical_results)

    # Save detailed summary CSV
    suite_summary_csv = os.path.join(suite_dir, "data_ablation_all_runs.csv")
    pd.DataFrame(suite_all_results).to_csv(suite_summary_csv, index=False)

    # Save aggregated statistics CSV (Mean +/- Std)
    suite_stats_csv = os.path.join(suite_dir, "data_ablation_aggregated_stats.csv")
    pd.DataFrame(summary_stats).to_csv(suite_stats_csv, index=False)

    print(f"\n{'='*80}")
    print(
        "[SUCCESS] Data Ablation Suite completed successfully across 5 runs!"
    )
    print(f"[DIRECTORY] Outputs structured in: '{suite_dir}'")
    print(f"[STATS CSV] Aggregated (Mean +/- Std) saved to: '{suite_stats_csv}'")
    print(f"{'='*80}")