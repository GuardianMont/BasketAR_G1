import argparse
import itertools
import os
import random
import sys
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
from data.experiment_config import ExperimentConfig
from data.samplers import BucketSampler
from data.temporal_sampling import LegacyUniformSampler
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

# Feature dataset di baseline: X3D-L, legacy_uniform16, letterbox 312, RGB, no tracker.
BASELINE_FEATURE_CONFIG = ExperimentConfig(
    sampler=LegacyUniformSampler(16).config(),
    spatial_mode="letterbox",
    spatial_size=312,
    color="rgb",
    backbone="x3d_l",
    tracker="none",
    extra={"transform": "none"},
)
FIXED_DATA_CONFIG = {
    "name": "DATA-01_Solo3D",
    "train_path": os.path.join("features", BASELINE_FEATURE_CONFIG.feature_filename("train")),
    "val_path": os.path.join("features", BASELINE_FEATURE_CONFIG.feature_filename("val")),
    "feature_dim": 432,
}

# Configurazione per le 3 fasi gerarchiche
PHASE_CONFIGS = {
    "p1": {
        "name": "p1 (Action Type)",
        "num_classes": 3,
        "filter_shots_only": False,
    },
    "p2": {
        "name": "p2 (Shot Type)",
        "num_classes": 3,
        "filter_shots_only": True,
    },
    "p3": {
        "name": "p3 (Shot Outcome)",
        "num_classes": 2,
        "filter_shots_only": True,
    },
}

# EXPERIMENTS CONFIGURATION PER ARCHITECTURE & LOSS ABLATION
EXPERIMENTS = {
    "ARCH-01_Identity_Mean": {
        "description": "Identity Neck + Mean Pooling Head (Temporal Baseline)",
        "components": {
            "neck": lambda params, in_dim: IdentityNeck(in_dim=in_dim),
            "head": lambda params, in_dim, num_classes: MeanPoolingClassifierHead(
                in_dim=in_dim, num_classes=num_classes, dropout=params["dropout"]
            ),
            "loss": lambda params, weights: nn.CrossEntropyLoss(weight=weights),
        },
        "grid": {
            "dropout": [0.0, 0.2, 0.4],
            "weight_decay": [0.0, 1e-4, 1e-3],
            "lr": [1e-3, 1e-4],
        },
    },
    "ARCH-02_BiGRU_Mean": {
        "description": "BiGRU Neck + Mean Pooling Head",
        "components": {
            "neck": lambda params, in_dim: BiGRUNeck(
                in_dim=in_dim,
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
            ),
            "head": lambda params, in_dim, num_classes: MeanPoolingClassifierHead(
                in_dim=params["hidden_dim"] * 2,
                num_classes=num_classes,
                dropout=0.0,
            ),
            "loss": lambda params, weights: nn.CrossEntropyLoss(weight=weights),
        },
        "grid": {
            "hidden_dim": [128, 256],
            "dropout": [0.0, 0.2, 0.4],
            "weight_decay": [0.0, 1e-4, 1e-3],
            "num_layers": [1, 2],
            "lr": [1e-3, 5e-4, 1e-4],
        },
    },
    "ARCH-03_TCN_Mean": {
        "description": "TCN Neck + Mean Pooling Head",
        "components": {
            "neck": lambda params, in_dim: TCNNeck(
                in_dim=in_dim,
                num_channels=params["num_channels"],
                kernel_size=params["kernel_size"],
                dropout=params["tcn_dropout"],
            ),
            "head": lambda params, in_dim, num_classes: MeanPoolingClassifierHead(
                in_dim=in_dim, num_classes=num_classes, dropout=0.0
            ),
            "loss": lambda params, weights: nn.CrossEntropyLoss(weight=weights),
        },
        "grid": {
            "kernel_size": [3, 5],
            "tcn_dropout": [0.0, 0.2],
            "num_channels": [[128, 256], [256, 512], [128, 256, 512]],
            "lr": [1e-3, 5e-4, 1e-4],
        },
    },
    "ARCH-04_TCN_Attention": {
        "description": "TCN Neck + Attention Classifier Head",
        "components": {
            "neck": lambda params, in_dim: TCNNeck(
                in_dim=in_dim,
                num_channels=[128, 256],
                kernel_size=5,
                dropout=0.2,
            ),
            "head": lambda params, in_dim, num_classes: AttentionClassifierHead(
                in_dim=in_dim, num_classes=num_classes, dropout=params["dropout"]
            ),
            "loss": lambda params, weights: nn.CrossEntropyLoss(weight=weights),
        },
        "grid": {
            "dropout": [0.0, 0.2],
            "lr": [1e-3, 5e-4, 1e-4],
        },
    },
    "ARCH-05_FocalLoss": {
        "description": "TCN Neck + Attention Head + Focal Loss (gamma=2.0)",
        "components": {
            "neck": lambda params, in_dim: TCNNeck(
                in_dim=in_dim,
                num_channels=[128, 256],
                kernel_size=5,
                dropout=0.2,
            ),
            "head": lambda params, in_dim, num_classes: MeanPoolingClassifierHead(
                in_dim=in_dim, num_classes=num_classes, dropout=0.0
            ),
            "loss": lambda params, weights: FocalLoss(
                gamma=params['gamma'], alpha=weights
            ),
        },
        "grid": {
            "lr": [1e-3, 5e-4, 1e-4],
            "gamma": [0.0, 1.0, 2.0, 3.0],
        },
    },
}

def set_seed(seed: int):
    """Sets random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_preextracted_features(pt_file_path: str, expected_dim: int):
    """Load and validate a real feature payload."""
    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(
            f"Feature file '{pt_file_path}' not found. Extract features before training."
        )

    payload = torch.load(pt_file_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid feature payload in '{pt_file_path}': expected a dictionary.")
    if payload.get("feature_dim") != expected_dim:
        raise ValueError(
            f"Expected feature dimension {expected_dim}, got "
            f"{payload.get('feature_dim')} in '{pt_file_path}'."
        )
    print(
        f"[INFO] Loaded {payload['num_samples']} samples (Dimension"
        f" D={payload['feature_dim']}) from '{pt_file_path}' "
        f"(metadata={payload.get('metadata', {})})"
    )

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Invalid or empty feature payload: '{pt_file_path}'")
    for index, sample in enumerate(samples):
        data = sample.get("data")
        if not torch.is_tensor(data) or data.ndim != 2 or data.shape[1] != expected_dim:
            raise ValueError(f"Invalid feature tensor in sample {index} of '{pt_file_path}'.")
    return samples


def build_dataloaders_for_phase(train_samples, val_samples, phase: str):
    """Filters dataset samples according to phase requirements and creates DataLoaders."""
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
            sequence_lengths=[int(s["data"].shape[0]) for s in train_filtered],
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
    feature_dim = FIXED_DATA_CONFIG["feature_dim"]

    phase_dir = os.path.join(exp_dir, phase)
    plots_dir = os.path.join(phase_dir, "plots")
    checkpoints_dir = os.path.join(phase_dir, "checkpoints")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(
        f"   STARTING PHASE: {p_cfg['name']} | Experiment: {experiment_name}"
        f" [{num_runs} Runs]"
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
                model.parameters(), lr=params.get("lr", 1e-3), weight_decay=params.get("weight_decay", 0.0)
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


def extract_best_configuration_per_phase(df_phase_results: pd.DataFrame) -> dict:
    """Raggruppa le run per configurazione e restituisce la migliore per Macro-F1 medio."""
    grouped = (
        df_phase_results.groupby("hyperparameters")
        .agg(
            mean_macro_f1=("val_macro_f1", "mean"),
            std_macro_f1=("val_macro_f1", "std"),
            mean_accuracy=("val_accuracy", "mean"),
            std_accuracy=("val_accuracy", "std"),
        )
        .reset_index()
    )

    best_row = grouped.loc[grouped["mean_macro_f1"].idxmax()]

    return {
        "best_hyperparameters": best_row["hyperparameters"],
        "mean_val_macro_f1": round(float(best_row["mean_macro_f1"]), 4),
        "std_val_macro_f1": round(float(best_row["std_macro_f1"]), 4),
        "mean_val_accuracy": round(float(best_row["mean_accuracy"]), 4),
        "std_val_accuracy": round(float(best_row["std_accuracy"]), 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Architecture & Loss Ablation Study per Experiment and/or Phase."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=list(EXPERIMENTS.keys()),
        help="Specify a single experiment to run (e.g. ARCH-01_Identity_Mean). If omitted, runs all.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        choices=list(PHASE_CONFIGS.keys()),
        help="Specify a single phase to run (p1, p2, or p3). If omitted, runs all phases.",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all configured experiments and exit.",
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

    if args.list_experiments:
        print("\nConfigured Architecture Experiments:")
        for exp_key, exp_val in EXPERIMENTS.items():
            print(f"  - {exp_key}: {exp_val['description']}")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device detected and in use: {device}")
    print(f"[INFO] Running multi-seed evaluation: {args.num_runs} runs per experiment")

    target_experiments = (
        {args.experiment: EXPERIMENTS[args.experiment]}
        if args.experiment
        else EXPERIMENTS
    )

    target_phases = [args.phase] if args.phase else ["p1", "p2", "p3"]

    # Costruzione dinamica del nome cartella di report
    exp_name_str = args.experiment if args.experiment else "ALL_EXP"
    phase_str = args.phase if args.phase else "ALL_PHASES"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    suite_dir = os.path.join(
        "experiments", f"ARCH_ABLATION_{exp_name_str}_{phase_str}_{timestamp}"
    )
    os.makedirs(suite_dir, exist_ok=True)

    # Caricamento delle feature X3D-L senza tracker (D=432)
    raw_train_samples = load_preextracted_features(
        FIXED_DATA_CONFIG["train_path"], expected_dim=FIXED_DATA_CONFIG["feature_dim"]
    )
    raw_val_samples = load_preextracted_features(
        FIXED_DATA_CONFIG["val_path"], expected_dim=FIXED_DATA_CONFIG["feature_dim"]
    )

    suite_all_results = []
    summary_stats = []
    best_config_summary = []

    for exp_name, config in target_experiments.items():
        print(f"\n\n{'#'*80}")
        print(
            f"   RUNNING ARCHITECTURE ABLATION EXPERIMENT: {exp_name} -"
            f" {config['description']}"
        )
        print(f"{'#'*80}")

        exp_dir = os.path.join(suite_dir, exp_name)
        csv_dir = os.path.join(exp_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)

        exp_hierarchical_results = []

        for phase in target_phases:
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

            df_phase = pd.DataFrame(phase_results)
            phase_csv_path = os.path.join(csv_dir, f"results_{phase}_all_runs.csv")
            df_phase.to_csv(phase_csv_path, index=False)

            best_config_info = extract_best_configuration_per_phase(df_phase)
            best_config_summary.append({
                "experiment_name": exp_name,
                "phase": phase,
                "best_hyperparameters": best_config_info["best_hyperparameters"],
                "mean_val_macro_f1": best_config_info["mean_val_macro_f1"],
                "std_val_macro_f1": best_config_info["std_val_macro_f1"],
                "mean_val_accuracy": best_config_info["mean_val_accuracy"],
                "std_val_accuracy": best_config_info["std_val_accuracy"],
                "macro_f1_formatted": f"{best_config_info['mean_val_macro_f1']:.4f} +/- {best_config_info['std_val_macro_f1']:.4f}",
            })

            acc_mean = round(float(df_phase["val_accuracy"].mean()), 4)
            acc_std = round(float(df_phase["val_accuracy"].std()), 4)
            f1_mean = round(float(df_phase["val_macro_f1"].mean()), 4)
            f1_std = round(float(df_phase["val_macro_f1"].std()), 4)

            summary_stats.append({
                "experiment_name": exp_name,
                "phase": phase,
                "feature_dim": FIXED_DATA_CONFIG["feature_dim"],
                "num_runs": args.num_runs,
                "mean_val_accuracy": acc_mean,
                "std_val_accuracy": acc_std,
                "mean_val_macro_f1": f1_mean,
                "std_val_macro_f1": f1_std,
                "accuracy_formatted": f"{acc_mean:.4f} +/- {acc_std:.4f}",
                "macro_f1_formatted": f"{f1_mean:.4f} +/- {f1_std:.4f}",
            })

            exp_hierarchical_results.extend(phase_results)

        exp_overall_csv = os.path.join(exp_dir, "overall_results.csv")
        pd.DataFrame(exp_hierarchical_results).to_csv(
            exp_overall_csv, index=False
        )

        exp_best_cfg_csv = os.path.join(exp_dir, "best_configurations_per_phase.csv")
        pd.DataFrame([b for b in best_config_summary if b["experiment_name"] == exp_name]).to_csv(
            exp_best_cfg_csv, index=False
        )

        suite_all_results.extend(exp_hierarchical_results)

    suite_summary_csv = os.path.join(suite_dir, "arch_ablation_all_runs.csv")
    pd.DataFrame(suite_all_results).to_csv(suite_summary_csv, index=False)

    suite_stats_csv = os.path.join(suite_dir, "arch_ablation_aggregated_stats.csv")
    pd.DataFrame(summary_stats).to_csv(suite_stats_csv, index=False)

    suite_best_cfg_csv = os.path.join(suite_dir, "best_configurations_per_phase_all_experiments.csv")
    df_best_cfg = pd.DataFrame(best_config_summary)
    df_best_cfg.to_csv(suite_best_cfg_csv, index=False)

    print(f"\n{'='*80}")
    print("[SUCCESS] Architecture Ablation completed successfully!")
    print(f"[DIRECTORY] Outputs structured in: '{suite_dir}'")
    print(f"[BEST CONFIGS CSV] Saved to: '{suite_best_cfg_csv}'")
    print(f"{'='*80}\n")

    print("SUMMARY OF BEST CONFIGURATIONS PER PHASE:")
    for row in best_config_summary:
        print(
            f"  - [{row['experiment_name']} | {row['phase'].upper()}]: "
            f"Config: {row['best_hyperparameters']} -> "
            f"Val Macro-F1: {row['macro_f1_formatted']}"
        )
