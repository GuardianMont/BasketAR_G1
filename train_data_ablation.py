import argparse
import itertools
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data.phase_manager import PHASE_CONFIGS, prepare_phase_data
from models.trainer import Trainer
from models.heads import AttentionClassifierHead
from models.necks import TCNNeck
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
from utils.feature_loader import FeatureLoader
from utils.logger import ExperimentLogger


def set_seed(seed: int):
    """Sets random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
BASELINE_FEATURE_CONFIG = ExperimentConfig(
    sampler=LegacyUniformSampler(16).config(),
    spatial_mode="letterbox",
    spatial_size=312,
    color="rgb",
    backbone="x3d_l",
    tracker="none",
    extra={"transform": "none"},
)
TRACKER_FEATURE_CONFIG = ExperimentConfig(
    sampler=LegacyUniformSampler(16).config(),
    spatial_mode="letterbox",
    spatial_size=312,
    color="rgb",
    backbone="x3d_l",
    tracker="yolov8n_coco",
    extra={"transform": "none"},
)

DATA_CONFIGS = {
    "DATA-01_Solo3D": (
        "3D Spatio-Temporal Features (X3D-L)",
        "features/train_3d_L.pt",
        "features/val_3d_L.pt",
        432,
    ),
    "DATA-02_3D_Tracker": (
        "3D Features (X3D-L) + Ball Tracking Geometry",
        "features/train_3d_tracker_L.pt",
        "features/val_3d_tracker_L.pt",
        437,
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
    train_samples: list[dict],
    val_samples: list[dict],
    device: torch.device,
    epochs: int,
    patience: int,
    num_runs: int = 5,
    base_seed: int = 42,
) -> list[dict]:
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

    train_loader, val_loader, class_weights = prepare_phase_data(
        train_samples, val_samples, phase=phase
    )

    keys, values = zip(*config["grid"].items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    phase_results = []

    for run_idx in range(1, num_runs + 1):
        seed = base_seed + run_idx
        set_seed(seed)

        for combo_idx, params in enumerate(combinations, 1):
            print(
                f"\n--> [{phase.upper()} | Run {run_idx}/{num_runs} | Seed {seed}]"
                f" Config: {params}"
            )

            tb_logger = ExperimentLogger(
                log_dir=os.path.join(phase_dir, "tb_runs"),
                experiment_name=f"{experiment_name}_run{run_idx}",
                run_id=combo_idx,
            )

            neck = config["components"]["neck"](params, in_dim=feature_dim)
            head = config["components"]["head"](
                params,
                in_dim=getattr(neck, "out_dim", feature_dim),
                num_classes=num_classes,
            )
            criterion = config["components"]["loss"](
                params, class_weights.to(device)
            )

            model = ARPipeline(backbone=None, neck=neck, head=head)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=params.get("lr", 1e-3)
            )

            checkpoint_path = os.path.join(
                checkpoints_dir,
                f"{experiment_name}_run{run_idx}_{phase}_best.pt",
            )
            plot_path = os.path.join(
                plots_dir, f"{experiment_name}_run{run_idx}_{phase}_cm.png"
            )

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            metrics = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                phase=phase,
                epochs=epochs,
                patience=patience,
                tb_logger=tb_logger,
                checkpoint_path=checkpoint_path,
                plot_path=plot_path,
                desc=f"[{phase.upper()} | Run {run_idx}/{num_runs}]",
            )

            tb_logger.close()

            run_record = {
                "phase": phase,
                "experiment_name": experiment_name,
                "feature_dim": feature_dim,
                "run_id": run_idx,
                "seed": seed,
                "hyperparameters": str(params),
                "checkpoint_path": checkpoint_path,
                "confusion_matrix_path": plot_path,
                **metrics,
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

        raw_train_samples = FeatureLoader.load(
            config["train_path"], expected_dim=config["feature_dim"]
        )
        raw_val_samples = FeatureLoader.load(
            config["val_path"], expected_dim=config["feature_dim"]
        )

        exp_hierarchical_results = []

        for phase in ["p1", "p2", "p3"]:
            phase_results = train_single_phase(
                phase=phase,
                experiment_name=exp_name,
                exp_dir=exp_dir,
                config=config,
                train_samples=raw_train_samples,
                val_samples=raw_val_samples,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                num_runs=args.num_runs,
            )

            df_phase = pd.DataFrame(phase_results)
            phase_csv_path = os.path.join(csv_dir, f"results_{phase}_all_runs.csv")
            df_phase.to_csv(phase_csv_path, index=False)

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

        exp_overall_csv = os.path.join(exp_dir, "overall_results.csv")
        pd.DataFrame(exp_hierarchical_results).to_csv(
            exp_overall_csv, index=False
        )

        suite_all_results.extend(exp_hierarchical_results)

    suite_summary_csv = os.path.join(suite_dir, "data_ablation_all_runs.csv")
    pd.DataFrame(suite_all_results).to_csv(suite_summary_csv, index=False)

    suite_stats_csv = os.path.join(suite_dir, "data_ablation_aggregated_stats.csv")
    pd.DataFrame(summary_stats).to_csv(suite_stats_csv, index=False)

    print(f"\n{'='*80}")
    print(
        f"[SUCCESS] Data Ablation Suite completed successfully across {args.num_runs} runs!"
    )
    print(f"[DIRECTORY] Outputs structured in: '{suite_dir}'")
    print(f"[STATS CSV] Aggregated (Mean +/- Std) saved to: '{suite_stats_csv}'")
    print(f"{'='*80}")
