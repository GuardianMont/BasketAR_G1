import argparse
from datetime import datetime
import os
import itertools
import pandas as pd
import torch
import torch.nn as nn

from data import PHASE_CONFIGS, prepare_phase_data
from models.trainer import Trainer
from models.pipeline import ARPipeline
from utils.feature_loader import FeatureLoader
from utils.logger import ExperimentLogger
from models.necks import IdentityNeck, BiGRUNeck, TCNNeck
from models.heads import MeanPoolingClassifierHead, AttentionClassifierHead

EXPERIMENTS = {
    "EXP-01_Baseline_Identity": {
        "components": {
            "neck": lambda params, in_dim: IdentityNeck(in_dim=in_dim),
            "head": lambda params, in_dim, num_classes: MeanPoolingClassifierHead(
                in_dim=in_dim, num_classes=num_classes, dropout=params["dropout"]
            ),
            "loss": lambda params, weights: nn.CrossEntropyLoss(weight=weights),
        },
        "grid": {
            "dropout": [0.0, 0.2, 0.4],
            "lr": [1e-3, 3e-4],
        },
    },
    "EXP-02_BiGRU_Neck": {
        "components": {
            "neck": lambda params, in_dim: BiGRUNeck(
                in_dim=in_dim,
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                dropout=params["gru_dropout"],
            ),
            "head": lambda params, in_dim, num_classes: MeanPoolingClassifierHead(
                in_dim=params["hidden_dim"] * 2, num_classes=num_classes, dropout=0.0
            ),
            "loss": lambda params, weights: nn.CrossEntropyLoss(weight=weights),
        },
        "grid": {
            "hidden_dim": [128, 256],
            "num_layers": [1, 2],
            "gru_dropout": [0.0, 0.2],
            "lr": [1e-3],
        },
    },
    "EXP-03_TCN_Neck": {
        "components": {
            "neck": lambda params, in_dim: TCNNeck(
                in_dim=in_dim,
                num_channels=params["num_channels"],
                kernel_size=params["kernel_size"],
                dropout=params["tcn_dropout"],
            ),
            "head": lambda params, in_dim, num_classes: MeanPoolingClassifierHead(
                in_dim=params["num_channels"][-1], num_classes=num_classes, dropout=0.3
            ),
            "loss": lambda params, weights: nn.CrossEntropyLoss(weight=weights),
        },
        "grid": {
            "kernel_size": [3, 5],
            "tcn_dropout": [0.1, 0.2],
            "num_channels": [[128, 256], [256, 512], [128, 256, 512]],
            "lr": [1e-3, 3e-4],
        },
    },
    "EXP-04_Attention_Head": {
        "components": {
            "neck": lambda params, in_dim: TCNNeck(
                in_dim=in_dim,
                num_channels=params["num_channels"],
                kernel_size=3,
                dropout=0.2,
            ),
            "head": lambda params, in_dim, num_classes: AttentionClassifierHead(
                in_dim=params["num_channels"][-1], num_classes=num_classes, dropout=params["dropout"]
            ),
            "loss": lambda params, weights: nn.CrossEntropyLoss(weight=weights),
        },
        "grid": {
            "num_channels": [[128, 256], [256, 512]],
            "dropout": [0.2, 0.4],
            "lr": [1e-3, 3e-4],
        },
    },
    "EXP-05_Hybrid_2D_3D": {
        "components": {
            "neck": lambda params, in_dim: TCNNeck(
                in_dim=in_dim,
                num_channels=params["num_channels"],
                kernel_size=params["kernel_size"],
                dropout=params["dropout"],
            ),
            "head": lambda params, in_dim, num_classes: AttentionClassifierHead(
                in_dim=params["num_channels"][-1], num_classes=num_classes, dropout=0.3
            ),
            "loss": lambda params, weights: nn.CrossEntropyLoss(weight=weights),
        },
        "grid": {
            "num_channels": [[256, 512], [256, 512, 512]],
            "kernel_size": [3, 5],
            "dropout": [0.2, 0.3],
            "lr": [1e-3, 3e-4],
        },
    },
}

def run_phase_experiments(phase, exp_name, config, exp_dir, train_samples, val_samples, device, args):
    train_loader, val_loader, class_weights = prepare_phase_data(train_samples, val_samples, phase)

    phase_dir = os.path.join(exp_dir, phase)
    plots_dir = os.path.join(phase_dir, "plots")
    checkpoints_dir = os.path.join(phase_dir, "checkpoints")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    keys, values = zip(*config["grid"].items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    phase_results = []

    for run_idx, params in enumerate(combinations, 1):
        tb_logger = ExperimentLogger(
            log_dir=os.path.join(phase_dir, "tb_runs"),
            experiment_name=exp_name,
            run_id=run_idx,
        )

        neck = config["components"]["neck"](params, in_dim=197)
        head = config["components"]["head"](
            params,
            in_dim=getattr(neck, "out_dim", 197),
            num_classes=PHASE_CONFIGS[phase]["num_classes"],
        )
        criterion = config["components"]["loss"](params, class_weights.to(device))
        model = ARPipeline(backbone=None, neck=neck, head=head)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 1e-3))

        ckpt_path = os.path.join(checkpoints_dir, f"{exp_name}_run{run_idx}_{phase}_best.pt")
        plot_path = os.path.join(plots_dir, f"{exp_name}_run{run_idx}_{phase}_cm.png")

        trainer = Trainer(model, optimizer, criterion, device)
        metrics = trainer.fit(
            train_loader, val_loader, phase, args.epochs, args.patience,
            tb_logger, ckpt_path, plot_path, desc=f"[{phase.upper()} | Run {run_idx}/{len(combinations)}]"
        )
        tb_logger.close()

        phase_results.append({
            "phase": phase,
            "experiment_name": exp_name,
            "run_id": run_idx,
            "hyperparameters": str(params),
            "checkpoint_path": ckpt_path,
            "confusion_matrix_path": plot_path,
            **metrics,
        })

    return phase_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"{args.experiment}_{timestamp}")
    csv_dir = os.path.join(exp_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    train_samples = FeatureLoader.load("features/train_3d_tracker.pt", expected_dim=197)
    val_samples = FeatureLoader.load("features/val_3d_tracker.pt", expected_dim=197)

    all_results = []
    for phase in ["p1", "p2", "p3"]:
        results = run_phase_experiments(phase, args.experiment, EXPERIMENTS[args.experiment], exp_dir, train_samples, val_samples, device, args)
        pd.DataFrame(results).to_csv(os.path.join(csv_dir, f"results_{phase}.csv"), index=False)
        all_results.extend(results)

    pd.DataFrame(all_results).to_csv(os.path.join(exp_dir, "overall_results.csv"), index=False)