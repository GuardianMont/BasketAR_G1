"""Train the Phase 1 cascade/oracle metrics and flat baseline on X3D-L features."""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data.dataset import ActionRecognitionDataset
from data.experiment_config import ExperimentConfig
from data.samplers import BucketSampler
from data.temporal_sampling import LegacyUniformSampler
from data.utils import compute_class_weights, pad_and_collate_video_batch
from models.heads import AttentionClassifierHead, MeanPoolingClassifierHead
from models.necks import BiGRUNeck, TCNNeck
from models.pipeline import ARPipeline
from utils.feature_loader import load_experiment_feature_pair


PHASES = {
    "flat": {"classes": 9, "target": "label_orig", "shot_only": False},
    "p1": {"classes": 3, "target": "label_p1", "shot_only": False},
    "p2": {"classes": 3, "target": "label_p2", "shot_only": True},
    "p3": {"classes": 2, "target": "label_p3", "shot_only": True},
}
ARCHITECTURES = ("bigru_mean", "tcn_attention")


def baseline_config():
    return ExperimentConfig(
        sampler=LegacyUniformSampler(16).config(),
        spatial_mode="letterbox",
        spatial_size=312,
        color="rgb",
        backbone="x3d_l",
        tracker="none",
        extra={"transform": "none"},
    )



def make_model(architecture: str, feature_dim: int, num_classes: int):
    if architecture == "bigru_mean":
        neck = BiGRUNeck(in_dim=feature_dim, hidden_dim=128, num_layers=1, dropout=0.0)
        head = MeanPoolingClassifierHead(in_dim=256, num_classes=num_classes, dropout=0.0)
    elif architecture == "tcn_attention":
        neck = TCNNeck(
            in_dim=feature_dim, num_channels=[128, 256], kernel_size=3, dropout=0.2
        )
        head = AttentionClassifierHead(in_dim=256, num_classes=num_classes, dropout=0.2)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return ARPipeline(backbone=None, neck=neck, head=head)


def make_loaders(train_samples, val_samples, phase: str, batch_size: int):
    phase_config = PHASES[phase]
    if phase_config["shot_only"]:
        train_samples = [s for s in train_samples if 1 <= int(s["label"]) <= 6]
        val_samples = [s for s in val_samples if 1 <= int(s["label"]) <= 6]
    if not train_samples or not val_samples:
        raise ValueError(f"No data available for phase {phase}.")

    train_dataset = ActionRecognitionDataset(train_samples, is_feature_dataset=True)
    val_dataset = ActionRecognitionDataset(val_samples, is_feature_dataset=True)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BucketSampler(
            [int(s["data"].shape[0]) for s in train_samples], batch_size=batch_size
        ),
        collate_fn=pad_and_collate_video_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_and_collate_video_batch,
    )
    label_key = phase_config["target"]
    class_weights = compute_class_weights(
        [int(train_dataset[i][label_key]) for i in range(len(train_dataset))],
        num_classes=phase_config["classes"],
    )
    return train_loader, val_loader, class_weights, label_key


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(
    train_loader,
    val_loader,
    label_key: str,
    feature_dim: int,
    phase: str,
    architecture: str,
    learning_rate: float,
    seed: int,
    epochs: int,
    patience: int,
    class_weights: torch.Tensor,
    output_dir: str,
    experiment_id: str,
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(architecture, feature_dim, PHASES[phase]["classes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_f1, best_epoch, stale = -1.0, 0, 0
    best_state = None
    best_truth, best_predictions = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            lengths = batch["lengths"]
            labels = batch[label_key].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths=lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        truth, predictions = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                lengths = batch["lengths"]
                labels = batch[label_key].cpu().numpy()
                logits = model(x, lengths=lengths)
                predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                truth.extend(labels.tolist())
        val_f1 = f1_score(truth, predictions, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_truth, best_predictions = truth, predictions
        else:
            stale += 1
        if stale >= patience:
            break

    model_key = f"{architecture}__{phase}__lr{learning_rate:g}__seed{seed}"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{model_key}.pt")
    torch.save(
        {"model_state_dict": best_state, "experiment_id": experiment_id},
        checkpoint_path,
    )
    cm = confusion_matrix(
        best_truth,
        best_predictions,
        labels=list(range(PHASES[phase]["classes"])),
    )
    np.savetxt(os.path.join(output_dir, f"{model_key}__confusion.csv"), cm, fmt="%d", delimiter=",")
    return {
        "phase": phase,
        "architecture": architecture,
        "learning_rate": learning_rate,
        "seed": seed,
        "best_epoch": best_epoch,
        "val_accuracy": accuracy_score(best_truth, best_predictions),
        "val_macro_f1": best_f1,
        "checkpoint": checkpoint_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", default=None)
    parser.add_argument("--val-features", default=None)
    parser.add_argument("--expected-experiment-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--phases", nargs="+", choices=tuple(PHASES), default=list(PHASES))
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3])
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = baseline_config()
    train_path = args.train_features or f"features/{config.feature_filename('train')}"
    val_path = args.val_features or f"features/{config.feature_filename('val')}"
    default_experiment_id = (
        config.feature_id()
        if not args.train_features and not args.val_features and not args.expected_experiment_id
        else None
    )
    train_payload, val_payload, experiment_id, experiment_config = load_experiment_feature_pair(
        train_path,
        val_path,
        default_experiment_id=default_experiment_id,
        expected_experiment_id=args.expected_experiment_id,
    )
    feature_dim = int(train_payload["feature_dim"])
    train_samples = train_payload["samples"]
    val_samples = val_payload["samples"]
    if experiment_config is None and experiment_id == config.feature_id():
        experiment_config = config.to_dict()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_parent = args.output_dir or "experiments"
    output_dir = os.path.join(output_parent, f"PHASE1_{experiment_id}_{timestamp}")
    if args.num_runs < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("num-runs, epochs, and patience must all be positive.")
    if any(lr <= 0 for lr in args.learning_rates):
        raise ValueError("All learning rates must be positive.")
    os.makedirs(output_dir, exist_ok=True)
    run_config = {
        "feature_config": experiment_config,
        "experiment_config": experiment_config,
        "experiment_id": experiment_id,
        "feature_experiment_id": experiment_id,
        "feature_dim": feature_dim,
        "train_features": train_path,
        "validation_features": val_path,
        "architectures": {
            "bigru_mean": {"neck": "BiGRU", "hidden_dim": 128, "layers": 1, "head": "mean"},
            "tcn_attention": {"neck": "TCN", "channels": [128, 256], "kernel_size": 3, "head": "attention"},
        },
        "phases": args.phases,
        "learning_rates": args.learning_rates,
        "num_runs": args.num_runs,
        "base_seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "validation_selection_metric": "macro_f1",
        "test_split_used": False,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    results = []
    for architecture in args.architectures:
        for phase in args.phases:
            train_loader, val_loader, weights, label_key = make_loaders(
                train_samples, val_samples, phase, args.batch_size
            )
            for learning_rate in args.learning_rates:
                for run_idx in range(args.num_runs):
                    seed = args.seed + run_idx
                    print(
                        f"[RUN] arch={architecture} phase={phase} lr={learning_rate:g} "
                        f"seed={seed} device={'cuda' if torch.cuda.is_available() else 'cpu'}"
                    )
                    result = run_one(
                        train_loader,
                        val_loader,
                        label_key,
                        feature_dim,
                        phase,
                        architecture,
                        learning_rate,
                        seed,
                        args.epochs,
                        args.patience,
                        weights,
                        output_dir,
                        experiment_id,
                    )
                    result["feature_experiment_id"] = experiment_id
                    results.append(result)

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "runs.csv")
    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = defaultdict(list)
    for result in results:
        key = (result["architecture"], result["phase"], result["learning_rate"])
        summary[key].append(result)
    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fields = ["architecture", "phase", "learning_rate", "runs", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (architecture, phase, learning_rate), rows in summary.items():
            accuracies = np.asarray([r["val_accuracy"] for r in rows], dtype=float)
            f1s = np.asarray([r["val_macro_f1"] for r in rows], dtype=float)
            writer.writerow({
                "architecture": architecture,
                "phase": phase,
                "learning_rate": learning_rate,
                "runs": len(rows),
                "accuracy_mean": accuracies.mean(),
                "accuracy_std": accuracies.std(ddof=1) if len(rows) > 1 else 0.0,
                "macro_f1_mean": f1s.mean(),
                "macro_f1_std": f1s.std(ddof=1) if len(rows) > 1 else 0.0,
            })
    print(f"[DONE] Fase 1 risultati: {output_dir}")
    print("[INFO] p2/p3 sono addestrati e valutati solo sui tiri con etichetta reale (p1-oracle).")
    print("[INFO] Il test split non viene caricato: resta intoccato.")
    print(
        "[ATTENZIONE] Il p1 a 3 classi accorpa idle e non-gioco in 'altro': "
        "non consente una cascata end-to-end valida a 9 classi. La metrica 9-classi "
        "viene quindi fornita dal modello flat."
    )


if __name__ == "__main__":
    main()
