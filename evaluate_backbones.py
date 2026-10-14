import argparse
import csv
import os
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from utils.feature_loader import FeatureLoader
from utils.plotting import save_confusion_matrix_plot, PHASE_CLASS_NAMES, CLASS_NAMES
from data.phase_manager import prepare_phase_data, PHASE_CONFIGS, ActionRecognitionDataset, pad_and_collate_video_batch
from models.probing import MLPProber, KNNProber, ProbingEvaluator, BaseProber


def extract_pooled_features_from_loader(
    data_loader: DataLoader, 
    phase: str, 
    mode: str = "mean"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterates over the DataLoader batches, applies temporal pooling on-the-fly, 
    and returns the concatenated NumPy arrays."""
    X_list, y_list = [], []

    target_key = "label_orig" if phase == "flat" else f"label_{phase}"
    for batch in data_loader:
        data = batch["x"] 
        labels = batch[target_key]
        
        if mode == "mean":
            pooled = torch.mean(data, dim=1)
        elif mode == "max":
            pooled = torch.max(data, dim=1).values
        else:
            raise ValueError(f"Modalità di pooling non supportata: {mode}")

        X_list.append(pooled.detach().cpu().numpy())
        y_list.append(labels.detach().cpu().numpy())

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    return X, y


def evaluate_single_phase(
    phase: str, 
    train_samples: List[Dict[str, Any]], 
    test_samples: List[Dict[str, Any]], 
    pooling_mode: str,
    times: int = 1
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Performs probing evaluation for a single phase (p1, p2, p3, or flat).
    "flat mode" means we evaluate on the entire dataset without filtering by phase."""
    if phase == "flat":
        num_classes = len(CLASS_NAMES)
        
        train_dataset = ActionRecognitionDataset(samples=train_samples, is_feature_dataset=True)
        val_dataset = ActionRecognitionDataset(samples=test_samples, is_feature_dataset=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=pad_and_collate_video_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=pad_and_collate_video_batch,
        )
    else:
        p_cfg = PHASE_CONFIGS[phase]
        num_classes = p_cfg["num_classes"]

        train_loader, val_loader, _ = prepare_phase_data(
            train_samples=train_samples,
            val_samples=test_samples,
            phase=phase,
        )  
    
    X_train, y_train = extract_pooled_features_from_loader(train_loader, phase=phase, mode=pooling_mode)
    X_test, y_test   = extract_pooled_features_from_loader(val_loader, phase=phase, mode=pooling_mode)

    in_features = X_train.shape[1]

    probing_suite: List[BaseProber] = [
        MLPProber(in_features=in_features, num_classes=num_classes, num_layers=1),
        MLPProber(in_features=in_features, num_classes=num_classes, num_layers=2, hidden_dim=256),
        MLPProber(in_features=in_features, num_classes=num_classes, num_layers=3, hidden_dim=256),
        KNNProber(k=3, metric="cosine")
    ]

    evaluator = ProbingEvaluator(probers=probing_suite)
    results = evaluator.evaluate(X_train, y_train, X_test, y_test, times=times)

    return results

import json
def save_experiment_results(
    all_results: Dict[str, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]], 
    metadata: Dict[str, Any],
    base_dir: str = "experiments/backbone_evaluation"
):
    """Saves experiment results: CSV with mean and std_dev, plus aggregated confusion matrices."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"results_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    metadata["timestamp"] = timestamp
    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(metadata_path, mode="w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "prober_name", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"])

        for phase, probers_res in all_results.items():
            phase_class_names = CLASS_NAMES if phase == "flat" else PHASE_CLASS_NAMES.get(phase)

            for prober_name, runs in probers_res.items():
                accs = [accuracy_score(y_te, preds) for y_te, preds in runs]
                f1s = [f1_score(y_te, preds, average="macro") for y_te, preds in runs]

                acc_mean, acc_std = np.mean(accs), np.std(accs)
                f1_mean, f1_std = np.mean(f1s), np.std(f1s)

                writer.writerow([
                    phase, prober_name, 
                    f"{acc_mean:.6f}", f"{acc_std:.6f}", 
                    f"{f1_mean:.6f}", f"{f1_std:.6f}"
                ])

                all_y_te = np.concatenate([y_te for y_te, _ in runs])
                all_preds = np.concatenate([preds for _, preds in runs])

                clean_prober_name = prober_name.replace(" ", "_").replace("/", "_")
                cm_save_path = os.path.join(run_dir, f"cm_{phase}_{clean_prober_name}.png")
                
                save_confusion_matrix_plot(
                    y_true=all_y_te, 
                    y_pred=all_preds, 
                    save_path=cm_save_path, 
                    class_names=phase_class_names
                )

    print(f"\n[INFO] Report and metadata saved successfully in: {run_dir}")


# Note for future you:
# This architecture doesn't allow to report train-validation loss curves since:
# 1. The validation split is used as test
# 2. The entire probing architecture should be re-designed to log train-validation loss curves 
#       but only for epoch-based probers (like MLP). Non-epoch-based probers
def main():
    parser = argparse.ArgumentParser(description="Valutazione Probing delle Feature Estratte da Backbone")
    parser.add_argument("--train_path", type=str, default="features/train/train__x3dl__trackernone__edgedense16s100__letterbox312__rgb__4cacfaa4.pt")
    parser.add_argument("--test_path", type=str, default="features/val/val__x3dl__trackernone__edgedense16s100__letterbox312__rgb__4cacfaa4.pt")
    parser.add_argument("--expected_dim", type=int, default=432)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--times", type=int, default=5, help="Numero di esecuzioni indipendenti per ciascun prober")
    parser.add_argument(
        "--phase", 
        type=str, 
        default="all", 
        choices=["p1", "p2", "p3", "flat", "all_phases", "all"],
        help="Alternatives: p1/p2/p3, flat (no phase filtering), all_phases (p1, p2, p3), all (p1, p2, p3, flat)"
    )
    args = parser.parse_args()

    train_samples = FeatureLoader.load(args.train_path, expected_dim=args.expected_dim)
    test_samples  = FeatureLoader.load(args.test_path , expected_dim=args.expected_dim)

    if args.phase == "all":
        phases_to_evaluate = ["p1", "p2", "p3", "flat"]
    elif args.phase == "all_phases":
        phases_to_evaluate = ["p1", "p2", "p3"]
    else:
        phases_to_evaluate = [args.phase]

    all_experiment_results = {}

    for phase in phases_to_evaluate:
        results = evaluate_single_phase(
            phase=phase,
            train_samples=train_samples,
            test_samples=test_samples,
            pooling_mode=args.pooling,
            times=args.times
        )
        all_experiment_results[phase] = results

        print("-" * 50)
        print(f"FASE: {phase.upper()} (Runs: {args.times})")
        for prober_name, runs in results.items():
            accs = [accuracy_score(y_te, preds) for y_te, preds in runs]
            f1s = [f1_score(y_te, preds, average="macro") for y_te, preds in runs]
            
            print(f"[{prober_name}]")
            print(f"  ├─ Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            print(f"  └─ F1-Macro: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    feature_name = os.path.basename(args.train_path)

    metadata = {
        "feature_name": feature_name,
        "train_path": args.train_path,
        "test_path": args.test_path,
        "expected_dim": args.expected_dim,
        "pooling": args.pooling,
        "phase_evaluated": args.phase,
        "times": args.times,
    }

    save_experiment_results(all_experiment_results, metadata=metadata)


if __name__ == "__main__":
    main()