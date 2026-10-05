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
from utils.plotting import save_confusion_matrix_plot, PHASE_CLASS_NAMES
from data.phase_manager import prepare_phase_data, PHASE_CONFIGS
from models.probing import MLPProber, KNNProber, ProbingEvaluator, BaseProber


def extract_pooled_features_from_loader(
    data_loader: DataLoader, 
    phase: str, 
    mode: str = "mean"
) -> Tuple[np.ndarray, np.ndarray]:
    """Itera sui batch del DataLoader, applica il temporal pooling on-the-fly

    e restituisce le matrici NumPy composte.
    """
    X_list, y_list = [], []
    target_key = f"label_{phase}"

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
    pooling_mode: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Esegue la pipeline di probing per una singola fase (p1, p2 o p3)."""
    p_cfg = PHASE_CONFIGS[phase]

    train_loader, val_loader, _ = prepare_phase_data(
        train_samples=train_samples,
        val_samples=test_samples,
        phase=phase,
    )   
    
    X_train, y_train = extract_pooled_features_from_loader(train_loader, phase=phase, mode=pooling_mode)
    X_test, y_test   = extract_pooled_features_from_loader(val_loader, phase=phase, mode=pooling_mode)

    in_features = X_train.shape[1]
    num_classes = p_cfg["num_classes"]

    probing_suite: List[BaseProber] = [
        MLPProber(in_features=in_features, num_classes=num_classes, num_layers=1),
        MLPProber(in_features=in_features, num_classes=num_classes, num_layers=2, hidden_dim=256),
        MLPProber(in_features=in_features, num_classes=num_classes, num_layers=3, hidden_dim=256),
        KNNProber(k=3, metric="cosine")
    ]

    evaluator = ProbingEvaluator(probers=probing_suite)
    results = evaluator.evaluate(X_train, y_train, X_test, y_test)

    return results


def save_experiment_results(
    all_results: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], 
    base_dir: str = "experiments/backbone_evaluation"
):
    """Salva i risultati dell'esperimento in una cartella timestampata:

    1. results.csv con accuratezza e F1-macro per ciascun prober e fase.
    2. Matrici di confusione normalizzate con le etichette testuali delle classi
       generate tramite utils.plotting.save_confusion_matrix_plot.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"results_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "results.csv")
    
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "prober_name", "accuracy", "macro f1-score"])

        for phase, probers_res in all_results.items():
            # Recupera i nomi testuali specifici per la fase (es. ["passaggio", "tiro", "altro"])
            phase_class_names = PHASE_CLASS_NAMES.get(phase)

            for prober_name, (y_te, preds) in probers_res.items():
                acc = accuracy_score(y_te, preds)
                f1 = f1_score(y_te, preds, average="macro")
                writer.writerow([phase, prober_name, f"{acc:.6f}", f"{f1:.6f}"])

                # Generazione e salvataggio della matrice di confusione tramite utils.plotting
                clean_prober_name = prober_name.replace(" ", "_").replace("/", "_")
                cm_save_path = os.path.join(run_dir, f"cm_{phase}_{clean_prober_name}.png")
                
                save_confusion_matrix_plot(
                    y_true=y_te, 
                    y_pred=preds, 
                    save_path=cm_save_path, 
                    class_names=phase_class_names
                )

    print(f"\n[INFO] Report salvato con successo in: {run_dir}")


# Note for future you:
# This architecture doesn't allow to report train-validation loss curves since:
# 1. The validation split is used as test
# 2. The entire probing architecture should be re-designed to log train-validation loss curves 
#       but only for epoch-based probers (like MLP). Non-epoch-based probers
def main():
    parser = argparse.ArgumentParser(description="Valutazione Probing delle Feature Estratte da Backbone")
    parser.add_argument("--train_path", type=str, default="features/train_3d_L.pt")
    parser.add_argument("--test_path", type=str, default="features/val_3d_L.pt")
    parser.add_argument("--expected_dim", type=int, default=432)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--phase", type=str, default="all", choices=["p1", "p2", "p3", "all"])
    args = parser.parse_args()

    train_samples = FeatureLoader.load(args.train_path, expected_dim=args.expected_dim)
    test_samples  = FeatureLoader.load(args.test_path , expected_dim=args.expected_dim)

    phases_to_evaluate = ["p1", "p2", "p3"] if args.phase == "all" else [args.phase]
    all_experiment_results = {}

    for phase in phases_to_evaluate:
        results = evaluate_single_phase(
            phase=phase,
            train_samples=train_samples,
            test_samples=test_samples,
            pooling_mode=args.pooling
        )
        all_experiment_results[phase] = results

        print("-" * 50)
        print(f"FASE: {phase.upper()}")
        for prober_name, (y_te, preds) in results.items():
            print(f"[{prober_name}]")
            print(f"  ├─ Accuracy: {accuracy_score(y_te, preds):.4f}")
            print(f"  └─ F1-Macro: {f1_score(y_te, preds, average='macro'):.4f}")

    save_experiment_results(all_experiment_results)


if __name__ == "__main__":
    main()