import os
import pandas as pd
from typing import Dict, Any, List


def save_run_record(csv_path: str, record: Dict[str, Any]):
    """Saves a single experiment run summary record into the master CSV file (append mode)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_row = pd.DataFrame([record])
    header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", index=False, header=header)


def analyze_grid_results(
    grid_dir: str, 
    phases: List[str], 
    target_metric: str = "val_macro_f1_mean", 
    mode: str = "max"
) -> Dict[str, Dict[str, Any]]:
    """
    Scans the aggregated grid search CSV results and finds the optimal configuration
    per hierarchical phase based on averaged metrics across repeat runs.

    Args:
        grid_dir: Directory containing the grid search results (e.g., 'experiments/neck_grid_search')
        phases: List of hierarchical phases (e.g., ['p1', 'p2', 'p3'])
        target_metric: Target aggregated metric column name (default: 'val_macro_f1_mean')
        mode: 'max' to maximize performance metrics, 'min' for loss metrics
    """
    csv_path = os.path.join(grid_dir, "grid_results.csv")
    if not os.path.exists(csv_path):
        print(f"[WARNING] Results CSV file not found at: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    best_configs = {}

    print("\n" + "=" * 75)
    print(f"   BEST CONFIGURATIONS REPORT (Target: {target_metric.upper()} | Mode: {mode.upper()})")
    print("=" * 75)

    for phase in phases:
        phase_df = df[df["phase"] == phase]
        if phase_df.empty:
            print(f"\n[PHASE {phase.upper()}] No data found.")
            continue

        # Select the best experiment index based on the mean target metric
        if mode == "max":
            best_idx = phase_df[target_metric].idxmax()
        else:
            best_idx = phase_df[target_metric].idxmin()

        best_row = phase_df.loc[best_idx].to_dict()
        best_configs[phase] = best_row

        print(f"\n>>> HIERARCHICAL PHASE: {phase.upper()}")
        print(f"  - Exp Config Index : Config #{best_row['exp_idx']}")
        print(f"  - Neck             : {best_row['neck_type']} (kwargs: {best_row['neck_kwargs']})")
        print(f"  - Head             : {best_row['head_type']}")
        print(f"  - Loss             : {best_row['loss_type']}")
        print(f"  - LR / WD          : {best_row['lr']} / {best_row['weight_decay']}")
        print(f"  - Repetitions      : {best_row.get('training_times', 'N/A')} runs (Base Seed: {best_row.get('base_seed', 'N/A')})")
        
        # Display aggregated metrics with std dev
        f1_mean = best_row.get("val_macro_f1_mean", 0.0)
        f1_std = best_row.get("val_macro_f1_std", 0.0)
        acc_mean = best_row.get("val_accuracy_mean", 0.0)
        acc_std = best_row.get("val_accuracy_std", 0.0)
        loss_mean = best_row.get("val_loss_mean", float("inf"))
        loss_std = best_row.get("val_loss_std", 0.0)
        epoch_mean = best_row.get("best_epoch_mean", 0.0)
        epoch_std = best_row.get("best_epoch_std", 0.0)

        print(f"  - Best Epoch (Avg) : {epoch_mean:.1f} ± {epoch_std:.1f}")
        print(f"  - Val Macro F1     : {f1_mean:.4f} ± {f1_std:.4f}  <-- BEST")
        print(f"  - Val Accuracy     : {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"  - Val Loss         : {loss_mean:.4f} ± {loss_std:.4f}")
        print(f"  - Best Checkpoint  : {best_row.get('best_checkpoint_path', 'N/A')}")

    print("=" * 75 + "\n")
    return best_configs