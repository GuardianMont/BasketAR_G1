import argparse
import os
import statistics
import yaml
from typing import Dict, Any
import torch

from scripts.experiment_generator import ExperimentGenerator

from data.phase_manager import PHASE_CONFIGS, prepare_phase_data
from utils.feature_loader import FeatureLoader
from models.pipeline import ARPipeline
from models.trainer import Trainer
from models.heads import HeadType, HeadFactory
from models.necks import NeckType, NeckFactory
from losses import LossType, LossFactory
from utils.logger import ExperimentLogger
from utils.metrics_analyzer import save_run_record, analyze_grid_results
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Action Recognition grid search.")
    parser.add_argument("--config_path", type=str, required=True, help="Config file path")
    parser.add_argument("--grid_name", type=str, required=True, help="Grid name in the config file")
    parser.add_argument("--global_config", type=str, default="global_train_params")
    parser.add_argument("--metadata", type=str, default="metadata")
    parser.add_argument("--experiment_name", type=str, default=None)

    parser.add_argument(
        "--report_only",
        action="store_true",
        help="If set, only runs reporting on grid_results.csv and exits without training.",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help="Optional explicit path to grid_results.csv for reporting mode.",
    )
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_yaml_config(args, yaml_object):
    if args.metadata not in yaml_object:
        raise KeyError(f"Metadata '{args.metadata}' not found in YAML file.")
    if args.global_config not in yaml_object:
        raise KeyError(f"Global train params section '{args.global_config}' not found in YAML file.")
    if args.grid_name not in yaml_object:
        raise KeyError(f"Grid search section '{args.grid_name}' not found in YAML file.")


def main():
    args = parse_args()
    
    full_yaml = load_yaml_config(args.config_path)
    parse_yaml_config(args, full_yaml)

    experiment_name = args.experiment_name if args.experiment_name else args.grid_name
    
    metadata:Dict[str, str|int] = full_yaml[args.metadata]
    global_train_params:Dict[str, Any] = full_yaml[args.global_config]
    grid_spec = full_yaml[args.grid_name]

    set_seed(metadata.get("base_seed", 42))
    
    training_times = global_train_params.get("training_times", 1)
    
    phases = metadata["phases"]

    grid_results_dir = os.path.join(metadata["root_dir"], experiment_name)
    master_csv_path = os.path.join(grid_results_dir, "grid_results.csv")

    if args.report_only:
        print(f"[INFO] Running in REPORT ONLY mode using results file: {master_csv_path}")
        analyze_grid_results(
            grid_dir=os.path.dirname(master_csv_path) if args.results_csv else grid_results_dir,
            phases=phases,
            target_metric="val_macro_f1_mean",
            mode="max",
        )
        return

    generator = ExperimentGenerator(grid_spec)
    experiments = generator.generate()

    total_fit_runs = len(phases) * len(experiments) * training_times
    run_counter = 1

    print(
        f"[INFO] Grid search start: {len(phases)} phases x {len(experiments)} configs "
        f"x {training_times} repetitions = {total_fit_runs} total fit runs."
    )

    for phase in phases:
        print(f"\n==================== Training: {phase.upper()} ====================")

        raw_train_samples = FeatureLoader.load(
            metadata["train_embeddings_path"],
            expected_dim=metadata["backbone_embed_dim"],
            expected_experiment_id=None,
        )
        raw_val_samples = FeatureLoader.load(
            metadata["val_embeddings_path"],
            expected_dim=metadata["backbone_embed_dim"],
            expected_experiment_id=None,
        )

        train_loader, val_loader, class_weights = prepare_phase_data(
            raw_train_samples, raw_val_samples, phase=phase, batch_size=global_train_params.get('batch_size', 8)
        )

        class_weights = class_weights.to(global_train_params["device"])
        
        for exp_idx, exp in enumerate(experiments, start=1):
            # Accumulators to store metrics across repeat runs
            val_losses = []
            val_accuracies = []
            val_macro_f1s = []
            best_epochs = []
            
            best_repeat_f1 = -1.0
            best_ckpt_path = ""

            for rep_idx in range(1, training_times + 1):
                exp_name = f"{metadata['train_embeddings_path'].split('/')[-1].replace('.pt', '')}_{phase}_exp{exp_idx}_rep{rep_idx}"
                
                tb_logger = ExperimentLogger(
                    log_dir=os.path.join(metadata['root_dir'], experiment_name, phase, "tb_runs"),
                    experiment_name=exp_name,
                    run_id=f"{exp_idx}_{rep_idx}",
                )
                
                neck, neck_out_dim = NeckFactory.create(
                    neck_type=NeckType(exp["neck_type"]),
                    kwargs=exp["neck_kwargs"],
                    in_dim=metadata["backbone_embed_dim"]
                )
                
                head = HeadFactory.create(
                    head_type=HeadType(exp["head_type"]),
                    kwargs=exp["head_kwargs"],
                    in_dim=neck_out_dim,
                    num_classes=PHASE_CONFIGS[phase]["num_classes"]
                )

                loss = LossFactory.create(
                    loss_type=LossType(exp["loss_type"]),
                    weights=class_weights,
                    kwargs=exp["loss_kwargs"],
                )
                
                model = ARPipeline(backbone=None, neck=neck, head=head)
                
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=exp["lr"],
                    weight_decay=exp["weight_decay"]
                )

                # NOTE: move in config
                """
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=global_train_params["epochs"],
                    eta_min=1e-6
                )"""

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",        
                    factor=0.5,        
                    patience=3,        
                    threshold=1e-3,    
                    cooldown=1,        
                    min_lr=1e-6,       
                )
                
                trainer = Trainer(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=loss,
                    device=global_train_params["device"],
                )

                ckpt_path = f"{metadata['root_dir']}/{experiment_name}/{phase}/{exp['neck_type']}_exp_{exp_idx}_rep_{rep_idx}.pt"
                plot_path = f"{metadata['root_dir']}/{experiment_name}/{phase}/{exp['neck_type']}_exp_{exp_idx}_rep_{rep_idx}.png"

                metrics = trainer.fit(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    phase=phase,
                    epochs=global_train_params["epochs"],
                    patience=global_train_params["patience"],
                    tb_logger=tb_logger,
                    checkpoint_path=ckpt_path,
                    plot_path=plot_path,
                    desc=f"[{phase.upper()} | Exp {exp_idx}/{len(experiments)} | Rep {rep_idx}/{training_times} (Run {run_counter}/{total_fit_runs})]",
                )

                val_loss = metrics["val_loss"]
                val_acc = metrics["val_accuracy"]
                val_f1 = metrics["val_macro_f1"]
                best_ep = metrics["best_epoch"]

                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                val_macro_f1s.append(val_f1)
                best_epochs.append(best_ep)

                if val_f1 > best_repeat_f1:
                    best_repeat_f1 = val_f1
                    best_ckpt_path = ckpt_path

                run_counter += 1

            # Compute mean and standard deviation across all repetitions for the configuration
            val_loss_mean = statistics.mean(val_losses)
            val_loss_std = statistics.stdev(val_losses) if len(val_losses) > 1 else 0.0

            val_acc_mean = statistics.mean(val_accuracies)
            val_acc_std = statistics.stdev(val_accuracies) if len(val_accuracies) > 1 else 0.0

            val_f1_mean = statistics.mean(val_macro_f1s)
            val_f1_std = statistics.stdev(val_macro_f1s) if len(val_macro_f1s) > 1 else 0.0

            best_epoch_mean = statistics.mean(best_epochs)
            best_epoch_std = statistics.stdev(best_epochs) if len(best_epochs) > 1 else 0.0

            record = {
                "phase": phase,
                "exp_idx": exp_idx,
                "exp_name": f"{metadata['train_embeddings_path'].split('/')[-1].replace('.pt', '')}_{phase}_exp{exp_idx}",
                "neck_type": exp["neck_type"],
                "neck_kwargs": str(exp["neck_kwargs"]),
                "head_type": exp["head_type"],
                "head_kwargs": str(exp["head_kwargs"]),
                "loss_type": exp["loss_type"],
                "lr": exp["lr"],
                "weight_decay": exp["weight_decay"],
                "training_times": training_times,
                "best_epoch_mean": best_epoch_mean,
                "best_epoch_std": best_epoch_std,
                "val_loss_mean": val_loss_mean,
                "val_loss_std": val_loss_std,
                "val_accuracy_mean": val_acc_mean,
                "val_accuracy_std": val_acc_std,
                "val_macro_f1_mean": val_f1_mean,
                "val_macro_f1_std": val_f1_std,
                "best_checkpoint_path": best_ckpt_path
            }
            
            save_run_record(master_csv_path, record)

    print("\n[INFO] Grid search completed. Analyzing averaged results...")
    analyze_grid_results(
        grid_dir=grid_results_dir, 
        phases=phases, 
        target_metric="val_macro_f1_mean", 
        mode="max"
    )


if __name__ == "__main__":
    main()