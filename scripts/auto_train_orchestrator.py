import ast
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch

# Aggiunta root del progetto al path per gli import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.heads import AttentionClassifierHead, MeanPoolingClassifierHead
from models.necks import BiGRUNeck, TCNNeck
from train_arch_ablation import (
    EXPERIMENTS,
    FIXED_DATA_CONFIG,
    build_dataloaders_for_phase,
    extract_best_configuration_per_phase,
    load_preextracted_features,
    train_single_phase,
)


def setup_logger(log_file):
    logger = logging.getLogger("Orchestrator")
    logger.setLevel(logging.INFO)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def update_global_best(
    global_bests, phase, exp_name, best_cfg, df_results, arch_desc
):
    """Aggiorna la migliore architettura mediamente vincente per la fase specificata."""
    mean_f1 = float(best_cfg["mean_val_macro_f1"])

    if (
        phase not in global_bests
        or mean_f1 > global_bests[phase]["mean_val_macro_f1"]
    ):
        best_hp = (
            ast.literal_eval(str(best_cfg["best_hyperparameters"]))
            if isinstance(best_cfg["best_hyperparameters"], str)
            else best_cfg["best_hyperparameters"]
        )

        param_col = next(
            (
                c
                for c in [
                    "hyperparameters",
                    "params",
                    "best_hyperparameters",
                    "config",
                ]
                if c in df_results.columns
            ),
            None,
        )

        if param_col:
            matched = df_results[
                df_results[param_col].apply(
                    lambda x: (
                        ast.literal_eval(str(x)) if isinstance(x, str) else x
                    )
                    == best_hp
                )
            ]
        else:
            matched = df_results

        if not matched.empty and "val_macro_f1" in matched.columns:
            best_run_row = matched.sort_values(
                by="val_macro_f1", ascending=False
            ).iloc[0]
            best_ckpt_path = best_run_row.get(
                "checkpoint_path", best_cfg.get("best_checkpoint_path", "N/A")
            )
            best_run_f1 = float(best_run_row["val_macro_f1"])
        else:
            best_ckpt_path = best_cfg.get("best_checkpoint_path", "N/A")
            best_run_f1 = mean_f1

        global_bests[phase] = {
            "experiment_name": exp_name,
            "architecture_description": arch_desc,
            "hyperparameters": best_hp,
            "mean_val_macro_f1": mean_f1,
            "best_run_val_macro_f1": best_run_f1,
            "best_checkpoint_path": str(best_ckpt_path),
        }


def save_experiment_csv(df_results, exp_dir, exp_name, phase):
    """Salva il DataFrame dei risultati dettagliati in formato CSV."""
    csv_name = f"results_{exp_name}_{phase}.csv"
    csv_path = os.path.join(exp_dir, csv_name)
    df_results.to_csv(csv_path, index=False)
    return csv_path


def run_orchestrator(epochs=30, patience=5, num_runs=3):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = os.path.join("experiments", f"ORCHESTRATOR_RUN_{timestamp}")
    os.makedirs(suite_dir, exist_ok=True)

    logger = setup_logger(os.path.join(suite_dir, "orchestrator.log"))
    logger.info("=" * 80)
    logger.info("🏀 Avvio Auto Train Orchestrator per BasketAR 🏀")
    logger.info(f"Suite Directory: {suite_dir}")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Caricamento dataset
    logger.info("Caricamento delle feature fisse...")
    raw_train_samples = load_preextracted_features(
        FIXED_DATA_CONFIG["train_path"], FIXED_DATA_CONFIG["feature_dim"]
    )
    raw_val_samples = load_preextracted_features(
        FIXED_DATA_CONFIG["val_path"], FIXED_DATA_CONFIG["feature_dim"]
    )

    best_params_per_phase = {}
    global_bests = {}

    # =========================================================================
    # STEP 1: ARCH-01 (Baseline - Estrazione Dropout e Weight Decay ottimali)
    # =========================================================================
    exp1_name = "ARCH-01_Identity_Mean"
    logger.info(f"\n[STEP 1] Esecuzione {exp1_name}")
    config1 = EXPERIMENTS[exp1_name]
    exp1_dir = os.path.join(suite_dir, exp1_name)
    os.makedirs(exp1_dir, exist_ok=True)

    for phase in ["p1", "p2", "p3"]:
        logger.info(f"Avvio {exp1_name} per fase {phase}...")
        train_loader, val_loader, class_weights = build_dataloaders_for_phase(
            raw_train_samples, raw_val_samples, phase
        )
        results = train_single_phase(
            phase,
            exp1_name,
            exp1_dir,
            config1,
            train_loader,
            val_loader,
            class_weights,
            device,
            epochs,
            patience,
            num_runs,
        )

        df_res = pd.DataFrame(results)
        csv_path = save_experiment_csv(df_res, exp1_dir, exp1_name, phase)
        logger.info(f"  -> Risultati CSV salvati in: {csv_path}")

        best_cfg = extract_best_configuration_per_phase(df_res)
        best_hp = ast.literal_eval(best_cfg["best_hyperparameters"])

        logger.info(
            f"Miglior configurazione {exp1_name} -> {phase}: {best_hp} (Macro-F1:"
            f" {best_cfg['mean_val_macro_f1']:.4f})"
        )

        # Propaghiamo solo dropout e weight_decay (il learning rate resta libero)
        best_params_per_phase[phase] = {
            "dropout": best_hp.get("dropout", 0.0),
            "weight_decay": best_hp.get("weight_decay", 0.0),
        }

        update_global_best(
            global_bests,
            phase,
            exp1_name,
            best_cfg,
            df_res,
            "Identity Neck + Mean Pooling Head",
        )

    # =========================================================================
    # STEP 2: ARCH-02 (BiGRU) e ARCH-03 (TCN)
    # =========================================================================
    logger.info(
        "\n[STEP 2] Esecuzione ARCH-02 e ARCH-03 iniettando (dropout,"
        " weight_decay) da ARCH-01"
    )

    neck_results = {"p1": {}, "p2": {}, "p3": {}}

    for exp_name in ["ARCH-02_BiGRU_Mean", "ARCH-03_TCN_Mean"]:
        base_config = EXPERIMENTS[exp_name]
        exp_dir = os.path.join(suite_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        for phase in ["p1", "p2", "p3"]:
            logger.info(f"Avvio {exp_name} per fase {phase}...")

            injected_config = dict(base_config)
            injected_grid = dict(base_config["grid"])

            b_drop = best_params_per_phase[phase]["dropout"]
            b_wd = best_params_per_phase[phase]["weight_decay"]

            # Blocco regolarizzazioni apprese in STEP 1
            for k in list(injected_grid.keys()):
                if "dropout" in k:
                    injected_grid[k] = [b_drop]
                    logger.info(f"  -> Injected {k} = {b_drop}")
            injected_grid["weight_decay"] = [b_wd]
            logger.info(f"  -> Injected weight_decay = {b_wd}")

            # Il learning rate NON viene sovrascritto: mantiene la sua griglia originale!

            # Iniezione del dropout di ARCH-01 nel Head di MeanPooling
            if "BiGRU" in exp_name:
                injected_config["components"]["head"] = (
                    lambda params, in_dim, num_classes, drop=b_drop: MeanPoolingClassifierHead(
                        in_dim=params["hidden_dim"] * 2,
                        num_classes=num_classes,
                        dropout=drop,
                    )
                )
            else:
                injected_config["components"]["head"] = (
                    lambda params, in_dim, num_classes, drop=b_drop: MeanPoolingClassifierHead(
                        in_dim=in_dim, num_classes=num_classes, dropout=drop
                    )
                )

            injected_config["grid"] = injected_grid

            train_loader, val_loader, class_weights = (
                build_dataloaders_for_phase(
                    raw_train_samples, raw_val_samples, phase
                )
            )
            results = train_single_phase(
                phase,
                exp_name,
                exp_dir,
                injected_config,
                train_loader,
                val_loader,
                class_weights,
                device,
                epochs,
                patience,
                num_runs,
            )

            df_res = pd.DataFrame(results)
            csv_path = save_experiment_csv(df_res, exp_dir, exp_name, phase)
            logger.info(f"  -> Risultati CSV salvati in: {csv_path}")

            best_cfg = extract_best_configuration_per_phase(df_res)
            logger.info(
                f"Miglior configurazione {exp_name} -> {phase}:"
                f" {best_cfg['best_hyperparameters']} (Macro-F1:"
                f" {best_cfg['mean_val_macro_f1']:.4f})"
            )

            neck_results[phase][exp_name] = best_cfg
            desc = (
                "BiGRU Neck + Mean Pooling Head"
                if "BiGRU" in exp_name
                else "TCN Neck + Mean Pooling Head"
            )
            update_global_best(
                global_bests, phase, exp_name, best_cfg, df_res, desc
            )

    # =========================================================================
    # STEP 3: ARCH-04 (Migliore Neck + Attention Head)
    # =========================================================================
    logger.info(
        "\n[STEP 3] Selezione Vincitore Neck (BiGRU vs TCN) e Avvio ARCH-04"
    )
    exp4_name = "ARCH-04_Hybrid_Attention"
    exp4_dir = os.path.join(suite_dir, exp4_name)
    os.makedirs(exp4_dir, exist_ok=True)

    best_necks_info = {}

    for phase in ["p1", "p2", "p3"]:
        f1_bigru = neck_results[phase]["ARCH-02_BiGRU_Mean"][
            "mean_val_macro_f1"
        ]
        f1_tcn = neck_results[phase]["ARCH-03_TCN_Mean"]["mean_val_macro_f1"]

        if f1_bigru >= f1_tcn:
            winner = "BiGRU"
            best_hp_neck = ast.literal_eval(
                neck_results[phase]["ARCH-02_BiGRU_Mean"][
                    "best_hyperparameters"
                ]
            )
            logger.info(
                f"Fase {phase} -> Vince {winner} (BiGRU: {f1_bigru:.4f} vs TCN:"
                f" {f1_tcn:.4f})"
            )
        else:
            winner = "TCN"
            best_hp_neck = ast.literal_eval(
                neck_results[phase]["ARCH-03_TCN_Mean"]["best_hyperparameters"]
            )
            logger.info(
                f"Fase {phase} -> Vince {winner} (TCN: {f1_tcn:.4f} vs BiGRU:"
                f" {f1_bigru:.4f})"
            )

        best_necks_info[phase] = {"winner": winner, "hp": best_hp_neck}

        def create_neck_fn(winner_type, hp):
            if winner_type == "BiGRU":
                return lambda params, in_dim: BiGRUNeck(
                    in_dim=in_dim,
                    hidden_dim=hp["hidden_dim"],
                    num_layers=hp["num_layers"],
                    dropout=hp.get("dropout", 0.0),
                )
            else:
                return lambda params, in_dim: TCNNeck(
                    in_dim=in_dim,
                    num_channels=hp["num_channels"],
                    kernel_size=hp["kernel_size"],
                    dropout=hp.get("tcn_dropout", 0.0),
                )

        b_wd = best_params_per_phase[phase]["weight_decay"]

        config4 = {
            "description": f"{winner} Neck + Attention Head",
            "components": {
                "neck": create_neck_fn(winner, best_hp_neck),
                "head": lambda params, in_dim, num_classes: AttentionClassifierHead(
                    in_dim=in_dim,
                    num_classes=num_classes,
                    dropout=params["attention_dropout"],
                ),
                "loss": lambda params, weights: torch.nn.CrossEntropyLoss(
                    weight=weights
                ),
            },
            "grid": {
                "attention_dropout": [0.0, 0.2, 0.4],
                "weight_decay": [b_wd],
                "lr": [1e-3, 5e-4, 1e-4],  # Griglia lr autonoma
            },
        }

        logger.info(f"Avvio {exp4_name} per fase {phase}...")
        train_loader, val_loader, class_weights = build_dataloaders_for_phase(
            raw_train_samples, raw_val_samples, phase
        )
        results = train_single_phase(
            phase,
            exp4_name,
            exp4_dir,
            config4,
            train_loader,
            val_loader,
            class_weights,
            device,
            epochs,
            patience,
            num_runs,
        )

        df_res = pd.DataFrame(results)
        csv_path = save_experiment_csv(df_res, exp4_dir, exp4_name, phase)
        logger.info(f"  -> Risultati CSV salvati in: {csv_path}")

        best_cfg = extract_best_configuration_per_phase(df_res)
        logger.info(
            f"Miglior configurazione {exp4_name} -> {phase}:"
            f" {best_cfg['best_hyperparameters']} (Macro-F1:"
            f" {best_cfg['mean_val_macro_f1']:.4f})"
        )

        best_hp_4 = ast.literal_eval(best_cfg["best_hyperparameters"])
        best_necks_info[phase]["attention_dropout"] = best_hp_4[
            "attention_dropout"
        ]
        update_global_best(
            global_bests,
            phase,
            exp4_name,
            best_cfg,
            df_res,
            f"{winner} Neck + Attention Head",
        )

    # =========================================================================
    # STEP 4: ARCH-05 (Focal Loss Tuning)
    # =========================================================================
    exp5_name = "ARCH-05_FocalLoss"
    logger.info(
        f"\n[STEP 4] Esecuzione {exp5_name} (Migliore Architettura + Focal Loss)"
    )
    exp5_dir = os.path.join(suite_dir, exp5_name)
    os.makedirs(exp5_dir, exist_ok=True)

    from losses import FocalLoss

    for phase in ["p1", "p2", "p3"]:
        winner = best_necks_info[phase]["winner"]
        hp_neck = best_necks_info[phase]["hp"]
        att_drop = best_necks_info[phase]["attention_dropout"]
        b_wd = best_params_per_phase[phase]["weight_decay"]

        def create_neck_fn_final(winner_type, hp):
            if winner_type == "BiGRU":
                return lambda params, in_dim: BiGRUNeck(
                    in_dim=in_dim,
                    hidden_dim=hp["hidden_dim"],
                    num_layers=hp["num_layers"],
                    dropout=hp.get("dropout", 0.0),
                )
            else:
                return lambda params, in_dim: TCNNeck(
                    in_dim=in_dim,
                    num_channels=hp["num_channels"],
                    kernel_size=hp["kernel_size"],
                    dropout=hp.get("tcn_dropout", 0.0),
                )

        config5 = {
            "description": f"{winner} Neck + Attention + Focal Loss",
            "components": {
                "neck": create_neck_fn_final(winner, hp_neck),
                "head": lambda params, in_dim, num_classes: AttentionClassifierHead(
                    in_dim=in_dim, num_classes=num_classes, dropout=att_drop
                ),
                "loss": lambda params, weights: FocalLoss(
                    gamma=params["gamma"], alpha=weights
                ),
            },
            "grid": {
                "gamma": [0.0, 1.0, 2.0, 3.0],
                "weight_decay": [b_wd],
                "lr": [1e-3, 5e-4, 1e-4],  # Griglia lr autonoma
            },
        }

        logger.info(f"Avvio {exp5_name} per fase {phase}...")
        train_loader, val_loader, class_weights = build_dataloaders_for_phase(
            raw_train_samples, raw_val_samples, phase
        )
        results = train_single_phase(
            phase,
            exp5_name,
            exp5_dir,
            config5,
            train_loader,
            val_loader,
            class_weights,
            device,
            epochs,
            patience,
            num_runs,
        )

        df_res = pd.DataFrame(results)
        csv_path = save_experiment_csv(df_res, exp5_dir, exp5_name, phase)
        logger.info(f"  -> Risultati CSV salvati in: {csv_path}")

        best_cfg = extract_best_configuration_per_phase(df_res)
        logger.info(
            f"Miglior configurazione FINALE {exp5_name} -> {phase}:"
            f" {best_cfg['best_hyperparameters']} (Macro-F1:"
            f" {best_cfg['mean_val_macro_f1']:.4f})"
        )
        update_global_best(
            global_bests,
            phase,
            exp5_name,
            best_cfg,
            df_res,
            f"{winner} Neck + Attention Head + Focal Loss",
        )

    # =========================================================================
    # RIEPILOGO FINALE
    # =========================================================================
    logger.info("=" * 80)
    logger.info("✅ Orchestrazione Completata! ✅")
    logger.info(f"Tutti i file salvati in: {suite_dir}")
    logger.info("=" * 80)
    logger.info(
        "🏆 MIGLIORI ARCHITETTURE MEDIE E CHECKPOINT VINCENTI PER FASE 🏆"
    )
    logger.info("=" * 80)

    for phase in ["p1", "p2", "p3"]:
        info = global_bests.get(phase, {})
        logger.info(f"\n--- FASE {phase.upper()} ---")
        logger.info(
            f"  • Esperimento Vincente  : {info.get('experiment_name')}"
        )
        logger.info(
            f"  • Descrizione Architet. :"
            f" {info.get('architecture_description')}"
        )
        logger.info(f"  • Iperparametri        : {info.get('hyperparameters')}")
        logger.info(
            "  • Mean Val Macro-F1    :"
            f" {info.get('mean_val_macro_f1', 0.0):.4f}"
        )
        logger.info(
            "  • Best Run Val Macro-F1:"
            f" {info.get('best_run_val_macro_f1', 0.0):.4f}"
        )
        logger.info(
            f"  • Path Checkpoint      : {info.get('best_checkpoint_path')}"
        )

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-runs", type=int, default=5)
    args = parser.parse_args()

    run_orchestrator(
        epochs=args.epochs, patience=args.patience, num_runs=args.num_runs
    )