import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import yaml
from typing import Dict, Any, Optional

from data.phase_manager import PHASE_CONFIGS, prepare_phase_data
from utils.feature_loader import FeatureLoader
from models.pipeline import ARPipeline
from models.trainer import Trainer
from models.heads import HeadType, HeadFactory
from models.necks import NeckType, NeckFactory
from losses import LossType, LossFactory

from data.dataset import ActionRecognitionDataset
from data.utils import pad_and_collate_video_batch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

from utils.plotting import save_confusion_matrix_plot, map_to_p2_labels, CLASS_NAMES_NOACTION, CLASS_NAMES_P2


class FeatureSlicerWrapper(nn.Module):
    """Wrapper module that slices visual features on-the-fly 

    if tracking features are disabled for a specific phase.
    """
    def __init__(self, model: nn.Module, target_dim: int):
        super().__init__()
        self.model = model
        self.target_dim = target_dim

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.shape[-1] > self.target_dim:
            x = x[..., :self.target_dim]
        return self.model(x, lengths=lengths)


def parse_args():
    parser = argparse.ArgumentParser(description="Action Recognition Test Evaluation.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to evaluation test configuration file",
    )
    parser.add_argument(
        "--test_name",
        type=str,
        default="best_architectures",
        help="Test configuration section name in the config file (e.g., best_architectures)",
    )
    parser.add_argument(
        "--global_config",
        type=str,
        default="global_params",
        help="Global params section name in the config file (default: global_params)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata",
        help="Metadata section name in the config file (default: metadata)",
    )
    parser.add_argument(
        "--downstream",
        action="store_true",
        help="Enable full downstream hierarchical inference (P1 -> P2, P3)",
    )
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_yaml_config(args, yaml_object):
    if args.metadata not in yaml_object:
        raise KeyError(f"Metadata '{args.metadata}' not found in YAML file.")
    if args.global_config not in yaml_object:
        raise KeyError(f"Global params section '{args.global_config}' not found in YAML file.")
    if args.test_name not in yaml_object:
        raise KeyError(f"Test section '{args.test_name}' not found in YAML file.")


def build_and_load_model(phase: str, arch_config: Dict[str, Any], metadata: Dict[str, Any], device: torch.device):
    """Utility function to instantiate, load weights, and wrap a single model phase."""
    use_tracker = arch_config.get("use_tracker", False)
    visual_dim = metadata.get("visual_embed_dim", 768)
    
    # Calculate effective input dimension based on tracking flag
    in_dim = metadata["backbone_embed_dim"] if use_tracker else visual_dim

    neck, neck_out_dim = NeckFactory.create(
        neck_type=NeckType(arch_config["neck_type"]),
        kwargs=arch_config["neck_kwargs"],
        in_dim=in_dim,
    )

    head = HeadFactory.create(
        head_type=HeadType(arch_config["head_type"]),
        kwargs=arch_config["head_kwargs"],
        in_dim=neck_out_dim,
        num_classes=PHASE_CONFIGS[phase]["num_classes"],
    )

    pipeline = ARPipeline(backbone=None, neck=neck, head=head)
    
    ckpt_path = arch_config["checkpoint_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint for phase '{phase}' not found at: {ckpt_path}")
        
    print(f"[INFO] Loading {phase.upper()} checkpoint (use_tracker={use_tracker}, in_dim={in_dim}): {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state_dict:
        pipeline.load_state_dict(state_dict["model_state_dict"])
    else:
        pipeline.load_state_dict(state_dict)

    # Wrap model to slice tracker features automatically if use_tracker is False
    model = FeatureSlicerWrapper(pipeline, target_dim=in_dim)
    model.to(device)
    model.eval()
    return model

def evaluate_p2_downstream(all_labels: list[int], all_preds: list[int], output_dir: str) -> dict:
    p2_true = map_to_p2_labels(all_labels)
    p2_pred = map_to_p2_labels(all_preds)

    acc = accuracy_score(p2_true, p2_pred)
    f1 = f1_score(p2_true, p2_pred, average="macro", zero_division=0)
    cm = confusion_matrix(p2_true, p2_pred, labels=list(range(len(CLASS_NAMES_P2))))

    plot_path = os.path.join(output_dir, "p2_collapsed_confusion_matrix.png")
    save_confusion_matrix_plot(p2_true, p2_pred, plot_path, class_names=CLASS_NAMES_P2)

    print("\n>>> DOWNSTREAM METRICS (P2 COLLAPSED - IGNORING P3):")
    print(f"  - P2 Collapsed Accuracy : {acc:.4f}")
    print(f"  - P2 Collapsed Macro F1 : {f1:.4f}")
    print("Confusion Matrix (P2 Collapsed):")
    print(cm)

    return {
        "p2_accuracy": round(acc, 4),
        "p2_macro_f1": round(f1, 4),
    }

def evaluate_downstream(args, metadata, global_params, test_architectures, raw_test_samples, output_dir):
    """Executes the full hierarchical downstream inference pipeline."""
    device = torch.device(global_params.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 75)
    print("        STARTING FULL DOWNSTREAM HIERARCHICAL INFERENCE")
    print("=" * 75)

    test_dataset = ActionRecognitionDataset(
        samples=raw_test_samples, 
        is_feature_dataset=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=global_params.get("batch_size", 32),
        shuffle=False,
        collate_fn=pad_and_collate_video_batch,
    )

    models = {}
    for phase in ["p1", "p2", "p3"]:
        if phase not in test_architectures:
            raise KeyError(f"Phase {phase} missing from test configurations. Cannot perform downstream inference.")
        models[phase] = build_and_load_model(phase, test_architectures[phase], metadata, device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[DOWNSTREAM INFERENCE]"):
            x = batch.get("x", batch.get("data")).to(device)
            lengths = batch.get("lengths", batch.get("length"))
            orig_labels = batch.get("label_orig")

            logits_p1 = models["p1"](x, lengths=lengths)
            preds_p1 = torch.argmax(logits_p1, dim=1)

            logits_p2 = models["p2"](x, lengths=lengths)
            preds_p2 = torch.argmax(logits_p2, dim=1)

            logits_p3 = models["p3"](x, lengths=lengths)
            preds_p3 = torch.argmax(logits_p3, dim=1)

            # label reconstruction
            for i in range(len(preds_p1)):
                orig = orig_labels[i].item()
                p1_val = preds_p1[i].item()
                p2_val = preds_p2[i].item()
                p3_val = preds_p3[i].item()

                if p1_val == 0:
                    pred_orig = 0
                elif p1_val == 1:
                    pred_orig = (p2_val * 2) + 1 + p3_val
                else:
                    pred_orig = 7
                
                if orig > 6:
                    orig = 7
                
                all_preds.append(pred_orig)
                all_labels.append(orig)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES_NOACTION))))
    save_confusion_matrix_plot(all_labels, all_preds, 
                               os.path.join(output_dir, 'total_confusion_matrix.png'),
                               CLASS_NAMES_NOACTION)

    print("\n>>> FINAL DOWNSTREAM METRICS:")
    print(f"  - Complete Original Label Accuracy : {acc:.4f}")
    print(f"  - Complete Original Label Macro F1 : {f1:.4f}  <-- FINAL SYSTEM SCORE")
    print(cm)

    p2_metrics = evaluate_p2_downstream(all_labels, all_preds, output_dir)
    results_csv_path = os.path.join(output_dir, "downstream_results.csv")
    df_results = pd.DataFrame([{
        "inference_type": "hierarchical_downstream",
        "full_accuracy": round(acc, 4),
        "full_macro_f1": round(f1, 4),
        "p2_collapsed_accuracy": p2_metrics["p2_accuracy"],
        "p2_collapsed_macro_f1": p2_metrics["p2_macro_f1"],
    }])
    df_results.to_csv(results_csv_path, index=False)
    print("=" * 75)
    print(f"Downstream evaluation completed. Results saved to: {results_csv_path}\n")


def main():
    args = parse_args()
    
    full_yaml = load_yaml_config(args.config_path)
    parse_yaml_config(args, full_yaml)

    metadata = full_yaml[args.metadata]
    global_params = full_yaml[args.global_config]
    test_architectures = full_yaml[args.test_name]

    device = torch.device(global_params.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(metadata.get("output_dir", "experiments/final_evaluation"), args.test_name)
    os.makedirs(output_dir, exist_ok=True)
    results_csv_path = os.path.join(output_dir, "test_results.csv")

    print(f"[INFO] Loading Train and Test embeddings...")
    raw_train_samples = FeatureLoader.load(
        metadata["train_embeddings_path"],
        expected_dim=metadata["backbone_embed_dim"],
        expected_experiment_id=None,
    )
    raw_test_samples = FeatureLoader.load(
        metadata["test_embeddings_path"],
        expected_dim=metadata["backbone_embed_dim"],
        expected_experiment_id=None,
    )

    if args.downstream:
        evaluate_downstream(args, metadata, global_params, test_architectures, raw_test_samples, output_dir)
        return

    test_results = []
    print("\n" + "=" * 75)
    print(f"       STARTING TEST EVALUATION: {args.test_name.upper()}")
    print("=" * 75)

    for phase in metadata["phases"]:
        if phase not in test_architectures:
            print(f"[WARNING] Skipping phase '{phase}': No architecture specified under '{args.test_name}'.")
            continue

        arch_config = test_architectures[phase]
        ckpt_path = arch_config["checkpoint_path"]

        print(f"\n==================== TEST PHASE: {phase.upper()} ====================")

        # Build test loader for the current phase
        _, test_loader, class_weights = prepare_phase_data(
            raw_train_samples, raw_test_samples, phase=phase
        )

        model = build_and_load_model(phase, arch_config, metadata, device)

        loss_fn = LossFactory.create(
            loss_type=LossType(arch_config["loss_type"]),
            weights=class_weights.to(device),
            kwargs=arch_config["loss_kwargs"],
        )

        trainer = Trainer(
            model=model,
            optimizer=None,
            criterion=loss_fn,
            device=device,
        )

        plot_path = os.path.join(output_dir, f"{phase}_confusion_matrix.png")

        # Call the evaluate method from Trainer
        metrics = trainer.evaluate(
            data_loader=test_loader,
            phase=phase,
            plot_path=plot_path,
            desc=f"[{phase.upper()} | Test]",
        )

        test_loss = metrics["val_loss"]
        test_acc = metrics["val_accuracy"]
        test_f1 = metrics["val_macro_f1"]

        record = {
            "phase": phase,
            "neck_type": arch_config["neck_type"],
            "neck_kwargs": str(arch_config["neck_kwargs"]),
            "head_type": arch_config["head_type"],
            "use_tracker": arch_config.get("use_tracker", False),
            "checkpoint_path": ckpt_path,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
        }
        test_results.append(record)

        print(f"\n>>> TEST RESULTS FOR PHASE {phase.upper()}:")
        print(f"  - Test Loss     : {test_loss:.4f}")
        print(f"  - Test Accuracy : {test_acc:.4f}")
        print(f"  - Test Macro F1 : {test_f1:.4f}  <-- FINAL SCORE")

    # Save summary report
    df_results = pd.DataFrame(test_results)
    df_results.to_csv(results_csv_path, index=False)

    print("\n" + "=" * 75)
    print(f"Final test evaluation completed. Results saved to: {results_csv_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()