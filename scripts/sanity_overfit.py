"""Memorize a small real feature subset to sanity-check the training path."""

import argparse
import random
from collections import defaultdict

import torch
from sklearn.metrics import accuracy_score

from data.dataset import ActionRecognitionDataset
from data.experiment_config import ExperimentConfig
from data.temporal_sampling import LegacyUniformSampler
from data.utils import pad_and_collate_video_batch
from models.heads import MeanPoolingClassifierHead
from models.necks import BiGRUNeck
from models.pipeline import ARPipeline
from utils.feature_loader import load_experiment_feature_pair


def stratified_subset(samples, target_size: int, seed: int):
    groups = defaultdict(list)
    for sample in samples:
        groups[int(sample["label"])].append(sample)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    labels = sorted(groups)
    while len(selected) < target_size:
        progressed = False
        for label in labels:
            if groups[label] and len(selected) < target_size:
                selected.append(groups[label].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) < 30:
        raise ValueError(
            f"Only {len(selected)} samples could be selected; at least 30 are required."
        )
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=None)
    parser.add_argument("--expected-experiment-id", default=None)
    parser.add_argument("--num-clips", type=int, default=40, choices=range(30, 51))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-accuracy", type=float, default=0.98)
    args = parser.parse_args()

    config = ExperimentConfig(
        sampler=LegacyUniformSampler(16).config(),
        spatial_mode="letterbox",
        spatial_size=312,
        color="rgb",
        backbone="x3d_l",
        tracker="none",
        extra={"transform": "none"},
    )
    feature_path = args.features or f"features/{config.feature_filename('train')}"
    default_experiment_id = (
        config.feature_id()
        if not args.features and not args.expected_experiment_id
        else None
    )
    payload, _, experiment_id, _ = load_experiment_feature_pair(
        feature_path,
        default_experiment_id=default_experiment_id,
        expected_experiment_id=args.expected_experiment_id,
    )
    samples = payload["samples"]
    print(f"[INFO] Sanity experiment_id={experiment_id}")

    subset = stratified_subset(samples, args.num_clips, args.seed)
    dataset = ActionRecognitionDataset(subset, is_feature_dataset=True)
    batch = pad_and_collate_video_batch([dataset[i] for i in range(len(dataset))])
    features = batch["x"]
    lengths = batch["lengths"]
    labels = batch["label_orig"]
    feature_dim = int(payload["feature_dim"])
    if int(features.shape[-1]) != feature_dim:
        raise ValueError("Collated feature dimension differs from payload feature_dim.")
    num_classes = int(max(int(sample["label"]) for sample in subset) + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    neck = BiGRUNeck(in_dim=feature_dim, hidden_dim=128, num_layers=1, dropout=0.0)
    head = MeanPoolingClassifierHead(in_dim=256, num_classes=num_classes, dropout=0.0)
    model = ARPipeline(backbone=None, neck=neck, head=head).to(device)
    features, lengths, labels = (
        features.to(device), lengths.to(device), labels.to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, lengths=lengths)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(features, lengths=lengths).argmax(dim=1)
            accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        best_accuracy = max(best_accuracy, accuracy)
        if epoch == 1 or epoch % 10 == 0 or accuracy >= args.target_accuracy:
            print(
                f"epoch={epoch:03d} loss={loss.item():.5f} "
                f"train_accuracy={accuracy:.4f} best={best_accuracy:.4f}"
            )
        if accuracy >= args.target_accuracy:
            print("OVERFIT_SANITY=PASS")
            return

    print(f"OVERFIT_SANITY=FAIL best_train_accuracy={best_accuracy:.4f}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
