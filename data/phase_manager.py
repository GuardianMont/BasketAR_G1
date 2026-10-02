import torch
from torch.utils.data import DataLoader

from data.dataset import ActionRecognitionDataset
from data.samplers import BucketSampler
from data.utils import compute_class_weights, pad_and_collate_video_batch

PHASE_CONFIGS = {
    "p1": {"name": "p1 (Action Type)", "num_classes": 3, "filter_shots_only": False},
    "p2": {"name": "p2 (Shot Type)", "num_classes": 3, "filter_shots_only": True},
    "p3": {"name": "p3 (Shot Outcome)", "num_classes": 2, "filter_shots_only": True},
}


def prepare_phase_data(
    train_samples: list[dict],
    val_samples: list[dict],
    phase: str,
    batch_size: int = 8,
):
    """Filters samples by phase, building Dataloaders and calculating class weights"""
    p_cfg = PHASE_CONFIGS[phase]

    if p_cfg["filter_shots_only"]:
        shot_labels = {1, 2, 3, 4, 5, 6}
        train_filtered = [s for s in train_samples if s["label"] in shot_labels]
        val_filtered = [s for s in val_samples if s["label"] in shot_labels]
    else:
        train_filtered, val_filtered = train_samples, val_samples

    train_dataset = ActionRecognitionDataset(samples=train_filtered, is_feature_dataset=True)
    val_dataset = ActionRecognitionDataset(samples=val_filtered, is_feature_dataset=True)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BucketSampler(
            sequence_lengths=[s["num_frames"] for s in train_filtered],
            batch_size=batch_size,
        ),
        collate_fn=pad_and_collate_video_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_and_collate_video_batch,
    )

    target_key = f"label_{phase}"
    train_phase_labels = [train_dataset[i][target_key] for i in range(len(train_dataset))]
    class_weights = compute_class_weights(
        train_phase_labels,
        num_classes=p_cfg["num_classes"],
        method="inverse",
    )

    return train_loader, val_loader, class_weights