import torch
from typing import List, Dict, Any, Tuple
import numpy as np


def pad_and_collate_video_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that groups variable length clips.
    Generates a padded tensor, the binary attention mask and lengths
    """
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    
    max_len = int(lengths.max().item())
    batch_size = len(batch)
    
    sample_elem = batch[0]["data"]
    
    if sample_elem.dim() == 2:
        # Pre-extracted: (T, D) -> Output Batch: (B, T_max, D)
        D = sample_elem.shape[1]
        padded_data = torch.zeros(batch_size, max_len, D, dtype=sample_elem.dtype)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        for i, b in enumerate(batch):
            seq_len = b["length"]
            padded_data[i, :seq_len] = b["data"]
            mask[i, :seq_len] = True  # False for padding
            
    elif sample_elem.dim() == 4:
        # Raw video: (C, T, H, W) -> Output Batch: (B, C, T_max, H, W)
        C, _, H, W = sample_elem.shape
        padded_data = torch.zeros(batch_size, C, max_len, H, W, dtype=sample_elem.dtype)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        for i, b in enumerate(batch):
            seq_len = b["length"]
            padded_data[i, :, :seq_len, :, :] = b["data"]
            mask[i, :seq_len] = True
    else:
        raise ValueError(f"Unsupported tensor dimension: {sample_elem.dim()}")

    return {
        "x": padded_data,
        "labels": labels,
        "lengths": lengths,
        "attention_mask": mask
    }


def compute_class_weights(labels: List[int], num_classes: int = 9, method: str = "inverse") -> torch.Tensor:
    """
    Returns the weights for Class-Weighted Cross-Entropy or Focal loss's alpha.
    
    Args:
        method: supported values are 'inverse' and 'effective_num'. The first calculates
            the weights based on w_c = N_tot / (K * N_c). The second uses a `beta` parameter
            hardcoded to 0.999 to reduce the impact of class imbalance. The second one is
            recommended for high class imbalance.
    """
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    if method == "inverse":
        weights = total_samples / (num_classes * np.maximum(counts, 1))
    elif method == "effective_num":
        beta = 0.999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / np.sum(weights) * num_classes
    else:
        raise ValueError(f"Unsupported method: {method}")
        
    return torch.tensor(weights, dtype=torch.float32)