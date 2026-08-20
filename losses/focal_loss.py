"""
Implementation of Multi-Class Focal Loss for handling severe class imbalance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss implementation.

    Role:
        Addresses class imbalance by down-weighting easy-to-classify (well-trained)
        examples and focusing model training on hard, rare instances (e.g., rare actions).

    Mathematical Formulation:
        FL(p_t) = - \alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)

    Attributes:
        alpha (Optional[torch.Tensor]): 1D Tensor of shape (K,) specifying class weights.
        gamma (float): Focusing parameter that controls the strength of down-weighting
                       easy examples. (Default: 2.0).
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """
    def __init__(
        self, 
        alpha: torch.Tensor = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the multi-class focal loss.

        Args:
            inputs (torch.Tensor): Raw, unnormalized model logits of shape (B, K)
                                  where B=Batch Size and K=Number of Classes.
            targets (torch.Tensor): Ground-truth class indices of shape (B,)
                                   with values in range [0, K-1].

        Returns:
            torch.Tensor: Computed loss value (scalar if reduction='mean' or 'sum').
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        # Apply class weights (alpha_t) if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        
        return focal_loss