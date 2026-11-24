"""Loss functions supporting class-imbalance mitigation and hard example focusing."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for mitigating extreme class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights well-classified common examples (like nevi) and concentrates
    model optimization on hard, ambiguous, and minority classes (melanoma, bcc, etc.).
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes focal loss between unnormalized logits and integer class targets.

        Args:
            inputs: Logit tensor of shape (batch_size, num_classes).
            targets: Long target tensor of shape (batch_size,).
        """
        # Cross entropy without reduction yields -log(p_t)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            alpha_tensor = self.alpha.to(device=inputs.device, dtype=inputs.dtype)
            alpha_t = alpha_tensor[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: '{self.reduction}'. Choose 'mean', 'sum', or 'none'.")


def get_loss_function(
    loss_type: str = "weighted_ce",
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    gamma: float = 2.0,
) -> nn.Module:
    """Factory creating loss function according to experiment configuration."""
    l_type = loss_type.lower().strip()

    if l_type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif l_type in ("weighted_ce", "class_weighted"):
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    elif l_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma, reduction="mean")
    else:
        supported = ["ce", "weighted_ce", "class_weighted", "focal"]
        raise ValueError(f"Unsupported loss_type: '{loss_type}'. Choose from: {supported}")
