"""Imbalance mitigation utilities: inverse-frequency weights and balanced sampling."""

from typing import Sequence, Union
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def compute_inverse_class_weights(
    train_labels: Union[Sequence[int], np.ndarray, torch.Tensor],
    num_classes: int = 7,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Computes inverse class frequency weights normalized for PyTorch loss functions.

    Formula:
        w_c = N_total / (num_classes * N_c)
    With optional smoothing:
        w_c = (N_total + smoothing * num_classes) / (num_classes * (N_c + smoothing))

    Ensures minority skin disease classes (e.g. dermatofibroma, vascular)
    receive proportional gradient penalties.
    """
    if isinstance(train_labels, torch.Tensor):
        labels_arr = train_labels.detach().cpu().numpy()
    else:
        labels_arr = np.array(train_labels)

    counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float64)
    total_samples = float(len(labels_arr))

    # Apply smoothing or clamp minimum count to 1 to prevent division by zero
    counts_smoothed = counts + smoothing
    counts_smoothed = np.maximum(counts_smoothed, 1.0)

    denominator = num_classes * counts_smoothed
    weights = (total_samples + smoothing * num_classes) / denominator

    # Normalize weights so that their mean is 1.0
    weights = weights / np.mean(weights)

    return torch.tensor(weights, dtype=torch.float32)


def create_balanced_sampler(
    train_labels: Union[Sequence[int], np.ndarray, torch.Tensor],
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Constructs a WeightedRandomSampler that samples classes with equal probability per batch.

    Assigns weight W_i = 1.0 / count(class(i)) to each individual training instance.
    """
    if isinstance(train_labels, torch.Tensor):
        labels_arr = train_labels.detach().cpu().numpy()
    else:
        labels_arr = np.array(train_labels)

    class_counts = np.bincount(labels_arr).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_arr]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=replacement,
    )
    return sampler
