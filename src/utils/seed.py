"""Deterministic seeding across Python, NumPy, and PyTorch (CPU and CUDA)."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """Enforces bitwise determinism across all random number generators.

    Args:
        seed: Integer seed value.
        deterministic_cudnn: Whether to enforce deterministic cuDNN convolution algorithms.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
