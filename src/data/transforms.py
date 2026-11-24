"""Image transformations and augmentation pipelines for dermatological classification."""

from typing import Optional, Sequence
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int = 224,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> T.Compose:
    """Constructs the stochastic augmentation pipeline for training skin lesion images.

    Includes full rotational invariance (0-360 deg), vertical/horizontal flips,
    subtle affine perturbations, and bounded color jitter to account for dermatoscope
    calibration variances while preserving diagnostic pigment variegation.
    """
    norm_mean = list(mean) if mean is not None else IMAGENET_MEAN
    norm_std = list(std) if std is not None else IMAGENET_STD

    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=180, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std),
        ]
    )


def get_eval_transforms(
    image_size: int = 224,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> T.Compose:
    """Constructs the deterministic preprocessing pipeline for validation and testing.

    Only performs resizing and normalization, guaranteeing reproducibility.
    """
    norm_mean = list(mean) if mean is not None else IMAGENET_MEAN
    norm_std = list(std) if std is not None else IMAGENET_STD

    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
