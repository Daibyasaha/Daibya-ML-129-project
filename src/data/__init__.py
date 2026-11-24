"""Data module for dermatological datasets, transforms, splitting, and imbalance handling."""

SKIN_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(SKIN_CLASSES)
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(SKIN_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(SKIN_CLASSES)}

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

from .dataset import SkinLesionDataset
from .transforms import get_train_transforms, get_eval_transforms
from .split import split_skin_dataset, save_splits
from .imbalance import compute_inverse_class_weights, create_balanced_sampler
from .synthetic import generate_synthetic_dermoscopy_dataset

__all__ = [
    "SKIN_CLASSES",
    "NUM_CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "CLASS_DESCRIPTIONS",
    "SkinLesionDataset",
    "get_train_transforms",
    "get_eval_transforms",
    "split_skin_dataset",
    "save_splits",
    "compute_inverse_class_weights",
    "create_balanced_sampler",
    "generate_synthetic_dermoscopy_dataset",
]
