"""PyTorch Dataset implementation for dermatological skin lesion classification."""

from pathlib import Path
from typing import Callable, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from . import CLASS_TO_IDX, SKIN_CLASSES


class SkinLesionDataset(Dataset):
    """Dataset for loading dermatoscopic images with corresponding diagnostic labels.

    Supports metadata dataframes containing either integer 'label' or categorical 'dx' columns.
    Handles multiple image extension conventions (.jpg, .jpeg, .png).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        image_ext: str = ".jpg",
    ):
        self.df = dataframe.copy().reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_ext = image_ext

        # Validate or populate the numeric label column
        if "label" not in self.df.columns:
            if "dx" in self.df.columns:
                self.df["label"] = self.df["dx"].map(CLASS_TO_IDX)
                if self.df["label"].isnull().any():
                    unmapped = self.df[self.df["label"].isnull()]["dx"].unique()
                    raise ValueError(f"Unrecognized diagnostic classes found in metadata: {unmapped}")
            else:
                raise ValueError("DataFrame must contain either 'label' or 'dx' column.")

        # Ensure label column is integer
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_id: str) -> Path:
        """Resolves the physical image path with extension fallback."""
        img_id_str = str(image_id).strip()
        candidate = self.image_dir / img_id_str
        if candidate.exists() and candidate.is_file():
            return candidate

        # Try appending default image extension
        candidate_ext = self.image_dir / f"{img_id_str}{self.image_ext}"
        if candidate_ext.exists() and candidate_ext.is_file():
            return candidate_ext

        # Try alternative common extensions
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            cand = self.image_dir / f"{img_id_str}{ext}"
            if cand.exists() and cand.is_file():
                return cand

        raise FileNotFoundError(
            f"Dermoscopy image '{img_id_str}' not found in '{self.image_dir}' "
            f"with extensions {['', self.image_ext, '.jpg', '.jpeg', '.png']}"
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        image_path = self._resolve_image_path(image_id)

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image from {image_path}: {e}") from e

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])
        return image, label

    def get_labels(self) -> np.ndarray:
        """Returns the numpy array of target labels across the dataset."""
        return self.df["label"].to_numpy()

    def get_class_counts(self) -> dict[str, int]:
        """Returns a dictionary mapping class names to sample counts in this dataset."""
        counts = self.df["label"].value_counts().to_dict()
        return {SKIN_CLASSES[idx]: counts.get(idx, 0) for idx in range(len(SKIN_CLASSES))}
