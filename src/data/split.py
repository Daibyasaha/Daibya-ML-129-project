"""Group-stratified partitioning to guarantee zero patient/lesion data leakage."""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import CLASS_TO_IDX


def split_skin_dataset(
    metadata_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partitions metadata into Train, Val, and Test splits.

    Guarantees:
      1. Zero lesion_id overlap between splits when lesion_id is present,
         preventing patient/lesion level data leakage.
      2. Stratified diagnostic class representation across all partitions.
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-5):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    df = metadata_df.copy().reset_index(drop=True)

    # Ensure label column exists
    if "label" not in df.columns and "dx" in df.columns:
        df["label"] = df["dx"].map(CLASS_TO_IDX)

    # Determine grouping column
    if "lesion_id" in df.columns:
        groups = df["lesion_id"].values
    else:
        # Fallback to image-level grouping if lesion_id is absent
        groups = df["image_id"].values if "image_id" in df.columns else np.arange(len(df))

    targets = df["label"].values if "label" in df.columns else df["dx"].values

    # Step 1: Split into Train and Holdout (Val + Test)
    holdout_ratio = val_ratio + test_ratio
    n_splits_stage1 = max(2, int(round(1.0 / holdout_ratio)))

    try:
        sgkf1 = StratifiedGroupKFold(n_splits=n_splits_stage1, shuffle=True, random_state=seed)
        train_idx, holdout_idx = next(sgkf1.split(df, targets, groups=groups))
    except ValueError:
        # Fallback to standard StratifiedKFold if group stratification fails due to extreme min counts
        skf1 = StratifiedKFold(n_splits=n_splits_stage1, shuffle=True, random_state=seed)
        train_idx, holdout_idx = next(skf1.split(df, targets))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    holdout_df = df.iloc[holdout_idx].copy().reset_index(drop=True)

    # Step 2: Split Holdout into Val and Test
    # Determine the proportion of val within holdout
    val_in_holdout_ratio = val_ratio / holdout_ratio
    n_splits_stage2 = max(2, int(round(1.0 / (1.0 - val_in_holdout_ratio)))) if val_in_holdout_ratio < 0.9 else 2

    holdout_groups = (
        holdout_df["lesion_id"].values
        if "lesion_id" in holdout_df.columns
        else (holdout_df["image_id"].values if "image_id" in holdout_df.columns else np.arange(len(holdout_df)))
    )
    holdout_targets = holdout_df["label"].values if "label" in holdout_df.columns else holdout_df["dx"].values

    try:
        sgkf2 = StratifiedGroupKFold(n_splits=n_splits_stage2, shuffle=True, random_state=seed)
        val_sub_idx, test_sub_idx = next(sgkf2.split(holdout_df, holdout_targets, groups=holdout_groups))
    except ValueError:
        skf2 = StratifiedKFold(n_splits=n_splits_stage2, shuffle=True, random_state=seed)
        val_sub_idx, test_sub_idx = next(skf2.split(holdout_df, holdout_targets))

    val_df = holdout_df.iloc[val_sub_idx].copy().reset_index(drop=True)
    test_df = holdout_df.iloc[test_sub_idx].copy().reset_index(drop=True)

    # Tag each DataFrame with split column
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Verification of zero lesion leakage if lesion_id is present
    if "lesion_id" in df.columns:
        train_lesions = set(train_df["lesion_id"])
        val_lesions = set(val_df["lesion_id"])
        test_lesions = set(test_df["lesion_id"])

        leak_train_val = train_lesions.intersection(val_lesions)
        leak_train_test = train_lesions.intersection(test_lesions)
        leak_val_test = val_lesions.intersection(test_lesions)

        if leak_train_val or leak_train_test or leak_val_test:
            # Resolving any rare boundary overlaps deterministically
            for l_id in leak_train_val.union(leak_train_test):
                # Move conflicting lesions entirely into training set
                val_df = val_df[val_df["lesion_id"] != l_id].reset_index(drop=True)
                test_df = test_df[test_df["lesion_id"] != l_id].reset_index(drop=True)
            for l_id in leak_val_test:
                # Move to val
                test_df = test_df[test_df["lesion_id"] != l_id].reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Union[str, Path],
) -> dict[str, Path]:
    """Saves split DataFrames to CSV files and returns a dictionary of file paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out_dir / "train.csv",
        "val": out_dir / "val.csv",
        "test": out_dir / "test.csv",
        "master": out_dir / "metadata_split.csv",
    }

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    master_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    master_df.to_csv(paths["master"], index=False)

    return paths
