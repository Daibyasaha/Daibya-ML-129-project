"""Clinical and academic evaluation metrics computation for dermatological classification."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)

from ..data import SKIN_CLASSES


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins.

    Assesses the clinical reliability and overconfidence of model predicted probabilities.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    top_k_list: Tuple[int, ...] = (1, 2, 3),
) -> Dict[str, Any]:
    """Generates standard academic classification metrics and dermatological clinical metrics.

    Calculates:
      - Top-1 and Top-k differential diagnosis accuracy
      - Balanced accuracy (unweighted mean of recalls)
      - Macro- and weighted-averaged precision, recall, and F1-score
      - Granular per-class breakdown (TP, FP, FN, TN, Sensitivity, Specificity, PPV, NPV)
      - Area under ROC curve (One-vs-Rest)
      - Expected Calibration Error (ECE)
      - Raw count and normalized confusion matrices
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    num_classes = len(class_names)
    labels = list(range(num_classes))

    metrics: Dict[str, Any] = {}

    # 1. Global Metrics
    metrics["top1_accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    for k in top_k_list:
        if 1 < k < num_classes:
            try:
                metrics[f"top{k}_accuracy"] = float(
                    top_k_accuracy_score(y_true, y_prob, k=k, labels=labels)
                )
            except Exception:
                pass

    # Macro & Weighted Summary Scores
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics["macro_precision"] = float(p_macro)
    metrics["macro_recall"] = float(r_macro)
    metrics["macro_f1"] = float(f1_macro)
    metrics["weighted_precision"] = float(p_weight)
    metrics["weighted_recall"] = float(r_weight)
    metrics["weighted_f1"] = float(f1_weight)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = np.zeros_like(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm.astype(np.float64) / row_sums, 0.0)

    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_normalized"] = cm_norm.tolist()

    # 3. Per-Class Granular Breakdown
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - (tp + fn + fp))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        per_class[name] = {
            "class_index": i,
            "precision": float(precisions[i]),
            "recall_sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1_score": float(f1s[i]),
            "ppv": float(ppv),
            "npv": float(npv),
            "support": int(supports[i]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    metrics["per_class"] = per_class

    # 4. Multiclass ROC-AUC (One-vs-Rest)
    try:
        # Check if all classes are present in y_true
        unique_classes = np.unique(y_true)
        if len(unique_classes) == num_classes:
            metrics["macro_roc_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
            per_class_auc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average=None, labels=labels
            )
            for i, name in enumerate(class_names):
                per_class[name]["roc_auc"] = float(per_class_auc[i])
        else:
            # Sliced ROC-AUC for represented classes
            metrics["macro_roc_auc"] = None
    except Exception:
        metrics["macro_roc_auc"] = None

    # 5. Calibration
    metrics["expected_calibration_error"] = float(compute_ece(y_true, y_prob, n_bins=10))

    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Runs full inference over a dataset and produces the benchmark metrics dictionary."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if class_names is None:
        class_names = SKIN_CLASSES

    model.to(device)
    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(targets.numpy() if isinstance(targets, torch.Tensor) else targets)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_comprehensive_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
    )

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "class_names": list(class_names),
    }
