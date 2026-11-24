"""Visualization and artifact generation for evaluation benchmark reports."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(
    cm_matrix: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
    normalized: bool = True,
    title: str = "Dermatology Benchmark Confusion Matrix",
) -> Path:
    """Renders a confusion matrix heatmap and saves it to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    fmt = ".2f" if normalized else "d"
    vmax = 1.0 if normalized else None

    sns.heatmap(
        cm_matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        vmin=0.0,
        vmax=vmax,
        linewidths=0.5,
        linecolor="lightgray",
    )

    plt.title(title, fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Diagnostic Class", fontsize=12, labelpad=8)
    plt.ylabel("True Pathologic Class", fontsize=12, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(out, dpi=300)
    plt.close()
    return out


def plot_training_curves(
    history: Dict[str, List[Any]],
    output_path: Union[str, Path],
) -> Path:
    """Generates dual-panel learning curves for loss and accuracy across epochs."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history.get("train_loss", []), label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history.get("val_loss", []), label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True)

    # Accuracy / Macro F1 curve
    ax2.plot(epochs, history.get("train_acc", []), label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(epochs, history.get("val_acc", []), label="Val Accuracy", color="#d62728", lw=2, linestyle="--")
    if "val_macro_f1" in history:
        ax2.plot(epochs, history.get("val_macro_f1", []), label="Val Macro-F1", color="#9467bd", lw=2, linestyle="-.")
    ax2.set_title("Classification Accuracy & Macro F1", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def generate_text_classification_report(metrics: Dict[str, Any], class_names: List[str]) -> str:
    """Formats a detailed human-readable clinical classification report."""
    lines = []
    lines.append("=" * 78)
    lines.append("        DERMATOLOGICAL SKIN LESION CLASSIFICATION BENCHMARK REPORT")
    lines.append("=" * 78)
    lines.append(f"Global Top-1 Accuracy:      {metrics.get('top1_accuracy', 0.0):.4f}")
    lines.append(f"Global Balanced Accuracy:  {metrics.get('balanced_accuracy', 0.0):.4f}")
    if "top2_accuracy" in metrics:
        lines.append(f"Differential Top-2 Accuracy: {metrics.get('top2_accuracy', 0.0):.4f}")
    if "top3_accuracy" in metrics:
        lines.append(f"Differential Top-3 Accuracy: {metrics.get('top3_accuracy', 0.0):.4f}")
    lines.append(f"Macro-Averaged F1-Score:    {metrics.get('macro_f1', 0.0):.4f}")
    lines.append(f"Weighted-Averaged F1-Score: {metrics.get('weighted_f1', 0.0):.4f}")
    lines.append(f"Expected Calibration Error: {metrics.get('expected_calibration_error', 0.0):.4f}")
    if metrics.get("macro_roc_auc") is not None:
        lines.append(f"Macro One-vs-Rest ROC-AUC:  {metrics.get('macro_roc_auc', 0.0):.4f}")
    lines.append("-" * 78)
    lines.append(f"{'Class':<8} {'Support':<8} {'Precision':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1-Score':<10}")
    lines.append("-" * 78)

    per_class = metrics.get("per_class", {})
    for cls in class_names:
        stats = per_class.get(cls, {})
        sup = stats.get("support", 0)
        p = stats.get("precision", 0.0)
        r = stats.get("recall_sensitivity", 0.0)
        s = stats.get("specificity", 0.0)
        f1 = stats.get("f1_score", 0.0)
        lines.append(f"{cls:<8} {sup:<8} {p:<10.4f} {r:<12.4f} {s:<12.4f} {f1:<10.4f}")

    lines.append("=" * 78)
    return "\n".join(lines)


def save_evaluation_artifacts(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    history: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, Path]:
    """Exports all benchmark artifacts: metrics JSON, text report, CSVs, and PNG figures."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = results.get("metrics", results)
    class_names = results.get("class_names", list(metrics.get("per_class", {}).keys()))

    artifacts = {}

    # 1. Save machine-readable metrics JSON
    metrics_json_path = out_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    artifacts["metrics_json"] = metrics_json_path

    # 2. Save text classification report
    report_text = generate_text_classification_report(metrics, class_names)
    report_txt_path = out_dir / "classification_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    artifacts["classification_report"] = report_txt_path

    # 3. Export raw and normalized confusion matrix CSVs
    raw_cm = np.array(metrics["confusion_matrix"])
    norm_cm = np.array(metrics["confusion_matrix_normalized"])

    raw_cm_df = pd.DataFrame(raw_cm, index=class_names, columns=class_names)
    raw_cm_path = out_dir / "confusion_matrix.csv"
    raw_cm_df.to_csv(raw_cm_path)
    artifacts["confusion_matrix_csv"] = raw_cm_path

    norm_cm_df = pd.DataFrame(norm_cm, index=class_names, columns=class_names)
    norm_cm_path = out_dir / "confusion_matrix_normalized.csv"
    norm_cm_df.to_csv(norm_cm_path)
    artifacts["confusion_matrix_normalized_csv"] = norm_cm_path

    # 4. Save confusion matrix heatmap PNG
    cm_png_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(norm_cm, class_names, cm_png_path, normalized=True)
    artifacts["confusion_matrix_png"] = cm_png_path

    # 5. Save training curves PNG if history is provided
    if history is not None and "epoch" in history and len(history["epoch"]) > 0:
        curves_path = out_dir / "training_curves.png"
        plot_training_curves(history, curves_path)
        artifacts["training_curves_png"] = curves_path

    return artifacts
