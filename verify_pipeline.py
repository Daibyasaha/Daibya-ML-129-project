"""Automated acceptance verification script for the academic skin disease classification pipeline.

Executes comprehensive end-to-end self-tests covering:
  1. Deterministic seeding
  2. Multi-backbone instantiation & forward pass shapes
  3. Regularization behavior (train vs eval mode)
  4. Class-weighted & focal loss functions
  5. Checkpoint serialization roundtrip
  6. Comprehensive metric & ECE calculations
  7. End-to-end 1-epoch training and evaluation dry-run
"""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import (
    SkinLesionDataset,
    get_train_transforms,
    get_eval_transforms,
    split_skin_dataset,
    compute_inverse_class_weights,
    generate_synthetic_dermoscopy_dataset,
    SKIN_CLASSES,
)
from src.models import build_classifier, get_loss_function, SkinClassifier
from src.training import Trainer
from src.evaluation import evaluate_model, save_evaluation_artifacts
from src.utils import seed_everything


def test_seeding_determinism() -> bool:
    """Verifies that seed_everything produces bitwise identical tensor operations."""
    print("  [1/7] Testing Seeding Determinism...", end=" ")
    seed_everything(42)
    t1 = torch.randn(10, 10)
    a1 = np.random.rand(10)

    seed_everything(42)
    t2 = torch.randn(10, 10)
    a2 = np.random.rand(10)

    assert torch.equal(t1, t2), "PyTorch RNG output mismatch under identical seed!"
    assert np.array_equal(a1, a2), "NumPy RNG output mismatch under identical seed!"
    print("PASS")
    return True


def test_backbone_instantiation() -> bool:
    """Verifies instantiation and output logit tensor shapes across candidate CNN backbones."""
    print("  [2/7] Testing Backbone Instantiation & Forward Pass...", end=" ")
    backbones = ["resnet18", "efficientnet_b0", "mobilenet_v2"]
    dummy_input = torch.randn(2, 3, 224, 224)

    for b in backbones:
        model = build_classifier(backbone=b, num_classes=7, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (2, 7), f"Expected shape (2, 7) for {b}, got {out.shape}"
    print("PASS")
    return True


def test_custom_head_regularization() -> bool:
    """Verifies dropout regularization differences between training and evaluation modes."""
    print("  [3/7] Testing Head Regularization (Train vs. Eval)...", end=" ")
    seed_everything(42)
    model = build_classifier("resnet18", num_classes=7, pretrained=False, dropout_rate=0.5)
    dummy_input = torch.randn(4, 3, 224, 224)

    # Train mode has stochastic dropout
    model.train()
    out1 = model(dummy_input)
    out2 = model(dummy_input)
    assert not torch.equal(out1, out2), "Dropout should produce stochastic variations in train mode!"

    # Eval mode is strictly deterministic
    model.eval()
    with torch.no_grad():
        eval1 = model(dummy_input)
        eval2 = model(dummy_input)
    assert torch.allclose(eval1, eval2, atol=1e-6), "Eval mode must be deterministic!"
    print("PASS")
    return True


def test_loss_functions() -> bool:
    """Verifies CrossEntropy, Class-Weighted CE, and Focal Loss computations."""
    print("  [4/7] Testing Loss Functions & Gradient Propagation...", end=" ")
    logits = torch.randn(4, 7, requires_grad=True)
    targets = torch.tensor([0, 1, 4, 6], dtype=torch.long)
    weights = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5, 0.8, 2.5], dtype=torch.float32)

    # 1. Standard CE
    ce = get_loss_function("ce")
    l_ce = ce(logits, targets)
    assert l_ce.item() > 0, "CE loss must be strictly positive"

    # 2. Weighted CE
    w_ce = get_loss_function("weighted_ce", class_weights=weights)
    l_wce = w_ce(logits, targets)
    assert l_wce.item() > 0, "Weighted CE loss must be strictly positive"

    # 3. Focal Loss
    focal = get_loss_function("focal", class_weights=weights, gamma=2.0)
    l_focal = focal(logits, targets)
    assert l_focal.item() > 0, "Focal loss must be strictly positive"

    # Verify backprop
    l_focal.backward()
    assert logits.grad is not None, "Gradients must propagate through Focal Loss"
    print("PASS")
    return True


def test_checkpoint_roundtrip() -> bool:
    """Verifies that model parameters serialize and restore with bitwise precision."""
    print("  [5/7] Testing Checkpoint Serialization Roundtrip...", end=" ")
    seed_everything(42)
    m1 = build_classifier("resnet18", num_classes=7, pretrained=False)
    with tempfile.TemporaryDirectory() as tmp_dir:
        chk_path = Path(tmp_dir) / "test_checkpoint.pth"
        torch.save({
            "model_state_dict": m1.state_dict(),
            "backbone": "resnet18",
            "class_names": SKIN_CLASSES,
        }, chk_path)

        m2 = build_classifier("resnet18", num_classes=7, pretrained=False)
        saved = torch.load(chk_path, weights_only=False)
        m2.load_state_dict(saved["model_state_dict"])

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2), "Restored parameters must match saved parameters exactly"
    print("PASS")
    return True


def test_evaluation_metrics() -> bool:
    """Verifies metric calculations against known classification ground-truth."""
    print("  [6/7] Testing Evaluation Metrics & Calibration...", end=" ")
    from src.evaluation.metrics import compute_comprehensive_metrics

    # Perfect prediction case
    y_true = np.array([0, 1, 2, 3, 4, 5, 6])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6])
    y_prob = np.eye(7)

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob, SKIN_CLASSES)
    assert metrics["top1_accuracy"] == 1.0, "Perfect predictions must yield accuracy 1.0"
    assert metrics["balanced_accuracy"] == 1.0, "Perfect predictions must yield balanced accuracy 1.0"
    assert metrics["macro_f1"] == 1.0, "Perfect predictions must yield macro F1 1.0"
    assert metrics["expected_calibration_error"] == 0.0, "Perfect predictions must yield ECE 0.0"
    print("PASS")
    return True


def test_end_to_end_fast_dry_run() -> bool:
    """Executes a 1-epoch train -> checkpoint -> evaluate cycle on a synthetic fixture."""
    print("  [7/7] Testing End-to-End Pipeline Dry-Run...", end=" ")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_dir = tmp_path / "synthetic_fixture"
        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"

        # 1. Generate micro dataset (28 samples, all 7 classes)
        df = generate_synthetic_dermoscopy_dataset(
            output_dir=data_dir,
            num_samples=28,
            seed=42,
            image_size=64,  # Fast spatial dimension for instantaneous test
        )
        assert len(df) == 28
        assert (data_dir / "metadata.csv").exists()

        # 2. Partition
        train_df, val_df, test_df = split_skin_dataset(df, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)
        assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0

        # Check zero lesion leakage
        train_lesions = set(train_df["lesion_id"])
        val_lesions = set(val_df["lesion_id"])
        test_lesions = set(test_df["lesion_id"])
        assert len(train_lesions.intersection(val_lesions)) == 0, "Train-Val lesion leakage detected!"
        assert len(train_lesions.intersection(test_lesions)) == 0, "Train-Test lesion leakage detected!"
        assert len(val_lesions.intersection(test_lesions)) == 0, "Val-Test lesion leakage detected!"

        # 3. DataLoaders
        train_transforms = get_train_transforms(image_size=64)
        eval_transforms = get_eval_transforms(image_size=64)

        train_ds = SkinLesionDataset(train_df, image_dir=data_dir / "images", transform=train_transforms)
        val_ds = SkinLesionDataset(val_df, image_dir=data_dir / "images", transform=eval_transforms)
        test_ds = SkinLesionDataset(test_df, image_dir=data_dir / "images", transform=eval_transforms)

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

        # 4. Train 1 epoch
        model = build_classifier("resnet18", num_classes=7, pretrained=False)
        weights = compute_inverse_class_weights(train_ds.get_labels(), num_classes=7)
        criterion = get_loss_function("weighted_ce", class_weights=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            output_dir=models_dir,
            config={"use_amp": False, "backbone": "resnet18"},
        )

        history = trainer.fit(num_epochs=1)
        assert (models_dir / "checkpoints" / "best_model.pth").exists(), "best_model.pth missing!"
        assert (models_dir / "training_history.json").exists(), "training_history.json missing!"

        # 5. Evaluate on test split
        eval_results = evaluate_model(model=model, data_loader=test_loader, class_names=SKIN_CLASSES)
        artifacts = save_evaluation_artifacts(eval_results, reports_dir, history=history)

        assert artifacts["metrics_json"].exists(), "metrics.json missing!"
        assert artifacts["classification_report"].exists(), "classification_report.txt missing!"
        assert artifacts["confusion_matrix_csv"].exists(), "confusion_matrix.csv missing!"
        assert artifacts["confusion_matrix_png"].exists(), "confusion_matrix.png missing!"

    print("PASS")
    return True


def run_all_verification_tests() -> int:
    print("=" * 70)
    print("   ACADEMIC SKIN DISEASE CLASSIFIER - AUTOMATED ACCEPTANCE VERIFICATION")
    print("=" * 70)

    tests = [
        test_seeding_determinism,
        test_backbone_instantiation,
        test_custom_head_regularization,
        test_loss_functions,
        test_checkpoint_roundtrip,
        test_evaluation_metrics,
        test_end_to_end_fast_dry_run,
    ]

    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
                break
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            break

    print("=" * 70)
    if all_passed:
        print("ALL 7 ACCEPTANCE CRITERIA VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    else:
        print("PIPELINE VERIFICATION ENCOUNTERED FAILURES.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_verification_tests())
