"""Pytest test suite for skin lesion classification pipeline."""

import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import (
    SkinLesionDataset,
    get_train_transforms,
    get_eval_transforms,
    split_skin_dataset,
    compute_inverse_class_weights,
    create_balanced_sampler,
    generate_synthetic_dermoscopy_dataset,
    SKIN_CLASSES,
)
from src.models import build_classifier, get_loss_function, SkinClassifier
from src.training import Trainer
from src.evaluation import evaluate_model, compute_comprehensive_metrics, save_evaluation_artifacts
from src.utils import seed_everything


class TestSkinDiseasePipeline:

    def test_seeding_determinism(self):
        seed_everything(123)
        a = torch.randn(5, 5)
        seed_everything(123)
        b = torch.randn(5, 5)
        assert torch.equal(a, b)

    @pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0", "mobilenet_v2"])
    def test_backbone_shapes(self, backbone):
        model = build_classifier(backbone=backbone, num_classes=7, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 7)

    def test_loss_functions(self):
        logits = torch.randn(4, 7, requires_grad=True)
        targets = torch.tensor([0, 2, 5, 6])
        weights = torch.ones(7)

        ce = get_loss_function("ce")
        w_ce = get_loss_function("weighted_ce", class_weights=weights)
        focal = get_loss_function("focal", class_weights=weights, gamma=2.0)

        assert ce(logits, targets) > 0
        assert w_ce(logits, targets) > 0
        loss_focal = focal(logits, targets)
        assert loss_focal > 0

        loss_focal.backward()
        assert logits.grad is not None

    def test_synthetic_data_generation_and_zero_leakage_split(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "synth"
            df = generate_synthetic_dermoscopy_dataset(out_dir, num_samples=35, seed=42, image_size=64)
            assert len(df) == 35
            assert (out_dir / "metadata.csv").exists()

            train_df, val_df, test_df = split_skin_dataset(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
            assert len(train_df) > 0
            assert len(val_df) > 0
            assert len(test_df) > 0

            # Verify lesion group isolation
            train_lesions = set(train_df["lesion_id"])
            val_lesions = set(val_df["lesion_id"])
            test_lesions = set(test_df["lesion_id"])

            assert len(train_lesions.intersection(val_lesions)) == 0
            assert len(train_lesions.intersection(test_lesions)) == 0
            assert len(val_lesions.intersection(test_lesions)) == 0

    def test_dataset_and_sampler(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "synth"
            df = generate_synthetic_dermoscopy_dataset(out_dir, num_samples=28, seed=42, image_size=64)
            eval_transforms = get_eval_transforms(image_size=64)

            dataset = SkinLesionDataset(df, image_dir=out_dir / "images", transform=eval_transforms)
            assert len(dataset) == 28

            img, label = dataset[0]
            assert isinstance(img, torch.Tensor)
            assert img.shape == (3, 64, 64)
            assert 0 <= label < 7

            labels = dataset.get_labels()
            weights = compute_inverse_class_weights(labels, num_classes=7)
            assert len(weights) == 7
            assert (weights > 0).all()

            sampler = create_balanced_sampler(labels)
            loader = DataLoader(dataset, batch_size=4, sampler=sampler)
            batch_images, batch_targets = next(iter(loader))
            assert batch_images.shape == (4, 3, 64, 64)
            assert batch_targets.shape == (4,)

    def test_metrics_computation(self):
        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])
        y_prob = np.eye(7)

        m = compute_comprehensive_metrics(y_true, y_pred, y_prob, SKIN_CLASSES)
        assert m["top1_accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["expected_calibration_error"] == 0.0

    def test_trainer_and_artifacts_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            df = generate_synthetic_dermoscopy_dataset(tmp_path / "data", num_samples=21, seed=42, image_size=64)
            train_df, val_df, test_df = split_skin_dataset(df, 0.5, 0.25, 0.25, seed=42)

            t_trans = get_train_transforms(image_size=64)
            e_trans = get_eval_transforms(image_size=64)

            train_ds = SkinLesionDataset(train_df, tmp_path / "data" / "images", transform=t_trans)
            val_ds = SkinLesionDataset(val_df, tmp_path / "data" / "images", transform=e_trans)
            test_ds = SkinLesionDataset(test_df, tmp_path / "data" / "images", transform=e_trans)

            train_loader = DataLoader(train_ds, batch_size=4)
            val_loader = DataLoader(val_ds, batch_size=4)
            test_loader = DataLoader(test_ds, batch_size=4)

            model = build_classifier("resnet18", num_classes=7, pretrained=False)
            criterion = get_loss_function("ce")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                config={"use_amp": False, "backbone": "resnet18"},
                output_dir=tmp_path / "output",
            )
            history = trainer.fit(num_epochs=1)

            assert (tmp_path / "output" / "checkpoints" / "best_model.pth").exists()
            assert (tmp_path / "output" / "training_history.json").exists()

            results = evaluate_model(model, test_loader, class_names=SKIN_CLASSES)
            artifacts = save_evaluation_artifacts(results, tmp_path / "reports", history=history)

            assert artifacts["metrics_json"].exists()
            assert artifacts["classification_report"].exists()
            assert artifacts["confusion_matrix_csv"].exists()
            assert artifacts["confusion_matrix_png"].exists()
