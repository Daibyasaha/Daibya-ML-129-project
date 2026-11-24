"""CLI Evaluation and Benchmarking Suite for Skin Disease Classification."""

import argparse
import json
import logging
from pathlib import Path
import sys
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (
    SkinLesionDataset,
    get_eval_transforms,
    split_skin_dataset,
    generate_synthetic_dermoscopy_dataset,
    SKIN_CLASSES,
)
from src.models import build_classifier
from src.evaluation import evaluate_model, save_evaluation_artifacts
from src.utils import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained CNN model on held-out skin disease test set.")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth", help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML config file")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing dataset images and metadata")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory where evaluation artifacts are saved")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--use-synthetic", action="store_true", help="Force use of synthetic dataset fixture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    args = parse_args()
    seed_everything(args.seed)

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = data_cfg.get("image_size", 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Resolve Dataset & Test Split
    data_dir = Path(args.data_dir or data_cfg.get("data_dir", "data"))
    splits_dir = data_dir / "processed" / "splits"
    raw_images_dir = data_dir / "raw" / "images"
    synthetic_dir = data_dir / "synthetic"

    use_synthetic = args.use_synthetic or not (splits_dir / "test.csv").exists() and not (raw_images_dir.exists())

    if use_synthetic:
        logger.info("Resolving synthetic test split...")
        if not (synthetic_dir / "metadata.csv").exists():
            generate_synthetic_dermoscopy_dataset(
                output_dir=synthetic_dir,
                num_samples=data_cfg.get("synthetic_samples", 140),
                seed=args.seed,
                image_size=image_size,
            )
        meta_df = pd.read_csv(synthetic_dir / "metadata.csv")
        image_dir = synthetic_dir / "images"
        _, _, test_df = split_skin_dataset(meta_df, seed=args.seed)
    else:
        if (splits_dir / "test.csv").exists():
            logger.info(f"Loading existing test split from: {splits_dir / 'test.csv'}")
            test_df = pd.read_csv(splits_dir / "test.csv")
            image_dir = raw_images_dir if raw_images_dir.exists() else synthetic_dir / "images"
        else:
            meta_df = pd.read_csv(data_dir / "raw" / "metadata.csv")
            image_dir = raw_images_dir
            _, _, test_df = split_skin_dataset(meta_df, seed=args.seed)

    logger.info(f"Test split size: {len(test_df)} samples")

    eval_transforms = get_eval_transforms(image_size=image_size)
    test_dataset = SkinLesionDataset(test_df, image_dir=image_dir, transform=eval_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    # 2. Load Model Architecture & Checkpoint
    checkpoint_path = Path(args.checkpoint)
    backbone_name = model_cfg.get("backbone", "resnet18")
    num_classes = model_cfg.get("num_classes", 7)

    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "backbone" in checkpoint:
            backbone_name = checkpoint["backbone"]
        if isinstance(checkpoint, dict) and "class_names" in checkpoint:
            class_names = checkpoint["class_names"]
        else:
            class_names = SKIN_CLASSES

        model = build_classifier(
            backbone=backbone_name,
            num_classes=num_classes,
            pretrained=False,
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        logger.info("Checkpoint weights loaded successfully.")
    else:
        logger.warning(
            f"Checkpoint '{checkpoint_path}' not found! "
            f"Instantiating baseline model '{backbone_name}' with transfer learning weights for evaluation benchmark."
        )
        class_names = SKIN_CLASSES
        model = build_classifier(
            backbone=backbone_name,
            num_classes=num_classes,
            pretrained=True,
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
        )

    # 3. Run Benchmark Inference & Metric Calculation
    logger.info("Executing evaluation on test split...")
    results = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        class_names=class_names,
    )

    # Check for historical training curves if available
    history = None
    history_json_candidate = Path("models") / "training_history.json"
    if history_json_candidate.exists():
        with open(history_json_candidate, "r", encoding="utf-8") as f:
            history = json.load(f)

    # 4. Save Artifacts to Disk
    artifacts = save_evaluation_artifacts(
        results=results,
        output_dir=output_dir,
        history=history,
    )

    logger.info("Evaluation complete! Benchmark artifacts saved:")
    for name, path in artifacts.items():
        logger.info(f"  - {name}: {path}")

    # Print summary report to stdout
    with open(artifacts["classification_report"], "r", encoding="utf-8") as f:
        print("\n" + f.read())

    return 0


if __name__ == "__main__":
    sys.exit(main())
