"""CLI Training script for academic skin disease classification with PyTorch transfer learning."""

import argparse
import logging
from pathlib import Path
import sys
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (
    SkinLesionDataset,
    get_train_transforms,
    get_eval_transforms,
    split_skin_dataset,
    save_splits,
    compute_inverse_class_weights,
    create_balanced_sampler,
    generate_synthetic_dermoscopy_dataset,
    SKIN_CLASSES,
)
from src.models import build_classifier, get_loss_function
from src.training import Trainer
from src.utils import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN transfer learning classifier for skin lesion diagnosis.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML config file")
    parser.add_argument("--backbone", type=str, default=None, help="CNN backbone (e.g. resnet18, efficientnet_b0, mobilenet_v2)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--loss-type", type=str, default=None, help="Loss function type: 'ce', 'weighted_ce', 'focal'")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing dataset images and metadata")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints and logs")
    parser.add_argument("--use-synthetic", action="store_true", help="Force use of synthetic benchmark dataset fixture")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for determinism")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI arguments override config file values
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    seed_everything(seed)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    backbone_name = args.backbone or model_cfg.get("backbone", "resnet18")
    num_classes = model_cfg.get("num_classes", 7)
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 5)
    batch_size = args.batch_size if args.batch_size is not None else data_cfg.get("batch_size", 16)
    learning_rate = args.lr if args.lr is not None else train_cfg.get("learning_rate", 1e-4)
    loss_type = args.loss_type or train_cfg.get("loss_type", "weighted_ce")
    output_dir = Path(args.output_dir or output_cfg.get("checkpoint_dir", "models"))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = data_cfg.get("image_size", 224)
    num_workers = data_cfg.get("num_workers", 0)

    # 1. Dataset Resolution & Ingestion
    data_dir = Path(args.data_dir or data_cfg.get("data_dir", "data"))
    raw_images_dir = data_dir / "raw" / "images"
    raw_meta_file = data_dir / "raw" / "metadata.csv"
    synthetic_dir = data_dir / "synthetic"

    use_synthetic = args.use_synthetic or not (raw_images_dir.exists() and raw_meta_file.exists())

    if use_synthetic:
        logger.info("Using reproducible synthetic dermoscopy dataset fixture for training...")
        if not (synthetic_dir / "metadata.csv").exists():
            generate_synthetic_dermoscopy_dataset(
                output_dir=synthetic_dir,
                num_samples=data_cfg.get("synthetic_samples", 140),
                seed=seed,
                image_size=image_size,
            )
        metadata_df = pd.read_csv(synthetic_dir / "metadata.csv")
        image_dir = synthetic_dir / "images"
    else:
        logger.info(f"Loading real dermatological dataset from: {raw_images_dir}")
        metadata_df = pd.read_csv(raw_meta_file)
        image_dir = raw_images_dir

    logger.info(f"Total dataset records: {len(metadata_df)}")

    # 2. Group-Stratified Partitioning (Zero-Leakage)
    splits_dir = data_dir / "processed" / "splits"
    if (splits_dir / "train.csv").exists() and not use_synthetic:
        logger.info(f"Loading existing split manifests from: {splits_dir}")
        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df = pd.read_csv(splits_dir / "val.csv")
    else:
        logger.info("Generating group-stratified zero-leakage splits on lesion_id...")
        train_df, val_df, test_df = split_skin_dataset(
            metadata_df,
            train_ratio=data_cfg.get("train_ratio", 0.70),
            val_ratio=data_cfg.get("val_ratio", 0.15),
            test_ratio=data_cfg.get("test_ratio", 0.15),
            seed=seed,
        )
        save_splits(train_df, val_df, test_df, splits_dir)

    logger.info(f"Partitions: Train={len(train_df)} | Val={len(val_df)}")

    # 3. Augmentation & DataLoader Setup
    train_transforms = get_train_transforms(image_size=image_size)
    eval_transforms = get_eval_transforms(image_size=image_size)

    train_dataset = SkinLesionDataset(train_df, image_dir=image_dir, transform=train_transforms)
    val_dataset = SkinLesionDataset(val_df, image_dir=image_dir, transform=eval_transforms)

    train_labels = train_dataset.get_labels()
    class_weights = compute_inverse_class_weights(train_labels, num_classes=num_classes)
    logger.info(f"Computed inverse class weights for loss balancing: {class_weights.tolist()}")

    use_balanced_sampler = data_cfg.get("use_balanced_sampler", True)
    if use_balanced_sampler:
        sampler = create_balanced_sampler(train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 4. Model Architecture & Loss Construction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} ({'CUDA acceleration' if device.type == 'cuda' else 'CPU'})")

    model = build_classifier(
        backbone=backbone_name,
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        dropout_rate=model_cfg.get("dropout_rate", 0.3),
        head_type=model_cfg.get("head_type", "linear"),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    )

    weights_on_device = class_weights.to(device)
    criterion = get_loss_function(
        loss_type=loss_type,
        class_weights=weights_on_device,
        label_smoothing=train_cfg.get("label_smoothing", 0.05),
        gamma=train_cfg.get("focal_gamma", 2.0),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=train_cfg.get("weight_decay", 1e-2),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    # 5. Trainer Engine Execution
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config={
            "backbone": backbone_name,
            "num_classes": num_classes,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
            "loss_type": loss_type,
            "use_amp": train_cfg.get("use_amp", True),
            "max_grad_norm": train_cfg.get("max_grad_norm", 1.0),
            "monitor_metric": train_cfg.get("monitor_metric", "val_macro_f1"),
            "monitor_mode": train_cfg.get("monitor_mode", "max"),
            "class_names": SKIN_CLASSES,
        },
        output_dir=output_dir,
    )

    history = trainer.fit(num_epochs=epochs)
    logger.info("Training complete! Model checkpoints and logs successfully exported.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
