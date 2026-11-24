# Project: Academic Skin Disease Classification with PyTorch & Transfer Learning

## Architecture
A modular, reproducible machine learning system for dermatological skin lesion classification based on the canonical HAM10000 7-class benchmark and ISIC standards. Designed for academic research, rigorous benchmark evaluation, and deterministic verification.

```
+------------------------------------------------------------------------------------+
|                                    User & CLI                                      |
+--------------------------+-----------------------------+---------------------------+
                           |                             |
                           v                             v
                   [ train.py ]                    [ evaluate.py ]
                           |                             |
                           +--------------+--------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
|                                   Core Modules                                     |
|  +---------------------------+  +--------------------------+  +------------------+ |
|  |         src/data/         |  |       src/models/        |  |  src/evaluation/ | |
|  | - dataset.py              |  | - classifier.py          |  | - metrics.py     | |
|  | - transforms.py           |  |   (ResNet18, EfficientNet|  | - plots.py       | |
|  | - split.py                |  |    MobileNetV2, Heads)   |  |   (Confusion Mat,| |
|  |   (Zero-leakage Group)    |  | - loss.py                |  |    ROC/PR Curves)| |
|  | - imbalance.py            |  |   (Weighted CE, Focal)   |  +------------------+ |
|  | - synthetic.py            |  +--------------------------+                       |
|  |   (Offline generator)     |                                                     |
|  +---------------------------+                                                     |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
|                           Deterministic Verification                               |
| - verify_pipeline.py / tests/test_pipeline.py                                      |
| - End-to-end data loading, forward pass, loss computation, checkpointing, metrics  |
+------------------------------------------------------------------------------------+
```

## Feature Inventory
Every feature from ORIGINAL_REQUEST.md mapped to architecture.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Dataset Acquisition & Caching | Ingestion of HAM10000 7-class benchmark images and metadata with integrity validation | M1 | R1, Survey |
| 2 | Offline Synthetic Benchmark Generator | Procedural generation of realistic dermoscopy images and metadata (< 3s) for 100% offline reproducible testing | M1 | R1, Survey |
| 3 | Group-Stratified Partitioning | Stratified partitioning into Train (70%), Val (15%), Test (15%) grouped by `lesion_id` to guarantee zero patient/lesion leakage | M1 | R1, Survey |
| 4 | Dermatological Augmentation & Preprocessing | Geometric rotational invariance (0-360 deg, flips) and bounded photometric transforms (mild jitter) with ImageNet normalization | M1 | R1, Survey |
| 5 | Class Imbalance Mitigation | Inverse-frequency class weights for CrossEntropyLoss and WeightedRandomSampler for balanced DataLoader batches | M1 | R1, Survey |
| 6 | CNN Transfer Learning Architecture | Configurable pre-trained backbones (`resnet18` primary baseline, `efficientnet_b0`, `mobilenet_v2`) with custom classification head (Dropout + Linear) | M2 | R2, Survey |
| 7 | End-to-End Training Pipeline | Configurable hyperparameters (lr, batch size, epochs, AdamW/SGD, CosineAnnealing/Plateau scheduler), AMP mixed precision, device-agnostic execution | M2 | R2, Survey |
| 8 | Model Checkpointing & History Logging | Atomic saving of best model (val macro-F1 / loss) and final model weights; logging history to JSON and CSV | M2 | R2, Survey |
| 9 | Comprehensive Evaluation Suite | Benchmark execution on held-out test set generating Top-1/Top-k accuracy, Balanced Accuracy, Macro & Weighted Precision, Recall, F1-score | M3 | R3, Survey |
| 10 | Confusion Matrix & Classification Reports | Programmatic export of raw count CSV, row-normalized CSV, heatmap visualization PNG, and structured classification report JSON/TXT | M3 | R3, Survey |
| 11 | Deterministic Seeding Engine | Reproducible random seed control across Python, NumPy, PyTorch CPU & CUDA (`seed_everything`) | M4 | Acceptance, Survey |
| 12 | Automated Verification Test Suite | End-to-end test script verifying data loading, model instantiation, forward pass, loss, checkpoint save/restore, and evaluation | M4 | Acceptance, Survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | R1: Dataset & Preprocessing Pipeline | `src/data/` (dataset, transforms, split, imbalance, synthetic) + synthetic fixture generation | none | COMPLETED |
| 2 | R2: CNN Architecture & Training Pipeline | `src/models/`, `src/training/`, `train.py`, checkpoints, training history logging | M1 | COMPLETED |
| 3 | R3: Evaluation & Benchmarking Suite | `src/evaluation/`, `evaluate.py`, metrics export, confusion matrix generation | M2 | COMPLETED |
| 4 | Verification & Acceptance Testing | `verify_pipeline.py`, `tests/test_pipeline.py`, requirements, configuration | M1, M2, M3 | COMPLETED |

## Interface Contracts

### 7-Class Dermatology Taxonomy
```python
SKIN_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(SKIN_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(SKIN_CLASSES)}
NUM_CLASSES = 7
```

### Module `src/data` Interface
- `SkinLesionDataset(dataframe: pd.DataFrame, image_dir: Path, transform: Callable) -> Dataset`: Returns `(image_tensor: Tensor[3, 224, 224], label: int)`.
- `get_train_transforms(image_size: int = 224) -> Transform`: Augmented torchvision composition.
- `get_eval_transforms(image_size: int = 224) -> Transform`: Deterministic resize + ImageNet normalization.
- `split_skin_dataset(metadata_df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`: Group-stratified on `lesion_id`.
- `compute_inverse_class_weights(train_labels: Sequence[int], num_classes=7) -> torch.Tensor`: Normalized inverse frequency tensor.
- `generate_synthetic_dermoscopy_dataset(output_dir: Path, num_samples=140, seed=42) -> pd.DataFrame`: Fast offline dataset generator.

### Module `src/models` Interface
- `build_classifier(backbone: str = "resnet18", num_classes: int = 7, pretrained: bool = True, dropout_rate: float = 0.3) -> nn.Module`: Instantiates transfer learning model.
- `get_loss_function(loss_type: str = "weighted_ce", class_weights: Optional[Tensor] = None, label_smoothing: float = 0.0) -> nn.Module`.

### Checkpoint Format Contract
Saved checkpoint dictionary keys:
```python
{
    "epoch": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "best_val_metric": float,
    "metric_name": str,
    "class_names": list[str],
    "backbone": str,
    "config": dict
}
```

### Module `src/evaluation` Interface
- `evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device, class_names: list[str]) -> dict`: Returns complete dictionary of overall metrics, per-class metrics, confusion matrix, and prediction arrays.
- `save_evaluation_artifacts(results: dict, output_dir: Path) -> None`: Exports `metrics.json`, `classification_report.txt`, `confusion_matrix.csv`, `confusion_matrix_normalized.csv`, `confusion_matrix.png`.

## Code Layout
```
c:\Users\SAYAN\Project\ML_Project_Daibya\
├── configs/
│   └── default_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── splits/
│   └── synthetic/
├── models/                     # Saved checkpoints (.pth)
├── reports/                    # Evaluation metrics, confusion matrix, logs
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   ├── split.py
│   │   ├── imbalance.py
│   │   ├── synthetic.py
│   │   └── downloader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   └── loss.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── logger.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── plots.py
│   └── utils/
│       ├── __init__.py
│       └── seed.py
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
├── train.py
├── evaluate.py
├── verify_pipeline.py
├── requirements.txt
└── README.md
```
