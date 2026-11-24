# Academic Skin Disease Classification with PyTorch & CNN Transfer Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.14+-3776ab.svg)](https://www.python.org/)

An academic research and benchmark system for multi-class dermatological skin lesion classification based on the canonical HAM10000 dataset and ISIC standards.

---

## 1. System Overview & Medical Context

Skin cancer is among the most prevalent human malignancies worldwide. Dermoscopy enables non-invasive visualization of subsurface skin structures, yet early visual discrimination between harmless melanocytic nevi and lethal malignant melanomas remains challenging.

This system provides a rigorous, modular, and reproducible machine learning pipeline implementing:
- **Zero-Leakage Group Partitioning**: Strict `lesion_id` grouping preventing patient-level image leakage across train/validation/test splits.
- **Dermatological Domain Augmentations**: Geometric rotational symmetry ($0^\circ\text{--}360^\circ$), horizontal/vertical flipping, and bounded photometric calibration.
- **Extreme Class Imbalance Mitigation**: Inverse-frequency loss weights (`CrossEntropyLoss`), `WeightedRandomSampler`, and `FocalLoss`.
- **Transfer Learning Backbones**: Pre-trained deep convolutional neural networks (`ResNet-18`, `EfficientNet-B0`, `MobileNet-V2`) with custom dropout-regularized classification heads.
- **Comprehensive Academic & Clinical Benchmarking**: Top-1 and differential Top-$k$ accuracy, Balanced Accuracy, per-class sensitivity/specificity, macro/weighted F1-scores, Expected Calibration Error (ECE), and programmatic confusion matrix artifacts.

---

## 2. 7-Class Dermatological Taxonomy

The diagnostic categories follow the international HAM10000 / ISIC standard:

| Class Code | Medical Diagnosis | Pathology Type | Clinical Significance |
| :--- | :--- | :--- | :--- |
| **`nv`** | Melanocytic nevi | Benign melanocytic proliferation | Harmless mole; majority class (~67%) |
| **`mel`** | Melanoma | Malignant melanocytic tumor | High mortality risk; critical false-negative penalty |
| **`bkl`** | Benign keratosis-like lesions | Benign non-melanocytic | Seborrheic keratoses, solar lentigines |
| **`bcc`** | Basal cell carcinoma | Malignant non-melanocytic | Common skin cancer; destructive if untreated |
| **`akiec`** | Actinic keratoses & Intraepithelial Ca | Pre-cancerous / In-situ | Squamous cell carcinoma precursor |
| **`vasc`** | Vascular lesions | Benign vascular | Hemangiomas, angiokeratomas |
| **`df`** | Dermatofibroma | Benign fibrohistiocytic | Extreme minority benign nodule (~1.1%) |

---

## 3. Directory Layout

```
ML_Project_Daibya/
├── configs/
│   └── default_config.yaml         # Configurable hyperparameters & paths
├── data/
│   ├── raw/                        # Real HAM10000 dataset (images/ and metadata.csv)
│   ├── processed/splits/           # Generated zero-leakage split manifests
│   └── synthetic/                  # Fast deterministic synthetic benchmark fixture
├── models/                         # Saved checkpoints (best_model.pth, final_model.pth)
├── reports/                        # Evaluation metrics, confusion matrices, logs
├── src/
│   ├── data/
│   │   ├── dataset.py              # PyTorch Dataset
│   │   ├── transforms.py           # Augmentations (Train vs Eval)
│   │   ├── split.py                # Zero-leakage lesion-grouped splitter
│   │   ├── imbalance.py            # Inverse frequency weights & balanced sampler
│   │   ├── synthetic.py            # Offline benchmark generator (< 3s)
│   │   └── downloader.py           # Dataset acquisition & SHA256 validation
│   ├── models/
│   │   ├── classifier.py           # Transfer learning CNN architecture
│   │   └── loss.py                 # Class-weighted CE & Focal Loss
│   ├── training/
│   │   ├── trainer.py              # Training loop, AMP, gradient clipping
│   │   └── logger.py               # History exporter (JSON & CSV)
│   ├── evaluation/
│   │   ├── metrics.py              # Top-1/k, Balanced Acc, F1, per-class sensitivity
│   │   └── plots.py                # Heatmaps, curves, and reports
│   └── utils/
│       └── seed.py                 # Deterministic seed_everything
├── tests/
│   └── test_pipeline.py            # Integration test suite
├── train.py                        # CLI training entrypoint
├── evaluate.py                     # CLI evaluation & benchmarking entrypoint
├── verify_pipeline.py              # End-to-end acceptance verification script
├── requirements.txt                # Pinned dependencies
└── README.md
```

---

## 4. Setup & Installation

### Virtual Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5. Usage & Execution

### 5.1 Automated Pipeline Acceptance Verification
To verify the entire pipeline (data loading, model forward pass, checkpoint save/restore, metric computation, and dry-run training) in under 15 seconds:
```powershell
python verify_pipeline.py
```
Or execute via pytest:
```powershell
pytest tests/test_pipeline.py -v
```

### 5.2 Training a Model
To train ResNet-18 with mixed precision on the benchmark:
```powershell
python train.py --config configs/default_config.yaml --backbone resnet18 --epochs 5 --batch-size 16
```

Supported Backbones:
- `resnet18` (Primary recommended baseline)
- `efficientnet_b0` (Parameter-efficient depthwise-separable CNN)
- `mobilenet_v2` (Lightweight edge-deployable CNN)

Supported Loss Functions:
- `weighted_ce` (Inverse-frequency class-weighted cross-entropy)
- `focal` (Multi-class focal loss with $\gamma=2.0$)
- `ce` (Standard cross-entropy with label smoothing)

### 5.3 Evaluating & Benchmarking
To benchmark a trained model on the held-out test split:
```powershell
python evaluate.py --checkpoint models/checkpoints/best_model.pth --output-dir reports
```

Generated evaluation artifacts:
- `reports/metrics.json`: Complete dictionary of all quantitative scores.
- `reports/classification_report.txt`: Formatted clinical summary table.
- `reports/confusion_matrix.csv`: Raw integer count confusion matrix.
- `reports/confusion_matrix_normalized.csv`: Row-normalized true-class sensitivity matrix.
- `reports/confusion_matrix.png`: High-resolution visual heatmap.
- `reports/training_curves.png`: Training & validation loss/accuracy learning curves.

---

## 6. Reproducibility & Determinism
All scripts accept a `--seed` argument (default: `42`). Setting the seed guarantees identical bitwise outputs across Python `random`, `numpy`, PyTorch CPU, and CUDA cuDNN routines.
