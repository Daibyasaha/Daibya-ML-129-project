"""CNN transfer learning architectures for dermatological skin lesion classification."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger(__name__)


def _get_backbone(backbone_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Instantiates a torchvision backbone and extracts (feature_extractor, in_features).

    Supports graceful offline fallback if pre-trained weights cannot be fetched from remote.
    """
    name = backbone_name.lower().strip()

    try:
        if name == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
        elif name == "resnet34":
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet34(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
        elif name == "resnet50":
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
        elif name == "efficientnet_b0":
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = tv_models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
        elif name == "mobilenet_v2":
            weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            model = tv_models.mobilenet_v2(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
        else:
            supported = ["resnet18", "resnet34", "resnet50", "efficientnet_b0", "mobilenet_v2"]
            raise ValueError(f"Unsupported backbone: '{name}'. Choose from: {supported}")
    except Exception as e:
        if pretrained:
            logger.warning(
                f"Failed to download pretrained weights for {name} ({e}). "
                "Falling back to randomly initialized backbone for offline execution."
            )
            return _get_backbone(name, pretrained=False)
        raise

    return model, in_features


class ClassificationHead(nn.Module):
    """Configurable classification head with dropout regularization."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        head_type: str = "linear",
        dropout_rate: float = 0.3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.head_type = head_type.lower()
        if self.head_type == "linear":
            self.net = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        elif self.head_type == "mlp":
            self.net = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2.0),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unsupported head_type: '{head_type}'. Choose 'linear' or 'mlp'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SkinClassifier(nn.Module):
    """End-to-End Transfer Learning Skin Lesion Classifier."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        head_type: str = "linear",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

        self.backbone, self.in_features = _get_backbone(backbone_name, pretrained=pretrained)
        self.classifier = ClassificationHead(
            in_features=self.in_features,
            num_classes=num_classes,
            head_type=head_type,
            dropout_rate=dropout_rate,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self) -> None:
        """Freezes all feature extractor parameters for linear probing."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes feature extractor parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float = 1e-2,
    ) -> List[Dict[str, Union[List[torch.nn.Parameter], float]]]:
        """Returns differential learning rate parameter groups for transfer learning."""
        groups = []
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            })
        groups.append({
            "params": list(self.classifier.parameters()),
            "lr": head_lr,
            "weight_decay": weight_decay,
        })
        return groups


def build_classifier(
    backbone: str = "resnet18",
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    head_type: str = "linear",
    freeze_backbone: bool = False,
) -> SkinClassifier:
    """Factory function to instantiate a configured SkinClassifier."""
    return SkinClassifier(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        head_type=head_type,
        freeze_backbone=freeze_backbone,
    )
