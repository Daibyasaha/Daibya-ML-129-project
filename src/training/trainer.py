"""End-to-End PyTorch Trainer supporting AMP, gradient clipping, checkpointing, and metric logging."""

import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from .logger import TrainingHistoryLogger
from ..data import SKIN_CLASSES

logger = logging.getLogger(__name__)


class Trainer:
    """Configurable training engine with automatic mixed precision and metric checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Union[str, Path] = "output",
    ):
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger = TrainingHistoryLogger(self.output_dir)

        # Mixed precision setup
        self.use_amp = bool(self.config.get("use_amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))

        # Checkpointing monitor
        self.monitor_metric = self.config.get("monitor_metric", "val_macro_f1")
        self.monitor_mode = self.config.get("monitor_mode", "max")
        if self.monitor_mode == "max":
            self.best_metric_value = -float("inf")
        else:
            self.best_metric_value = float("inf")

        self.class_names = self.config.get("class_names", SKIN_CLASSES)
        self.backbone_name = getattr(self.model, "backbone_name", "unknown")

    def _is_improvement(self, current: float) -> bool:
        if self.monitor_mode == "max":
            return current > self.best_metric_value
        return current < self.best_metric_value

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs one full training epoch over the dataset."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        start_time = time.time()

        for step, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = outputs.argmax(dim=-1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy())

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(1, total_samples)
        acc = float(accuracy_score(all_targets, all_preds))
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "epoch_time_sec": elapsed,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluates model performance on the validation split."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = outputs.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / max(1, total_samples)
        acc = float(accuracy_score(all_targets, all_preds))
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "macro_f1": macro_f1,
        }

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        epoch: int,
        is_best: bool = False,
    ) -> Path:
        """Saves model weights and optimizer state following the project interface contract."""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_val_metric": self.best_metric_value,
            "metric_name": self.monitor_metric,
            "class_names": self.class_names,
            "backbone": self.backbone_name,
            "config": self.config,
        }

        torch.save(checkpoint, save_path)
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

        return save_path

    def fit(self, num_epochs: int) -> Dict[str, Any]:
        """Executes full training across the specified number of epochs."""
        logger.info(
            f"Starting training for {num_epochs} epochs on device '{self.device}' "
            f"(AMP: {self.use_amp}, Backbone: {self.backbone_name})"
        )

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Record history
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_acc=train_metrics["accuracy"],
                train_macro_f1=train_metrics["macro_f1"],
                val_loss=val_metrics["loss"],
                val_acc=val_metrics["accuracy"],
                val_macro_f1=val_metrics["macro_f1"],
                lr=current_lr,
                epoch_time_sec=train_metrics["epoch_time_sec"],
            )

            # Determine monitored metric value
            current_monitor_val = val_metrics.get(
                "macro_f1" if self.monitor_metric == "val_macro_f1" else (
                    "loss" if self.monitor_metric == "val_loss" else "accuracy"
                ),
                val_metrics["macro_f1"],
            )

            is_best = self._is_improvement(current_monitor_val)
            if is_best:
                self.best_metric_value = current_monitor_val
                self.save_checkpoint(
                    self.checkpoint_dir / "best_model.pth",
                    epoch=epoch,
                    is_best=True,
                )

            logger.info(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['macro_f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['macro_f1']:.4f} "
                f"({'★ BEST' if is_best else ''}) LR: {current_lr:.6f} Time: {train_metrics['epoch_time_sec']:.1f}s"
            )

        # Save final model checkpoint
        self.save_checkpoint(
            self.checkpoint_dir / "final_model.pth",
            epoch=num_epochs,
            is_best=False,
        )

        # Save training logs to disk
        self.logger.save()

        return self.logger.history
