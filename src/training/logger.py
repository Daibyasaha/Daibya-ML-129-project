"""Metric tracking and history export to JSON and CSV formats."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
import pandas as pd


class TrainingHistoryLogger:
    """Records per-epoch training and validation metrics, exporting to JSON and CSV."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, List[Any]] = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "train_macro_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "lr": [],
            "epoch_time_sec": [],
        }

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        train_macro_f1: float,
        val_loss: float,
        val_acc: float,
        val_macro_f1: float,
        lr: float,
        epoch_time_sec: float,
        extra_metrics: Dict[str, Any] = None,
    ) -> None:
        """Appends metrics for a single training epoch."""
        self.history["epoch"].append(int(epoch))
        self.history["train_loss"].append(float(train_loss))
        self.history["train_acc"].append(float(train_acc))
        self.history["train_macro_f1"].append(float(train_macro_f1))
        self.history["val_loss"].append(float(val_loss))
        self.history["val_acc"].append(float(val_acc))
        self.history["val_macro_f1"].append(float(val_macro_f1))
        self.history["lr"].append(float(lr))
        self.history["epoch_time_sec"].append(float(epoch_time_sec))

        if extra_metrics:
            for k, v in extra_metrics.items():
                if k not in self.history:
                    # Pad previous epochs with None
                    self.history[k] = [None] * (len(self.history["epoch"]) - 1)
                self.history[k].append(v)

    def save(self) -> tuple[Path, Path]:
        """Exports the recorded history to both JSON and CSV files."""
        json_path = self.output_dir / "training_history.json"
        csv_path = self.output_dir / "training_history.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        df = pd.DataFrame(self.history)
        df.to_csv(csv_path, index=False)

        return json_path, csv_path

    def get_latest(self) -> Dict[str, Any]:
        """Returns the most recent epoch's metrics."""
        if not self.history["epoch"]:
            return {}
        return {k: v[-1] for k, v in self.history.items()}
