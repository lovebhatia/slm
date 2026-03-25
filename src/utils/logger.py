# src/utils/logger.py
import os
import json
from datetime import datetime


class TrainingLogger:
    """
    Logs training metrics to a JSON file.
    Used to plot loss curves and track experiments.
    """

    def __init__(self, log_dir: str = "logs/"):
        self.log_dir  = log_dir
        self.log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.records  = []
        os.makedirs(log_dir, exist_ok=True)

    def log(self, step: int, train_loss: float, val_loss: float = None):
        record = {"step": step, "train_loss": round(train_loss, 6)}
        if val_loss is not None:
            record["val_loss"] = round(val_loss, 6)
        self.records.append(record)
        tag = f"val: {val_loss:.4f}" if val_loss else ""
        print(f"step {step:5d} | train loss: {train_loss:.4f} {tag}")

    def save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"Log saved → {self.log_file}")