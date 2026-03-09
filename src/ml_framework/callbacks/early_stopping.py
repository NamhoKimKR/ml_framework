from __future__ import annotations

from typing import Any, Dict, Optional

from ml_framework.core.callbacks import Callback
from ml_framework.core.base_trainer import BaseTrainer


class EarlyStoppingCallback(Callback):
    """
    Early stopping based on monitored metric.

    Default behavior:
        - Monitor validation loss
        - Stop training ig no improvement is seen for 'patience' epochs
    """

    def __init__(
            self,
            *,
            monitor: str = "loss",
            mode: str = "min",
            patience: int = 10,
            min_delta: float = 0.0,
    ) -> None:
        assert mode in ("min", "max")

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best: Optional[float] = None
        self.num_bad_epochs = 0

    def on_epoch_end(
            self,
            trainer: BaseTrainer,
            *,
            split: str,
            logs: Dict[str, float],
            **_: Any,   
    ) -> None:
        if split != "val":
            return
        
        current = logs.get(self.monitor)
        if self.best is None:
            self.best = current
            return
        
        improved = (
            current < self.best - self.min_delta
            if self.mode == "min"
            else current > self.best + self.min_delta
        )

        if improved:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            trainer.should_stop = True

        