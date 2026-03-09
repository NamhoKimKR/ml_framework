from __future__ import annotations

from typing import Any, Dict

from ml_framework.core.callbacks import Callback
from ml_framework.core.base_trainer import BaseTrainer


class LoggingCallback(Callback):
    """
    Simple stdout logging callback.

    Intended for:
        - Debugging
        - Minimal local experiments
    """

    def on_epoch_end(
            self,
            trainer: BaseTrainer,
            *,
            epoch: int,
            split: str,
            logs: Dict[str, float],
            **_: Any,
    ) -> None:
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in logs.items())
        print(f"[Epoch {epoch:05d}][{split:5s}]\t{metrics}")

    # def on_train_end(
    #         self,
    #         trainer: BaseTrainer,
    #         *args: Any,
    #         **kwargs: Any,
    # ) -> None:
        
    #     metrics = ", ".join(f"{k}={v:.4f}" for k, v in logs.items())
    #     print(f"[Epoch {epoch:05d}][{split:5s}]\t{metrics}")