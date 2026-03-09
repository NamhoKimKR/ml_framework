from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ml_framework.core.callbacks import Callback
from ml_framework.core.base_trainer import BaseTrainer


class ClearMLCallback(Callback):
    """
    ClearML monitoring callback.

    What it logs:
        - Train/val loss per epoch (report_scalar)
        - Val metrics per epoch (report_scalar)  # any keys in logs except "loss"
        - LR per epoch (report_scalar)
        - Epoch time (seconds) per epoch (report_scalar)
        - Evaluation summary (report_single_value) for each metric

    Notes:
        - This callback assumes trainer calls:
            on_epoch_end(epoch=..., split=..., logs=...)
            on_evaluate_end(logs=...)
        - It is robust to missing keys and missing ClearML installation.
        - It will NEVER crash training due to ClearML exceptions.
    """

    def __init__(
            self,
            *,
            project_name: str,
            task_name: str,
            task_type: str = "training",
            tags: Optional[list[str]] = None,
            reuse_last_task_id: bool = False,
            auto_connect_frameworks: bool = False,
            enabled: bool = True,
    ) -> None:
        """
        Args:
            project_name: ClearML project name.
            task_name: ClearML task name.
            task_type: ClearML task type string ("training", "testing", ...).
            tags: Optional list of tags attached to the task.
            reuse_last_task_id: Whether to reuse last task id (ClearML feature).
            auto_connect_frameworks: Let ClearML auto-connect frameworks.
            enabled: If False, this callback becomes a no-op.
        """
        self.enabled = bool(enabled)

        self.project_name = project_name
        self.task_name = task_name
        self.task_type = task_type
        self.tags = tags
        self.reuse_last_task_id = reuse_last_task_id
        self.auto_connect_frameworks = auto_connect_frameworks

        # ClearML objects (lazy-init)
        self.task: Any = None
        self.logger: Any = None
        self._clearml_available: bool = False

        # Timing
        self._epoch_start_time: Optional[float] = None

        # Optional dependency import (do not crash framework import)
        try:
            import clearml  # type: ignore  # noqa: F401
            self._clearml_available = True
        except Exception:
            self._clearml_available = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_task(
            self,
    ) -> None:
        """
        Lazily create ClearML Task and logger.
        """
        if not self.enabled:
            return
        if not self._clearml_available:
            return
        if self.task is not None and self.logger is not None:
            return
        
        try:
            from clearml import Task # type: ignore

            self.task = Task.init(
                project_name=self.project_name,
                task_name=self.task_name,
                task_type=self.task_type,
                reuse_last_task_id=self.reuse_last_task_id,
                auto_connect_frameworks=self.auto_connect_frameworks,
            )

            if self.tags:
                try:
                    self.task.add_tags(self.tags)
                except Exception:
                    pass

            try:
                self.logger = self.task.get_logger()
            except Exception:
                self.logger = None

        except Exception:
            # Hard fail protection: never block training
            self.task = None
            self.logger = None

    def _report_scalar(
            self,
            *,
            title: str,
            series: str,
            value: float,
            iteration: int,
    ) -> None:
        if not self.enabled:
            return
        if self.logger is None:
            return
        try:
            self.logger.report_scalar(
                title=str(title),
                series=str(series), 
                value=float(value),
                iteration=int(iteration)),
        except Exception:
            return
        
    def _report_single_value(
            self, 
            *, 
            name: str, 
            value: float,
    ) -> None:
        if not self.enabled:
            return
        if self.logger is None:
            return
        try:
            self.logger.report_single_value(
                name=str(name),
                value=float(value),
            )
        except Exception:
            return
        
    def _get_lr(
            self,
            trainer: BaseTrainer,
    ) -> Optional[float]:
        """
        Extract learning rate from trainer.optimizer if present.

        Notes:
            - Works for torch optimizers (param_groups).
            - Returns None if unavailable.
        """
        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            return None
        groups = getattr(opt, "param_groups", None)
        if not groups:
            return None
        try:
            lr = groups[0].get("lr", None)
            return float(lr) if lr is not None else None
        except Exception:
            return None
        
    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def on_train_start(
            self,
            trainer: BaseTrainer,
            **kwargs: Any,
    ) -> None:
        """
        Initialize ClearML task at training start.
        """
        self._ensure_task()

        # Best-effort: connect config (optional)
        cfg = getattr(trainer, "cfg", None)
        if cfg is not None and self.task is not None:
            try:
                from omegaconf import OmegaConf # type: ignore

                text = OmegaConf.to_yaml(cfg, resolve=True)
                self.task.connect_configuration(text)
            except Exception:
                pass    
    
    def on_epoch_start(
            self,
            trainer: BaseTrainer,
            **kwargs: Any,
    ) -> None:
        """
        Start epoch timer.
        """
        self._ensure_task()
        self._epoch_start_time = time.perf_counter()

    def on_epoch_end(
            self,
            trainer: BaseTrainer,
            *,
            epoch: int,
            split: str,
            logs: Dict[str, Any],
            **kwargs: Any,
    ) -> None:
        """
        Log epoch-level scalars.
        
        Expected logs example:
            {"loss": 0.123, "acc": 0.95}
        """
        self._ensure_task()

        ep = int(epoch)
        sp = str(split)

        # 1) loss
        loss_val = logs.get("loss", None)
        if loss_val is not None:
            try:
                self._report_scalar(
                    title="Loss",
                    series=sp,
                    value=float(loss_val),
                    iteration=ep,
                )
            except Exception:
                pass

        # 2) metrics (everyting except loss)
        for k, v in logs.items():
            if str(k) == "loss":
                continue
            try:
                self._report_scalar(
                    title="Metrics",
                    series=f"{sp}/{k}",
                    value=float(v),
                    iteration=ep,
                )
            except Exception:
                pass

        # 3) lr
        if sp == "train":
            lr = self._get_lr(trainer)
            if lr is not None:
                try:
                    self._report_scalar(
                        title="Learning rate",
                        series="lr",
                        value=float(lr),
                        iteration=ep,
                    )
                except Exception:
                    pass

        # 4) epoch time
        if sp == "train":
            if self._epoch_start_time is not None:
                sec = time.perf_counter() - self._epoch_start_time
                self._report_scalar(
                    title="Epoch time",
                    series = "epoch_sec",
                    value=float(sec),
                    iteration=ep,
                )

    def on_evaluate_end(
            self,
            trainer: BaseTrainer,
            *,
            logs: Dict[str, Any],
            **kwargs: Any,
    ) -> None:
        """
        Log evaluation summary via report_single_value.

        Notes:
            - ClearML report_single_value is suitable for "final numbers"
              rather than time-series curves.
            - We log each numeric metric under "eval/<key>".
        """
        self._ensure_task()

        for k, v in logs.items():
            try:
                fv = float(v)
            except Exception:
                continue
            self._report_single_value(name=f"eval/{k}", value=fv)