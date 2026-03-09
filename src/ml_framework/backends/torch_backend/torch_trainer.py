from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml_framework.core.base_trainer import BaseTrainer, CfgType
from ml_framework.core.callbacks import Callback
from ml_framework.core.base_model import BaseModel
from ml_framework.core.build import build_from_cfg
from ml_framework.core.registry.instances import LOSS_REGISTRY, OPTIM_REGISTRY, SCHED_REGISTRY



# ----------------------------------------------------------------------
# 0) Default registry entries (short-name -> torch built-ins)
# ----------------------------------------------------------------------
def _try_register_defaults(
) -> None:
    """
    Register common torch built-ins into registries to support shrot-name YAML.

    Notes:
        - This keeps YAML simple (name: "adam") while still using registries.
        - If the key already exists, it is left untouched.
    """
    # Losses


# ----------------------------------------------------------------------
# 1) Data containers
# ----------------------------------------------------------------------
@dataclass
class TrainStepOutput:
    """
    Container for a single training step output.

    Notes:
        - Keep it minimal for v0.
        - You can extend this to include arbitrary tensors for debugging.
    """
    loss: torch.Tensor
    metrics: Dict[str, float]


@dataclass
class EpochOutput:
    """
    Aggregated epoch output (train/val/test).

    Attributes:
        loss: Average loss across steps.
        metrics: Average metrics across steps.
        num_steps: Number of steps aggregated.
    """
    loss: float
    metrics: Dict[str, float]
    num_steps: int


# ----------------------------------------------------------------------
# 2) TorchTrainer
# ----------------------------------------------------------------------
class TorchTrainer(BaseTrainer):
    """
    Torch trainer implementaiton for a single experiment run.

    This class is responsible for:
        - Building optimizer / criterion / scheduler / AMP scaler from cfg
        - Running epoch/step loops with optimizer + AMP support
        - Calling callbacks at standard hook points
        - Managing resumable trainer states (epoch/global_step/best)

    This class should NOT:
        - Implement task-specific dataset details (provide hooks)
        - Implement task-specific forward/metric logic (provide hooks)
    """
    backend: str = "torch"

    def __init__(
            self,
            cfg: CfgType,
            model: BaseModel,
            callbacks: Optional[list[Callback]] = None,
    ) -> None:
        super().__init__(cfg=cfg, model=model, callbacks=callbacks)

        # Normalize cfg to "params" level if needed
        # If cfg contains {"name": ..., "params": {....}}, use cfg["params"].
        # Ohterwise, use cfg as-is.
        self.p = self._unwrap_params(cfg)

        # Training state
        self.epoch: int = 0
        self.global_step: int = 0
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None

        # Runtime components (initialized in fit()).
        self.device: torch.device = torch.device("cpu")
        self.criterion: Optional[Any] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self._amp_enabled: bool = False
        self._amp_dtype: torch.dtype = torch.float16

        # Dataloaders
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

        self.should_stop: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
            self,
    ) -> Dict[str, float]:
        """
        Run a full training loop for ONE experiment.

        Expected cfg schema (params-level):
            - seed (optional)
            - device (optional)
            - epochs
            - loss: {name: ..., ...} or {target: ..., ...}
            - optimizer: {name: ..., ...} or {target: ..., ...}
            - scheduler (optional): {name: ..., ...} or {target: ..., ...}
            - amp (optional): {enabled: bool, dtype: str}
        
        Returns:
            Dict[str, float]: Final validation metrics if available,
                otherwise final training metrics.
        """
        # Seed
        seed = self._get(self.p, "seed", None)
        if seed is not None:
            torch.manual_seed(seed)

        # Build runtime components
        self.device = self._configure_device()
        self.model.to_device(self.device)

        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler, self._amp_enabled, self._amp_dtype = self._build_amp()

        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()

        # Callbacks: train start
        self._callback("on_train_start")

        epochs = int(self._get(self.p, "epochs", 1))
        final_logs: Dict[str, float] = {}

        for ep in range(self.epoch, epochs):
            self.epoch = ep
            self._callback("on_epoch_start", epoch=self.epoch)

            train_out = self._run_train_epoch()
            self._callback(
                "on_epoch_end",
                epoch=self.epoch,
                split="train",
                logs={"loss": train_out.loss, **{f"{k}": v for k, v in train_out.metrics.items()}},
            )

            val_out = None
            if self.val_loader is not None:
                val_out = self._run_eval_epoch(self.val_loader, split="val")
                self._callback(
                    "on_epoch_end",
                    epoch=self.epoch,
                    split="val",
                    logs={"loss": val_out.loss, **{f"{k}": v for k, v in val_out.metrics.items()}},
                )

                # Optional best tracking (override _score_from_metrics if needed)
                score = self._score_from_metrics(val_out)
                if self.best_score is None or score > self.best_score:
                    self.best_score = score
                    self.best_epoch = self.epoch
                    self._callback(
                        "on_best_update",
                        epoch=self.epoch,
                        score=self.best_score,
                        logs=val_out
                    )

            # Scheduler policy (default: epoch-based)
            if self.scheduler is not None:
                self._step_scheduler(val_out)

            # Update final logs preference: val if avaiable else train
            final_logs = self._flatten_epoch_logs(val_out if val_out is not None else train_out)

            # Early stopping
            if self.should_stop:
                break

        self._callback("on_train_end")
        return final_logs
    
    def evaluate(
            self,
    ) -> Dict[str, float]:
        """
        Evaluate on test set (or val set if test is not provided).

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        if self.device is None:
            self.device = self._configure_device()
            self.model.to_device(self.device)

        # Ensure criterion exists for eval_step defaults
        if self.criterion is None:
            self.criterion = self._build_criterion()
        
        # Build loaders if not built
        if self.test_loader is None and self.val_loader is None and self.train_loader is None:
            self.train_loader, self.val_loader, self.test_loader = self.build_dataloaders()
        
        loader = self.test_loader or self.val_loader
        if loader is None:
            raise RuntimeError("No test/val loader is available for evaluation.")
        
        self._callback("on_evaluate_start")
        out = self._run_eval_epoch(loader, split="test" if loader is self.test_loader else "val")
        logs = self._flatten_epoch_logs(out)
        self._callback("on_evaluate_end", logs=logs)
        return logs
    
    # ------------------------------------------------------------------
    # Abstract hooks (task-specific)
    # ------------------------------------------------------------------
    def build_dataloaders(
            self,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Build train/val/test dataloaders.

        Returns:
            (train_loader, val_loader, test_loader)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.build_dataloaders() must be implemented in subclasses."
        )

    def train_step(
            self,
            batch: Any,
    ) -> TrainStepOutput:
        """
        One training step.

        Notes:
            Override in subclasses for complex batches / custom metrics.
        """
        # Default: (x, y) classification-like batch
        x, y = batch
        assert self.criterion is not None
        pred = self.model(x)
        loss = self.criterion(pred, y)
        return TrainStepOutput(loss=loss, metrics={})

    @torch.no_grad()
    def eval_step(
            self,
            batch: Any,
    ) -> TrainStepOutput:
        """
        One evaluation step.

        Notes:
            Override in subclasses for complex batches / custom metrics.
        """
        x, y = batch
        assert self.criterion is not None
        pred = self.model(x)
        loss = self.criterion(pred, y)
        return TrainStepOutput(loss=loss, metrics={})
    
    # ------------------------------------------------------------------
    # Trainer state
    # ------------------------------------------------------------------
    def get_state(
            self,
    ) -> Dict[str, Any]:
        """
        Return trainer-only state for resume/checkpoint.

        Returns:
            Dict[str, Any]
        """
        state: Dict[str, Any] = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
        }

        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()        
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            state["scheduler"] = self.scheduler.state_dict()
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        
        return state
    
    def set_state(
            self,
            state: Dict[str, Any],
    ) -> None:
        """
        Restore trainer-only state.

        Args:
            state: Dict returned by get_state().
        """
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.best_score = state.get("best_score", None)
        self.best_epoch = state.get("best_epoch", None)

        if self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and "scheduler" in state and hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

    # ------------------------------------------------------------------
    # Internal epoch loops
    # ------------------------------------------------------------------
    def _run_train_epoch(
            self,
    ) -> EpochOutput:
        assert self.train_loader is not None
        assert self.optimizer is not None

        self.model.train()

        total_loss = 0.0
        metric_sums: Dict[str, float] = {}
        num_steps = 0

        for step_idx, batch in enumerate(self.train_loader):
            self._callback(
                "on_train_batch_begin",
                epoch=self.epoch,
                step=step_idx,
                global_step=self.global_step,
            )
            
            batch = self._to_device(batch, self.device)

            out = self._train_step_impl(batch)

            loss_value = float(out.loss.detach().cpu().item())
            total_loss += loss_value

            for k, v in out.metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)

            num_steps += 1
            self.global_step += 1

            self._callback(
                "on_train_batch_end",
                epoch=self.epoch,
                step=step_idx,
                global_step=self.global_step,
                loss=loss_value,
                metrics=out.metrics,
            )

        avg_loss = total_loss / max(1, num_steps)
        avg_metrics = {k: v / max(1, num_steps) for k, v in metric_sums.items()}    
        return EpochOutput(loss=avg_loss, metrics=avg_metrics, num_steps=num_steps)
    
    def _run_eval_epoch(
            self,
            loader: DataLoader,
            *,
            split: str,
    ) -> EpochOutput:
        self.model.eval()

        total_loss = 0.0
        metric_sums: Dict[str, float] = {}
        num_steps = 0

        for step_idx, batch in enumerate(loader):
            self._callback(
                "on_eval_batch_begin",
                epoch=self.epoch,
                step=step_idx,
                split=split,
            )

            batch = self._to_device(batch, self.device)

            out = self._eval_step_impl(batch)

            loss_value = float(out.loss.detach().cpu().item())
            total_loss += loss_value

            for k, v in out.metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)

            num_steps += 1

            self._callback(
                "on_eval_batch_end",
                epoch=self.epoch,
                step=step_idx,
                split=split,
                loss=loss_value,
                metrics=out.metrics,
            )

        avg_loss = total_loss / max(1, num_steps)
        avg_metrics = {k: v / max(1, num_steps) for k, v in metric_sums.items()}
        return EpochOutput(loss=avg_loss, metrics=avg_metrics, num_steps=num_steps)


    # ------------------------------------------------------------------
    # Internal step impl (AMP + optimizer)
    # ------------------------------------------------------------------
    def _train_step_impl(
            self,
            batch: Any,
    ) -> TrainStepOutput:
        assert self.optimizer is not None

        self.optimizer.zero_grad(set_to_none=True)

        if self._amp_enabled and self.scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype):
                out = self.train_step(batch)
            self.scaler.scale(out.loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            out = self.train_step(batch)
            out.loss.backward()
            self.optimizer.step()

        return out

    def _eval_step_impl(
            self, 
            batch: Any,
    ) -> TrainStepOutput:
        if self._amp_enabled:
            with torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype):
                return self.eval_step(batch)
        return self.eval_step(batch)
    
    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_criterion(
            self,
    ) -> Any:
        """
        Build criterion from cfg.loss using LOSS_REGISTRY.

        Returns:
            Any: Callable loss module/function.
        """
        assert self.p is not None
        loss_cfg = self._get(self.p, "loss", None)
        if loss_cfg is None:
            raise ValueError("cfg.loss is required for TorchTrainer.")
        
        return build_from_cfg(
            loss_cfg,
            registry=LOSS_REGISTRY,
        )
    
    def _build_optimizer(
            self,
    ) -> torch.optim.Optimizer:
        """
        Build optimizer from cfg.optimizer using OPTIM_REGISTRY.

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        assert self.p is not None
        optim_cfg = self._get(self.p, "optimizer", None)
        if optim_cfg is None:
            raise ValueError("cfg.optimizer is required for TorchTrainer.")
        
        return build_from_cfg(
            optim_cfg,
            registry=OPTIM_REGISTRY,
            params=self.model.parameters(),
        )
    
    def _build_scheduler(
            self,
    ) -> Optional[Any]:
        """
        Build scheduler from cfg.scheduler using SCHED_REGISTRY (optional).

        Returns:
            Any: Scheduler instance or None.
        """
        assert self.p is not None
        sched_cfg = self._get(self.p, "scheduler", None)
        if sched_cfg is None:
            return None
        
        # Allow explicit disable:
        name = self._get(sched_cfg, "name", None)
        if name in ("none", "None", None, "", "null"):
            return None
        
        assert self.optimizer is not None
        return build_from_cfg(
            sched_cfg,
            registry=SCHED_REGISTRY,
            optimizer=self.optimizer,
        )
    
    def _build_amp(
            self,
    ) -> Tuple[Optional[torch.amp.GradScaler], bool, torch.dtype]:
        """
        Build AMP GradScaler from cfg.amp (optional).

        Returns:
            Tuple[Optional[torch.cuda.amp.GradScaler], bool, torch.dtype]:
                (scaler, enabled, dtype)
        """
        assert self.p is not None
        amp_cfg = self._get(self.p, "amp", None)
        enabled = bool(self._get(amp_cfg, "enabled", False)) if amp_cfg is not None else False
        dtype_str = str(self._get(amp_cfg, "dtype", "float16")).lower() if amp_cfg is not None else "float16"
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype_str, torch.float16)

        if not enabled:
            return None, False, dtype
        
        if not torch.cuda.is_available():
            return None, False, dtype

        return torch.amp.GradScaler("cuda"), True, dtype
    
    def _configure_device(
            self,
    ) -> torch.device:
        """
        Configure device based on cfg.device (default: cuda if available else cpu).

        Returns:
            torch.device
        """
        assert self.p is not None
        device_str = str(self._get(self.p, "device", None)).lower()
        if device_str is not None:
            return torch.device(str(device_str))
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # ------------------------------------------------------------------
    # Scheduler stepping policy
    # ------------------------------------------------------------------
    def _step_scheduler(
            self,
            val_out: Optional[EpochOutput] = None,
    ) -> None:
        """
        Scheduler stepping policy.

        Notes:
            - Default: epoch-based stepping for most schedulers.
            - Override if you want step-based or metric-based schedules.
        """
        if self.scheduler is None:
            return

        # ReduceLROnPlateau-style schedulers require a metric.
        if hasattr(self.scheduler, "step") and "ReduceLROnPlateau" in type(self.scheduler).__name__:
            metric = val_out.loss if val_out is not None else None
            if metric is None:
                return
            self.scheduler.step(metric)
            return
        
        self.scheduler.step()

    def _score_from_metrics(
            self,
            val_out: EpochOutput,
    ) -> float:
        """
        Default best-score policy.

        Notes:
            - Default: use negative loss as score (higher is better).
            - Override in subclasses for custom metrics.
        """
        return -float(val_out.loss)
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap_params(
            cfg: Any,
    ) -> Any:
        """
        Unwrap cfg.params if it exists; otherwise return cfg.

        Supports:
            - dict / Mapping
            - omegaconf DictConfig (via getattr / get)
        """
        if cfg is None:
            return {}
        # Mapping
        if isinstance(cfg, Mapping):
            return cfg.get("params", cfg)
        # DictConfig-like: has get()
        get_fn = getattr(cfg, "get", None)
        if callable(get_fn):
            try:
                params = get_fn("params", None)
                return params if params is not None else cfg
            except Exception:
                return cfg
        # Attribute-like
        params = getattr(cfg, "params", None)
        return params if params is not None else cfg

    @staticmethod
    def _get(
            obj: Any, 
            key: str, 
            default: Any = None,
    ) -> Any:
        """
        Safe config getter for Mapping/DictConfig-like objects.
        """
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        get_fn = getattr(obj, "get", None)
        if callable(get_fn):
            try:
                return get_fn(key, default)
            except Exception:
                return default
        return getattr(obj, key, default)

    @staticmethod
    def _to_device(
            x: Any, 
            device: torch.device,
    ) -> Any:
        """
        Move a batch to device.

        Supports:
            - torch.Tensor
            - (list/tuple) of nested structures
            - dict-like mappings
        """
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, (list, tuple)):
            return type(x)(TorchTrainer._to_device(v, device) for v in x)
        if isinstance(x, Mapping):
            return type(x)({k: TorchTrainer._to_device(v, device) for k, v in x.items()})
        return x
    
    @staticmethod
    def _flatten_epoch_logs(
            out: EpochOutput,
    ) -> Dict[str, float]:
        """
        Flatten EpochOutput into a single dict for return/logging.
        """
        logs: Dict[str, float] = {"loss": float(out.loss)}
        for k, v in out.metrics.items():
            logs[str(k)] = float(v)
        return logs