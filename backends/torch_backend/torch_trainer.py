from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Iterable

import torch
from torch.utils.data import DataLoader

from src.core.base_trainer import BaseTrainer, CfgType
from src.core.callbacks import Callback
from src.core.base_model import BaseModel


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
class EvalOutput:
    """
    Container for evaluation results.

    Notes:
        - Used for validation/test epoch aggregation.
    """
    loss: float
    metrics: Dict[str, float]


class TorchTrainer(BaseTrainer):
    """
    Torch trainer implementaiton for a single experiment run.

    Responsibilities:
        - Build dataloaders (train/val/test)
        - Run epoch/step loops
        - Call optimizer/scheduler
        - Trigger callbacks for logging/monitoring
        - Manage trainer state (epoch/global_step/best_metrics)

    This class should NOT:
        - Reconstruct model architecture from YAML
        - Implement project-specific dataset logic in the base implementation
            (provide hooks for subclasses to override)
    """
    backend: str = "torch"

    def __init__(
            self,
            cfg: CfgType,
            model: BaseModel,
            callbacks: Optional[list[Callback]] = None,
    ) -> None:
        super().__init__(cfg=cfg, model=model, callbacks=callbacks)

        # Basic training state
        self.epoch: int = 0
        self.global_step:int = 0

        # Best tracking (optional)
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None

        # Runtime objects (built in fit())
        self.device: torch.device = torch.device("cpu")
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None # torch.optim.lr_scheduler._LRScheduler or others
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None # optional AMP

        # Dataloaders
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    # ------------------------------------------------------------------
    # 0) Hooks: build components (override in concrete trainers)
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
    
    def build_optimizer(
            self,
    ) -> torch.optim.Optimizer:
        """
        Build optimizer for the current model parameters.

        Notes:
            - You typically read optimizer settings form cfg.
            - Must use model parameters (self.model.parameters()).

        Returns:
            torch.optim.Optimizer
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.build_optimizer() must be implemented in subclasses."
        )
    
    def build_scheduler(
            self,
    ) -> Optional[Any]:
        """
        Build LR scheduler.

        Returns:
            Scheduler object or None.
        """
        return None
    
    def build_amp_scaler(
            self,
    ) -> Optional[torch.cuda.amp.GradScaler]:
        """
        Build AMP scaler if mixed precision is enabled.

        Returns:
            GradScaler on None.
        """
        return None
    
    def configure_device(
            self,
    ) -> torch.device:
        """
        Decide which device to use for training.
        
        Returns:
            torch.device
        """
        # v0 default: use cfg.device if exists else cuda if available else cpu
        device_str = None
        try:
            device_str = self.cfg.get("device", None) # type: ignore[attr-defined]
        except Exception:
            device_str = None

        if device_str is not None:
            return torch.device(str(device_str))
        
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------------------------------------------------------
    # 1) Core step functions (override if you need project-specific logic)
    # ------------------------------------------------------------------
    def train_step(
            self,
            batch: Any,
    ) -> TrainStepOutput:
        """
        Single training step.

        Notes:
            - This default implementation assumes:
                - batch -> (x, y) like structure
                - model(x) returns predictions
                - loss_fn(pred, y) exists (provided by subclass via self.loss_fn)
            - Most projects oberride this to handle complex batch structure.

        Returns:
            TrainStepOutput
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.train_step() must be implemented in subclasses."
        )
    
    @torch.no_grad()
    def eval_step(
            self,
            batch: Any,
    ) -> TrainStepOutput:
        """
        Single evaluation step (val/test).

        Returns:
            TrainStepOutput
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.eval_step() must be implemented in subclasses."
        )
    
    # ------------------------------------------------------------------
    # 2) Epoch routines
    # ------------------------------------------------------------------
    def _run_train_epoch(
            self,
    ) -> EvalOutput:
        """
        Run one training epoch over train_laoder.

        Returns:
            EvalOutput: averaged loss/metrics for the epoch
        """
        assert self.train_loader is not None
        assert self.optimizer is not None

        self.model.train()

        total_loss = 0.0
        metric_sums: Dict[str, float] = {}
        num_steps = 0

        for step_idx, batch in enumerate(self.train_loader):
            self._callback("on_train_batch_begin", batch=batch, step=step_idx, epoch=self.epoch)

            # Move batch to device if needed (override if batch is nested)
            batch = self._to_device(batch, self.device)

            out = self._train_step_impl(batch)

            loss_value = float(out.loss.detach().cpu().item())
            total_loss += loss_value
            for k, v in out.metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)

            num_steps += 1
            self.global_step += 1

            # Logging hook point (callbacks typically do logging)
            self._callback(
                "on_train_batch_end",
                batch=batch,
                step=step_idx,
                epoch=self.epoch,
                loss=loss_value,
                metrics=out.metrics,
                global_step=self.global_step,
            )

        avg_loss = total_loss / max(1, num_steps)
        avg_metrics = {k: v / max(1, num_steps) for k, v in metric_sums.items()}
        return EvalOutput(loss=avg_loss, metrics=avg_metrics)
    
    @torch.no_grad()
    def _run_eval_epoch(
            self,
            loader: DataLoader,
            *,
            split: str,
    ) -> EvalOutput:
        """
        Run one evaluation epoch over given loader.

        Args:
            loader: Dataloader for evaluation.
            split: "val" or "test" (for callback semantics)

        Returns:
            EvalOutput
        """
        self.model.eval()

        total_loss = 0.0
        metric_sums: Dict[str, float] = {}
        num_steps = 0

        for step_idx, batch in enumerate(loader):
            self._callback(f"on_{split}_batch_begin", batch=batch, step=step_idx, epoch=self.epoch)

            batch = self._to_device(batch, self.device)
            out = self.eval_step(batch)

            loss_value = float(out.loss.detach().cpu().item())
            total_loss += loss_value
            for k, v in out.metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)

            num_steps += 1

            self._callback(
                f"on_{split}_btach_end",
                batch=batch,
                step=step_idx,
                epoch=self.epoch,
                loss=loss_value,
                metrics=out.metrics,
                gloabl_step=self.global_step,
            )

        avg_loss = total_loss / max(1, num_steps)
        avg_metrics = {k: v / max(1, num_steps) for k, v in metric_sums.items()}
        return EvalOutput(loss=avg_loss, metrics=avg_metrics)
    

    # ------------------------------------------------------------------
    # 3) Fit/Evaluate public API
    # ------------------------------------------------------------------
    def fit(
            self,
    ) -> Dict[str, float]:
        """
        Run the full training routine for a single experiment run.

        Returns:
            Dict[str, float]: final metrics (val/test optional)
        """
        self.device = self.configure_device()
        self.mdoel.to_devoce(self.device) # TrochModel should override to_device()

        self._callback("on_fit_begin", trainer=self)

        self.train_loader,self.val_loader, self.test_loader = self.build_dataloaders()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_amp_scaler()

        # Read epochs from cfg (v0 default)
        epochs = 1
        try:
            epochs = int(self.cfg.get("epochs", 1)) # type: ignore[attr-defined]
        except Exception:
            epochs = 1

        # Optional: validation frequency
        val_every = 1
        try:
            val_every = int(self.cfg.get("val_every", 1)) # type: ignore[attr-defined]
        except Exception:
            val_every = 1

        final_out: Dict[str, float] = {}

        for ep in range(self.epoch, epochs):
            self.epoch = ep
            self._callback("on_train_epoch_begin", epoch=self.epoch)

            train_out = self._run_train_epoch()

            # Scheduler step policy:
            # - If ReduceLROnPlateau: step(metric)
            # - Else: step() each epoch
            self._step_scheduler(train_out=train_out)

            self._callback(
                "on_train_epoch_end",
                epoch=self.epoch,
                loss=train_out.loss,
                metrics=train_out.metrics,
                global_step=self.global_step,
                lr=self._get_lr(),
            )

            # Validation
            if self.val_loader is not None and((self.epoch + 1) % val_every == 0):
                self._callback("on_val_epoch_begin", epoch=self.epoch)
                val_out = self._run_eval_epoch(self.val_loader, split="val")

                self._callback(
                    "on_val_epoch_end",
                    epoch=self.epoch,
                    loss=val_out.loss,
                    metrics=val_out.metrics,
                    global_step=self.global_step,
                )

                # Update best tracking (default metric: -val_loss)
                score = self._default_score(val_out)
                if self.best_score is None or score > self.best_score:
                    self.best_score = score
                    self.best_epoch = self.epoch
                    self._callback(
                        "on_best_updated",
                        epch=self.epoch,
                        score=score,
                        best_score=self.best_score,
                    )

                final_out.update({f"val_loss": float(val_out.loss)})
                for k, v in val_out.metrics.items():
                    final_out[f"val_{k}"] = float(v)

            # Always expose train results
            final_out.update({f"train_loss": float(train_out.loss)})
            for k, v in train_out.metrics.items():
                final_out[f"train_{k}"] = float(v)

        self._callback("on_fit_end", trainer=self, final=final_out)
        return final_out
    
    def evaluate(
            self,
    ) -> Dict[str, float]:
        """
        Run evaluation routine (usually on test set).

        Returns:
            Dict[str, float]: evaluation metrics
        """
        self._callback("on_evaluate_begin", trainer=self)

        # Build loaders if not built
        if self.test_loader is None:
            _, _, self.test_loader = self.build_dataloaders()

        assert self.test_loader is not None

        out = self._run_eval_epoch(self.test_loader, split="test")

        result: Dict[str, float] = {"test_loss": float(out.loss)}
        for k, v in out.metrics.items():
            result[f"test_{k}"] = float(v)

        self._callback("on_evaluation_end", trainer=self, result=result)
        return result
    
    # ------------------------------------------------------------------
    # 4) Trainer state (checkpoint / resume)
    # ------------------------------------------------------------------
    def get_state(
            self,
    ) -> Dict[str, Any]:
        """
        Return trainer-only state.

        Notes:
            - Model weights are NOT included here.
            - Optimizer/scheduler states are included if available.
        """
        state: Dict[str, Any] = {
            "epoch": int(self.epoch),
            "gloabl_step": int(self.global_step),
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
        """
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.best_score = state.get("best_score", None)
        self.best_epoch = state.get("best_epoch", None)

        if self.optimzier is not None and "optimzier" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        
        if self.scheduler is not None and "scheduler" in state and hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(state["scheduler"])

        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

    # ------------------------------------------------------------------
    # 5) Internal helpers
    # ------------------------------------------------------------------
    def _train_step_imlp(
            self,
            batch: Any,
    ) -> TrainStepOutput:
        """
        Internal wrapper to support optional AMP.

        Returns:
            TrainStepOutput
        """
        assert self.optimizer is not None

        use_amp = self.scaler is not None

        self.optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                out = self.train_step(batch)
            self.scaler.scale(out.loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            out = self.train_step(batch)
            out.loss.backward()
            self.optimizer.step()

        return out
    
    def _step_scheduler(
            self,
            train_out: EvalOutput,
    ) -> None:
        """
        Step LR scheduler with a conservative policy.

        Notes:
            - If scheduler is ReduceLROnPlateau-like, step with metric.
            - Otherwise, step per epoch.
        """
        if self.scheduler is None:
            return
        
        # ReduceLROnPlateau signature: step(metrics, epoch=None)
        if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
            self.scheduler.step(train_out.loss)
        else:
            try:
                self.scheduler.step()
            except TypeError:
                # Some schedulers require metric; fallback to loss
                self.scheduler.step(train_out.loss)

    def _get_lr(
            self,
    ) -> Optional[float]:
        """
        Get current learning rate (first param group).
        """
        if self.optimizer is None:
            return None
        try:
            return float(self.optimizer.param_groups[0].get("lr", None))
        except Exception:
            return None
        
    def _default_score(
            self,
            val_out: EvalOutput,
    ) -> float:
        """
        Default best-score definition.

        Notes:
            - v0 default: maximize (-val_loss)
            - You can override to use accuracy/F1 etc.
        """
        return -float(val_out.loss)
    
    def _to_device(
            self,
            batch: Any,
            device: torch.device,
    ) -> Any:
        """
        Move batch to target device.

        Notes:
            - Handles common nested structures.
            - Override for custom batch containers.
        """
        if torch.is_tensor(batch):
            return batch.to(device, non_blocking=True)
        
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(x, device) for x in batch)
        
        if isinstance(batch, dict):
            return {k: self._to_device(v, device) for k, v in batch.items()}
        
        # Non-tensor object (e.g., strings, indices); return as-is.
        return batch
    
    def _callback(
            self,
            name: str, 
            **kwargs
    ) -> None:
        """
        Dispatch callback events if callbacks are configured.
        """
        if not getattr(self, "callbacks", None):
            return
        
        for cb in self.callbacks:
            fn = getattr(cb, name, None)
            if fn is not None:
                fn(**kwargs)