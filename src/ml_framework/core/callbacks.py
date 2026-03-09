from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_trainer import BaseTrainer


class Callback:
    """
    Base Callback class.

    A callback is a collection of hook methods that are called by the Trainer
    at specific points during the training and evaluation process.

    All methods are optional. Subclasses can override only the hooks they need.
    Typical use cases include:
        - Logging (e.g., ClearML, WandB, TernsorBoard)
        - Checkpointing
        - Early stopping
        - Learning rate monitoring

    The Trainer is responsible for invkoing these hooks, for example:
        - on_train_start: before the training loop starts
        - on_epoch_start: at the beginning of each epoch
        - on_epoch_end: at the end of each epoch (with logs)
        - on_train_end: after the training loop finished
        - on_validation_end: after validation step is done
        - on_batch_end: after each training batch (optional)
    """

    # ------------------------------------------------------------------
    # High-level training lifecycle
    # ------------------------------------------------------------------
    def on_train_start(
            self,
            trainer: BaseTrainer,
    ) -> None:
        """
        Callled at the very beginning of training, before any epoch starts.

        Typical usage:
            - Initialize logging resources
            - Log configuration or model summary
            - Reset internal states of the callback

        Args:
            trainer: The trainer instance that owns this callback.
        """
        pass

    def on_train_end(
            self,
            trainer: BaseTrainer,
    ) -> None:
        """
        Called at the very end of training, after all epochs are finished.

        Typical usage:
            - Finalize logging
            - Close any opened resources
            - Log final summary or metrics

        Args:
            trainer: The trainer instance that owns this callback.
        """
        pass

    # ------------------------------------------------------------------
    # Epoch-level hooks
    # ------------------------------------------------------------------
    def on_epoch_start(
            self,
            trainer: BaseTrainer,
            epoch: int,
            logs: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Called at the beginning of each epoch.

        Args:
            trainer: The trainer instance that owns this callback.
            epoch: Current epoch index (0-based or 1-based, depending on Trainer).
        """
        pass

    def on_epoch_end(
            self,
            trainer: BaseTrainer,
            epoch: int,
            logs: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Called at the end of each epoch.

        The 'logs' dictionary is expected to contain aggregated statistics
        from this epoch, such as:
            - Training loss and metrics
            - Validation loss anf metrics
            - Learning rates, etc.

        Example of 'logs':
            {
                "loss_train": 0.123,
                "loss_val": 0.234,
                "metric_train_acc": 0.95,
                "metric_val_acc": 0.92,
                "lr_group0": 1e-3,
            }

        Args:
            trainer: The trainer instance that owns this callback.
            epoch: Current epoch index.
            logs: Dictionary of scalar metrics for this epoch.
        """
        pass

    # ------------------------------------------------------------------
    # Validation / evaluation hooks
    # ------------------------------------------------------------------
    def on_validation_start(
            self,
            trainer: BaseTrainer,
            epoch: Optional[int] = None,
    ) -> None:
        """
        Called right before starting a validation phase.

        This may be called during training (e.g., after each epoch)
        or during a separate evaluation routine.

        Args:
            trainer: The trainer instance that owns this callback.
            epoch: Epoch index at which validation is triggered,
                or None if validation is not r]tied to epochs.
        """
        pass

    def on_validation_end(
            self,
            trainer: BaseTrainer,
            epoch: Optional[int] = None,
            logs: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Called at the end of a validation phase.

        'logs' may include validation metrics such as:
            - val_loss
            - val_accuracy
            - any other validation-related statistics.

        Args:
            trainer: The trainer instance that owns this callback.
            epoch: Epoch index at which validation finished,
                or None if not tied to epochs.
            logs: Dictionary of validation metrics.
        """
        pass

    # ------------------------------------------------------------------
    # Batch-level hooks (optional)
    # ------------------------------------------------------------------
    def on_batch_start(
            self,
            trainer: BaseTrainer,
            step: int,
            logs: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Called at the beginning of a training batch.

        This hook is optional and may not be used in all Trainers.

        Args:
            trainer: The trainer instance that owns this callback.
            step: Global step index (e.g., total number of batches seen so far).
            logs: Optional dictionary with batch-level information.
        """
        pass

    def on_batch_end(
            self,
            trainer: BaseTrainer,
            step: int,
            logs: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Called at the end of a training batch.

        'logs' may include batch-level statistics such as :
            - batch_loss
            - batch_accuracy
            - gradient norms, etc.

        This can be used for fine-grained logging or debugging.

        Args:
            trainer: The trainer instance that owns this callback.
            step: Gloabel step index.
            logs: Dictionary of batch-level metrics.
        """
        pass

    # ------------------------------------------------------------------
    # Exception hook (optional)
    # ------------------------------------------------------------------
    def on_exception(
            self,
            trainer: BaseTrainer,
            exceotion: BaseException,
    ) -> None:
        """
        Called when an exception is raised during training or evaluation.

        Typical usage:
            - Save last checkpoint
            - Flush logs
            - Mark the task as failed in external tracking systems

        Args:
            trainer: The trainer instance that owns this callback.
            exception: The exception that was raised.
        """
        pass