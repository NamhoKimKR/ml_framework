from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from omegaconf import DictConfig
    CfgType: TypeAlias = DictConfig | Mapping[str, Any]
else:
    CfgType: TypeAlias = Mapping[str, Any]

from .base_model import BaseModel
from .callbacks import Callback


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers in this framework.

    A single BaseTrainer instance is responsible for:
        - Handling ONE experiment run
            (ONE config / ONE hyperparameter set / ONE seed)
        - Running the training loop (optionally multi-epoch)
        - Running evaluation (validation / test)
        - Maning training-related states (optimizer, scheduler, etc.)

    Backend (torch / sklearn / ...) specific logic
    should be implemented in sublcasses such as TorchTrainer, SklearnTrainer, etc.
    """
    #: Optional metadata for backend type. (e.g. "torch", "sklearn", ...)
    backend: str = "base"

    def __init__(
            self,
            cfg: CfgType,
            model: BaseModel,
            callbacks: Optional[list[Callback]] = None,
    ) -> None:
        """
        Initialize BaseTrainer.

        Args:
            cfg: Trainer configuration object.
                Typically includes training hyperparameters
                such as epohcs, batch size, optimizer settings, etc.
            model: Model instance that inherits from BaseModel.
            callbacks: List of callback instances.
                Each callback can implement hook methods such as 
                'on_train_start', 'on_epoch_end', etc.
                If None, an empty list is used.
        """
        self.cfg = cfg
        self.model = model
        self.callbacks: list[Callback] = callbacks or []

    # ------------------------------------------------------------------
    # 1) Public API: Single experiment run
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(
            self,
    ) -> Dict[str, float]:
        """
        Run the full training loop for a single experiment.

        This method is responsible for:
            - Building data loaders
            - Iterating over epochs and steps
            - Ubdating model parameters
            - (Optionally) running validation during training
            - Returning final metrics as a dictionary

        During the training loop, subclasses are expected to call
        callback hooks at appropriate points. Typical hooks include:
            - 'on_train_start'
            - 'on_epoch_start'
            - 'on_epoch_end' with logs (loss, metrics, lr, etc.)
            - 'on_train_end'

        Examples:
        - TorchTrainer: multi-epoch loop with optimizer + scheduler.
        - SklearnTrainer: usually a single 'fit()' call on estimator.

        Returns:
            Dict[str, float]: Final metrics of the experiment
                (e.g., {"val_loss": 0.123, "val_acc": 0.987}
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.fit() must be implemented in subclasses."
        )
        
    @abstractmethod
    def evaluate(
            self,
    ) -> Dict[str, float]:
        """
        Run evaluate routine (usually on test set).

        This method is separated from 'fit()' to allow:
            - Independent test-time evaluation
            - Post-training evaluation with loaded checkpoints

        Subclasses may also invoke callbacks here if needed
        (e.g., 'on_validation_end' or custom hooks).

        Examples:
            - TorchTrainer: evaluates on test DataLoader with no_grad().
            - SklearnTrainer: calls estimator.score() or custom metrics.

        Returns:
            Dict[str, float]: Evaluation metrcis
                (e.g., {"test_loss": 0.123})

        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.evaluate() must be implemented in subclasses."
        )
    
    # ------------------------------------------------------------------
    # 2) Trainer-level state (for checkpoint / resume)
    # ------------------------------------------------------------------
    @abstractmethod
    def get_state(
            self,
    ) -> Dict[str, Any]:
        """
        Return the internal traning state as a dictionary.

        This method should include ONLY trainer-related states, such as:
            - Current epoch / global step index
            - Optimizer state (e.g., optimizer.state_dict())
            - Scheduler state
            - Best metric and early-stopping related flags

        Model parameters should NOT be included here.
        Use 'BaseModel.get_state()' for model weights.

        Returns:
            Dict[str, Any]: Trainer state that can be used to resume training.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__}.get_state() mush be implemented in subclasses."
        )
    
    @abstractmethod
    def set_state(
            self,
            state: Dict[str, Any],
    ) -> None:
        """
        Restore the training state from a dictionary.

        This method should perform the inverse of 'get_state()', i.e.:
            - Restore epoch / step index
            - Load optimizer / scheduler states
            - Restore best metric and early-stopping flags

        Args:
            state: Trainer state dictionary returned by 'get_state()'.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.set_state() must be implemented in subclasses."
        )
    
    # ------------------------------------------------------------------
    # 3) Callback utilities
    # ------------------------------------------------------------------
    def add_callbacks(
            self,
            callback: Callback,
    ) -> None:
        """
        Register an additional callback at runtime.

        Args:
            callback: Callback instance to be appended to the callback list.
        """
        self.callbacks.append(callback)

    def _callback(
            self,
            hook_name: str,
            **kwargs: Any,
    ) -> None:
        """
        Internal helper to invoke a callback hook on all registered callbacks.

        This method looks up the method with name 'hook_name' on each callback
        and calls it if it exists.

        Typical usage inside subclasses:
            - self._callback("on_train_start")
            - self._callback("on_epoch_end", epoch=epoch, logs=epoch_logs)

        Args:
            hook_name: Name of the callback method to invoke.
            **kwargs: Arbitrary keyword arguments passed to the callback.
                The first argument 'trainer' will alwayes be injected.
        """
        for cb in self.callbacks:
            fn = getattr(cb, hook_name, None)
            if fn is None:
                continue
            fn(trainer=self, **kwargs)
    # ------------------------------------------------------------------
    # 4) Debug / representation
    # ------------------------------------------------------------------
    def extra_repr(
            self,
    ) -> str:
        """
        Additional representation string used in __repr__ or logging.

        This is optional and can be overridden in subclasses to show
        backend-specific information or key hyperparameters.

        Returns:
            str: Additional info string. Default is backend name.
        """
        return f"backend={self.backend!r}"

    def __repr__(
            self,
    ) -> str:
        """
        String representation for debugging and logging.

        By default, it shows:
            - Trainer class name
            - Model name
            - Extra representation string (e.g., backend info)
        """
        base = f"{self.__class__.__name__}(model={self.model.name!r})"
        extra = self.extra_repr()
        return f"{base} - {extra}" if extra else base