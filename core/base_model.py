from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Mapping


class BaseModel(ABC):
    """
    Abstract base class for all models in this framework.

    A BaseModel is designed so that Trainer can operate only through its
    public methods, regardless of the concrete backend implementation
    (TorchModel, SklearnModel, etc.).

    The model class should focus on:
        - 'Input -> output' computation
        - Getting / setting model state (parameters)

    Training logic such as optimizer steps, epoch loops, and scheduling
    should be handled by Trainer, not by BaseModel.

    Backend-specific behavior (torch / sklearn / ...) must be implemented
    in subclasses such as TorchModel, SklearnModel, etc.
    """
    #: Optional metadata for backend type. (e.g. "torch", "sklearn", ...)
    backend: str = "base"

    def __init__(
            self,
            cfg: Optional[Mapping[str, Any]] = None,
            name: Optional[str] = None,
    ) -> None:
        """
        Initialize BaseModel.

        Args:
            cfg: Configuration used to construct the model
                (e.g. omegaconf.DictConfig or plain dict).
                Not required, but recommended for reproducibility/logging.
            name: Model name. If None, the class name is used.
        """
        self.cfg = cfg
        self.name = name or self.__class__.__name__

    # ------------------------------------------------------------------
    # 1) Inference / forward methods
    # ------------------------------------------------------------------
    @abstractmethod
    def forward(
            self,
            x: Any,
    ) -> Any:
        """
        Forward propagation for input x.

        Examples:
            - TorchModel: Equivalent to nn.Module.forward().
            - SklearnModel: Similar to predict(), but may return logits/proba
                depending on subclass implementation.

        Args:
            x: Input data.

        Returns:
            Any: Model output.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() must be implemented in subclasses."
        )
    
    def __call__(
            self,
            x: Any,
    ) -> Any:
        """
        Inference method for Trainer or external users.

        By default, this simply calls 'forward(x)', but it can be
        overridden in subclasses if needed.

        Examples:
            - TorchModel: Could wrap forward() with eval() and no_grad().
            - SklearnModel: Could delegate to estimator.predict().

        Args:
            x: Input data.

        Returns:
            Any: Model output.
        """
        return self.forward(x)
    
    # ------------------------------------------------------------------
    # 2) State methods
    # ------------------------------------------------------------------
    @abstractmethod
    def get_state(
            self,
    ) -> Dict[str, Any]:
        """
        Return the trainable state of the model as a dictionary.

        This is typically used for checkpointing and serialization.

        Examples:
            - TorchModel: Returns state_dict() as-is or with light processing.
            - SklearnModel: Returns the estimator in a picklable form,
                or get_params() + additional information composed as a dict.

        Returns:
            Dict[str, Any]: Model state dictionary.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.get_state() must be implemented in subclasses."
        )
    
    @abstractmethod
    def set_state(
            self,
            state: Dict[str, Any],
    ) -> None:
        """
        Restore the model state from the given state dictionary.

        This method should be the inverse counterpart of 'get_state()'.

        Examples:
            - TorchModel: Calls load_state_dict(state).
            - SklearnModel: Restores estimator via set_params() or unpickling.

        Args:
            state: Model state dictionary returned by 'get_state()'.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.set_state() must be implemented in subclasses."
        )
    
    @abstractmethod
    def load_state(
            self,
            path: str,
            cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Load a state from disk and inject it into this model instance.

        This method is a convenience API that performs:
            1) Backend-specific file I/O
            2) State normalization (if needed)
            3) Calling 'set_state(state)'

        It muust NOT reconstruct the model architecture. Model construction
        (e.g., from YAML/Hydra config) is handled outside of the model class.

        Typical usage pattern:
            - model = build_from_cfg(cfg.model, ...)
            - model.load_state("ckpt_model.pt", cfg={"map_location": "cpu"})

        Args:
            path: Checkpoint file path.
            cfg: Optional configuration for backend-specific loading options
                (e.g., map_location, strict flags, etc.).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.load_state() must be implemented in subclasses."
        )
    
    # ------------------------------------------------------------------
    # 3) Save utility
    # ------------------------------------------------------------------
    @abstractmethod
    def save(
            self,
            path: str
    ) -> None:
        """
        Save the model to the given path.

        By default, this should store the result of 'get_state()' in some
        backend-specific format. The actual file format
        (e.g. torch.save, joblib.dump, custom pickle) is expected to be
        defined in subclasses.

        Args:
            path: File path to save the model to.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.save() must be implemented in subclasses."
        )    

    # ------------------------------------------------------------------
    # 4) Device / resource utilities
    # ------------------------------------------------------------------
    def to_device(
            self,
            device: Any
    ) -> BaseModel:
        """
        Common interface for device transfer.

        Examples:
            - TorchModel: Calls model.to(device).
            - SklearnModel: No device concept; returns self as-is (no-op).

        Args:
            device: Target device to move the model to.

        Returns:
            BaseModel: The same model instance (possibly moved to device).
        """
        return self
    
    # ------------------------------------------------------------------
    # 5) Util methods: etc.
    # ------------------------------------------------------------------
    def extra_repr(
            self,
    ) -> str:
        """
        Additional representation string used in __repr__ or logging.

        Subclasses can override this method to expose extra information
        such as key hyperparameters or architecture details.

        Returns:
            str: Additional info string. Default is an empty string.
        """
        return ""
    
    def __repr__(
            self,
    ) -> str:
        """
        String representation for debugging and logging.

        By default, it shows:
            - Model class name
            - Model name
            - Extra representation string (if provided)
        """
        base = f"{self.__class__.__name}(name={self.name!r})"
        extra = self.extra_repr()
        return f"{base} - {extra}" if extra else base


