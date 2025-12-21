from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn

from src.core.base_model import BaseModel


class TorchModel(nn.Module, BaseModel):
    """
    Torch backend base model.

    This class inherits both:
        - torch.nn.Module: for full compatibility with the PyTorch ecosystem
            (state_dict, parameters, .to(), DDP, torch.compile, hooks, etc.)
        - BaseModel: for framework-level backend-egonostic intergace
            (get_state/set_state, save/load, to_device, metadata)

    Notes:
        - In this multiple-inheritance layout, nn.Module should come first to ensure
            torch-native methods like __call__ take precedence in MRO.
        - BaseModel.__call__ is effectively ignored here; nn.Module.__call__ will be used.
        - You should implement forward() in concrete subclasses (or override here).
    """
    backend: str = "torch"

    def __init__(
            self,
            cfg: Optional[Mapping[str, Any]] = None,
            name: Optional[str] = None,
    ) -> None:
        """
        Initialize TorchModel.

        Important:
            nn.Module.__init__ MUST be called to properly register submodules/parameters.
            BaseModel.__init__ sets metadata (cfg, name) for reproducibility/logging.

        Args:
            cfg: Optional config mapping used to construct the model.
            name: Optional model name. If None, the class name is used.
        """
        nn.Module.__init__(self)
        BaseModel.__init__(self, cfg=cfg, name=name)

    # ------------------------------------------------------------------
    # 1) Inference / forward method
    # ------------------------------------------------------------------
    def get_state(
            self,
    ) -> Dict[str, Any]:
        """
        Return model state in a framework-friendly dict.

        Returns:
            Dict[str, Any]: Contains 'state_dict' by default.
        """
        return {"state_dict": self.state_dict()}
    
    def set_state(
            self,
            state: Dict[str, Any],
    ) -> None:
        """
        Restore model state from a framework-friendly dict.

        Args:
            state: Expected to contain 'state_dict'.

        Raises:
            KeyError: If 'state_dict' is missing.
        """
        sd = state.get("state_dict", None)
        if sd is None:
            raise KeyError(
                "TorchModel.set_state expects state['state_dict']."
            )
        self.load_state_dict(sd)

    def load_state(
            self,
            path: str,
            cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Load a checkpoint from disk and injectt it into this model instance.

        Accepted checkpoint formats:
            - {"state_dict": <state_dict>, ...}
            - <raw_state_dict> (will be wrapped automatically)

        Args:
            path: Checkpoint file path
            cfg: Optional config for loading options.
                If cfg contains "map_location", it will be passed to torch.load().
        """
        map_location: Any = "cpu"
        if cfg is not None:
            # cfg might be a DictConfig for dict-like; use .get if available.
            try:
                map_location = cfg.get("map_location", map_location) # type: ignore[attr-defiend]
            except Exception:
                map_location = map_location
        
        loaded = torch.load(path, map_lcoation=map_location)

        if isinstance(loaded, dict) and "state_dict" in loaded:
            state = loaded
        else:
            state = {"state_dict": loaded}
        
        self.set_state(state)

    # ------------------------------------------------------------------
    # 3) Save utility
    # ------------------------------------------------------------------
    def save(
            self,
            path: str
    ) -> None:
        """
        Save model state to disk.

        Args:
            path: Output file path.
        """
        torch.save(self.get_state(), path)

    # ------------------------------------------------------------------
    # 4) Device / resource utilities
    # ------------------------------------------------------------------
    def to_device(
            self,
            device: Any,
    ) -> TorchModel:
        """
        Move model to the given device.

        Args:
            device: torch.device or string (e.g., 'cuda', 'cpu').

        Returns:
            TorchModel: Self (for chaining).
        """
        self.to(device)
        return self

    # ------------------------------------------------------------------
    # 5) Debug / representation
    # ------------------------------------------------------------------
    def extra_repr(
            self,
    ) -> str:
        """
        Extra info used in __repr__ / logging.

        Returns:
            str: Additional representation info.
        """
        try:
            n_params = sum(p.numel() for p in self.parameters())
            return f"backend={self.backend!r}, params={n_params}"
        except Exception:
            return f"backend={self.backend!r}"