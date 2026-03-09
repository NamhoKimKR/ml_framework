"""
Torch backend public API.

This module re-exports key backend classes so users can import them via:
    from ml_framework.backends.torch_backend import TorchModel, TorchTrainer
"""

from .torch_model import TorchModel
from .torch_trainer import TorchTrainer

__all__ = [
    "TorchModel",
    "TorchTrainer",
]