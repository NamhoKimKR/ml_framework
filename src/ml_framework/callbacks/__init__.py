"""
Built-in callback implementations.

This package provides reusable callbacks that can be plugged into Trainer
without modifying core training logic.
"""

from .checkpoint import CheckpointCallback
from .early_stopping import EarlyStoppingCallback
from .logging import LoggingCallback

__all__ = [
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LoggingCallback",
]
