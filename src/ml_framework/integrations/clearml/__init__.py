"""
ClearML integration package.

Notes:
    - This package is optional. The framework should still work even if ClearML
      is not installed.
"""

from __future__ import annotations

from .callbacks import ClearMLCallback

__all__ = ["ClearMLCallback"]