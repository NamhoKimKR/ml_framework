from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ml_framework.core.callbacks import Callback
from ml_framework.core.base_trainer import BaseTrainer

class CheckpointCallback(Callback):
    """
    Callback for saving model/trainer checkpoints.

    v0 policy:
        - Save format is fixed as:
            {
                "model": model.get_state(),
                "trainer": trainer.get_state()
            }

        - Actual serialization backend (torch.save) is assumed.
    """
    
    def __init__(
          self,
          save_dir: str | Path,
          *,
          save_last: bool = True,
          save_best: bool = True,
          save_every: Optional[int] = None,
    ) -> None:
        self.output_dir = Path(save_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.model_dir = self.output_dir / "models"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_last = save_last
        self.save_best = save_best
        self.save_every = save_every

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def on_epoch_end(
            self,
            trainer: BaseTrainer,
            *,
            epoch: int,
            split: str,
            logs: Dict[str, float],
            **_: Any,
    ) -> None:
        # Only trigger checkpointing on validation end
        if split != "val":
            return
        
        state = self._collect_state(trainer)

        # Save last
        if self.save_last:
            self._save(state, self.model_dir / "last.pt")

        # Save every N epochs
        if self.save_every is not None and epoch % self.save_every == 0:
            self._save(state, self.ckpt_dir / f"epoch_{epoch:04d}.pt")

    def on_best_update(
            self,
            trainer: BaseTrainer,
            *,
            epoch: int,
            score: float,
            **_: Any,
    ) -> None:
        if not self.save_best:
            return
        
        state = self._collect_state(trainer)
        self._save(state, self.model_dir / "best.pt")
    
    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _collect_state(
            self,
            trainer: BaseTrainer,
    ) -> Dict[str, Any]:
        return {
            "model": trainer.model.get_state(),
            "trainer": trainer.get_state(),
        }
    
    def _save(
            self,
            state: Dict[str, Any],
            path: Path,
    ) -> None:
        torch.save(state, path)
