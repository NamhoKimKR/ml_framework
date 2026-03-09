from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from omegaconf import OmegaConf
from hydra.utils import get_original_cwd


def resolve_path(
    path: str | Path,
) -> Path:
    """
    Resolve a path relative to Hydra original working directory.
    
    Notes:
        Hydra changes cwd to a run directory by default.
        This helper makes relative paths stable by anchoring them to original cwd.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(get_original_cwd()) / p).resolve()


def ensure_dir(
    path: str | Path,
) -> Path:
    """
    Create ditectory if not exists and return resolved Path.    
    """
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_resolved_config(
    cfg: Any,
    out_dir: str | Path,
    filename: str = "config_resolved.yaml",
) -> Path:
    """
    Save the resolved OmegaConf config under out_dir/configs/.

    Saves:
        - configs/config_resolved.yaml
        - configs/config_raw.yaml (optional if you want; see below)

    Returns:
        Path to saved resolved config.
    """
    out_dir = ensure_dir(out_dir)
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    resolved_path = cfg_dir / filename
    OmegaConf.save(cfg, resolved_path, resolve=True)
    return resolved_path


def save_raw_config(
    cfg: Any,
    out_dir: str | Path,
    filename: str = "config_raw.yaml",
) -> Path:
    """
    Save the raw (unresolved) OmegaConf config under out_dir/configs/.
    """
    out_dir = ensure_dir(out_dir)
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    raw_path = cfg_dir / filename
    OmegaConf.save(cfg, raw_path, resolve=False)
    return raw_path