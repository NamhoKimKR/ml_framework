from __future__ import annotations

import importlib
from typing import Any, Mapping, Optional

from .regisrty.registry import Registry


def import_from_string(
        path: str,
) -> Any:
    """
    Import a Python object from a fully-qualified string.

    Examples:
        cls = import_from_string("torch.optim.Adam")
        obj = cls(lr=1e-3)
    """
    try:
        module_path, attr_name = path.rsplit(".", 1)
    except ValueError:
        raise ValueError(
            f"Invalid import path: {path!r}."
            f"Expected a fully-qualified path like 'torch.nn.CrossEntropyLoss'."
        )
    
    module = importlib.import_module(module_path)

    try:
        return getattr(module, attr_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{attr_name}'. "
            f"(from path={path!r})"
        ) from e
    

def build_from_cfg(
        cfg: Mapping[str, Any],
        registry: Optional[Registry] = None,
        *,
        name_key: str = "name",
        target_key: str = "target",
        **kwargs: Any,
) -> Any:
    """
    Build an object from a config mapping using either:
        1) 'target': a fully-qualified import path (importlib-based)
        2) 'name': a registry key (registry-based)

    Exactly one of ('target_key', 'name_key') must be present in cfg.

    Resolution rules:
        - If 'target_key' exists in cfg, importlib is used.
        - Otherwise, 'name_key' is used to look up the object in the registry.
        - If resolution fails, an error is raised.

    Dependencies that cannot be expressed purely in config
    (e.g., model parameters, optimizer instance, callbacks)
    should be passed explicitly via '**kwargs'.

    Examples:
        # Registry-based (custom component)
        model = build_from_cfg(
            cfg.model,
            registry=MODEL_REGISTRY,
        )

        # Import-based (torch built-in)
        optimizer = build_from_cfg(
            cfg.optimizer,
            registry=None,
            params=model.parameters(),
        )
        
    Args:
        cfg:
            Configuration mapping that describes how to build the object.
        registry:
            Registry instance used when 'name_key' is specified.
           Can be None if only 'target_key' is used.
        name_key:
            Key in cfg for registry lookup (default: "name").
        type_key:
            Fallback key if 'name_key' is not found.
        target_key:
            Key in cfg for importlib lookup (default: "target").
        **kwargs:
            Additional keyword arguments passed at build time.
            Typical exmaples:
                - params=model.parameters() (for opimizers)
                - optimzier=optimizer (for schedulers)
                - model=model, callbacks=callbacks (for trainers)

    Returns:
        Any: Instantiated object.

    Raises:
        KeyError:
            If neither 'target_key' nor 'name_key' is found in cfg.
        ValueError:
            If registry is required but not provided.
        TypeError:
            If the resolved object is not callable.

    """
    # Make a shallow copy to avoid modifying the original config.
    cfg_dict = dict(cfg)

    has_target = target_key in cfg_dict
    has_name = name_key in cfg_dict

    if not (has_target or has_name):
        raise KeyError(
            f"Config must contain either '{target_key}' or '{name_key}': cfg={cfg}"
        )
    
    if has_target:
        target_path = cfg_dict.pop(target_key)
        resolved = import_from_string(target_path)
    
    else:
        if registry is None:
            raise ValueError(
                f"Registry is None but '{name_key}' is specified: cfg={cfg}"
            )
        name = cfg_dict.pop(name_key)
        resolved = registry.get(name)
    
    if not callable(resolved):
        raise TypeError(
            f"Resolved object is not callable: {resolved!r} (cfg={cfg})"
        )
    
    return resolved(**cfg_dict, **kwargs)