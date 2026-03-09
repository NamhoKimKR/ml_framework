from __future__ import annotations

import importlib
from typing import Any, Mapping, Optional, Iterable

from .registry.registry import Registry


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
    except ValueError as e:
        raise ValueError(
            f"Invalid import path: {path!r}."
            f"Expected a fully-qualified path like 'torch.nn.CrossEntropyLoss'."
        ) from e
    
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
        build_mode_key: str = "build_mode",
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

    Build mode:
        - Default ("params"): call as resolved(**cfg_payload, **kwargs).
          This is compatible with torch built-ins and most constructors.
        - "cfg": call as resolved(cfg=cfg, **kwargs).
          This is useful for components that want to own config parsing
          (e.g., Trainer classes reading cfg.params internally).

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

        # cfg-mode build (trainer keeps cfg and parses params internally)
        trainer = build_from_cfg(
            cfg.trainer,
            registry=TRAINER_REGISTRY,
            model=model)
        
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
        build_mode_key:
            Key in cfg for build mode (default: "build_mode").
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

    # Read build mode first (but don't pass it into constructor).
    build_mode = str(cfg_dict.pop(build_mode_key, "params")).lower().strip()

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
    
    # Build-mode dispatch:
    # - cfg: pass the whole cfg (original mapping) into the constructor
    # - params (default): expand remaining keys as kwargs
    if build_mode == "cfg":
        return resolved(cfg=cfg, **kwargs)

    if build_mode not in ("params", ""):
        raise ValueError(
            f"Invalid build_mode={build_mode!r}. Expected 'params' or 'cfg'. (cfg={cfg})"
        )

    return resolved(**cfg_dict, **kwargs)


def build_list_from_cfg(
    cfg_list: Optional[Iterable[Mapping[str, Any]]],
    *,
    registry: Optional[Registry] = None,
    **kwargs: Any,
) -> list[Any]:
    """
    Build a list of objects from a list of cfg mappings.

    Notes:
        - This is mainly intended for callbacks, but can be reused for
          metrics/transforms/etc.
        - Each item is built by build_from_cfg(item, registry=..., **kwargs).

    Args:
        cfg_list: Iterable of config mappings. If None, returns [].
        registry: Optional registry to resolve "name" keys.
        **kwargs: Injected runtime dependencies (e.g., output_dir, model, task)

    Returns:
        list[Any]: Built objects in the same order as cfg_list.
    """
    if cfg_list is None:
        return []

    objs: list[Any] = []
    for item in cfg_list:
        objs.append(build_from_cfg(item, registry=registry, **kwargs))
    return objs