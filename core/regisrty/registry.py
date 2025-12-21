from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, TypeVar

T = TypeVar("T")


class Registry:
    """
    Simple name-to-object registry.

    This calss is intended to be used as a central registry
    for models, trainers, callbacks, losses, etc.

    Typical usage pattern:

        MODEL_REGISTRY = Registry("MODEL")

        @MODEL_REGISTRY.register()
        class MyModel(BaseModel):
            ...

        # Or with explicit name:
        @MODEL_REGISTRY.register("resnet18")
        class ResNet18(BaseModel):
            ...
        
        cls = MODEL_REGISTRY.get("MyModel")
        model = cls(cfg=cfg)

    The registry itself does not impose any base class constraint.
    It simply maps string keys to arbitrary Python objects
    """
    def __init__(
            self,
            name: str,
    ) -> None:
        """
        Initialize a Registry.

        Args:
            name: Nmae of this registry.
                Used only for debugging / representation.
        """
        self.name = name
        self._items: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1) Registration API
    # ------------------------------------------------------------------
    def register(
            self,
            name: Optional[str] = None,
            *,
            override: bool = False,
    ) -> Callable[[T], T]:
        """
        Decorator to register a class or function into this registry.

        Examples:
            registry = Registry("model")

            @registry.register()
            class MyModel(BaseModel):
                ...

            @registry.register("resnet18")
            class ResNet18(BaseModel):
                ...
        
        Args:
            name: Optional name to register the object under.
                If None, obj.__name__ is used.
            override: If False (default), raise an error when the name
                already exists. If True, overwrite the existing entry.

        Returns:
            Callable[[T], T]: A decorator that registers the given object
                and retuens it unchanged.

        Riases:
            KeyError: If the name already exists and override=False.
        """
        def decorator(
                obj: T,
        ) -> T:
            key = name or obj.__name__
            if (not override) and (key in self._items):
                raise KeyError(
                    f"[Registry: {self.name}] An object named '{key}' is already registered."
                )
            self._items[key] = obj
            return obj
        
        return decorator
    
    def register_obj(
            self,
            obj: Any,
            name: Optional[str] = None,
            *,
            override: bool = False,
    ) -> None:
        """
        Register an object without using the decorator syntax.

        This is useful when the object is created dynamically or when
        decorator-style registration is not convenient.

        Args:
            obj: Object to register.
            name: Optional name to register under. If None, obj.__name__ is used
                when available.
            override: Whether to allow overriding existing entries.
            
        Raises:
            ValueError: If name is None and obj does not have __name__.
            KeyError: If the name already exists and override=False.
        """
        key = name or getattr(obj, "__name__", None)
        if key is None:
            raise ValueError(
                f"[Registry: {self.name}] Cannot infer name for object {obj!r}."
            )
        if (not override) and (key in self._items):
            raise KeyError(
                f"[Registry: {self.name}] An object named '{key}' is already registered."
            )
        self._items[key] = obj

    # ------------------------------------------------------------------
    # 2) Lookup API
    # ------------------------------------------------------------------
    def get(
            self,
            name: str,
    ) -> Any:
        """
        Retrieve a registered object by name.

        Args:
            name: Name of the registered object.

        Returns:
            Any: The registered object.

        Raises:
            KeyError: If the name is not found in the registry.
        """
        if name not in self._items:
            raise KeyError(
                f"[Registry: {self.name}] No object named '{name}' found."
                f"Available: {list(self._items.keys())}"
            )
        return self._items[name]
    
    def __contains__(
            self,
            name: str,
    ) -> bool:
        """
        Check whether a given name is registered.

        This enables syntax such as :
            if "MyModel" in MODEL_REGISTRY: ...

        Args:
            name: Name to check.

        Returns:
            bool: True if the name exists in this registry, otherwise False.
        """
        return name in self._items
    
    def __len__(
            self,
    ) -> int:
        """
        Return the number of registered items.

        Returns:
            int: Number of registered entries.
        """
        return len(self._items)
    
    def keys(
            self,
    ) -> list[str]:
        """
        Return a list of registered names.

        Returns:
            list[str]: List of keys in this registry.
        """
        return list(self._items.keys())
    
    def items(
            self,
    ) -> list[tuple[str, Any]]:
        """
        Return a list of (name, object) pairs.

        Returns:
            list[tuple[str, Any]]: List of (key, value) pairs.
        """
        return list(self._items.items())
    
    def __repr__(
            self,
    ) -> str:
        """
        String representation for debugging and logging.

        By default, it shows:
            - Registry class name
            - Registry name
            - List of registered keys
        """
        cls_name = self.__class__.__name__
        return f"{cls_name}(name={self.name!r}, items={list(self._items.keys())})"