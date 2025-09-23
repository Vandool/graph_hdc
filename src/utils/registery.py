from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

ModelType = Literal["MLP", "BAH", "GIN-F", "GIN-C", "GIN-LF", "NVP"]

# Store CLASSES (constructors), not instances
_MODEL_REGISTRY: dict[ModelType, type[LightningModule]] = {}


def register_model(name: ModelType) -> Callable[[type[LightningModule]], type[LightningModule]]:
    """Decorator: register a LightningModule subclass under `name`."""
    def deco(ctor: type[LightningModule]) -> type[LightningModule]:
        _MODEL_REGISTRY[name] = ctor
        return ctor
    return deco


def resolve_model(name: ModelType, **kwargs: Any) -> pl.LightningModule:
    """Instantiate the registered class with kwargs."""
    cls = _MODEL_REGISTRY.get(name)
    if cls is None:
        msg = f"Unknown model '{name}'. Registered: {list(_MODEL_REGISTRY)}"
        raise KeyError(msg)
    return cls(**kwargs)


def retrieve_model(name: ModelType) -> type[LightningModule]:
    """Return the registered class (not instantiated)."""
    cls = _MODEL_REGISTRY.get(name)
    if cls is None:
        msg = f"Unknown model '{name}'. Registered: {list(_MODEL_REGISTRY)}"
        raise KeyError(msg)
    return cls
