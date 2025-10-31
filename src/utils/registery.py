from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Union

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

ModelType = Literal["MLP", "BAH", "GIN-F", "GIN-C", "GIN-LF", "NVP", "LPR", "PR"]

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


def get_model_type(path: Union[Path, str]) -> ModelType:
    res: ModelType = "MLP"
    if "bah" in str(path):
        res = "BAH"
    elif "gin-c" in str(path):
        res = "GIN-C"
    elif "gin-f" in str(path):
        res = "GIN-F"
    elif "nvp" in str(path):
        res = "NVP"
    elif "lpr" in str(path):
        res = "LPR"
    elif "nvp" in str(path):
        res = "NVP"
    return res
