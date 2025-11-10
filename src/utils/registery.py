from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Union

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

ModelType = Literal["MLP", "BAH", "GIN-F", "GIN-C", "GIN-LF", "NVP", "NVP-V3", "LPR", "PR"]

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
    s = str(path)
    if "bah" in s:
        res = "BAH"
    elif "gin-c" in s:
        res = "GIN-C"
    elif "gin-f" in s:
        res = "GIN-F"
    elif "lpr" in s:
        res = "LPR"
    elif "nvp_v3" in s:
        res = "NVP-V3"
    elif "nvp" in s:
        res = "NVP"
    return res
