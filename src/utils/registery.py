# ------------------------- registry -------------------------
from collections.abc import Callable
from typing import Literal

import pytorch_lightning as pl

ModelType = Literal["MLP", "BAH", "GIN-F", "GIN-C", "NVP"]

_MODEL_REGISTRY: dict[ModelType, Callable[..., pl.LightningModule]] = {}


def register_model(name: ModelType):
    def deco(ctor: Callable[..., pl.LightningModule]):
        _MODEL_REGISTRY[name] = ctor
        return ctor

    return deco


def resolve_model(name: ModelType, **kwargs) -> pl.LightningModule:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)
