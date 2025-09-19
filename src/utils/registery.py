# ------------------------- registry -------------------------
from typing import Callable, Literal

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

ModelType = Literal["MLP", "BAH", "GIN-F", "GIN-C", "NVP"]

_MODEL_REGISTRY: dict[ModelType, Callable[..., pl.LightningModule]] = {}


def register_model(name: ModelType):
    def deco(ctor: Callable[..., pl.LightningModule]):
        _MODEL_REGISTRY[name] = ctor
        return ctor

    return deco


def resolve_model(name: ModelType, **kwargs) -> pl.LightningModule:
    if name not in _MODEL_REGISTRY:
        msg = f"Unknown model '{name}'. Registered: {list(_MODEL_REGISTRY)}"
        raise KeyError(msg)
    return _MODEL_REGISTRY[name](**kwargs)

def retrieve_model(name: ModelType) -> Callable[[...], LightningModule]:
    if name not in _MODEL_REGISTRY:
        msg = f"Unknown model '{name}'. Registered: {list(_MODEL_REGISTRY)}"
        raise KeyError(msg)
    return _MODEL_REGISTRY[name]