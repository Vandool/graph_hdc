# ------------------------- registry -------------------------
from typing import Callable

from torch import nn

_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    def deco(ctor: Callable[..., nn.Module]):
        _MODEL_REGISTRY[name] = ctor
        return ctor

    return deco


def resolve_model(name: str, **kwargs) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)
