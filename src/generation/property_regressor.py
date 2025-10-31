"""
Multi-Property Regressor for Molecular Properties

Supports regression on multiple molecular properties:
- logp: Octanol-water partition coefficient
- sa_score: Synthetic accessibility score
- qed: Quantitative Estimate of Drug-likeness
- max_ring_size: Maximum ring size in the molecule

This is a backward-compatible extension of LogPRegressor (LPR).
"""
from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torchmetrics import R2Score

from src.utils.registery import register_model

# Reuse the same activation and normalization options as LPR
ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leakyrelu": lambda: nn.LeakyReLU(0.1),
}

NORMS = {
    "lay_norm": nn.LayerNorm,
    "none": None,
}


class MolecularProperty(str, Enum):
    """Supported molecular properties for regression."""

    LOGP = "logp"
    SA_SCORE = "sa_score"
    QED = "qed"
    MAX_RING_SIZE = "max_ring_size"

    @classmethod
    def from_string(cls, s: str) -> "MolecularProperty":
        """Convert string to enum, case-insensitive."""
        s_lower = s.lower()
        for prop in cls:
            if prop.value == s_lower:
                return prop
        raise ValueError(f"Unknown property: {s}. Supported: {[p.value for p in cls]}")


def _instantiate(factory):  # noqa: ANN202
    """Instantiate activation/norm factory."""
    return factory() if callable(factory) else factory


def _make_mlp(
    in_dim: int,
    hidden_dims: Iterable[int],
    out_dim: int = 1,
    *,
    activation: str = "gelu",
    dropout: float = 0.0,
    norm: str = "lay_norm",
) -> nn.Sequential:
    """Build MLP with configurable architecture."""
    layers: list[nn.Module] = []
    act_factory = ACTS.get(activation, nn.GELU)
    norm_factory = NORMS.get(norm, nn.LayerNorm)

    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if norm_factory is not None:
            layers.append(norm_factory(h))
        layers.append(_instantiate(act_factory))
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


@register_model("PR")
class PropertyRegressor(pl.LightningModule):
    """
    Multi-property regressor for molecular properties.

    Backward compatible with LogPRegressor (LPR) - defaults to logp if no property specified.
    """

    def __init__(
        self,
        input_dim: int = 3 * 1600,
        hidden_dims: Iterable[int] = (1024, 256, 64),
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm: str = "lay_norm",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        target_property: str = "logp",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Parse target property
        self.target_property = MolecularProperty.from_string(target_property)

        self.net = _make_mlp(
            in_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=1,
            activation=activation,
            dropout=dropout,
            norm=norm,
        )
        self.loss_fn = nn.MSELoss()
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

    def on_train_epoch_start(self):
        self.train_r2.reset()

    def on_validation_epoch_start(self):
        self.val_r2.reset()

    def on_test_epoch_start(self):
        self.test_r2.reset()

    def _flat_from_batch(self, batch) -> torch.Tensor:
        """Extract and flatten hypervector features from batch."""
        D = self.hparams.input_dim // 2
        B = batch.num_graphs
        e = batch.edge_terms.as_subclass(torch.Tensor)
        g = batch.graph_terms.as_subclass(torch.Tensor)

        # Cast to module dtype
        td = self.dtype
        if e.dtype.is_floating_point:
            e = e.to(td)
        if g.dtype.is_floating_point:
            g = g.to(td)

        # Reshape if needed (optimized: direct view)
        e = e.view(B, D)
        g = g.view(B, D)
        return torch.cat([e, g], dim=-1).contiguous()

    def forward(self, batch) -> torch.Tensor:
        """Forward pass on PyG batch."""
        x = self._flat_from_batch(batch)  # [B, 2D]
        return self.net(x).squeeze(-1)

    def gen_forward(self, batch):
        """Forward pass on pre-flattened features."""
        return self.net(batch).squeeze(-1)

    def _get_target(self, batch) -> torch.Tensor:
        """Extract target property from batch based on configuration."""
        # Map property to batch attribute
        if self.target_property == MolecularProperty.LOGP:
            target = batch.logp
        elif self.target_property == MolecularProperty.SA_SCORE:
            target = batch.sa_score
        elif self.target_property == MolecularProperty.QED:
            target = batch.qed
        elif self.target_property == MolecularProperty.MAX_RING_SIZE:
            target = batch.max_ring_size
        else:
            raise ValueError(f"Unsupported property: {self.target_property}")

        return target.as_subclass(torch.Tensor).to(self.dtype).view(-1)

    def _step(self, batch, stage: str):
        """Generic training/validation/test step."""
        y = self._get_target(batch)
        y_hat = self.forward(batch)
        loss = self.loss_fn(y_hat, y)

        with torch.no_grad():
            mae = F.l1_loss(y_hat, y)
            rmse = torch.sqrt(F.mse_loss(y_hat, y))

        B = batch.num_graphs
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}_mae", mae, prog_bar=(stage != "train"), on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)

        # R² update + epoch log
        if stage == "train":
            self.train_r2.update(y_hat.detach(), y.detach())
            self.log("train_r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        elif stage == "val":
            self.val_r2.update(y_hat.detach(), y.detach())
            self.log("val_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        elif stage == "test":
            self.test_r2.update(y_hat.detach(), y.detach())
            self.log("test_r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):  # noqa: ARG002
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            foreach=True,  # Faster multi-tensor operations (PyTorch 2.0+)
        )

    def on_after_batch_transfer(self, batch, _: int):
        """Cast batch to model dtype for mixed precision training."""

        def cast(x):
            return x.to(self.dtype) if torch.is_tensor(x) and x.dtype.is_floating_point else x

        if isinstance(batch, dict):
            return {k: cast(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(cast(v) for v in batch)
        return cast(batch)
