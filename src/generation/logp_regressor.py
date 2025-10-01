from __future__ import annotations

from collections.abc import Iterable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torchmetrics import R2Score

from src.utils.registery import register_model

ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leakyrelu": lambda: nn.LeakyReLU(0.1),
}

NORMS = {
    "batch_norm": nn.BatchNorm1d,
    "lay_norm": nn.LayerNorm,  # as specified
    "none": None,
}


def _instantiate(factory):  # noqa: ANN202
    # factory can be a class (nn.ReLU) or a zero-arg callable (lambda: nn.LeakyReLU(0.1))
    return factory() if callable(factory) else factory


def _make_mlp(
    in_dim: int,
    hidden_dims: Iterable[int],
    out_dim: int = 1,
    *,
    activation: str = "gelu",
    dropout: float = 0.0,
    norm: str = "lay_norm",  # exclusive: choose exactly one or use None to disable
) -> nn.Sequential:
    layers: list[nn.Module] = []
    act_factory = ACTS.get(activation, nn.GELU)
    norm_factory = NORMS.get(norm, nn.LayerNorm)

    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if norm_factory is not None:
            # BatchNorm1d and LayerNorm both accept feature dim h
            layers.append(norm_factory(h))
        layers.append(_instantiate(act_factory))
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


@register_model("LPR")
class LogPRegressor(pl.LightningModule):
    """Simple MLP regressor for cLogP with sweep-ready knobs (exclusive `norm`)."""

    def __init__(
        self,
        input_dim: int = 3200,
        hidden_dims: Iterable[int] = (1024, 256, 64),
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm: str = "lay_norm",  # 'batch_norm' | 'lay_norm' | None/'' to disable
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

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
        D = self.hparams.input_dim // 2
        B = batch.num_graphs
        n, g = batch.node_terms.as_subclass(torch.Tensor), batch.graph_terms.as_subclass(torch.Tensor)
        if n.dim() == 1:
            n = n.view(B, D)
        if g.dim() == 1:
            g = g.view(B, D)
        return torch.cat([n, g], dim=-1)

    def forward(self, batch) -> torch.Tensor:
        # batch.node_terms: [B*D], batch.graph_terms: [B*D]
        x = self._flat_from_batch(batch)  # [B, 2D]
        return self.net(x).squeeze(-1)

    def gen_forward(self, batch):
        return self.net(batch).squeeze(-1)

    def _step(self, batch, stage: str):
        y = batch.logp.float().view(-1)  # [B]
        y_hat = self.forward(batch)  # [B]
        loss = self.loss_fn(y_hat, y)
        with torch.no_grad():
            mae = F.l1_loss(y_hat, y)
            rmse = torch.sqrt(F.mse_loss(y_hat, y))
        B = batch.num_graphs
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}_mae", mae, prog_bar=(stage != "train"), on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)

        # ---- R² update + epoch log ----
        if stage == "train":
            self.train_r2.update(y_hat.detach(), y.detach())
            self.log("train_r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        elif stage == "val":
            self.val_r2.update(y_hat.detach(), y.detach())
            self.log("val_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        elif stage == "test":
            self.test_r2.update(y_hat.detach(), y.detach())
            self.log("test_r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)

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
        )
