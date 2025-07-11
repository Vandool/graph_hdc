import pytorch_lightning as pl
import torch
from torch import Tensor

from src.normalizing_flow.config import FlowConfig


class AbstractNFModel(pl.LightningModule):
    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.flow = None  # Must be set by subclasses

    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        batch = x.shape[0]
        flat = x.view(batch, -1)
        if context is not None:
            return self.flow.forward(flat, context)
        return self.flow.forward(flat)

    def forward_kld(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        flat = x.view(batch, -1)
        return self.flow.forward_kld(flat)

    def sample(self, num_samples: int) -> Tensor:
        z, _ = self.flow.sample(num_samples)
        if self.cfg.input_shape is not None:
            return z.view(num_samples, *self.cfg.input_shape)
        return z

    def training_step(self, batch, batch_idx):
        loss = self.forward_kld(batch).mean()
        if torch.isfinite(loss):
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        else:
            self.log("train_loss", torch.tensor(0.0), prog_bar=True)
            self.logger.warning(f"Skipping NaN/Inf loss at step {batch_idx}")
            return None

    def validation_step(self, batch, batch_idx):
        loss = self.forward_kld(batch).mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

    def on_train_epoch_start(self):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log("lr", float(lr), on_epoch=True, prog_bar=True)