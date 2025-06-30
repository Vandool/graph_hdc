import normflows as nf
import pytorch_lightning as pl
import torch
from torch import Tensor

from src.normalizing_flow.config import SpiralFlowConfig


class NeuralSplineLightning(pl.LightningModule):
    def __init__(self, cfg: SpiralFlowConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        latent_dim = cfg.num_input_channels
        flows = []

        for _ in range(cfg.num_flows):
            spline = nf.flows.AutoregressiveRationalQuadraticSpline(
                num_input_channels=cfg.num_input_channels,
                num_blocks=cfg.num_blocks,
                num_hidden_channels=cfg.num_hidden_channels,
                num_context_channels=cfg.num_context_channels,
                num_bins=cfg.num_bins,
                tail_bound=cfg.tail_bound,
                activation=cfg.activation,
                dropout_probability=cfg.dropout_probability,
                permute_mask=cfg.permute,
            )
            flows.append(spline)
            flows.append(nf.flows.LULinearPermute(latent_dim))
            # flows.append(nf.flows.Permute(latent_dim, mode="shuffle"))

        base = nf.distributions.DiagGaussian(latent_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)

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
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def on_train_epoch_start(self):
        # Put any logic you want here: custom metric resets, LR logging, sampling, etc.
        # For demonstration, we log the current learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', float(lr), on_epoch=True, prog_bar=True)
