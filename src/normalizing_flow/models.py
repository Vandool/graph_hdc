from pathlib import Path

import normflows as nf
import pytorch_lightning as pl
import torch
from torch import Tensor

from src.normalizing_flow.config import FlowConfig


class AbstractNFModel(pl.LightningModule):
    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.flow = None  # Must be initialized in subclass

    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        batch = x.shape[0]
        flat = x.view(batch, -1)
        return self.flow.forward(flat, context) if context is not None else self.flow.forward(flat)

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
        print(f"Skipping NaN/Inf loss at step {batch_idx}")
        return None

    def validation_step(self, batch, batch_idx):
        loss = self.forward_kld(batch).mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def on_train_epoch_start(self):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log("lr", float(lr), on_epoch=True, prog_bar=True)

    def load_from_path(self, path: str):
        """
        Load the model state and hyperparameters from file.
        """
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["state_dict"])
        self.cfg = FlowConfig(**checkpoint["hyper_parameters"]["cfg"])

    def save_to_path(self, path: str):
        """
        Save the model state and hyperparameters to file.
        """
        ckpt = {
            "state_dict": self.state_dict(),
            "hyper_parameters": {"cfg": self.cfg.dict()}
        }
        torch.save(ckpt, path)


class RealNVPLightning(AbstractNFModel):
    def __init__(self, cfg: FlowConfig):
        super().__init__(cfg)


        # flatten dimensionality
        latent_dim = int(np.prod(cfg.input_shape))

        # build alternating mask [1,0,1,0,...] and register as buffer
        mask = torch.tensor(
            [1 if i % 2 == 0 else 0 for i in range(latent_dim)],
            dtype=torch.float32
        )
        self.register_buffer('mask', mask)

        flows = []
        for _ in range(cfg.num_flows):
            # 2 hidden layers: [latent_dim → hidden → hidden → latent_dim]
            mlp_layers = [latent_dim,
                          cfg.num_hidden_channels,
                          cfg.num_hidden_channels,
                          latent_dim]

            t_net = nf.nets.MLP(mlp_layers, init_zeros=True)
            s_net = nf.nets.MLP(mlp_layers, init_zeros=True)

            flows.append(
                nf.flows.MaskedAffineFlow(self.mask, t=t_net, s=s_net)
            )
            flows.append(
                nf.flows.Permute(latent_dim, mode='swap')
            )

        base = nf.distributions.DiagGaussian(latent_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)


class NeuralSplineLightning(AbstractNFModel):
    def __init__(self, cfg: FlowConfig):
        super().__init__(cfg)

        latent_dim = cfg.num_input_channels
        flows = []
        for _ in range(cfg.num_flows):
            flows.append(nf.flows.AutoregressiveRationalQuadraticSpline(
                num_input_channels=cfg.num_input_channels,
                num_blocks=cfg.num_blocks,
                num_hidden_channels=cfg.num_hidden_channels,
                num_context_channels=cfg.num_context_channels,
                num_bins=cfg.num_bins,
                tail_bound=cfg.tail_bound,
                activation=cfg.activation,
                dropout_probability=cfg.dropout_probability,
                permute_mask=cfg.permute,
            ))
            flows.append(nf.flows.Permute(latent_dim, mode="shuffle"))

        base = nf.distributions.DiagGaussian(latent_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)