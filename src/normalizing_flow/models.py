import contextlib
import math
from dataclasses import dataclass
from pathlib import Path

import normflows as nf
import pytorch_lightning as pl
import torch
from torch import Tensor

from src.encoding.the_types import VSAModel
from src.utils.registery import register_model


## Real NVP
@dataclass
class FlowConfig:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"

    hv_dim: int = 7744
    vsa: VSAModel = VSAModel.HRR
    num_flows: int = 8
    num_hidden_channels: int = 512

    smax_initial = 1.0
    smax_final = 6.0
    smax_warmup_epochs = 15


class AbstractNFModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.flow = None  # child must build a flow

    # Child classes will override training_step / validation_step to accept PyG Batch.
    # These helpers operate on *flat* tensors already computed by the child.

    def nf_forward_kld(self, flat):
        return self.flow.forward_kld(flat)  # returns per-sample KL, normflows API

    def sample(self, num_samples: int):
        z, logs = self.flow.sample(num_samples)
        return z, logs

    def split(self, flat: torch.Tensor) -> tuple[Tensor, ...]:
        D = self.D
        if self.dim_multiplier == 3:
            return (
                flat[:, :D].contiguous(),  # node terms
                flat[:, D : 2 * D].contiguous(),  # edge terms
                flat[:, 2 * D :].contiguous(),  # graph terms
            )
        return (
            flat[:, :D].contiguous(),
            flat[:, D:].contiguous(),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    @classmethod
    def load_from_path(cls, path: str | Path) -> "AbstractNFModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = FlowConfig(**checkpoint["hyper_parameters"]["cfg"])
        model = cls(cfg)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def save_to_path(self, path: str):
        """
        Save the model state and hyperparameters to file.
        """
        ckpt = {"state_dict": self.state_dict(), "hyper_parameters": {"cfg": self.cfg.__dict__()}}
        torch.save(ckpt, path)


class BoundedMLP(torch.nn.Module):
    def __init__(self, dims, smax=6.0):
        super().__init__()
        self.net = nf.nets.MLP(dims, init_zeros=True)  # keeps identity at start
        self.smax = float(smax)
        self.last_pre = None  # for logging

    def forward(self, x):
        pre = self.net(x)
        self.last_pre = pre
        return torch.tanh(pre) * self.smax  # bound log-scale in [-smax, smax]


@register_model("NVP")
class RealNVPV2Lightning(AbstractNFModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        D = int(cfg.hv_dim)
        self.D = D
        dim_multiplier = 2 if not hasattr(cfg, "hv_count") else cfg.hv_count
        self.hv_count = dim_multiplier
        self.flat_dim = dim_multiplier * D

        mask = (torch.arange(self.flat_dim) % 2).to(torch.float32)
        self.register_buffer("mask0", mask)

        # per-feature standardization params (fill later)
        default = torch.get_default_dtype()  # follows trainer precision if set early
        self.register_buffer("mask0", (torch.arange(self.flat_dim) % 2).to(default))
        self.register_buffer("mu", torch.zeros(self.flat_dim, dtype=default))
        self.register_buffer("log_sigma", torch.zeros(self.flat_dim, dtype=default))

        self.s_modules = []  # keep handles for warmup/logging
        flows = []

        # ActNorm layers
        use_act_norm = cfg.use_act_norm
        if use_act_norm and hasattr(nf.flows, "ActNorm"):
            flows.append(nf.flows.ActNorm(self.flat_dim))  # learns per-feature shift/scale

        hidden = int(cfg.num_hidden_channels)
        for _ in range(int(cfg.num_flows)):
            layers = [self.flat_dim, hidden, hidden, self.flat_dim]
            t_net = nf.nets.MLP(layers, init_zeros=True)
            s_net = BoundedMLP(layers, smax=getattr(cfg, "smax_final", 6))
            self.s_modules.append(s_net)

            flows.append(nf.flows.MaskedAffineFlow(self.mask0, t=t_net, s=s_net))
            flows.append(nf.flows.Permute(self.flat_dim, mode="shuffle"))

        base = nf.distributions.DiagGaussian(self.flat_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)

    def set_standardization(self, mu, sigma, eps=1e-6):
        tgt_dtype = self.mu.dtype  # match buffer dtype (fp32/fp64)
        mu = torch.as_tensor(mu, dtype=tgt_dtype, device=self.device)
        sigma = torch.as_tensor(sigma, dtype=tgt_dtype, device=self.device)
        self.mu.copy_(mu)
        self.log_sigma.copy_(torch.log(torch.clamp(sigma, min=eps)))

    def _pretransform(self, x):
        x = x.to(self.mu.dtype)
        """z = (x - mu) / sigma ; returns (z, +sum(log sigma)) for log-det correction."""
        z = (x - self.mu) * torch.exp(-self.log_sigma)
        # log|det ∂z/∂x| = -sum(log_sigma); NLL must ADD +sum(log_sigma)
        return z, float(self.log_sigma.sum().item())

    def _posttransform(self, z):
        return self.mu + z * torch.exp(self.log_sigma)

    def decode_from_latent(self, z_std):
        # z_std in standardized latent space -> x in data-space (differentiable)
        x_std = self.flow.forward(z_std)  # normflows forward: z_std -> x_std
        return self._posttransform(x_std)

    @torch.no_grad()
    def sample_split(self, num_samples: int) -> dict:
        """
        Returns:
          node_terms:  [num_samples, D]
          graph_terms: [num_samples, D]
          logs
        """
        z, _logs = self.sample(num_samples)  # standardized space
        x = self._posttransform(z)  # back to data space
        if self.hv_count == 3:
            node_terms, edge_terms, graph_terms = self.split(x)
            return {
                "node_terms": node_terms,
                "edge_terms": edge_terms,
                "graph_terms": graph_terms,
                "logs": _logs,
            }
        edge_terms, graph_terms = self.split(x)
        return {
            "edge_terms": edge_terms,
            "graph_terms": graph_terms,
            "logs": _logs,
        }

    def nf_forward_kld(self, flat):
        """Example: exact NLL with pre-transform correction."""
        z, log_det_corr = self._pretransform(flat)
        # If your nf API returns -log p(z) per-sample, add the correction:
        nll = -self.flow.log_prob(z) + log_det_corr
        return nll

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # 5% warmup then cosine
        steps_per_epoch = max(
            1, getattr(self.trainer, "estimated_stepping_batches", 1000) // max(1, self.trainer.max_epochs)
        )
        warmup = int(0.05 * self.trainer.max_epochs) * steps_per_epoch
        total = self.trainer.max_epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: min(1.0, step / max(1, warmup))
            * 0.5
            * (1 + math.cos(math.pi * max(0, step - warmup) / max(1, total - warmup))),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_fit_start(self):
        # H100 tip: get tensor cores w/o AMP
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")

    def on_train_epoch_start(self):
        # log-det warmup for stability: ramp smax from small → final
        warm = int(getattr(self.cfg, "smax_warmup_epochs", 15))
        s0 = float(getattr(self.cfg, "smax_initial", 1.0))
        s1 = float(getattr(self.cfg, "smax_final", 6.0))
        if warm > 0:
            t = min(1.0, self.current_epoch / max(1, warm))
            smax = (1 - t) * s0 + t * s1
            for m in self.s_modules:
                m.smax = smax

    # Lightning will move the PyG Batch; just ensure we return batch on device
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def _flat_from_batch(self, batch) -> torch.Tensor:
        D = self.D
        B = batch.num_graphs
        n, e, g = batch.node_terms, batch.edge_terms, batch.graph_terms
        if n.dim() == 1:
            n = n.view(B, D)
        if e.dim() == 1:
            e = e.view(B, D)
        if g.dim() == 1:
            g = g.view(B, D)

        if self.hv_count == 3:
            return torch.cat([n, e, g], dim=-1)
        return torch.cat([e, g], dim=-1)

    def training_step(self, batch, batch_idx):
        flat = self._flat_from_batch(batch)
        obj = self.nf_forward_kld(flat)  # [B]
        # keep only finite samples (safety)
        obj = obj[torch.isfinite(obj)]
        if obj.numel() == 0:
            # log a plain float
            self.log("nan_loss_batches", 1.0, on_step=True, prog_bar=True, batch_size=flat.size(0))
            return None

        loss = obj.mean()
        # *** log pure Python float and pass batch_size ***
        self.log(
            "train_loss",
            float(loss.detach().cpu().item()),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=flat.size(0),
        )

        # optional: monitor scale pre-activation magnitude (float)
        with torch.no_grad():
            s_absmax = 0.0
            for m in getattr(self, "s_modules", []):
                if getattr(m, "last_pre", None) is not None and torch.isfinite(m.last_pre).any():
                    s_absmax = max(s_absmax, float(m.last_pre.detach().abs().max().cpu().item()))
        self.log("s_pre_absmax", s_absmax, on_step=True, prog_bar=True, batch_size=flat.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        flat = self._flat_from_batch(batch)
        obj = self.nf_forward_kld(flat)
        obj = obj[torch.isfinite(obj)]
        val = float("nan") if obj.numel() == 0 else float(obj.mean().detach().cpu().item())
        self.log("val_loss", val, on_epoch=True, prog_bar=True, batch_size=flat.size(0))
        return torch.tensor(val, device=flat.device) if val == val else None  # return NaN-safe

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        # Cast floats to the module’s dtype (bf16/fp32/fp64 depending on Trainer precision)
        return _cast_to_dtype(batch, self.dtype)


def _cast_to_dtype(x, dtype):
    if torch.is_tensor(x):
        return x.to(dtype) if x.dtype.is_floating_point else x
    if isinstance(x, dict):
        return {k: _cast_to_dtype(v, dtype) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_cast_to_dtype(v, dtype) for v in x)
    return x
