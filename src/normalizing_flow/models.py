import contextlib
import math
from dataclasses import dataclass
from pathlib import Path

import normflows as nf
import pytorch_lightning as pl
import torch
from normflows.flows.neural_spline.coupling import PiecewiseRationalQuadraticCoupling
from normflows.nets import ResidualNet
from torch import Tensor, nn

from src.encoding.configs_and_constants import SupportedDataset
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
        if self.hv_count == 3:
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

        # Per-term standardization support (separate mu/sigma for edge_terms and graph_terms)
        self.per_term_standardization = getattr(cfg, "per_term_standardization", False)
        self._per_term_split = None  # Will be set during fit_per_term_standardization

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
        """z = (x - mu) / sigma ; returns (z, +sum(log sigma)) for log-det correction."""
        if self.per_term_standardization and self._per_term_split is not None:
            # Per-term standardization: separate mu/sigma for edge and graph terms
            split = self._per_term_split
            edge = x[..., :split]
            graph = x[..., split:]

            mu_edge = self.mu[:split]
            mu_graph = self.mu[split:]
            log_sigma_edge = self.log_sigma[:split]
            log_sigma_graph = self.log_sigma[split:]

            z_edge = (edge - mu_edge) * torch.exp(-log_sigma_edge)
            z_graph = (graph - mu_graph) * torch.exp(-log_sigma_graph)
            z = torch.cat([z_edge, z_graph], dim=-1)
        else:
            # Global standardization: all dimensions together
            z = (x - self.mu) * torch.exp(-self.log_sigma)

        # log|det ∂z/∂x| = -sum(log_sigma); NLL must ADD +sum(log_sigma)
        return z, float(self.log_sigma.sum().item())

    def _posttransform(self, z):
        """x = mu + z * sigma ; inverse of _pretransform."""
        if self.per_term_standardization and self._per_term_split is not None:
            # Per-term inverse: separate for edge and graph terms
            split = self._per_term_split
            z_edge = z[..., :split]
            z_graph = z[..., split:]

            mu_edge = self.mu[:split]
            mu_graph = self.mu[split:]
            log_sigma_edge = self.log_sigma[:split]
            log_sigma_graph = self.log_sigma[split:]

            edge = mu_edge + z_edge * torch.exp(log_sigma_edge)
            graph = mu_graph + z_graph * torch.exp(log_sigma_graph)
            return torch.cat([edge, graph], dim=-1)
        # Global inverse: all dimensions together
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
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            foreach=True,  # Faster multi-tensor operations (PyTorch 2.0+)
        )
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
        e, g = batch.edge_terms, batch.graph_terms
        # if e.dim() == 1:
        e = e.view(B, D)
        # if g.dim() == 1:
        g = g.view(B, D)

        # if self.hv_count == 3:
        #     n = batch.node_terms
        #     if n.dim() == 1:
        #         n = n.view(B, D)
        #     return torch.cat([n, e, g], dim=-1)
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


@register_model("NVP-V3")
class RealNVPV3Lightning(RealNVPV2Lightning):  # Inherit all helpers
    def __init__(self, cfg):
        # Call AbstractNFModel init, skipping V2's init
        AbstractNFModel.__init__(self, cfg)

        D = int(cfg.hv_dim)  # This is 256
        self.D = D
        dim_multiplier = 2 if not hasattr(cfg, "hv_count") else cfg.hv_count
        self.hv_count = dim_multiplier
        self.flat_dim = dim_multiplier * D  # This is 512

        default = torch.get_default_dtype()

        # --- Semantic Masks ---
        # Mask A: 0s for edge_terms, 1s for graph_terms
        # We will compute s,t from graph_terms (mask=1)
        # We will apply s,t to edge_terms (mask=0)
        mask_a = torch.zeros(self.flat_dim, dtype=default)
        mask_a[D:] = 1.0
        self.register_buffer("mask_a", mask_a)

        # Mask B: 1s for edge_terms, 0s for graph_terms
        # (Opposite of above)
        mask_b = 1.0 - mask_a
        self.register_buffer("mask_b", mask_b)

        # --- Buffers ---
        self.register_buffer("mu", torch.zeros(self.flat_dim, dtype=default))
        self.register_buffer("log_sigma", torch.zeros(self.flat_dim, dtype=default))
        self.per_term_standardization = getattr(cfg, "per_term_standardization", False)
        self._per_term_split = None

        self.s_modules = []  # To hold BoundedMLP modules for warmup
        flows = []

        # --- ActNorm ---
        use_act_norm = getattr(cfg, "use_act_norm", True)
        if use_act_norm and hasattr(nf.flows, "ActNorm"):
            flows.append(nf.flows.ActNorm(self.flat_dim))

        # --- Optuna Hyperparameters for Conditioner MLP ---
        hidden_dim = int(getattr(cfg, "hidden_dim", 1024))
        num_hidden_layers = int(getattr(cfg, "num_hidden_layers", 3))

        # --- *** CORRECTED IMPLEMENTATION *** ---

        # The t and s networks for MaskedAffineFlow MUST take flat_dim as input.
        # The mask ensures only the unmasked inputs are "seen" by the weights.

        # Input: self.flat_dim (512)
        # Hidden: [1024, 1024, 1024] (example)
        # Output: self.flat_dim (512)
        mlp_layers = [self.flat_dim] + [hidden_dim] * num_hidden_layers + [self.flat_dim]

        # --- V3 (Real NVP) Flow Loop ---
        for i in range(int(cfg.num_flows)):
            smax = getattr(cfg, "smax_final", 6)

            # Create the two networks, identical to NVP-V2's method
            t_net = nf.nets.MLP(mlp_layers.copy(), init_zeros=True)
            s_net = BoundedMLP(mlp_layers.copy(), smax=smax)

            self.s_modules.append(s_net)  # Add for smax warmup

            # Alternate the SEMANTIC mask
            mask = self.mask_a if i % 2 == 0 else self.mask_b

            # Use the SAME class as NVP-V2
            flows.append(nf.flows.MaskedAffineFlow(mask, t=t_net, s=s_net))

        # --- Base Distribution ---
        base = nf.distributions.DiagGaussian(self.flat_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)


@dataclass
class SFConfig:
    """Configuration for the SplineFlowLightning Model."""

    exp_dir_name: str | None = None
    seed: int = 42
    # epochs: int = 1200
    epochs: int = 2
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    device: str = "cuda"
    dropout_probability: float = 0.0

    hv_dim: int = 256
    hv_count: int = 2
    vsa: str = "HRR"
    dataset: SupportedDataset = SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4

    per_term_standardization: bool = True
    use_act_norm: bool = True

    num_flows: int = 8
    num_hidden_channels: int = 512
    num_bins: int = 8
    num_blocks: int = 2  # Number of residual blocks in the conditioner


@register_model("SplineFlow")
class SplineFlowLightning(RealNVPV2Lightning):  # Inherits from your base class
    def __init__(self, cfg):
        AbstractNFModel.__init__(self, cfg)
        D = int(cfg.hv_dim)
        self.D = D
        dim_multiplier = 2 if not hasattr(cfg, "hv_count") else cfg.hv_count
        self.hv_count = dim_multiplier
        self.flat_dim = dim_multiplier * D

        # --- Buffers ---
        default = torch.get_default_dtype()
        self.register_buffer("mu", torch.zeros(self.flat_dim, dtype=default))
        self.register_buffer("log_sigma", torch.zeros(self.flat_dim, dtype=default))

        self.per_term_standardization = getattr(cfg, "per_term_standardization", True)
        self._per_term_split = D

        # --- Semantic Masks ---
        mask_a = torch.zeros(self.flat_dim, dtype=default)
        mask_a[D:] = 1.0  # Uses graph_terms (1s) to update edge_terms (0s)
        self.register_buffer("mask_a", mask_a)

        mask_b = 1.0 - mask_a  # Uses edge_terms (1s) to update graph_terms (0s)
        self.register_buffer("mask_b", mask_b)

        # --- Build the Spline Flow ---
        flows = []

        use_act_norm = getattr(cfg, "use_act_norm", True)
        if use_act_norm:
            flows.append(nf.flows.ActNorm(self.flat_dim))

        # --- Get HPO parameters from config ---
        hidden = int(getattr(cfg, "num_hidden_channels", 512))
        num_bins = int(getattr(cfg, "num_bins", 8))
        num_blocks = int(getattr(cfg, "num_blocks", 2))  # Get the new HPO param
        dropout_prob = float(getattr(cfg, "dropout_probability", 0.0))

        for i in range(int(cfg.num_flows)):
            # The API needs a function that *creates* the net
            def create_conditioner(in_features, out_features):
                """
                in_features will be D (256)
                out_features will be D * ((3*num_bins)+1)
                """
                # Use ResidualNet, which has the correct forward(inputs, context=None) signature
                return ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden,  # HPO param
                    num_blocks=num_blocks,  # HPO param
                    activation=nn.ReLU(),
                    dropout_probability=dropout_prob,
                    use_batch_norm=False,
                    context_features=None,  # We are not using external context
                )

            mask = self.mask_a if i % 2 == 0 else self.mask_b

            flows.append(
                PiecewiseRationalQuadraticCoupling(
                    mask=mask,
                    transform_net_create_fn=create_conditioner,
                    num_bins=num_bins,
                    tails="linear",
                    # This bound is now for the *central* spline region.
                    # 3.0 is a safe and robust choice that covers
                    # >99% of a standard Gaussian.
                    tail_bound=3.0,
                )
            )

            flows.append(nf.flows.Permute(self.flat_dim, mode="shuffle"))

        base = nf.distributions.DiagGaussian(self.flat_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)

    def set_standardization(self, mu, sigma, eps=1e-6):
        tgt_dtype = self.mu.dtype
        mu = torch.as_tensor(mu, dtype=tgt_dtype, device=self.device)
        sigma = torch.as_tensor(sigma, dtype=tgt_dtype, device=self.device)
        self.mu.copy_(mu)
        self.log_sigma.copy_(torch.log(torch.clamp(sigma, min=eps)))

    def _pretransform(self, x):
        if self.per_term_standardization and self._per_term_split is not None:
            split = self._per_term_split
            edge, graph = x[..., :split], x[..., split:]
            mu_edge, mu_graph = self.mu[:split], self.mu[split:]
            log_sigma_edge, log_sigma_graph = self.log_sigma[:split], self.log_sigma[split:]
            z_edge = (edge - mu_edge) * torch.exp(-log_sigma_edge)
            z_graph = (graph - mu_graph) * torch.exp(-log_sigma_graph)
            z = torch.cat([z_edge, z_graph], dim=-1)
        else:
            z = (x - self.mu) * torch.exp(-self.log_sigma)
        return z, float(self.log_sigma.sum().item())

    def _posttransform(self, z):
        if self.per_term_standardization and self._per_term_split is not None:
            split = self._per_term_split
            z_edge, z_graph = z[..., :split], z[..., split:]
            mu_edge, mu_graph = self.mu[:split], self.mu[split:]
            log_sigma_edge, log_sigma_graph = self.log_sigma[:split], self.log_sigma[split:]
            edge = mu_edge + z_edge * torch.exp(log_sigma_edge)
            graph = mu_graph + z_graph * torch.exp(log_sigma_graph)
            return torch.cat([edge, graph], dim=-1)
        return self.mu + z * torch.exp(self.log_sigma)

    def decode_from_latent(self, z_std):
        x_std = self.flow.forward(z_std)
        return self._posttransform(x_std)

    def on_train_epoch_start(self):
        """
        Override parent's method.
        The parent method warms up s_modules, which we don't have.
        So we just do nothing.
        """

    def training_step(self, batch, batch_idx):
        """
        Override parent's method to remove s_modules logging.
        """
        flat = self._flat_from_batch(batch)
        obj = self.nf_forward_kld(flat)  # [B]
        obj = obj[torch.isfinite(obj)]
        if obj.numel() == 0:
            self.log("nan_loss_batches", 1.0, on_step=True, prog_bar=True, batch_size=flat.size(0))
            return None

        loss = obj.mean()
        self.log(
            "train_loss",
            float(loss.detach().cpu().item()),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=flat.size(0),
        )

        # The s_pre_absmax logging from the parent is now gone.

        return loss

    @torch.no_grad()
    def sample_split(self, num_samples: int) -> dict:
        z, _logs = self.sample(num_samples)
        x = self._posttransform(z)
        edge_terms, graph_terms = self.split(x)
        return {"edge_terms": edge_terms, "graph_terms": graph_terms, "logs": _logs}

    def nf_forward_kld(self, flat):
        z, log_det_corr = self._pretransform(flat)
        nll = -self.flow.log_prob(z) + log_det_corr
        return nll

    # --- Other helpers from V2 ---
    def on_fit_start(self):
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        return _cast_to_dtype(batch, self.dtype)


# Flow mathing -------------------------------------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t is shape (B,)
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# --- The Core Velocity Network ---
# This is the "MLP" that Flow Matching trains.
# It predicts the vector field v(z_t, t, e)
class VelocityNet(nn.Module):
    def __init__(
        self,
        data_dim: int,  # Dimension of flat (e+g), e.g., 512
        hidden_dim: int,  # HPO param, e.g., 1024
        num_hidden_layers: int,  # HPO param, e.g., 4
        time_emb_dim: int = 64,  # HPO param, e.g., 64
    ):
        super().__init__()

        self.time_embedder = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU()
        )

        # Input is the flat data + time
        input_dim = data_dim + time_emb_dim
        output_dim = data_dim  # It predicts velocity for the flat vector

        mlp_layers = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]

        self.net = nf.nets.MLP(mlp_layers)

    def forward(self, z_t: Tensor, t: Tensor) -> Tensor:
        """
        Inputs:
          z_t:    The noisy flat vector at time t. Shape (B, data_dim)
          t:      The time step. Shape (B,)
        Output:
          v_pred: The predicted velocity. Shape (B, data_dim)
        """
        t_emb = self.time_embedder(t)  # (B, time_emb_dim)
        net_input = torch.cat([z_t, t_emb], dim=-1)  # (B, data_dim + time_emb_dim)
        v_pred = self.net(net_input)
        return v_pred


@dataclass
class FMConfig:
    exp_dir_name: str = None
    seed: int = 42
    epochs: int = 1800
    batch_size: int = 256  # MSE loss is stable, can often use larger BS
    lr: float = 3e-4
    weight_decay: float = 1e-5
    device: str = "cuda"

    hv_dim: int = 256  # This is D_edge and D_graph
    hv_count: int = 2  # (edge_terms, graph_terms)
    vsa: VSAModel = VSAModel.HRR

    dataset: SupportedDataset = SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4

    # --- New HPO Params ---
    hidden_dim: int = 1024
    num_hidden_layers: int = 4
    time_emb_dim: int = 64

    per_term_standardization: bool = True


# --- The PyTorch Lightning Model ---
@register_model("FM")
class FlowMatchingLightning(AbstractNFModel):  # Inherit from your base class
    def __init__(self, cfg):
        super().__init__(cfg)  # This calls self.save_hyperparameters()

        self.D = int(getattr(cfg, "hv_dim", 256))
        self.hv_count = int(getattr(cfg, "hv_count", 2))
        self.flat_dim = self.hv_count * self.D  # 512

        self.model = VelocityNet(
            data_dim=self.flat_dim,
            hidden_dim=int(getattr(cfg, "hidden_dim", 1024)),
            num_hidden_layers=int(getattr(cfg, "num_hidden_layers", 4)),
            time_emb_dim=int(getattr(cfg, "time_emb_dim", 64)),
        )

        self.loss_fn = nn.MSELoss()

        # --- Setup standardization (from V2) ---
        default = torch.get_default_dtype()
        self.register_buffer("mu", torch.zeros(self.flat_dim, dtype=default))
        self.register_buffer("log_sigma", torch.zeros(self.flat_dim, dtype=default))
        self.per_term_standardization = getattr(cfg, "per_term_standardization", True)
        self._per_term_split = None  # Will be set by trainer

    def _flat_from_batch(self, batch) -> torch.Tensor:
        # This method is now inherited from AbstractNFModel, but if it weren't,
        # you'd need the same one from V2.
        D = self.D
        B = batch.num_graphs
        e, g = batch.edge_terms, batch.graph_terms
        e = e.view(B, D)
        g = g.view(B, D)
        return torch.cat([e, g], dim=-1)

    def training_step(self, batch, batch_idx):
        flat_raw = self._flat_from_batch(batch)
        B = flat_raw.shape[0]

        # Standardize the target data
        flat_std, _ = self._pretransform(flat_raw)

        # 1. Sample noise (z0)
        z0_noise = torch.randn_like(flat_std)

        # 2. Sample time t ~ U[0, 1]
        t = torch.rand(B, 1, device=self.device) * (1.0 - 1e-4) + 1e-4

        # 3. Create the OT path *in the standardized space*
        z_t = (1 - t) * z0_noise + t * flat_std

        # 4. Define the target velocity *in the standardized space*
        u_t_target = flat_std - z0_noise

        # 5. Predict velocity
        v_t_pred = self.model(z_t, t.squeeze(-1))

        # 6. Compute loss
        loss = self.loss_fn(v_t_pred, u_t_target)

        self.log(
            "train_loss", float(loss.detach().cpu().item()), on_step=True, on_epoch=True, prog_bar=True, batch_size=B
        )
        return loss

    def validation_step(self, batch, batch_idx):
        flat_raw = self._flat_from_batch(batch)
        B = flat_raw.shape[0]

        flat_std, _ = self._pretransform(flat_raw)

        z0_noise = torch.randn_like(flat_std)
        t = torch.rand(B, 1, device=self.device) * (1.0 - 1e-4) + 1e-4
        z_t = (1 - t) * z0_noise + t * flat_std
        u_t_target = flat_std - z0_noise
        v_t_pred = self.model(z_t, t.squeeze(-1))

        loss = self.loss_fn(v_t_pred, u_t_target)

        self.log("val_loss", float(loss.detach().cpu().item()), on_epoch=True, prog_bar=True, batch_size=B)
        return loss

    @torch.no_grad()
    def sample(self, num_samples: int, n_steps: int = 100) -> Tensor:
        """
        Generate flat_raw data.
        """
        B = num_samples
        dt = 1.0 / n_steps

        # Start from pure noise at t=0
        z = torch.randn(B, self.flat_dim, device=self.device)

        # Solve from t=0 to t=1
        for i in range(n_steps):
            t_now = i * dt
            t_vec = torch.ones(B, device=self.device) * t_now

            # Predict velocity
            v = self.model(z, t_vec)

            # Euler step: z_{i+1} = z_i + v * dt
            z = z + v * dt

        # z is now flat_std at t=1
        flat_std = z

        # De-standardize the result
        flat_raw = self._posttransform(flat_std)

        return flat_raw

    @torch.no_grad()
    def sample_split(self, num_samples: int) -> dict:
        """
        This is the new method your evaluation script needs.
        """
        flat_raw = self.sample(num_samples)

        # Use the split method from the parent class
        edge_terms, graph_terms = self.split(flat_raw)

        return {
            "edge_terms": edge_terms,
            "graph_terms": graph_terms,
            "logs": None,  # No log-det to return
        }

    # --- Methods for standardization (must be in this class) ---

    def set_standardization(self, mu, sigma, eps=1e-6):
        tgt_dtype = self.mu.dtype
        mu = torch.as_tensor(mu, dtype=tgt_dtype, device=self.device)
        sigma = torch.as_tensor(sigma, dtype=tgt_dtype, device=self.device)
        self.mu.copy_(mu)
        self.log_sigma.copy_(torch.log(torch.clamp(sigma, min=eps)))

    def _pretransform(self, x):
        if self.per_term_standardization and self._per_term_split is not None:
            split = self._per_term_split
            edge = x[..., :split]
            graph = x[..., split:]
            mu_edge = self.mu[:split]
            mu_graph = self.mu[split:]
            log_sigma_edge = self.log_sigma[:split]
            log_sigma_graph = self.log_sigma[split:]
            z_edge = (edge - mu_edge) * torch.exp(-log_sigma_edge)
            z_graph = (graph - mu_graph) * torch.exp(-log_sigma_graph)
            z = torch.cat([z_edge, z_graph], dim=-1)
        else:
            z = (x - self.mu) * torch.exp(-self.log_sigma)
        return z, 0.0  # Return 0 for log_det

    def _posttransform(self, z):
        if self.per_term_standardization and self._per_term_split is not None:
            split = self._per_term_split
            z_edge = z[..., :split]
            z_graph = z[..., split:]
            mu_edge = self.mu[:split]
            mu_graph = self.mu[split:]
            log_sigma_edge = self.log_sigma[:split]
            log_sigma_graph = self.log_sigma[split:]
            edge = mu_edge + z_edge * torch.exp(log_sigma_edge)
            graph = mu_graph + z_graph * torch.exp(log_sigma_graph)
            return torch.cat([edge, graph], dim=-1)
        return self.mu + z * torch.exp(self.log_sigma)

    # --- Inherit optimizer and other helpers ---
    def configure_optimizers(self):
        # This is copied from your V2 and is a great scheduler
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            foreach=True,
        )
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

    # --- Other helpers from V2 ---
    def on_fit_start(self):
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        return _cast_to_dtype(batch, self.dtype)
