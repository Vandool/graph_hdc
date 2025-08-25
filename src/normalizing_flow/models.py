import normflows as nf
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor

from src.normalizing_flow.config import FlowConfig


import torch
import pytorch_lightning as pl
from dataclasses import asdict

class AbstractNFModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.flow = None  # child must build a flow

    # Child classes will override training_step / validation_step to accept PyG Batch.
    # These helpers operate on *flat* tensors already computed by the child.

    def nf_forward(self, flat):
        # log q(x), z = f(x), etc. (normflows interface returns log prob with signs)
        return self.flow.forward(flat)

    def nf_forward_kld(self, flat):
        return self.flow.forward_kld(flat)  # returns per-sample KL, normflows API

    def sample(self, num_samples: int):
        z, logs = self.flow.sample(num_samples)
        return z, logs

    def configure_optimizers(self):
        import torch
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    @classmethod
    def load_from_path(cls, path: str):
        ckpt = torch.load(path, map_location="cpu")
        cfg_dict = ckpt.get("hyper_parameters", {}).get("cfg", {})
        model = cls(type("X", (), cfg_dict))  # minimal duck-typed cfg if needed
        model.load_state_dict(ckpt["state_dict"])
        return model

    def save_to_path(self, path: str):
        ckpt = {
            "state_dict": self.state_dict(),
            "hyper_parameters": {"cfg": asdict(self.cfg) if hasattr(self.cfg, "__dataclass_fields__") else dict(self.cfg.__dict__)},
        }
        torch.save(ckpt, path)



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
