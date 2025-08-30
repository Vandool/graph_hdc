from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch
from torch import nn
from torch_geometric.data import Batch

from src.encoding.graph_encoders import AbstractGraphEncoder
from src.encoding.the_types import VSAModel
from src.utils.utils import DataTransformer, pick_device_str


class Oracle:
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.encoder = None


    def is_induced_graph(self, small_gs: list[nx.Graph], final_h: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # h1, h2: tensors of shape [batch, hv_dim]
            batch_size = len(small_gs)
            small_hs = [DataTransformer.nx_to_pyg(g) for g in small_gs]
            small_hs_batch = Batch.from_data_list(small_hs)
            h1 = self.encoder.forward(small_hs_batch)["graph_embedding"]

            # No copy (broadcasted view) -> fastest
            h2 = final_h.unsqueeze(0).expand(batch_size, -1)

            cosines = torch.nn.functional.cosine_similarity(h1, h2, dim=-1)
            print("cosine similarities:", (cosines > 0).tolist())

            logits = self.model(h1, h2)
            return torch.sigmoid(logits)


## -------- MLP Classifier -------

@dataclass
class MLPConfig:
    # General
    project_dir: Path | None = None
    result_dir_name: Path | None = None
    seed: int = 42
    epochs: int = 5
    batch_size: int = 256
    train_parents: int = 2000
    valid_parents: int = 500

    # Resume
    last_checkpoint: Path | None = None

    hidden_dims: list[int] = (4096, 2048, 512, 128)

    # HDC / encoder
    hv_dim: int = 88 * 88  # 7744
    vsa: VSAModel = VSAModel.HRR
    device: str = pick_device_str()

    # Optim
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Loader
    num_workers: int = 0
    prefetch_factor: int = 1
    pin_memory: bool = False
    micro_bs: int = 64
    hv_scale: float | None = None

    # Checkpointing
    save_every_seconds: int = 1800  # every 30 minutes
    keep_last_k: int = 2  # rolling snapshots to keep


class MLPClassifier(nn.Module):
    def __init__(self, hv_dim: int, hidden_dims: list[int]):
        """
        hv_dim: dimension of each HRR vector (e.g., 7744)
        hidden_dims: e.g., [4096, 2048, 512, 128]
        """
        super().__init__()
        d_in = hv_dim * 2
        layers: list[nn.Module] = []
        last = d_in
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.GELU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        # h1,h2: [B, hv_dim]
        x = torch.cat([h1, h2], dim=-1)  # [B, 2*D]
        return self.net(x).squeeze(-1)  # [B]


## -------- Mirror MLP Classifier -------
@dataclass
class MirrorMLPConfig:
    # General
    project_dir: Path | None = None
    seed: int = 42
    epochs: int = 5
    batch_size: int = 256
    train_parents: int = 2000
    valid_parents: int = 500

    hidden_dims: list[int] = (4096, 2048, 512, 128)

    # HDC / encoder
    hv_dim: int = 88 * 88  # 7744
    vsa: VSAModel = VSAModel.HRR
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optim
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Loader
    num_workers: int = 0
    prefetch_factor: int = 1
    pin_memory: bool = False
    micro_bs: int = 64
    hv_scale: float | None = None


class MirrorMLP(nn.Module):
    def __init__(self, hv_dim: int, hidden_dims: list[int], *, use_layernorm: bool = True):
        super().__init__()
        self.use_layernorm = use_layernorm
        in_dim = 4 * hv_dim + 1  # h1,h2,|h1-h2|,h1⊙h2, + cosine scalar
        layers: list[nn.Module] = []
        last = in_dim
        if use_layernorm:
            layers.append(nn.LayerNorm(in_dim))
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.GELU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, h1, h2):
        diff = torch.abs(h1 - h2)
        had = h1 * h2
        cos = torch.nn.functional.cosine_similarity(h1, h2, dim=-1, eps=1e-8).unsqueeze(-1)
        x = torch.cat([h1, h2, diff, had, cos], dim=-1)
        return self.net(x).squeeze(-1)


@dataclass
class TwoTowerWithInteractionsConfig:
    # General
    project_dir: Path
    seed: int = 42
    epochs: int = 5
    batch_size: int = 256
    train_parents: int = 2000
    valid_parents: int = 500

    hidden_dims: list[int] = (4096, 2048, 512, 128)

    # HDC / encoder
    hv_dim: int = 88 * 88  # 7744
    vsa: VSAModel = VSAModel.HRR
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optim
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Loader
    num_workers: int = 0
    prefetch_factor: int = 1
    pin_memory: bool = False
    micro_bs: int = 64
    hv_scale: float | None = None


class TwoTowerWithInteractions(nn.Module):
    """
    Sizes:
    	•	S (9.3M): preproj_dim=1024, tower_dims=(1024, 256, 64), head_dims=(256, 64)
	    •	M (23M): preproj_dim=2048, tower_dims=(2048, 1024, 256), head_dims=(512, 128)
	    •	L (36–37M): no pre-proj, tower_dims=(4096, 1024, 256), head_dims=(512, 128)
    """

    def __init__(
            self,
            hv_dim: int,
            tower_dims: list[int] = (2048, 1024, 256),
            head_dims: list[int] = (512, 128),
            use_layernorm: bool = True,
            preproj_dim: int | None = None,
            freeze_preproj: bool = False,
            untie_towers: bool = False,
    ):
        super().__init__()
        self.use_ln = use_layernorm
        d_in = hv_dim

        # optional pre-projection (helps a lot when hv_dim is big)
        if preproj_dim is not None:
            self.pre = nn.Linear(d_in, preproj_dim, bias=False)
            if freeze_preproj:
                for p in self.pre.parameters():
                    p.requires_grad = False
            d_in = preproj_dim
        else:
            self.pre = None

        # small helper to build an MLP block list
        def mlp(d0, dims):
            layers = []
            if self.use_ln:
                layers.append(nn.LayerNorm(d0))
            last = d0
            for h in dims:
                layers += [nn.Linear(last, h), nn.GELU()]
                last = h
            return nn.Sequential(*layers), last

        # (shared) tower
        self.tower, d_out = mlp(d_in, list(tower_dims))
        if untie_towers:
            self.tower_b, _ = mlp(d_in, list(tower_dims))
        else:
            self.tower_b = self.tower

        # interaction head on [z1, z2, |z1-z2|, z1⊙z2, cos(z1,z2)]
        head_in = 4 * d_out + 1
        head_layers = []
        if self.use_ln:
            head_layers.append(nn.LayerNorm(head_in))
        last = head_in
        for h in head_dims:
            head_layers += [nn.Linear(last, h), nn.GELU()]
            last = h
        head_layers += [nn.Linear(last, 1)]
        self.head = nn.Sequential(*head_layers)

    def encode(self, x):
        if self.pre is not None:
            x = self.pre(x)
        return self.tower(x)

    def forward(self, h1, h2):
        z1 = self.encode(h1)
        # note: if untied_towers=True, use the second tower for h2
        z2 = self.tower_b(self.pre(h2) if self.pre is not None else h2)

        diff = (z1 - z2).abs()
        had = z1 * z2
        cos = torch.nn.functional.cosine_similarity(z1, z2, dim=-1, eps=1e-8).unsqueeze(-1)
        x = torch.cat([z1, z2, diff, had, cos], dim=-1)
        return self.head(x).squeeze(-1)
