from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import networkx as nx
import torch
from torch import nn
from torch_geometric.data import Batch

from src.encoding.the_types import VSAModel
from src.utils.registery import register_model
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
            # print("cosine similarities:", (cosines > 0).tolist())

            logits = self.model(h1, h2)
            return torch.sigmoid(logits)


# ---------- tiny logger ----------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


## -------- MLP Classifier -------
@register_model("MLPClassifier")
class MLPClassifier(nn.Module):
    def __init__(
            self,
            hv_dim: int = 88 * 88,
            hidden_dims: list[int] | None = None,
            use_layer_norm: bool = False,
            use_batch_norm: bool = False,
    ) -> None:
        """
        hv_dim: dimension of each HRR vector (e.g., 7744)
        hidden_dims: e.g., [4096, 2048, 512, 128]
        """
        super().__init__()
        hidden_dims = hidden_dims or [2048, 1024, 512, 128]
        d_in = hv_dim * 2
        layers: list[nn.Module] = []
        log(f"Using Layer Normalization: {use_layer_norm}\nUsing Batch Normalization: {use_batch_norm}")
        if use_layer_norm:
            layers.append(nn.LayerNorm(d_in))
        last = d_in
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        # h1,h2: [B, hv_dim]
        x = torch.cat([h1, h2], dim=-1)  # [B, 2*D]
        return self.net(x).squeeze(-1)  # [B]

