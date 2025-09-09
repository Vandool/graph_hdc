
from datetime import datetime
from typing import Literal

import networkx as nx
import pytorch_lightning as pl
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from src.encoding.graph_encoders import AbstractGraphEncoder
from src.utils.registery import register_model
from src.utils.utils import DataTransformer

ModelKind = Literal["gin", "mlp"]

class Oracle:
    """
    Thin wrapper that adapts inputs to the underlying model.

    - For ConditionalGIN:
        small_gs (list[nx.Graph]) -> PyG Batch with neutral edge attrs/weights
        cond := repeat(final_h, B) attached as data.cond
        logits := model(batch)["graph_prediction"].squeeze(-1)

    - For MLP:
        h1 := encoder(Batch(small_gs))["graph_embedding"]
        h2 := expand(final_h, B, -1)
        logits := model(h1, h2)
    """

    def __init__(
        self,
        model: nn.Module | pl.LightningModule,
        model_type: ModelKind,
        encoder: AbstractGraphEncoder | None = None,
    ):
        self.model = model.eval()
        self.encoder = encoder
        self.model_type = model_type

    @staticmethod
    def _ensure_graph_fields(g: Data) -> Data:
        E = g.edge_index.size(1)
        if getattr(g, "edge_attr", None) is None:
            g.edge_attr = torch.ones(E, 1, dtype=torch.float32)
        if getattr(g, "edge_weights", None) is None:
            g.edge_weights = torch.ones(E, dtype=torch.float32)
        return g

    @torch.inference_mode()
    def is_induced_graph(self, small_gs: list[nx.Graph], final_h: torch.Tensor) -> torch.Tensor:
        """
        Returns probabilities (sigmoid of logits) with shape [B].
        `final_h` is the hypervector (condition) of the *big* graph.
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        # Build PyG batch of candidate graphs (g1)
        pyg_list = [self._ensure_graph_fields(DataTransformer.nx_to_pyg(g)) for g in small_gs]
        g1_b = Batch.from_data_list(pyg_list).to(device)

        if self.model_type == "gin":
            # Prepare condition: [B, D]
            cond = final_h.detach().to(device=device, dtype=dtype)
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)  # [1, D]
            cond = cond.expand(g1_b.num_graphs, -1)  # [B, D]
            g1_b.cond = cond

            out = self.model(g1_b)  # dict
            logits = out["graph_prediction"].squeeze(-1)  # [B]
            return torch.sigmoid(logits)

        if self.model_type == "mlp":
            assert self.encoder is not None, "encoder is required for MLP oracle"
            h1 = self.encoder.forward(g1_b)["graph_embedding"].to(device=device, dtype=dtype)  # [B, D]
            h2 = final_h.detach().to(device=device, dtype=dtype)
            if h2.dim() == 1:
                h2 = h2.unsqueeze(0)
            h2 = h2.expand(h1.size(0), -1)  # [B, D]
            logits = self.model(h1, h2)  # [B] or [B,1]
            logits = logits.squeeze(-1)
            return torch.sigmoid(logits)

        raise ValueError(f"Unknown model_type: {self.model_type!r}")


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
