import math
from datetime import datetime

import networkx as nx
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.nn.conv import GATv2Conv

# === BEGIN NEW ===
from torchmetrics import AUROC, AveragePrecision

from src.encoding.graph_encoders import AbstractGraphEncoder
from src.exp.classification_v2.classification_utils import Config
from src.utils.registery import ModelType, register_model
from src.utils.utils import DataTransformer


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
        model_type: ModelType,
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

        if "gin" in self.model_type.lower():
            # Prepare condition: [B, D]
            cond = final_h.detach().to(device=device, dtype=dtype)
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)  # [1, D]
            cond = cond.expand(g1_b.num_graphs, -1)  # [B, D]
            g1_b.cond = cond

            out = self.model(g1_b)  # dict
            logits = out["graph_prediction"].squeeze(-1)  # [B]
            return torch.sigmoid(logits)

        if self.model_type == "MLP":
            assert self.encoder is not None, "encoder is required for MLP oracle"
            h1 = self.encoder.forward(g1_b)["graph_embedding"].to(device=device, dtype=dtype)  # [B, D]
            h2 = final_h.detach().to(device=device, dtype=dtype)
            if h2.dim() == 1:
                h2 = h2.unsqueeze(0)
            h2 = h2.expand(h1.size(0), -1)  # [B, D]
            logits = self.model(h1, h2)  # [B] or [B,1]
            logits = logits.squeeze(-1)
            return torch.sigmoid(logits)

        msg = f"Unknown model_type: {self.model_type!r}"
        raise ValueError(msg)


# ---------- tiny logger ----------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


## -------- MLP Classifier -------
# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
# MLP classifier on concatenated (h1, h2) – no normalization, GELU, no dropout
class MLPClassifier(nn.Module):
    def __init__(
        self,
        hv_dim: int = 88 * 88,
        hidden_dims: list[int] | None = None,
        *,
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


# ---------------------------------------------------------------------
# Lightning wrapper
# ---------------------------------------------------------------------
@register_model("MLP")
class LitMLPClassifier(pl.LightningModule):
    """
    Expect batches as (h1, h2, y) or {"h1":..., "h2":..., "y":...}, with h1/h2 ~ [B, D].
    If collate yields [B, 1, D], we squeeze the middle dim.
    """

    def __init__(
        self,
        *,
        hv_dim: int,
        hidden_dims: list[int] | None = None,
        use_layer_norm: bool = False,
        use_batch_norm: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        pos_weight: float | None = None,  # None -> unweighted BCE
    ) -> None:
        super().__init__()
        # Save EVERYTHING for checkpoint load
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = MLPClassifier(
            hv_dim=hv_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
        )

        # Register pos_weight as a buffer so device moves are handled automatically
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
        else:
            self.pos_weight = None  # type: ignore[assignment]

        # Metrics
        self.val_auc = AUROC(task="binary")
        self.val_ap = AveragePrecision(task="binary")

    # --------- helpers ---------
    @staticmethod
    def _fix_shapes(h1: torch.Tensor, h2: torch.Tensor, y: torch.Tensor):
        # squeeze [B,1,D] -> [B,D] emitted by your dataset/loader
        if h1.ndim == 3 and h1.size(1) == 1:
            h1 = h1.squeeze(1)
        if h2.ndim == 3 and h2.size(1) == 1:
            h2 = h2.squeeze(1)
        # y -> [B], float
        if y.ndim > 1:
            y = y.squeeze(-1)
        return h1, h2, y.float()

    # --------- Lightning API ---------
    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        return self.model(h1, h2)

    def training_step(self, batch, batch_idx: int):
        h1, h2, y = batch  # tuple from Dataset
        h1, h2, y = self._fix_shapes(h1, h2, y)
        logits = self(h1, h2)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.size(0), logger=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        h1, h2, y = batch  # keep the same structure as training
        h1, h2, y = self._fix_shapes(h1, h2, y)

        logits = self(h1, h2)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)

        # epoch-level val_loss for EarlyStopping
        sync = getattr(self.trainer, "world_size", 1) > 1
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=y.size(0),
            sync_dist=sync,
        )

        # metrics expect probabilities and int/bool targets
        probs = logits.sigmoid()
        self.val_auc.update(probs, y.int())
        self.val_ap.update(probs, y.int())

    def on_validation_epoch_end(self):
        # make sure these get logged too (optional)
        auc = self.val_auc.compute()
        ap = self.val_ap.compute()
        self.log("val_auc", auc, prog_bar=True)
        self.log("val_ap", ap, prog_bar=True)
        self.val_auc.reset()
        self.val_ap.reset()

    def test_step(self, batch, batch_idx: int):
        # mirror validation for later testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class BiaffineHead(nn.Module):
    def __init__(
        self,
        hv_dim: int,
        proj_dim: int = 1024,
        n_heads: int = 8,
        proj_hidden: int | None = None,
        dropout: float = 0.0,
        tau_init: float = 8.0,  # large τ to tame early logits
        *,
        share_proj: bool = False,
        norm: bool = True,
        use_layernorm: bool = True,
        use_temperature: bool = True,
    ):
        super().__init__()
        self.norm = norm
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.n_heads = n_heads
        P = proj_dim

        def proj_mlp() -> Module:
            if proj_hidden is None:
                return nn.Linear(hv_dim, P, bias=False)
            return nn.Sequential(
                nn.Linear(hv_dim, proj_hidden, bias=True),
                nn.GELU(),
                nn.Linear(proj_hidden, P, bias=False),
            )

        self.p1 = proj_mlp()
        self.p2 = self.p1 if share_proj else proj_mlp()

        self.ln1 = nn.LayerNorm(P) if use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(P) if use_layernorm else nn.Identity()

        self.W = nn.Parameter(torch.empty(n_heads, P, P))
        nn.init.xavier_uniform_(self.W)

        self.gate = nn.Linear(3, n_heads, bias=True)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        self.diag_w = nn.Parameter(torch.zeros(P))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.zeros(()))

        self.use_temperature = use_temperature
        if use_temperature:
            # τ = softplus(log_tau)  with τ≈tau_init at start
            self.log_tau = nn.Parameter(torch.tensor(math.log(math.exp(tau_init) - 1.0)))

        # init: orthogonal for any Linear with bias=False (covers final layers)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is None:
                nn.init.orthogonal_(m.weight)

    def forward(self, h1, h2):
        # accept [B,1,D]
        if h1.ndim == 3 and h1.size(1) == 1:
            h1 = h1.squeeze(1)
        if h2.ndim == 3 and h2.size(1) == 1:
            h2 = h2.squeeze(1)

        if self.norm:
            h1 = F.normalize(h1, dim=-1)
            h2 = F.normalize(h2, dim=-1)

        u = self.ln1(self.p1(h1))
        v = self.ln2(self.p2(h2))
        u = self.dropout(u)
        v = self.dropout(v)

        # biaffine heads
        s_heads = torch.einsum("bp,hpq,bq->bh", u, self.W, v)  # [B,H]

        # tiny feature gate per head
        uv = u * v
        cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1) + 1e-12)
        feat = torch.stack([uv.mean(-1), (u - v).abs().mean(-1), cos], dim=-1)  # [B,3]
        gates = torch.softmax(self.gate(feat), dim=-1)  # [B,H]

        s_biaff = (gates * s_heads).sum(-1)  # [B]
        s_diag = (uv * self.diag_w).sum(-1)  # [B]
        s_dot = self.alpha * (u * v).sum(-1)  # [B]
        logits = s_biaff + s_diag + s_dot + self.bias

        if self.use_temperature:
            tau = F.softplus(self.log_tau) + 1e-6
            logits = logits / tau
        return logits


# ---------------------------------------------------------------------
# Lightning wrapper
# ---------------------------------------------------------------------
@register_model("BAH")
class LitBAHClassifier(pl.LightningModule):
    """
    Expect batches as (h1, h2, y) or {"h1":..., "h2":..., "y":...}, with h1/h2 ~ [B, D].
    If collate yields [B, 1, D], we squeeze the middle dim.
    """

    def __init__(
        self,
        *,
        hv_dim: int,
        proj_dim: int = 1024,  # 1024–1536 are good starting points
        n_heads: int = 8,  # 4–16; more heads => more expressiveness
        norm: bool = True,
        proj_hidden: int | None = None,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        pos_weight: float | None = None,  # None -> unweighted BCE
        share_proj: bool = False,  # or e.g. 2048 for 2-layer projections if you want even more capacity_
        use_layernorm: bool = True,
        use_temperature: bool = True,
    ) -> None:
        super().__init__()
        # Save EVERYTHING for checkpoint load
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = BiaffineHead(
            hv_dim=hv_dim,
            proj_dim=proj_dim,
            n_heads=n_heads,
            share_proj=share_proj,
            norm=norm,
            use_layernorm=use_layernorm,
            proj_hidden=proj_hidden,
            dropout=dropout,
            use_temperature=use_temperature,
        )

        # Register pos_weight as a buffer so device moves are handled automatically
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
        else:
            self.pos_weight = None  # type: ignore[assignment]

        # Metrics
        self.val_auc = AUROC(task="binary")
        self.val_ap = AveragePrecision(task="binary")

    # --------- helpers ---------
    @staticmethod
    def _fix_shapes(h1: torch.Tensor, h2: torch.Tensor, y: torch.Tensor):
        # squeeze [B,1,D] -> [B,D] emitted by your dataset/loader
        if h1.ndim == 3 and h1.size(1) == 1:
            h1 = h1.squeeze(1)
        if h2.ndim == 3 and h2.size(1) == 1:
            h2 = h2.squeeze(1)
        # y -> [B], float
        if y.ndim > 1:
            y = y.squeeze(-1)
        return h1, h2, y.float()

    # --------- Lightning API ---------
    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        return self.model(h1, h2)

    def training_step(self, batch, batch_idx: int):
        h1, h2, y = batch  # tuple from Dataset
        h1, h2, y = self._fix_shapes(h1, h2, y)
        logits = self(h1, h2)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.size(0), logger=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        h1, h2, y = batch  # keep the same structure as training
        h1, h2, y = self._fix_shapes(h1, h2, y)

        logits = self(h1, h2)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)

        # epoch-level val_loss for EarlyStopping
        sync = getattr(self.trainer, "world_size", 1) > 1
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=y.size(0),
            sync_dist=sync,
        )

        # metrics expect probabilities and int/bool targets
        probs = logits.sigmoid()
        self.val_auc.update(probs, y.int())
        self.val_ap.update(probs, y.int())

    def on_validation_epoch_end(self):
        # make sure these get logged too (optional)
        self.log("val_auc", self.val_auc.compute(), prog_bar=True)
        self.log("val_ap", self.val_ap.compute(), prog_bar=True)
        self.val_auc.reset()
        self.val_ap.reset()

    def test_step(self, batch, batch_idx: int):
        h1, h2, y = batch
        h1, h2, y = self._fix_shapes(h1, h2, y)
        logits = self(h1, h2)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        probs = logits.sigmoid()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_auc", self.val_auc(probs, y.int()), prog_bar=True)
        self.log("test_ap", self.val_ap(probs, y.int()), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


## GNN Classifiers


class FilmConditionalLinear(nn.Module):
    """
    This is a conditional variant of the default ``Linear`` layer using the FiLM conditioning mechanism.

    As a conditional layer, this layer requires 2 different input tensors. The first is the actual input
    tensor to be transformed into the output tensor and the second is the condition vector that should
    modify the behavior of the linear layer. The implementation follows the Feature-wise Linear Modulation
    (FiLM) approach, which applies an affine transformation (scale and shift) to the output of a linear
    layer based on the conditioning vector.

    :param in_features: Number of input features
    :param out_features: Number of output features
    :param condition_features: Number of features in the conditioning vector
    :param film_units: List of hidden unit sizes for the FiLM network
    :param film_use_norm: Whether to use batch normalization in the FiLM network
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        condition_features: int,
        film_units: list[int] = [128],
        film_use_norm: bool = True,
        **kwargs,
    ):
        """
        Initialize the FiLM conditional linear layer.

        :param in_features: Number of input features
        :param out_features: Number of output features
        :param condition_features: Number of features in the conditioning vector
        :param film_units: List of hidden unit sizes for the FiLM network
        :param film_use_norm: Whether to use batch normalization in the FiLM network
        :param kwargs: Additional keyword arguments to pass to the parent class
        """
        nn.Module.__init__(self, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.condition_features = condition_features
        self.film_units = film_units
        self.film_use_norm = film_use_norm
        # The final activation we actually want to be Tanh because the output values should
        # be in the range of [-1, 1], both for the bias as well as the multiplicative factor.
        self.lay_final_activation = nn.Tanh()

        ## -- Main Linear Layer --
        # Ultimately, the FiLM layer is just a variation of a linear layer where the output
        # is additionally modified by the activation. So what we define here is the core
        # linear layer itself.
        self.lay_linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
        )
        self.dim = out_features

        ## -- FiLM Layers --
        # These are the layers that will be used to create the FiLM activation modifier tensors.
        # They take as the input the condition vector and transform that into the additive and
        # multiplicative modifiers which than perform the affine transformation on the output
        # of the actual linear layer.
        # This can even be a multi-layer perceptron by itself, depending on how difficult the
        # condition function is to learn.
        self.film_layers = nn.ModuleList()
        prev_features = condition_features
        for num_features in film_units:
            if self.film_use_norm:
                lay = nn.Sequential(
                    nn.Linear(in_features=prev_features, out_features=num_features),
                    nn.BatchNorm1d(num_features),
                    nn.ReLU(),
                )
            else:
                lay = nn.Sequential(
                    nn.Linear(in_features=prev_features, out_features=num_features),
                    nn.ReLU(),
                )

            self.film_layers.append(lay)
            prev_features = num_features

        # Finally, at the end of this MLP we need the final layer to be one that outputs a
        # vector of the size that is twice the size of the output of the core linear layer.
        # From this output we need to derive the additive and the multiplicative modifier
        # and we do this by using the first half of the output as the multiplicative
        # modifier and the second half as the additive modifier.
        self.film_layers.append(
            nn.Linear(
                in_features=prev_features,
                out_features=self.dim * 2,
            )
        )

    def forward(self, input: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FiLM conditional linear layer.

        The forward method applies the core linear transformation to the input tensor,
        then modifies the result based on the condition tensor through a FiLM (Feature-wise
        Linear Modulation) mechanism, which performs an affine transformation with parameters
        derived from the condition.

        :param input: Input tensor of shape (batch_size, in_features)
        :param condition: Condition tensor of shape (batch_size, condition_features)

        :returns: Output tensor of shape (batch_size, out_features)
        """

        ## -- getting the modifier from the condition --
        # We need the film layers to create the activation modifier tensor.
        # This actually may or may not be a multi layer perceptron.
        modifier = condition
        for lay in self.film_layers:
            modifier = lay(modifier)

        modifier = 2 * self.lay_final_activation(modifier)

        # -- getting the output from the linear layer --
        output = self.lay_linear(input)

        # -- applying the modifier to the output --
        # And then finally we split the modifier vector into the two equally sized distinct vectors where one of them
        # is the multiplicative modification and the other is the additive modification to the output activation.
        factor = modifier[:, : self.dim]
        bias = modifier[:, self.dim :]
        output = (factor * output) + bias

        return output


class ConditionalGraphAttention(MessagePassing):
    """
    A conditional graph attention layer that extends PyTorch Geometric's MessagePassing base class.

    This layer implements a message passing mechanism where attention coefficients are computed
    for each edge based on the features of the connected nodes and edge attributes, modified by
    a condition vector. The attention mechanism helps the network focus on the most relevant
    parts of the graph structure for the given task and condition.

    :param in_dim: Dimension of input node features
    :param out_dim: Dimension of output node features
    :param edge_dim: Dimension of edge features
    :param cond_dim: Dimension of the condition vector
    :param hidden_dim: Dimension of hidden layers
    :param eps: Epsilon value for residual connections
    :param film_units: List of hidden unit sizes for the FiLM networks
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        cond_dim: int,
        hidden_dim: int = 64,
        eps: float = 0.1,
        film_units: list[int] = [],
        **kwargs,
    ):
        """
        Initialize the conditional graph attention layer.

        :param in_dim: Dimension of input node features
        :param out_dim: Dimension of output node features
        :param edge_dim: Dimension of edge features
        :param cond_dim: Dimension of the condition vector
        :param hidden_dim: Dimension of hidden layers
        :param eps: Epsilon value for residual connections
        :param film_units: List of hidden unit sizes for the FiLM networks
        :param kwargs: Additional keyword arguments to pass to the parent class
        """
        kwargs.setdefault("aggr", "add")
        MessagePassing.__init__(self, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.film_units = film_units

        self._attention_logits = None
        self._attention = None

        ## -- Initial Embedding Layer --
        self.message_dim = in_dim * 2 + edge_dim
        self.lay_message_lin_1 = FilmConditionalLinear(
            in_features=self.message_dim,
            out_features=self.hidden_dim,
            condition_features=self.cond_dim,
            film_units=self.film_units,
        )
        self.lay_message_bn = nn.BatchNorm1d(self.hidden_dim)
        self.lay_message_act = nn.LeakyReLU()
        self.lay_message_lin_2 = FilmConditionalLinear(
            in_features=self.hidden_dim,
            out_features=self.hidden_dim,
            condition_features=self.cond_dim,
            film_units=self.film_units,
        )

        # -- Attention Layer --
        # This layer will produce the attention coefficients which will then be used in the
        # attention-weighted message accumulation step.
        self.lay_attention_lin_1 = FilmConditionalLinear(
            in_features=self.message_dim,
            out_features=self.hidden_dim,
            condition_features=self.cond_dim,
            film_units=self.film_units,
        )
        self.lay_attention_bn = nn.BatchNorm1d(self.hidden_dim)
        self.lay_attention_act = nn.LeakyReLU()
        self.lay_attention_lin_2 = FilmConditionalLinear(
            in_features=self.hidden_dim,
            out_features=1,  # attention logits
            condition_features=self.cond_dim,
            film_units=self.film_units,
        )

        # -- Final Transform Layer --
        # In the end we add an additional transformation on the attention weighted aggregation
        # of the message to determine the update to the node features.
        self.lay_transform_lin_1 = FilmConditionalLinear(
            in_features=self.hidden_dim + self.in_dim,
            out_features=self.hidden_dim,
            condition_features=self.cond_dim,
            film_units=self.film_units,
        )
        self.lay_transform_bn = nn.BatchNorm1d(self.hidden_dim)
        self.lay_transform_act = nn.LeakyReLU()
        self.lay_transform_lin_2 = FilmConditionalLinear(
            in_features=self.hidden_dim,
            out_features=self.out_dim,
            condition_features=self.cond_dim,
            film_units=self.film_units,
        )

    def message(
        self,
        x_i,
        x_j,
        condition_i,
        condition_j,
        edge_attr,
        edge_weights,
    ) -> torch.Tensor:
        """
        Compute the message for each edge in the message passing step.

        This method is called for each edge during the propagation step of message passing.
        It computes attention coefficients based on the features of connected nodes and the edge,
        then uses these coefficients to weight the message being passed.

        :param x_i: Features of the target node
        :param x_j: Features of the source node
        :param condition_i: Condition vector for the target node
        :param condition_j: Condition vector for the source node
        :param edge_attr: Edge attributes
        :param edge_weights: Optional edge weights to further modulate the messages

        :returns: The weighted message to be passed along the edge
        """

        message = torch.cat([x_i, x_j, edge_attr], dim=-1)

        attention_logits = self.lay_attention_lin_1(message, condition_i)
        attention_logits = self.lay_attention_bn(attention_logits)
        attention_logits = self.lay_attention_act(attention_logits)
        attention_logits = self.lay_attention_lin_2(attention_logits, condition_i)
        self._attention_logits = attention_logits
        self._attention = torch.sigmoid(self._attention_logits)

        message_transformed = self.lay_message_lin_1(message, condition_i)
        message_transformed = self.lay_message_bn(message_transformed)
        message_transformed = self.lay_message_act(message_transformed)
        message_transformed = self.lay_message_lin_2(message_transformed, condition_i)

        result = self._attention * message_transformed

        if edge_weights is not None:
            if edge_weights.dim() == 1:
                edge_weights = torch.unsqueeze(edge_weights, dim=-1)

            result *= edge_weights

        return result

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the conditional graph attention layer.

        This method implements the full message passing operation, including propagation of messages
        along edges and aggregation of these messages at each node. The final node embeddings are
        computed by transforming the aggregated messages together with the original node features.

        :param x: Input node features
        :param condition: Condition vector for all nodes
        :param edge_attr: Edge attributes
        :param edge_index: Graph connectivity
        :param edge_weights: Optional edge weights
        :param kwargs: Additional keyword arguments

        :returns: A tuple containing the updated node embeddings and attention logits
        """

        self._attention = None
        self._attention_logits = None

        # node_embedding: (B * V, out)
        node_embedding = self.propagate(
            edge_index,
            x=x,
            condition=condition,
            edge_attr=edge_attr,
            edge_weights=edge_weights,
        )

        # node_embedding = self.lay_act(node_embedding)
        x = self.lay_transform_lin_1(torch.cat([node_embedding, x], dim=1), condition)
        x = self.lay_transform_bn(x)
        x = self.lay_transform_act(x)
        x = self.lay_transform_lin_2(x, condition)

        # Residual connection to make the gradient flow more stable.
        # node_embedding += self.eps * x
        node_embedding = x

        return node_embedding, self._attention_logits


@register_model("GIN-F")
class ConditionalGIN(pl.LightningModule):
    """
    A conditional Graph Isomorphism Network (GIN) implemented using PyTorch Lightning.

    This model performs message passing on graph structured data conditioned on an external
    vector. It uses the conditional graph attention mechanism to propagate information through
    the graph. The model is designed for graph binary classification tasks, predicting a binary
    label for the entire graph based on the learned node representations and the condition vector.

    :param input_dim: Dimension of input node features
    :param edge_dim: Dimension of edge features
    :param condition_dim: Dimension of the condition vector
    :param cond_units: List of hidden unit sizes for the condition embedding network
    :param conv_units: List of hidden unit sizes for the graph convolution layers
    :param film_units: List of hidden unit sizes for the FiLM networks in the graph attention layers
    :param pred_units: List of hidden unit sizes for the graph prediction network
    :param learning_rate: Learning rate for the optimizer
    """

    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        condition_dim: int,
        cond_units: list[int] = [256, 128],
        conv_units: list[int] = [64, 64, 64],
        film_units: list[int] = [128],
        pred_units: list[int] = [256, 64, 1],
        learning_rate: float = 0.0001,
        cfg: Config | None = None,
    ):
        """
        Initialize the conditional GIN model.

        :param input_dim: Dimension of input node features
        :param edge_dim: Dimension of edge features
        :param condition_dim: Dimension of the condition vector
        :param cond_units: List of hidden unit sizes for the condition embedding network
        :param conv_units: List of hidden unit sizes for the graph convolution layers
        :param film_units: List of hidden unit sizes for the FiLM networks in the graph attention layers
        :param pred_units: List of hidden unit sizes for the graph prediction network
        :param learning_rate: Learning rate for the optimizer
        """

        super().__init__()

        self.save_hyperparameters(
            "input_dim",
            "edge_dim",
            "condition_dim",
            "cond_units",
            "conv_units",
            "film_units",
            "pred_units",
            "learning_rate",
        )

        self.cfg = cfg or Config()
        num = float(self.cfg.n_per_parent) if self.cfg.n_per_parent else 0.0
        den = float(self.cfg.p_per_parent) if self.cfg.p_per_parent else 1.0
        ratio = num / max(1.0, den)
        self.register_buffer("pos_weight", torch.tensor([ratio], dtype=torch.float32))

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.conv_units = conv_units
        self.learning_rate = learning_rate

        ## == LAYER DEFINITIONS ==

        ## -- Condition Layers --

        # These will be the layers (the mlp) which will be used to create an overall lower-dimensional
        # embedding representation of the (very high-dimensional) condition vector. It is then this
        # embedding that will be used in the individual FiLM conditioning layers.
        self.cond_layers = nn.ModuleList()
        prev_units = condition_dim
        for units in cond_units:
            self.cond_layers.append(
                nn.Linear(prev_units, units),
            )
            prev_units = units

        self.cond_embedding_dim = prev_units

        ## -- Graph Convolutional Layers --

        # These will be the actual convolutional layers that will be used as the message passing
        # operations on the given graph.
        self.conv_layers = nn.ModuleList()
        prev_units = input_dim
        for units in conv_units:
            lay = ConditionalGraphAttention(
                in_dim=prev_units,
                out_dim=units,
                edge_dim=edge_dim,
                cond_dim=self.cond_embedding_dim,
                film_units=film_units,
            )
            self.conv_layers.append(lay)
            prev_units = units

        # --- Binary Classifier ---

        # Finally, after the message passing and so on, we firstly need to reduce the node
        # representations of each individual graph object into a single graph vector and then
        # perform a binary classification based on that graph vector.

        # Aggregates the node representations into
        self.lay_pool = SumAggregation()

        # A multi layer perceptron made up of linear layers with batch norm and
        # relu activation up until the very last layer transition, which outputs the
        # single classification logit.
        self.pred_units = pred_units
        self.pred_layers = nn.ModuleList()
        for units in pred_units[:-1]:
            lay = nn.Sequential(
                nn.Linear(
                    in_features=prev_units,
                    out_features=units,
                ),
                nn.BatchNorm1d(units),
                nn.ReLU(),
            )
            self.pred_layers.append(lay)
            prev_units = units

        # Final layer outputs single logit for binary classification
        lay = nn.Linear(
            in_features=prev_units,
            out_features=pred_units[-1],
        )
        self.pred_layers.append(lay)

    def forward(self, data: Data):
        """
        Forward pass of the conditional GIN model.

        This method processes the input graph data through the condition embedding network,
        the graph convolutional layers, and finally the graph prediction network to predict
        a binary classification label for the entire graph.

        :param data: PyTorch Geometric Data object containing the graph

        :returns: Dictionary containing the graph prediction logit
        """

        # -- Embedding the condition --
        # 1) cond_graph: [num_graphs_in_batch, condition_dim]
        cond_graph: torch.Tensor = data.cond  # e.g. shape [B, 7744]
        for lay in self.cond_layers:
            cond_graph = lay(cond_graph)  # -> [B, cond_emb_dim]

        # 2) Broadcast to nodes via batch vector
        cond_nodes = cond_graph[data.batch]  # -> [N_total_nodes, cond_emb_dim]

        # -- Message passing
        # 3) Use node-wise condition in conv layers
        node_embedding = data.x
        for lay_conv in self.conv_layers:
            node_embedding, _ = lay_conv(
                x=node_embedding,
                condition=cond_nodes,  # <= crucial
                edge_attr=data.edge_attr,
                edge_index=data.edge_index,
                edge_weights=data.edge_weights,
            )

        graph_embedding = self.lay_pool(node_embedding, data.batch)

        output = graph_embedding
        for lay in self.pred_layers:
            output = lay(output)

        return {
            "graph_prediction": output,
        }

    def training_step(self, batch: Data, batch_idx):
        """
        Perform a single training step.

        This method is called by PyTorch Lightning during training. It computes the forward pass
        and calculates the binary cross-entropy loss between the predicted graph probabilities
        and the target graph labels.

        :param batch: PyTorch Geometric Data object containing a batch of graphs
        :param batch_idx: Index of the current batch

        :returns: Loss value for the current training step
        """
        logits = self(batch)["graph_prediction"].squeeze(-1).float()  # [B]
        target = batch.y.float()  # [B]
        loss = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight.to(logits.dtype), reduction="mean"
        )
        batch_size = int(getattr(batch, "num_graphs", batch.y.size(0)))
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)["graph_prediction"].squeeze(-1).float()  # [B]
        target = batch.y.float()  # [B]
        val_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        batch_size = int(getattr(batch, "num_graphs", batch.y.size(0)))
        self.log("val_loss", val_loss, prog_bar=True, logger=True, batch_size=batch_size, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        """
        Configure optimizers for the model.

        This method is called by PyTorch Lightning to set up the optimizer
        for training the model.

        :returns: The configured optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


@register_model("GIN-C")
class ConditionalGINConcat(pl.LightningModule):
    """
    A conditional Graph Isomorphism Network (GIN) implemented using PyTorch Lightning.

    This model performs message passing on graph structured data conditioned on an external
    vector. It uses the conditional graph attention mechanism to propagate information through
    the graph. The model is designed for graph binary classification tasks, predicting a binary
    label for the entire graph based on the learned node representations and the condition vector.

    :param input_dim: Dimension of input node features
    :param edge_dim: Dimension of edge features
    :param condition_dim: Dimension of the condition vector
    :param cond_units: List of hidden unit sizes for the condition embedding network
    :param conv_units: List of hidden unit sizes for the graph convolution layers
    :param film_units: List of hidden unit sizes for the FiLM networks in the graph attention layers
    :param pred_units: List of hidden unit sizes for the graph prediction network
    :param learning_rate: Learning rate for the optimizer
    """

    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        condition_dim: int,
        cond_units: list[int] = [256, 64],
        conv_units: list[int] = [128, 128, 128],
        pred_units: list[int] = [256, 64],
        film_units: list[int] | None = None,
        num_heads: int = 10,
        learning_rate: float = 0.0001,
        cfg: "Config" = None,
    ):
        """
        Initialize the conditional GIN model.

        :param input_dim: Dimension of input node features
        :param edge_dim: Dimension of edge features
        :param condition_dim: Dimension of the condition vector
        :param cond_units: List of hidden unit sizes for the condition embedding network
        :param conv_units: List of hidden unit sizes for the graph convolution layers
        :param film_units: List of hidden unit sizes for the FiLM networks in the graph attention layers
        :param pred_units: List of hidden unit sizes for the graph prediction network
        :param learning_rate: Learning rate for the optimizer
        """

        super().__init__()

        self.cfg = cfg
        num = float(self.cfg.n_per_parent) if self.cfg.n_per_parent else 0.0
        den = float(self.cfg.p_per_parent) if self.cfg.p_per_parent else 1.0
        ratio = num / max(1.0, den)
        self.register_buffer("pos_weight", torch.tensor([ratio], dtype=torch.float32))

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.conv_units = conv_units
        self.learning_rate = learning_rate

        ## == LAYER DEFINITIONS ==

        ## -- Condition Layers --

        # These will be the layers (the mlp) which will be used to create an overall lower-dimensional
        # embedding representation of the (very high-dimensional) condition vector. It is then this
        # embedding that will be used in the individual FiLM conditioning layers.
        self.cond_layers = nn.ModuleList()
        prev_units = condition_dim
        for units in cond_units:
            self.cond_layers.append(
                nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                    nn.ReLU(),
                )
            )
            prev_units = units

        self.cond_embedding_dim = prev_units

        ## -- Graph Convolutional Layers --

        # These will be the actual convolutional layers that will be used as the message passing
        # operations on the given graph.
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        prev_units = input_dim
        for units in conv_units:
            lay = GATv2Conv(
                in_channels=prev_units + self.cond_embedding_dim,
                out_channels=units,
                edge_dim=edge_dim,
                heads=num_heads,
                concat=False,
                add_self_loops=True,
                dropout=0.0,
            )
            self.conv_layers.append(lay)

            self.bn_layers.append(nn.BatchNorm1d(units))

            prev_units = units

        # --- Binary Classifier ---

        # Finally, after the message passing and so on, we firstly need to reduce the node
        # representations of each individual graph object into a single graph vector and then
        # perform a binary classification based on that graph vector.

        # Aggregates the node representations into
        self.lay_pool = SumAggregation()

        # A multi layer perceptron made up of linear layers with batch norm and
        # relu activation up until the very last layer transition, which outputs the
        # single classification logit.
        self.pred_units = pred_units
        self.pred_layers = nn.ModuleList()
        for units in pred_units:
            lay = nn.Sequential(
                nn.Linear(
                    in_features=prev_units,
                    out_features=units,
                ),
                nn.BatchNorm1d(units),
                nn.ReLU(),
            )
            self.pred_layers.append(lay)
            prev_units = units

        # Final layer outputs single logit for binary classification
        lay = nn.Linear(
            in_features=prev_units,
            out_features=1,
        )
        self.pred_layers.append(lay)

    def forward(self, data: Data):
        """
        Forward pass of the conditional GIN model.

        This method processes the input graph data through the condition embedding network,
        the graph convolutional layers, and finally the graph prediction network to predict
        a binary classification label for the entire graph.

        :param data: PyTorch Geometric Data object containing the graph

        :returns: Dictionary containing the graph prediction logit
        """

        # -- Embedding the condition --
        # 1) cond_graph: [num_graphs_in_batch, condition_dim]
        cond_graph: torch.Tensor = data.cond  # e.g. shape [B, 7744]
        for lay in self.cond_layers:
            cond_graph = lay(cond_graph)  # -> [B, cond_emb_dim]

        # 2) Broadcast to nodes via batch vector
        cond_nodes = cond_graph[data.batch]  # -> [N_total_nodes, cond_emb_dim]

        # -- Message passing
        # 3) Use node-wise condition in conv layers
        node_embedding = data.x
        for lay_conv in self.conv_layers:
            node_embedding = lay_conv(
                x=torch.cat([node_embedding, cond_nodes], dim=-1),
                edge_attr=data.edge_attr,
                edge_index=data.edge_index,
            )
            node_embedding = F.leaky_relu(node_embedding)

        graph_embedding = self.lay_pool(node_embedding, data.batch)

        output = graph_embedding
        for lay in self.pred_layers:
            output = lay(output)

        return {
            "graph_prediction": output,
        }

    def training_step(self, batch: Data, batch_idx):
        """
        Perform a single training step.

        This method is called by PyTorch Lightning during training. It computes the forward pass
        and calculates the binary cross-entropy loss between the predicted graph probabilities
        and the target graph labels.

        :param batch: PyTorch Geometric Data object containing a batch of graphs
        :param batch_idx: Index of the current batch

        :returns: Loss value for the current training step
        """
        logits = self(batch)["graph_prediction"].squeeze(-1).float()  # [B]
        target = batch.y.float()  # [B]
        loss = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight.to(logits.dtype), reduction="mean"
        )
        batch_size = int(getattr(batch, "num_graphs", batch.y.size(0)))
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)["graph_prediction"].squeeze(-1).float()  # [B]
        target = batch.y.float()  # [B]
        val_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        batch_size = int(getattr(batch, "num_graphs", batch.y.size(0)))
        self.log("val_loss", val_loss, prog_bar=True, logger=True, batch_size=batch_size, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        """
        Configure optimizers for the model.

        This method is called by PyTorch Lightning to set up the optimizer
        for training the model.

        :returns: The configured optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
