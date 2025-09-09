import argparse
import contextlib
import itertools
import json
import os
import random
import shutil
import string
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

# === BEGIN NEW ===
from torch_geometric.nn.aggr import SumAggregation
from torchhd import HRRTensor

from src.datasets.zinc_pairs_v2 import ZincPairsV2
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import ZINC_SMILES_HRR_7744_CONFIG, Features
from src.encoding.decoder import greedy_oracle_decoder
from src.encoding.graph_encoders import AbstractGraphEncoder, load_or_create_hypernet
from src.encoding.oracles import Oracle
from src.encoding.the_types import VSAModel
from src.exp.classification_v3_gnn.classification_utils_gnn import (
    exact_representative_validation_indices,
    stratified_per_parent_indices_with_caps,
)
from src.utils.utils import GLOBAL_MODEL_PATH, DataTransformer, pick_device, str2bool

with contextlib.suppress(RuntimeError):
    mp.set_sharing_strategy("file_system")


# ---------------------------------------------------------------------
# Utils & Config
# ---------------------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def pick_precision():
    # Works on A100/H100 if BF16 is supported by the PyTorch/CUDA build.
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed"  # safest + fast on H100/A100
        return "16-mixed"  # widely supported fallback
    return 32  # CPU or MPS


@dataclass
class Config:
    # General
    project_dir: Path | None = None
    exp_dir_name: str | None = None
    seed: int = 42
    epochs: int = 5
    batch_size: int = 256
    is_dev: bool = False

    # Model (shared knobs)
    cond_units: list[int] = field(default_factory=lambda: [256, 128])
    cond_emb_dim: int = 128
    film_units: list[int] = field(default_factory=lambda: [128])
    conv_units: list[int] = field(default_factory=lambda: [64, 64, 64])
    pred_head_units: list[int] = field(default_factory=lambda: [256, 64, 1])

    # Evals
    oracle_num_evals: int = 1
    oracle_beam_size: int = 8

    # HDC / encoder
    hv_dim: int = 88 * 88  # 7744
    vsa: VSAModel = VSAModel.HRR

    # Optim
    lr: float = 1e-4
    weight_decay: float = 0.0

    # Loader
    num_workers: int = 0
    prefetch_factor: int = 1
    pin_memory: bool = False

    # Checkpointing
    continue_from: Path | None = None
    resume_retrain_last_epoch: bool = False

    # Stratification
    stratify: bool = True
    p_per_parent: int = 20
    n_per_parent: int = 20
    exclude_negs: list[int] = field(default_factory=list)
    resample_training_data_on_batch: bool = False


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def setup_exp(dir_name: str | None = None) -> dict:
    script_path = Path(__file__).resolve()
    experiments_path = script_path.parent
    script_stem = script_path.stem

    base_dir = experiments_path / "results" / script_stem
    base_dir.mkdir(parents=True, exist_ok=True)

    log(f"Setting up experiment in {base_dir}")
    if dir_name:
        exp_dir = base_dir / dir_name
    else:
        slug = (
            f"{datetime.now(UTC).strftime('%Y-%m-%d_%H-%M-%S')}_{''.join(random.choices(string.ascii_lowercase, k=4))}"
        )
        exp_dir = base_dir / slug
    exp_dir.mkdir(parents=True, exist_ok=True)
    log(f"Experiment directory created: {exp_dir}")

    dirs = {
        "exp_dir": exp_dir,
        "models_dir": exp_dir / "models",
        "evals_dir": exp_dir / "evaluations",
        "artefacts_dir": exp_dir / "artefacts",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(script_path, exp_dir / script_path.name)
        log(f"Saved a copy of the script to {exp_dir / script_path.name}")
    except Exception as e:
        log(f"Warning: Failed to save script copy: {e}")

    return dirs


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
"""
Conditional Graph Neural Network Implementation for Graph Binary Classification

This module implements a conditional graph neural network architecture based on the message passing
paradigm, specifically designed for graph binary classification tasks. The implementation uses PyTorch
Geometric and PyTorch Lightning frameworks.

The key components of this implementation are:

1. FilmConditionalLinear: A conditional linear layer using Feature-wise Linear Modulation (FiLM)
   that allows neural network behavior to be conditioned on external inputs.

2. ConditionalGraphAttention: A graph attention layer that extends PyTorch Geometric's MessagePassing
   class, incorporating attention mechanisms and conditional processing via FiLM.

3. ConditionalGIN: A Graph Isomorphism Network that uses the conditional graph attention mechanism
   for message passing and is trained to perform binary classification on entire graphs.

The module demonstrates how to:
- Create conditional neural network layers with FiLM
- Implement custom message passing mechanisms with attention
- Apply graph neural networks to graph binary classification tasks
- Generate and visualize mock graph data for testing
"""


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

        self.cfg = cfg or Config()
        num = float(self.cfg.n_per_parent) if self.cfg.n_per_parent else 0.0
        den = float(self.cfg.p_per_parent) if self.cfg.p_per_parent else 1.0
        ratio = (num / max(1.0, den)) if self.cfg.stratify else (37483079 / 8266188)
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
            loss.detach().as_subclass(torch.Tensor),
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
        self.log("val_loss", val_loss.detach().as_subclass(torch.Tensor), prog_bar=True, logger=True, batch_size=batch_size)
        return val_loss

    def configure_optimizers(self):
        """
        Configure optimizers for the model.

        This method is called by PyTorch Lightning to set up the optimizer
        for training the model.

        :returns: The configured optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# ---------------------------------------------------------------------
# Dataset and loaders
# ---------------------------------------------------------------------
class PairsGraphsEncodedDataset(Dataset):
    """
    Returns a single PyG Data per pair:
      data.x, data.edge_index, data.edge_attr(=1), data.edge_weights(=1),
      data.cond (encoded g2 -> [D]),
      data.y (float), data.parent_idx (long)
    Encoding is done on-the-fly with the provided encoder (no grads).
    """

    def __init__(
        self,
        pairs_ds: ZincPairsV2,
        *,
        encoder: AbstractGraphEncoder,
        device: torch.device,
        add_edge_attr: bool = True,
        add_edge_weights: bool = True,
    ):
        self.ds = pairs_ds
        self.encoder = encoder.eval()
        self.device = device
        self.add_edge_attr = add_edge_attr
        self.add_edge_weights = add_edge_weights
        for p in self.encoder.parameters():
            p.requires_grad = False

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def _ensure_graph_fields(g: Data, *, add_edge_attr: bool, add_edge_weights: bool) -> Data:
        E = g.edge_index.size(1)
        if add_edge_attr and getattr(g, "edge_attr", None) is None:
            g.edge_attr = torch.ones(E, 1, dtype=torch.float32)
        if add_edge_weights and getattr(g, "edge_weights", None) is None:
            g.edge_weights = torch.ones(E, dtype=torch.float32)
        return g

    @torch.no_grad()
    def __getitem__(self, idx):
        item = self.ds[idx]

        # g1 (candidate subgraph)
        g1 = Data(x=item.x1, edge_index=item.edge_index1)
        g1 = self._ensure_graph_fields(g1, add_edge_attr=self.add_edge_attr, add_edge_weights=self.add_edge_weights)

        # g2 (condition) -> encode to cond
        g2 = Data(x=item.x2, edge_index=item.edge_index2)

        # Encode a single graph safely
        batch_g2 = Batch.from_data_list([g2]).to(self.device)
        h2 = self.encoder.forward(batch_g2)["graph_embedding"]  # [1, D] on device
        cond = h2.detach().cpu()  # let PL move the whole Batch later

        # target/meta
        y = float(item.y.view(-1)[0].item())
        parent_idx = int(item.parent_idx.view(-1)[0].item()) if hasattr(item, "parent_idx") else -1

        # Attach fields to g1
        g1.cond = cond
        g1.y = torch.tensor(y, dtype=torch.float32)
        g1.parent_idx = torch.tensor(parent_idx, dtype=torch.long)
        return g1


# ---------------- DataModule with per-epoch resampling ----------------
class PairsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: Config,
        *,
        encoder: AbstractGraphEncoder,
        device: torch.device,
        is_dev: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.device = device
        self.is_dev = is_dev

        # set in setup()
        self.train_full = None
        self.valid_full = None
        self.valid_loader = None

    def setup(self, stage=None):
        log("Loading pair datasets …")
        self.train_full = ZincPairsV2(split="train", base_dataset=ZincSmiles(split="train"), dev=self.is_dev)
        self.valid_full = ZincPairsV2(split="valid", base_dataset=ZincSmiles(split="valid"), dev=self.is_dev)
        log(f"Pairs loaded. train_pairs={len(self.train_full)} valid_pairs={len(self.valid_full)}")

        # Precompute validation indices (fixed selection); loaders are built in *_dataloader()
        self._valid_indices = None
        if self.cfg.stratify:
            self._valid_indices = exact_representative_validation_indices(
                ds=self.valid_full,
                target_total=2_000_000 if not self.is_dev else 500,
                exclude_neg_types=self.cfg.exclude_negs,
                by_neg_type=True,
                seed=self.cfg.seed,
            )


    def train_dataloader(self):
        train_base = self.train_full
        if self.cfg.stratify:
            sampling_seed = random.randint(self.cfg.seed + 1, 10_000_000)
            train_indices = stratified_per_parent_indices_with_caps(
                ds=self.train_full,
                pos_per_parent=self.cfg.p_per_parent,
                neg_per_parent=self.cfg.n_per_parent,
                exclude_neg_types=set(self.cfg.exclude_negs),
                seed=sampling_seed,
            )
            train_base = torch.utils.data.Subset(self.train_full, train_indices)


        train_ds = PairsGraphsEncodedDataset(
            train_base, encoder=self.encoder, device=self.device, add_edge_attr=True, add_edge_weights=True
        )
        return DataLoader(  # this is torch_geometric.loader.DataLoader
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(torch.cuda.is_available() and self.cfg.pin_memory),
        )

    def val_dataloader(self):
        valid_base = (
            torch.utils.data.Subset(self.valid_full, self._valid_indices)
            if self._valid_indices is not None
            else self.valid_full
        )
        valid_ds = PairsGraphsEncodedDataset(
            valid_base, encoder=self.encoder, device=self.device, add_edge_attr=True, add_edge_weights=True
        )
        return DataLoader(
            valid_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(torch.cuda.is_available() and self.cfg.pin_memory),
        )


# ---------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------


def _sanitize_for_parquet(d: dict) -> dict:
    """Make dict Arrow-friendly (Path/Enum/etc → str, tensors → int/float)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, VSAModel):
            out[k] = v.value
        elif torch.is_tensor(v):
            out[k] = v.item() if v.numel() == 1 else v.detach().cpu().tolist()
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_as_oracle(
    model, encoder, oracle_num_evals: int = 8, oracle_beam_size: int = 8, oracle_threshold: float = 0.5
):
    log(f"Evaluation classifier as oracle for {oracle_num_evals} examples @threshold:{oracle_threshold}...")

    # Helpers
    # Real Oracle
    def is_final_graph(G_small: nx.Graph, G_big: nx.Graph) -> bool:
        """NetworkX VF2: is `G_small` an induced, label-preserving subgraph of `G_big`?"""
        if (
            G_small.number_of_nodes() == G_big.number_of_nodes()
            and G_small.number_of_edges() == G_big.number_of_edges()
        ):
            nm = lambda a, b: a["feat"] == b["feat"]
            GM = nx.algorithms.isomorphism.GraphMatcher(G_big, G_small, node_match=nm)
            return GM.subgraph_is_isomorphic()
        return False

    model.eval()
    encoder.eval()
    ys = []

    zinc_smiles = ZincSmiles(split="valid")[:oracle_num_evals]
    dataloader = DataLoader(dataset=zinc_smiles, batch_size=oracle_num_evals, shuffle=False)
    batch = next(iter(dataloader))

    # Encode the whole graph in one HV
    graph_term = encoder.forward(batch)["graph_embedding"]
    graph_terms_hd = graph_term.as_subclass(HRRTensor)

    # Create Oracle
    oracle = Oracle(model=model, encoder=encoder, model_type="gin")

    ground_truth_counters = {}
    datas = batch.to_data_list()
    for i in range(oracle_num_evals):
        full_graph_nx = DataTransformer.pyg_to_nx(data=datas[i])
        node_multiset = DataTransformer.get_node_counter_from_batch(batch=i, data=batch)

        nx_GS: list[nx.Graph] = greedy_oracle_decoder(
            node_multiset=node_multiset,
            oracle=oracle,
            full_g_h=graph_terms_hd[i],
            beam_size=oracle_beam_size,
            oracle_threshold=oracle_threshold,
            strict=True,
        )
        nx_GS = list(filter(None, nx_GS))
        if len(nx_GS) == 0:
            ys.append(0)
            continue
        ps = []
        for j, g in enumerate(nx_GS):
            is_final = is_final_graph(g, full_graph_nx)
            # print("Is Induced subgraph: ", is_final)
            ps.append(int(is_final))
        correct_p = int(sum(ps) >= 1)
        if correct_p:
            log(f"Correct prediction for sample #{i} from ZincSmiles validation dataset.")
        ys.append(correct_p)
    acc = 0.0 if len(ys) == 0 else float(sum(ys) / len(ys))
    log(f"Oracle Accuracy within the graph decoder : {acc:.4f}")
    return acc


class MetricsPlotsAndOracleCallback(Callback):
    def __init__(
        self,
        *,
        encoder: AbstractGraphEncoder,
        cfg: Config,
        evals_dir: Path,
        artefacts_dir: Path,
        oracle_on_val_end: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.oracle_on_val_end = oracle_on_val_end
        self.evals_dir = Path(evals_dir)
        self.artefacts_dir = Path(artefacts_dir)
        # accumulators
        self._ys = []
        self._logits = []
        self._train_losses = []
        self._val_losses = []
        self._epoch_rows = []
        self._pr_rows = []  # list of dicts: epoch, thr, prec, rec
        self._roc_rows = []  # list of dicts: epoch, thr, tpr, fpr

    def _save_parquet_or_csv(self, df: pd.DataFrame, path: Path):
        try:
            df.to_parquet(path, index=False)
        except Exception:
            df.with_columns = None  # just to silence static analyzers
            df.to_csv(path.with_suffix(".csv"), index=False)

    def on_train_epoch_end(self, trainer, pl_module):
        # Lightning names can vary; try common keys
        cm = trainer.callback_metrics
        tr = None
        for k in ("loss_epoch", "train_loss_epoch", "loss"):
            if k in cm and torch.is_tensor(cm[k]):
                tr = float(cm[k].detach().cpu().item())
                break
        if tr is not None:
            self._train_losses.append(tr)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._ys.clear()
        self._logits.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            out = pl_module(batch)
            logits = out["graph_prediction"].squeeze(-1).detach().float().cpu()
            y = batch.y.detach().float().cpu()
        self._logits.append(logits)
        self._ys.append(y)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._ys:
            return

        # concat
        y = torch.cat(self._ys).numpy().astype(int)
        z = torch.cat(self._logits).numpy().astype(np.float32)
        p = 1.0 / (1.0 + np.exp(-z))  # sigmoid

        # unweighted val loss for comparability
        val_loss = float(
            F.binary_cross_entropy_with_logits(torch.from_numpy(z), torch.from_numpy(y.astype(np.float32)))
        )
        self._val_losses.append(val_loss)

        # prevalence
        pi = float(y.mean())

        # robust metrics (handle single-class batches)
        if np.unique(y).size < 2:
            auc = ap = float("nan")
            prec = rec = thr_pr = None
            fpr = tpr = thr_roc = None
        else:
            auc = float(roc_auc_score(y, p))
            ap = float(average_precision_score(y, p))
            prec, rec, thr_pr = precision_recall_curve(y, p)
            fpr, tpr, thr_roc = roc_curve(y, p)

        # Brier
        brier = float(brier_score_loss(y, p)) if np.unique(y).size == 2 else float("nan")

        # @0.5 metrics
        yhat05 = (p >= 0.5).astype(int)
        acc05 = float((yhat05 == y).mean())
        f105 = float(f1_score(y, yhat05, zero_division=0))
        bal05 = float(balanced_accuracy_score(y, yhat05))
        try:
            mcc05 = float(matthews_corrcoef(y, yhat05))
        except Exception:
            mcc05 = 0.0

        # --- Confusion matrix @0.5 ---
        cm05 = confusion_matrix(y, yhat05, labels=[0, 1])
        tn05, fp05, fn05, tp05 = [int(v) for v in cm05.ravel()]

        # best-F1 from PR thresholds
        if prec is not None and len(prec) > 1:
            f1s = 2 * prec[1:] * rec[1:] / (prec[1:] + rec[1:] + 1e-12)
            best_i = int(np.nanargmax(f1s))
            best_thr = float(thr_pr[best_i])
            f1_best = float(f1s[best_i])
            # stash PR/ROC rows for parquet
            epoch = int(trainer.current_epoch)
            self._pr_rows.extend(
                [
                    {"epoch": epoch, "threshold": float(t), "precision": float(pr), "recall": float(rc)}
                    for pr, rc, t in zip(prec[1:], rec[1:], thr_pr, strict=False)
                ]
            )
            if fpr is not None:
                self._roc_rows.extend(
                    [
                        {"epoch": epoch, "threshold": float(t), "tpr": float(tp), "fpr": float(fp)}
                        for fp, tp, t in zip(fpr, tpr, thr_roc, strict=False)
                    ]
                )
        else:
            best_thr = float("nan")
            f1_best = float("nan")
            epoch = int(trainer.current_epoch)

        # --- Confusion matrix @best F1 threshold (if available) ---
        if np.isfinite(best_thr):
            yhat_best = (p >= best_thr).astype(int)
            cmb = confusion_matrix(y, yhat_best, labels=[0, 1])
            tnb, fpb, fnb, tpb = [int(v) for v in cmb.ravel()]
        else:
            tnb = fpb = fnb = tpb = np.nan

        # log to Lightning
        metrics = {
            "val_loss": val_loss,
            "val_auc": auc,
            "val_ap": ap,
            "val_brier": brier,
            "val_prevalence": pi,
            "val_acc@0.5": acc05,
            "val_f1@0.5": f105,
            "val_bal_acc@0.5": bal05,
            "val_mcc@0.5": mcc05,
            # Confusion matrix counts (scalars => safe for Lightning/CSVLogger)
            "val_tn@0.5": tn05,
            "val_fp@0.5": fp05,
            "val_fn@0.5": fn05,
            "val_tp@0.5": tp05,
            "val_best_f1": f1_best,
            "val_best_thr": best_thr,
            # confusion matrix at best threshold
            "val_tn@best": tnb,
            "val_fp@best": fpb,
            "val_fn@best": fnb,
            "val_tp@best": tpb,
        }
        pl_module.log_dict(metrics, prog_bar=True, logger=True)

        # persist epoch summary row now (so crashes don’t lose it)
        row = {"epoch": epoch, **metrics}
        self._epoch_rows.append(row)
        df_epoch = pd.DataFrame([row])
        self._save_parquet_or_csv(df_epoch, self.evals_dir / "epoch_metrics.parquet")

        # optional: Oracle eval with best_thr
        if self.oracle_on_val_end and np.isfinite(best_thr):
            with torch.no_grad():
                oracle_acc = evaluate_as_oracle(
                    model=pl_module,
                    encoder=self.encoder,
                    oracle_num_evals=self.cfg.oracle_num_evals,
                    oracle_beam_size=self.cfg.oracle_beam_size,
                    oracle_threshold=best_thr,
                )
            pl_module.log("val_oracle_acc", float(oracle_acc), prog_bar=False, logger=True)
            # also store it
            self._epoch_rows[-1]["val_oracle_acc"] = float(oracle_acc)
            df_epoch = pd.DataFrame([self._epoch_rows[-1]])
            self._save_parquet_or_csv(df_epoch, self.evals_dir / "epoch_metrics.parquet")

        # store last-epoch arrays for plotting
        self._last_y = y
        self._last_p = p
        self._last_pr = (prec, rec, thr_pr) if prec is not None else None
        self._last_roc = (fpr, tpr, thr_roc) if fpr is not None else None

    def on_fit_end(self, trainer, pl_module):
        # Write full PR/ROC tables (all epochs) once
        if self._pr_rows:
            self._save_parquet_or_csv(pd.DataFrame(self._pr_rows), self.evals_dir / "pr_curve.parquet")
        if self._roc_rows:
            self._save_parquet_or_csv(pd.DataFrame(self._roc_rows), self.evals_dir / "roc_curve.parquet")

        # Write consolidated epoch metrics once (idempotent)
        if self._epoch_rows:
            df_all = pd.DataFrame(self._epoch_rows).drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
            self._save_parquet_or_csv(df_all, self.evals_dir / "epoch_metrics.parquet")

        # ---- Plots ----
        self.artefacts_dir.mkdir(parents=True, exist_ok=True)

        # 1) Loss curves
        epochs = np.arange(len(self._val_losses))
        plt.figure()
        if self._train_losses:
            plt.plot(np.arange(len(self._train_losses)), self._train_losses, label="train_loss")
        plt.plot(epochs, self._val_losses, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.artefacts_dir / "loss_curves.png")
        plt.close()

        # 2) AUC/AP over epochs (if available)
        if self._epoch_rows and "val_auc" in self._epoch_rows[0]:
            df_all = pd.DataFrame(self._epoch_rows).sort_values("epoch")
            if df_all["val_auc"].notna().any():
                plt.figure()
                plt.plot(df_all["epoch"], df_all["val_auc"])
                plt.xlabel("epoch")
                plt.ylabel("AUC")
                plt.tight_layout()
                plt.savefig(self.artefacts_dir / "auc_by_epoch.png")
                plt.close()
            if df_all["val_ap"].notna().any():
                plt.figure()
                plt.plot(df_all["epoch"], df_all["val_ap"])
                plt.xlabel("epoch")
                plt.ylabel("AP")
                plt.tight_layout()
                plt.savefig(self.artefacts_dir / "ap_by_epoch.png")
                plt.close()

        # 3) PR/ROC for last epoch
        if getattr(self, "_last_pr", None):
            prec, rec, _ = self._last_pr
            plt.figure()
            plt.plot(rec, prec)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.tight_layout()
            plt.savefig(self.artefacts_dir / "pr_curve_last.png")
            plt.close()
        if getattr(self, "_last_roc", None):
            fpr, tpr, _ = self._last_roc
            plt.figure()
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], "--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.tight_layout()
            plt.savefig(self.artefacts_dir / "roc_curve_last.png")
            plt.close()

        # 4) Calibration (reliability) for last epoch
        if getattr(self, "_last_p", None) is not None:
            p = self._last_p
            y = self._last_y
            bins = np.linspace(0, 1, 11)
            idx = np.digitize(p, bins) - 1
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            frac_pos = np.array([y[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(bin_centers))])
            plt.figure()
            plt.plot([0, 1], [0, 1], "--")
            plt.plot(bin_centers, frac_pos, marker="o")
            plt.xlabel("Predicted probability")
            plt.ylabel("Fraction positive")
            plt.tight_layout()
            plt.savefig(self.artefacts_dir / "calibration_last.png")
            plt.close()

            # 5) Confusion matrix heatmap (last epoch, @0.5)
        if getattr(self, "_last_p", None) is not None:
            yhat = (self._last_p >= 0.5).astype(int)
            cm = confusion_matrix(self._last_y, yhat, labels=[0, 1])

            plt.figure()
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix @0.5 (last epoch)")
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])
            for i, j in itertools.product(range(2), range(2)):
                plt.text(j, i, cm[i, j], ha="center", va="center")
            plt.tight_layout()
            plt.savefig(self.artefacts_dir / "confusion_matrix_last.png")
            plt.close()


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------


class TimeLoggingCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        trainer.logger.log_metrics({"elapsed_time_sec": elapsed}, step=trainer.current_epoch)


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def run_experiment(cfg: Config, is_dev: bool = False):
    log("Parsing args done. Starting run_experiment …")
    dirs = setup_exp(cfg.exp_dir_name)
    exp_dir = Path(dirs["exp_dir"])
    models_dir = Path(dirs["models_dir"])
    evals_dir = Path(dirs["evals_dir"])
    artefacts_dir = Path(dirs["artefacts_dir"])

    # Save the config
    def _json_sanitize(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, VSAModel):
            return obj.value
        return obj

    (evals_dir / "run_config.json").write_text(
        json.dumps({k: _json_sanitize(v) for k, v in asdict(cfg).items()}, indent=2)
    )

    seed_everything(cfg.seed)

    # Dataset & Encoder (HRR @ 7744)
    ds_cfg = ZINC_SMILES_HRR_7744_CONFIG
    device = pick_device()
    log(f"Using device: {device!s}")

    log("Loading/creating hypernet …")
    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds_cfg).to(device=device).eval()
    log("Hypernet ready.")
    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.ATOM_TYPE][0].codebook)
    encoder = hypernet.to(device).eval()

    # datamodule with per-epoch resampling
    dm = PairsDataModule(cfg, encoder=encoder, device=device, is_dev=is_dev)

    # ----- model + optim -----
    model = ConditionalGIN(
        input_dim=4,
        edge_dim=1,
        condition_dim=cfg.hv_dim,
        cond_units=cfg.cond_units,
        conv_units=cfg.conv_units,
        film_units=cfg.film_units,
        pred_units=[128, 64, 1],
        cfg=cfg,
    ).to(device)

    log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    log(f"Model on: {next(model.parameters()).device}")

    # ---- Callbacks
    csv_logger = CSVLogger(save_dir=str(evals_dir), name="logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=str(models_dir),
        auto_insert_metric_name=False,
        filename="epoch{epoch:02d}-val{val_loss:.4f}",
        save_last=True,
        save_on_train_epoch_end=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    time_logger = TimeLoggingCallback()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=2,
        min_delta=0.0,
        check_finite=True,  # stop if val becomes NaN/Inf
        verbose=True,
    )

    val_metrics_cb = MetricsPlotsAndOracleCallback(
        encoder=encoder, cfg=cfg, evals_dir=evals_dir, artefacts_dir=artefacts_dir, oracle_on_val_end=True
    )

    # Training
    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=[csv_logger],
        callbacks=[checkpoint_callback, lr_monitor, time_logger, early_stopping, val_metrics_cb],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        # gradient_clip_val=1.0,  # Do we need this?
        log_every_n_steps=100 if not is_dev else 1,
        enable_progress_bar=True,
        deterministic=False,
        precision=pick_precision(),
        reload_dataloaders_every_n_epochs=1,
    )

    # --- Train
    resume_path: Path | None = str(cfg.continue_from) if cfg.continue_from else None
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path)
    log("Finished training.")

    # Final plots


if __name__ == "__main__":
    # ----------------- CLI parsing that never clobbers defaults -----------------
    def _parse_int_list(s: str) -> list[int]:
        # accept "4096,2048,512,128" or with spaces
        return [int(tok) for tok in s.replace(" ", "").split(",") if tok]

    def _parse_vsa(s: str) -> VSAModel:
        # Accepts e.g. "HRR", not VSAModel.HRR
        if isinstance(s, VSAModel):
            return s
        return VSAModel(s)

    def get_args(argv: list[str] | None = None) -> Config:
        """
        Build a Config by starting from dataclass defaults and then
        applying ONLY the CLI options the user actually provided.
        NOTE: For --vsa, pass a string like "HRR", not VSAModel.HRR.
        """
        cfg = Config()  # start with your defaults

        p = argparse.ArgumentParser(description="Experiment Config (unified)")

        # IMPORTANT: default=SUPPRESS so unspecified flags don't overwrite dataclass defaults
        p.add_argument(
            "--project_dir",
            "-pdir",
            type=Path,
            default=argparse.SUPPRESS,
            help="Project root (will be created if missing)",
        )
        p.add_argument("--exp_dir_name", type=str, default=argparse.SUPPRESS)

        p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
        p.add_argument("--epochs", "-e", type=int, default=argparse.SUPPRESS)
        p.add_argument("--batch_size", "-bs", type=int, default=argparse.SUPPRESS)
        p.add_argument("--is_dev", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)

        # Evals
        p.add_argument("--oracle_num_evals", type=int, default=argparse.SUPPRESS)
        p.add_argument("--oracle_beam_size", type=int, default=argparse.SUPPRESS)

        # Model knobs
        p.add_argument(
            "--film_units",
            type=_parse_int_list,
            default=argparse.SUPPRESS,
            help="Comma-separated: e.g. '128,64'",
        )
        p.add_argument(
            "--cond_units", type=_parse_int_list, default=argparse.SUPPRESS, help="Comma-separated: e.g. '256,128'"
        )
        p.add_argument(
            "--cond_emb_dim",
            type=int,
            default=argparse.SUPPRESS,
            help="If omitted but --cond_units is given, will default to last(cond_units)",
        )
        p.add_argument(
            "--conv_units", type=_parse_int_list, default=argparse.SUPPRESS, help="Comma-separated: e.g. '64,64,64'"
        )
        p.add_argument(
            "--pred_head_units",
            type=_parse_int_list,
            default=argparse.SUPPRESS,
            help="Comma-separated: e.g. '256,64,1'",
        )

        # HDC
        p.add_argument("--hv_dim", "-hd", type=int, default=argparse.SUPPRESS)
        p.add_argument("--vsa", "-v", type=_parse_vsa, default=argparse.SUPPRESS)

        # Optim
        p.add_argument("--lr", type=float, default=argparse.SUPPRESS)
        p.add_argument("--weight_decay", "-wd", type=float, default=argparse.SUPPRESS)

        # Loader
        p.add_argument("--num_workers", type=int, default=argparse.SUPPRESS)
        p.add_argument("--prefetch_factor", type=int, default=argparse.SUPPRESS)
        p.add_argument("--pin_memory", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)

        # Checkpointing
        p.add_argument("--continue_from", type=Path, default=argparse.SUPPRESS)
        p.add_argument("--resume_retrain_last_epoch", type=str2bool, default=argparse.SUPPRESS)

        # Stratification
        p.add_argument("--stratify", type=str2bool, default=argparse.SUPPRESS)
        p.add_argument("--p_per_parent", type=int, default=argparse.SUPPRESS)
        p.add_argument("--n_per_parent", type=int, default=argparse.SUPPRESS)
        p.add_argument(
            "--exclude_negs", type=_parse_int_list, default=argparse.SUPPRESS, help="Comma-separated ints, e.g. '1,2,3'"
        )
        p.add_argument("--resample_training_data_on_batch", type=str2bool, default=argparse.SUPPRESS)

        ns = p.parse_args(argv)
        provided = vars(ns)  # only the keys the user actually passed

        # Apply only provided keys onto cfg
        for k, v in provided.items():
            # Make sure VSAModel parsed if user typed the enum value directly
            if k == "vsa" and isinstance(v, str):
                v = VSAModel(v)
            setattr(cfg, k, v)

        return cfg

    log(f"Running {Path(__file__).resolve()}")
    is_dev = os.getenv("LOCAL_HDC_", False)

    if is_dev:
        log("Running in local HDC (DEV) ...")
        cfg: Config = Config(
            exp_dir_name="overfitting_batch_norm",
            seed=42,
            epochs=1,
            batch_size=128,
            hv_dim=88 * 88,
            vsa=VSAModel.HRR,
            lr=1e-4,
            weight_decay=0.0,
            num_workers=0,
            prefetch_factor=1,
            pin_memory=False,
            continue_from=None,
            resume_retrain_last_epoch=False,
            stratify=True,
            p_per_parent=2,
            n_per_parent=2,
            oracle_beam_size=8,
            oracle_num_evals=8,
            resample_training_data_on_batch=True,
        )
    else:
        log("Running in cluster ...")
        cfg = get_args()

    pprint(asdict(cfg), indent=2)
    run_experiment(cfg, is_dev=is_dev or cfg.is_dev)
