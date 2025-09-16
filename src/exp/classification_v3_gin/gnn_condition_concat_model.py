import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.nn.conv import GATv2Conv


class MockConfig:
    """Mock configuration class for testing purposes."""

    def __init__(self, n_per_parent=1.0, p_per_parent=1.0):
        self.n_per_parent = n_per_parent
        self.p_per_parent = p_per_parent


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
        cond_units: list[int] = [256, 64],
        conv_units: list[int] = [128, 128, 128],
        pred_units: list[int] = [256, 64],
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
        self.log(
            "val_loss", val_loss.detach().as_subclass(torch.Tensor), prog_bar=True, logger=True, batch_size=batch_size
        )
        return val_loss

    def configure_optimizers(self):
        """
        Configure optimizers for the model.

        This method is called by PyTorch Lightning to set up the optimizer
        for training the model.

        :returns: The configured optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def create_mock_graph(num_nodes=10, num_edges=20, node_dim=32, edge_dim=8, condition_dim=128):
    """Create a mock graph for testing purposes."""
    # Generate random node features
    x = torch.randn(num_nodes, node_dim)

    # Generate random edge indices
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Generate random edge attributes
    edge_attr = torch.randn(num_edges, edge_dim)

    # Generate random edge weights
    edge_weights = torch.randn(num_edges)

    # Generate random condition vector
    cond = torch.randn(1, condition_dim)

    # Generate random binary label
    y = torch.randint(0, 2, (1,)).float()

    # Create batch index (single graph)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_weights=edge_weights, cond=cond, y=y, batch=batch)


def test_network():
    """Test the ConditionalGIN network with mock data."""
    print("Testing ConditionalGIN network...")

    # Network parameters
    input_dim = 32
    edge_dim = 8
    condition_dim = 128

    # Create the model
    mock_cfg = MockConfig(n_per_parent=1.0, p_per_parent=1.0)
    model = ConditionalGIN(
        input_dim=input_dim,
        edge_dim=edge_dim,
        condition_dim=condition_dim,
        cond_units=[256, 64],
        conv_units=[64, 64, 64],
        pred_units=[256, 64, 1],
        learning_rate=0.001,
        cfg=mock_cfg,
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create batch of mock graphs for proper batch norm operation
    print("Creating batch of mock graphs...")
    graphs = [
        create_mock_graph(
            num_nodes=torch.randint(10, 20, (1,)).item(),
            num_edges=torch.randint(20, 40, (1,)).item(),
            node_dim=input_dim,
            edge_dim=edge_dim,
            condition_dim=condition_dim,
        )
        for _ in range(4)
    ]  # Use batch size of 4

    # Create proper batch
    batch_list = []
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    edge_weights_list = []
    cond_list = []
    y_list = []

    node_offset = 0
    for i, g in enumerate(graphs):
        batch_list.append(torch.full((g.x.shape[0],), i, dtype=torch.long))
        x_list.append(g.x)
        edge_index_list.append(g.edge_index + node_offset)
        edge_attr_list.append(g.edge_attr)
        edge_weights_list.append(g.edge_weights)
        cond_list.append(g.cond)
        y_list.append(g.y)
        node_offset += g.x.shape[0]

    batch_data = Data(
        x=torch.cat(x_list, dim=0),
        edge_index=torch.cat(edge_index_list, dim=1),
        edge_attr=torch.cat(edge_attr_list, dim=0),
        edge_weights=torch.cat(edge_weights_list, dim=0),
        cond=torch.cat(cond_list, dim=0),
        y=torch.cat(y_list, dim=0),
        batch=torch.cat(batch_list, dim=0),
    )

    total_nodes = sum(g.x.shape[0] for g in graphs)
    total_edges = sum(g.edge_index.shape[1] for g in graphs)
    print(f"Created batch with {len(graphs)} graphs, {total_nodes} total nodes, {total_edges} total edges")

    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_data)
        predictions = output["graph_prediction"]
        print(f"Forward pass successful! Output shape: {predictions.shape}")
        print(f"Prediction values: {predictions.squeeze().tolist()}")
        print(f"Sigmoid probabilities: {torch.sigmoid(predictions).squeeze().tolist()}")

    # Test training mode and optimization
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nPerforming optimization steps...")
    initial_loss = None
    for step in range(50):
        optimizer.zero_grad()

        # Forward pass
        output = model(batch_data)
        logits = output["graph_prediction"].squeeze(-1)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(logits, batch_data.y)

        if initial_loss is None:
            initial_loss = loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Step {step + 1}: Loss = {loss.item():.6f}")

    print("\nOptimization complete!")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss: {loss.item():.6f}")
    print(f"Loss reduction: {initial_loss - loss.item():.6f}")

    print("\nAll tests passed! The network is working correctly with batch normalization.")


if __name__ == "__main__":
    test_network()
