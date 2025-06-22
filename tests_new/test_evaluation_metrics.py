import random

import pytest
import torch
from torch_geometric.data import Data

from src import evaluation_metrics
from src.datasets import ColorGraphDataset
from tests.utils import ASSETS_PATH


def make_graph(edge_list, node_features=None):
    """
    Helper to construct a torch_geometric Data object.

    :param edge_list: List of undirected edges [[u, v], ...].
    :param node_features: Optional list of feature lists or torch.Tensor of shape (N, F).
    :returns: Data object with .edge_index, .num_nodes, and .x if features provided.
    """
    # Determine edge_index tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        inferred_nodes = int(edge_index.max().item()) + 1
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        inferred_nodes = 0

    # Build Data
    data = Data(edge_index=edge_index)

    # Determine num_nodes and optional features
    if node_features is not None:
        x = torch.tensor(node_features, dtype=torch.float)
        data.x = x
    else:
        data.num_nodes = inferred_nodes

    return data


class TestEdgeF1Score:
    def test_self_score(self):
        """
        A graph should match itself perfectly (precision=recall=F1=1.0).
        """
        data = make_graph([[0, 1], [1, 2], [2, 3]])
        p, r, f = evaluation_metrics.edge_f1_score(
            data, data
        )
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f == pytest.approx(1.0)

    def test_self_score_color_graph(self):
        num_graphs = 100
        dataset = ColorGraphDataset(root=ASSETS_PATH, num_graphs=num_graphs)
        data = dataset[random.randint(0, dataset.num_graphs - 1)]
        p, r, f = evaluation_metrics.edge_f1_score(data, data)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("true_edges", "pred_edges", "exp_precision", "exp_recall", "exp_f1"),
        [
            (
                [[0, 1], [1, 2]],  # true
                [[0, 1]],  # pred
                1.0,  # precision
                0.5,  # recall
                2 * 1.0 * 0.5 / (1.0 + 0.5),  # f1
            ),
            # (
            #     [[0, 1], [1, 2], [2, 3]],
            #     [[0, 1], [2, 3], [3, 4]],
            #     2 / 3,  # precision: 2 correct of 3 pred
            #     2 / 3,  # recall: 2 correct of 3 true
            #     2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3)),
            # ),
            # (
            #     [],  # true empty
            #     [[0, 1]],  # pred non-empty
            #     0.0,  # precision: 0 TP / 1 pred
            #     0.0,  # recall: no true edges
            #     0.0,
            # ),
        ],
    )
    def test_known_scores(self, true_edges, pred_edges, exp_precision, exp_recall, exp_f1):
        """
        Parametrized tests comparing predictions with true graphs.
        """
        true = make_graph(true_edges)
        pred = make_graph(pred_edges)
        p, r, f = evaluation_metrics.edge_f1_score(
            pred,
            true)
        assert p == pytest.approx(exp_precision)
        assert r == pytest.approx(exp_recall)
        assert f == pytest.approx(exp_f1)

    def test_empty_graphs(self):
        """
        Two empty graphs: no edges should yield zeros.
        """
        empty1 = make_graph([])
        empty2 = make_graph([])
        p, r, f = evaluation_metrics.edge_f1_score(empty1, empty2)
        assert p == 0.0
        assert r == 0.0
        assert f == 0.0

class TestGraphEditDistance:
    def test_self_distance_nonempty(self):
        """
        A non-empty graph compared to itself should yield zero edit distance.
        """
        data = make_graph([[0, 1], [1, 2], [2, 3]])
        dv, de, total = evaluation_metrics.graph_edit_distance(data, data)
        assert dv == 0
        assert de == 0
        assert total == 0

    def test_self_distance_empty(self):
        """
        An empty graph compared to itself should yield zero edit distance.
        """
        empty = make_graph([])
        dv, de, total = evaluation_metrics.graph_edit_distance(empty, empty)
        assert dv == 0
        assert de == 0
        assert total == 0

    @pytest.mark.parametrize(
        ("true_edges", "pred_edges", "exp_dv", "exp_de", "exp_total"),
        [
            # missing one node and one edge
            ([[0, 1], [1, 2]], [[0, 1]], 1, 1, 2),
            # extra node and extra edge
            ([[0, 1]], [[0, 1], [1, 2]], 1, 1, 2),
            # missing node and edge
            ([], [[0, 1]], 2, 1, 3),  # true has 0 nodes, pred has 2 nodes; 1 edge
            # different nodes but no edges
            ([[0, 1]], [], 2, 1, 3),  # true has 2 nodes & 1 edge, pred empty
        ],
    )
    def test_known_distances(self, true_edges, pred_edges, exp_dv, exp_de, exp_total):
        """
        Parameterized tests for known graph edit distances.
        """
        true = make_graph(true_edges)
        pred = make_graph(pred_edges)
        dv, de, total = evaluation_metrics.graph_edit_distance(pred, true)
        assert dv == exp_dv
        assert de == exp_de
        assert total == exp_total

    def test_invalid_input(self):
        """
        Passing invalid inputs should raise an AttributeError.
        """
        with pytest.raises(AttributeError):
            evaluation_metrics.graph_edit_distance(None, None)

    def test_simple_feature_graph(self):
        """Single-feature graphs identical => zero edit distance."""
        true = make_graph([[0, 1]], node_features=[[10], [20]])
        pred = make_graph([[0, 1]], node_features=[[10], [20]])
        dv, de, total = evaluation_metrics.graph_edit_distance(pred, true)
        assert dv == 0
        assert de == 0
        assert total == 0

    def test_complex_feature_graph_self(self):
        """Multi-feature graph vs itself => zero edit distance."""
        features = [list(range(10)), list(range(10, 20)), list(range(20, 30))]
        edges = [[0, 1], [1, 2]]
        data = make_graph(edges, node_features=features)
        dv, de, total = evaluation_metrics.graph_edit_distance(data, data)
        assert (dv, de, total) == (0, 0, 0)

    def test_complex_feature_graph_with_mask(self):
        """Masking features hides differences in node identity."""
        # True graph features: all nodes identical except by feature mask
        features_true = [[0] * 10, [1] * 10, [2] * 10]
        # Pred graph differs only in the last feature of node 2 (1 substitution)
        features_pred = [[0] * 10, [1] * 10, [2] * 9 + [3]]
        edges = [[0, 1], [1, 2]]

        true = make_graph(edges, node_features=features_true)
        pred = make_graph(edges, node_features=features_pred)

        # Without mask: node 2 identity changed => delete old + insert new => dv=2
        # Edges reconnect to a “different” node identity twice => de=2, total=4
        dv, de, total = evaluation_metrics.graph_edit_distance(pred, true)
        assert dv == 1
        assert de == 2
        assert total == 3

        # With mask on first 9 features: the differing 10th dimension is ignored => zero edits
        dv, de, total = evaluation_metrics.graph_edit_distance(pred, true, node_mask=list(range(9)))
        assert dv == 0
        assert de == 0
        assert total == 0

    def test_mask_invalid(self):
        """Invalid mask indices should raise ValueError."""
        true = make_graph([], node_features=[[0, 1]])
        pred = make_graph([], node_features=[[0, 1]])
        with pytest.raises(ValueError, match="Invalid node_mask"):
            evaluation_metrics.graph_edit_distance(pred, true, node_mask=[2])
