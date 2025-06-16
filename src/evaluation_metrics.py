from collections import Counter
from typing import Optional, Union, Sequence

import torch
from torch_geometric.data import Data

from src.utils.utils import DataTransformer




def edge_f1_score(pred: Data, actual: Data) -> tuple[float, float, float]:
    # pred_set = to_edge_set(pred.edge_index)
    # true_set = to_edge_set(actual.edge_index)
    #
    # tp = len(pred_set & true_set)
    # fp = len(pred_set - true_set)
    # fn = len(true_set - pred_set)
    #
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    #
    # return precision, recall, f1
    return calculate_p_a_f1(
        pred=Counter(DataTransformer.to_tuple_list(pred.edge_index)),
        true=Counter(DataTransformer.to_tuple_list(actual.edge_index)),
    )


def graph_edit_distance(
    data_pred: Data, data_true: Data, node_mask: Optional[Union[Sequence[int], torch.Tensor]] = None
) -> tuple[int, int, int]:
    """
    Compute an approximate Graph Edit Distance (GED) between two graphs.

    COST MODEL:
      • Node substitution   (A→B)            = 1
      • Node insertion/deletion              = 1 each
      • Edge insertion/deletion              = 1 each

    NODE IDENTITY:
      If `data.x` exists, we derive each node's “label” from its feature vector.
      Optionally you can provide `node_mask` to select a subset of feature dims;
      otherwise all features are used.  If no `data.x`, we fallback to integer
      node indices as labels.

    RETURNS:
      delta_v = total node edit cost  (#subs + #ins + #dels)
      delta_e = total edge edit cost  (#edge ins + #edge dels)
      total   = delta_v + delta_e

    :param data_pred: Predicted graph
    :param data_true: Ground-truth graph
    :param node_mask:
        Optional list/tensor of feature-indices to use for node labels.
        If None and `data.x` is present, all dims are used.
    :raises ValueError: for invalid mask or missing features
    """

    # --- 1) Build a Counter of “node labels” and retain the raw label list
    def _get_node_labels(data: Data) -> tuple[Counter[list], list[int]]:
        if hasattr(data, "x") and data.x is not None:
            x = data.x.int()
            if node_mask is not None:
                mask = (
                    torch.tensor(node_mask, dtype=torch.long) if not isinstance(node_mask, torch.Tensor) else node_mask
                )
                if mask.ndim != 1 or mask.min() < 0 or mask.max() >= x.size(1):
                    msg = f"Invalid node_mask: {node_mask}"
                    raise ValueError(msg)
                x = x[:, mask]
            # convert each feature-row to a tuple so it’s hashable hence countable
            labels = [tuple(vec.tolist()) for vec in x]
        else:
            labels = list(range(data.num_nodes))

        return Counter(labels), labels

    counter_p, labels_p = _get_node_labels(data_pred)
    counter_t, labels_t = _get_node_labels(data_true)

    # --- 2) Node edit cost
    # raw_sym_diff = # of label mismatches = deletions + insertions
    union_labels = set(counter_p) | set(counter_t)
    # here the ins/del are counted as 1, but subs is counted as two - so we correct it
    raw_sym_diff = sum(abs(counter_p[label] - counter_t[label]) for label in union_labels)

    # the difference in total node‐counts must be pure ins/dels
    n_diff = abs(data_pred.num_nodes - data_true.num_nodes)

    num_subs = (raw_sym_diff - n_diff) // 2

    # total node cost = subs + unmatched ins/dels
    delta_v = num_subs + n_diff

    # --- 3) Edge edit cost (undirected)
    def _edge_counter(data: Data, labels: list) -> Counter[list]:
        # map each edge (u,v) → sorted pair of node-labels
        # Each node is identified from it's labels. We use the node labels instead of indexes.
        pairs = [tuple(sorted((labels[u], labels[v]))) for u, v in data.edge_index.t().tolist()]
        return Counter(pairs)

    edge_p = _edge_counter(data_pred, labels_p)
    edge_t = _edge_counter(data_true, labels_t)
    all_edges = set(edge_p) | set(edge_t)

    # deletions + insertions for edges
    delta_e = sum(abs(edge_p[e] - edge_t[e]) for e in all_edges)

    return delta_v, delta_e, (delta_v + delta_e)


def calculate_p_a_f1(pred: Counter, true: Counter) -> tuple[float, float, float]:
    # True positives: how many of each key we correctly predicted
    tp = sum(min(pred[k], true.get(k, 0)) for k in pred)

    # False positives: all predicted minus the true positives
    fp = sum(pred.values()) - tp

    # False negatives: all actual minus the true positives
    fn = sum(true.values()) - tp

    # Precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1
