from typing import Sequence, Optional

import networkx as nx
import matplotlib.pyplot as plt


ATOM_TYPES = ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']  # index -> symbol
FORMAL_CHARGE_IDX_TO_VAL: dict[int, str] = {0: "0", 1: "+", 2: "-"}


def feature_label_text(G: nx.Graph, n: int,
                       atom_symbols: Sequence[str] = ATOM_TYPES,
                       show_current_degree: bool = False,
                       show_residual: bool = False) -> str:
    """
    Build a compact, readable label string for a node.

    The label always reflects **frozen** features from ``node['feat']``.
    Optionally, include the *current* degree and residual for debugging.

    :param G: Graph with node attributes ``feat: Feat`` and ``target_degree``.
    :param n: Node id.
    :param atom_symbols: Mapping from atom index to chemical symbol.
    :param show_current_degree: If ``True``, append current degree.
    :param show_residual: If ``True``, append residual stubs.
    :returns: Label string like ``C | deg*:3 | H:1 | q:0 (cur:2,res:1)``.
    """
    f = G.nodes[n]["feat"]
    at_idx, deg_idx, ch_idx, hs = f.to_tuple()

    atom = atom_symbols[at_idx] if 0 <= at_idx < len(atom_symbols) else f"X{at_idx}"
    deg_star = getattr(f, "target_degree", None) or (deg_idx + 1)
    charge = FORMAL_CHARGE_IDX_TO_VAL.get(ch_idx, ch_idx)

    parts = [f"{atom}", f"deg*:{deg_star}", f"H:{hs}", f"q:{charge}"]
    base = " | ".join(parts)

    if show_current_degree or show_residual:
        cur = G.degree[n]
        res = int(G.nodes[n]["target_degree"]) - cur
        extra = []
        if show_current_degree:
            extra.append(f"cur:{cur}")
        if show_residual:
            extra.append(f"res:{res}")
        if extra:
            base = f"{base} ({','.join(extra)})"

    return base


def compute_layout(G: nx.Graph, layout: str = "spring", seed: int = 42) -> dict[int, tuple[float, float]]:
    """
    Compute deterministic 2D positions for drawing.

    :param G: Graph.
    :param layout: One of ``'spring'``, ``'kamada_kawai'``, ``'circular'``, ``'shell'``.
    :param seed: Seed for deterministic layouts.
    :returns: Mapping ``node -> (x, y)``.
    """
    layout = layout.lower()
    if layout == "spring":
        return nx.spring_layout(G, seed=seed, k=None)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    if layout == "circular":
        return nx.circular_layout(G)
    if layout == "shell":
        return nx.shell_layout(G)
    raise ValueError(f"Unknown layout: {layout}")



def draw_nx_with_atom_colorings(G: nx.Graph,
                                *,
                                atom_symbols: Sequence[str] = ATOM_TYPES,
                                layout: str = "spring",
                                seed: int = 42,
                                node_size: int = 500,
                                font_size: int = 9,
                                show_current_degree: bool = False,
                                show_residual: bool = False,
                                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Variant with simple categorical coloring by atom type index.

    :param atom_symbols:
    :param G: Graph with node attribute ``feat: Feat``.
    :returns: Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    pos = compute_layout(G, layout=layout, seed=seed)

    # Map atom_type -> color using a minimal palette (cycled).
    palette = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#ff7f0e"]
    colors: list[str] = []
    for n in G.nodes:
        at_idx = G.nodes[n]["feat"].atom_type
        colors.append(palette[at_idx % len(palette)])

    nx.draw_networkx_edges(G, pos, ax=ax, width=1.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, edgecolors="#333333",
                           linewidths=1.0, node_size=node_size)

    labels = {
        n: feature_label_text(G, n,
                              atom_symbols=atom_symbols,
                              show_current_degree=show_current_degree,
                              show_residual=show_residual)
        for n in G.nodes
    }
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=font_size)

    ax.axis("off")
    ax.set_title("Graph (colored by atom type, frozen features)", fontsize=11)
    return ax