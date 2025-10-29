import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ──────────────────────────────────────────────────────────────────────────────
ATOM_TYPES = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]

QM9_SMILE_ATOM_TO_IDX: dict[str, int] = {"C": 0, "N": 1, "O": 2, "F": 3}
QM9_SMILE_IDX_TO_ATOM: dict[int, str] = {v: k for k, v in QM9_SMILE_ATOM_TO_IDX.items()}
ZINC_SMILE_ATOM_TO_IDX: dict[str, int] = {sym: i for i, sym in enumerate(ATOM_TYPES)}
ZINC_SMILE_IDX_TO_ATOM: dict[int, str] = {i: sym for sym, i in ZINC_SMILE_ATOM_TO_IDX.items()}
FORMAL_CHARGE_IDX_TO_VAL: dict[int, str] = {0: "0", 1: "+", 2: "-"}


def _symbols_from_idx_map(idx_to_atom: dict[int, str]) -> list[str]:
    if not idx_to_atom:
        return []
    max_idx = max(idx_to_atom.keys())
    return [idx_to_atom.get(i, f"X{i}") for i in range(max_idx + 1)]


def feature_label_text(
    G: nx.Graph,
    n: int,
    atom_symbols: Sequence[str] = ATOM_TYPES,
    show_current_degree: bool = False,
    show_residual: bool = False,
) -> str:
    """
    Build a compact label from frozen node features.

    :param G: Graph with ``node['feat']`` and (optionally) ``target_degree``.
    :param n: Node id.
    :param atom_symbols: Dense mapping (index → symbol).
    :param show_current_degree: Append current degree if ``True``.
    :param show_residual: Append degree residual if ``True``.
    :returns: Readable label string.
    """
    f = G.nodes[n]["feat"]
    tup = f.to_tuple()
    if len(tup) == 4:
        at_idx, deg_idx, ch_idx, hs = tup
    if len(tup) == 5:
        at_idx, deg_idx, ch_idx, hs, _ = tup
    atom = atom_symbols[at_idx] if 0 <= at_idx < len(atom_symbols) else f"X{at_idx}"
    deg_star = getattr(f, "target_degree", None) or (deg_idx + 1)
    charge = FORMAL_CHARGE_IDX_TO_VAL.get(ch_idx, ch_idx)

    parts = [f"{atom}", f"deg*:{deg_star}", f"q:{charge}", f"H:{hs}"]
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
    Deterministic 2D positions.

    :param G: Graph.
    :param layout: ``'spring'``, ``'kamada_kawai'``, ``'circular'``, or ``'shell'``.
    :param seed: Seed for reproducible layouts that support it.
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


def _pad_axes_to_text(ax: plt.Axes, pad: float = 0.05) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for t in ax.texts:
        bb = t.get_window_extent(renderer=renderer).expanded(1.02, 1.05)
        lx0, ly0 = inv.transform((bb.x0, bb.y0))
        lx1, ly1 = inv.transform((bb.x1, bb.y1))
        x0 = min(x0, lx0)
        x1 = max(x1, lx1)
        y0 = min(y0, ly0)
        y1 = max(y1, ly1)
    xr, yr = x1 - x0, y1 - y0
    ax.set_xlim(x0 - pad * xr, x1 + pad * xr)
    ax.set_ylim(y0 - pad * yr, y1 + pad * yr)


def draw_nx_with_atom_colorings(
    H: nx.Graph,
    *,
    dataset: Literal["ZincSmiles", "QM9Smiles"] = "ZincSmiles",
    atom_symbols: Sequence[str] = ATOM_TYPES,
    layout: str = "spring",
    seed: int = 42,
    figsize: tuple[float, float] = (12, 9),
    node_size: int = 600,
    font_size: int = 10,
    show_current_degree: bool = False,
    show_residual: bool = False,
    label: str | None = None,
    ax: plt.Axes | None = None,
    node_key: str = "feat",
    # Highlighting
    highlight_edges: list[tuple[int, int]] | None = None,
    highlight_color: str = "#ff3b30",
    highlight_width: float = 4.0,
    # Full-graph overlay (backdrop)
    overlay_full_graph: nx.Graph | None = None,
    overlay_edge_color: str = "#737373",
    overlay_edge_width: float = 3.5,
    overlay_node_size: int = 900,
    overlay_alpha: float = 0.4,
    overlay_draw_nodes: bool = True,
    overlay_labels: bool = True,
    # Styling / visibility
    subgraph_edge_width: float = 2.0,
    node_border_color: str = "#222222",
    node_border_width: float = 1.4,
    label_box: bool = True,
    label_box_alpha: float = 0.85,
) -> plt.Axes:
    """
    Draw a large, clearly legible subgraph ``H`` on top of an optional, thick/faint
    overlay of the full graph—perfectly aligned.

    Rendering order:
      1. Overlay edges/nodes (backdrop)
      2. Subgraph edges/nodes
      3. Labels
      4. Highlighted edges

    Parameters
    ----------
    H :
        Subgraph to emphasize (colored, labeled).
    dataset :
        ``"ZincSmile"`` or ``"QM9Smiles"`` to select atom symbol mapping.
    atom_symbols :
        Dense mapping (index → symbol) when ``dataset!="QM9Smiles"``.
    layout, seed :
        Layout algorithm + seed. One layout is computed on the *layout graph*
        (``overlay_full_graph`` if provided, else ``H``) to guarantee alignment.
    figsize :
        Figure size in inches.
    node_size, font_size :
        Visual sizes for ``H``.
    show_current_degree, show_residual :
        Toggle extra debug info in labels.
    node_key :
        Node attribute with ``.atom_type`` and ``.to_tuple()``.
    highlight_edges, highlight_color, highlight_width :
        Edges in ``H`` to accentuate; drawn last with rounded caps.
    overlay_full_graph :
        If given, draw as thick/faint backdrop using the same positions.
    overlay_* :
        Styling for the backdrop (edge color/width, node size, alpha).
    subgraph_edge_width, node_border_color, node_border_width :
        Styling for emphasized subgraph.
    label_box, label_box_alpha :
        Draw semi-transparent label boxes for readability.

    Returns
    -------
    matplotlib.axes.Axes
        The axes used for drawing.
    """

    # --- small helpers for mpl collections/text returned by networkx draw ---
    def _set_zorder(obj, z: float) -> None:
        if obj is None:
            return
        # Some draw_* return a LineCollection/PathCollection; arrows may return list-like.
        try:
            # Try as iterable of artists
            for c in obj:
                with contextlib.suppress(Exception):
                    c.set_zorder(z)
        except TypeError:
            # Single collection
            with contextlib.suppress(Exception):
                obj.set_zorder(z)

    def _set_capstyle(obj, style: str) -> None:
        if obj is None:
            return
        try:
            for c in obj:
                with contextlib.suppress(Exception):
                    c.set_capstyle(style)
        except TypeError:
            with contextlib.suppress(Exception):
                obj.set_capstyle(style)

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    atom_symbols_eff = _symbols_from_idx_map(QM9_SMILE_IDX_TO_ATOM) if dataset == "QM9Smiles" else list(atom_symbols)

    # Shared layout → alignment guaranteed
    layout_graph = overlay_full_graph if overlay_full_graph is not None else H
    pos = compute_layout(layout_graph, layout=layout, seed=seed)

    # 1) Backdrop overlay (thick but faint)
    if overlay_full_graph is not None:
        overlay_edge_coll = nx.draw_networkx_edges(
            overlay_full_graph,
            pos,
            ax=ax,
            width=overlay_edge_width,
            edge_color=overlay_edge_color,
            alpha=overlay_alpha,
        )
        _set_zorder(overlay_edge_coll, 0)

        if overlay_draw_nodes:
            overlay_nodes = nx.draw_networkx_nodes(
                overlay_full_graph,
                pos,
                ax=ax,
                node_size=overlay_node_size,
                node_color="#e5e5e5",
                edgecolors="#9ca3af",
                linewidths=0.8,
                alpha=overlay_alpha,
            )
            _set_zorder(overlay_nodes, 0.1)

        if overlay_labels and overlay_full_graph is not None:
            import numpy as np

            # knobs
            avoid_H_nodes = False  # skip labels where H already labels
            outward_frac = 0.15  # ~8% of axis span away from center
            label_alpha = max(0.45, overlay_alpha)
            font = max(6, font_size - 2)
            z = 1.8  # above subgraph nodes (2.1), below H labels (3)

            # center of layout to compute "outward" direction
            xs, ys = zip(*pos.values(), strict=False)
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            dx_span, dy_span = (max(xs) - min(xs)), (max(ys) - min(ys))
            # distance in data units to push labels
            push_x, push_y = outward_frac * dx_span, outward_frac * dy_span

            H_nodes = set(H.nodes)

            def _safe_overlay_label(n: int) -> str:
                try:
                    return feature_label_text(
                        overlay_full_graph,
                        n,
                        atom_symbols=atom_symbols_eff,
                        show_current_degree=show_current_degree,
                        show_residual=show_residual,
                    )
                except Exception:
                    return str(n)

            for n, (x, y) in pos.items():
                if avoid_H_nodes and n in H_nodes:
                    continue  # keep subgraph labels dominant

                # outward direction from center → node
                vx, vy = x - cx, y - cy
                norm = np.hypot(vx, vy) or 1.0
                ux, uy = vx / norm, vy / norm

                # place label *away* from node in data coords
                lx = x + ux * push_x
                ly = y + uy * push_y

                txt = ax.text(
                    lx,
                    ly,
                    _safe_overlay_label(n),
                    fontsize=font,
                    alpha=label_alpha,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65),
                    clip_on=False,
                )
                try:
                    import matplotlib.patheffects as pe  # type: ignore

                    txt.set_path_effects([pe.withStroke(linewidth=1.2, foreground="black", alpha=0.5)])
                except Exception:
                    pass
                txt.set_zorder(z)

                # optional leader line from label to node (helps when displaced far)
                ax.plot([x, lx], [y, ly], lw=1.0, alpha=min(0.35, label_alpha), color="#9ca3af", solid_capstyle="round")

    # 2) Emphasized subgraph H (color by atom type)
    palette = [
        "#7f7f7f",
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#bcbd22",
        "#17becf",
        "#ff7f0e",
    ]
    try:
        node_colors = [palette[H.nodes[n][node_key].atom_type % len(palette)] for n in H.nodes]
    except Exception:
        node_colors = [palette[H.nodes[n][node_key][0] % len(palette)] for n in H.nodes]

    sub_edges = nx.draw_networkx_edges(H, pos, ax=ax, width=subgraph_edge_width)
    _set_zorder(sub_edges, 2)

    sub_nodes = nx.draw_networkx_nodes(
        H,
        pos,
        ax=ax,
        node_color=node_colors,
        edgecolors=node_border_color,
        linewidths=node_border_width,
        node_size=node_size,
    )
    _set_zorder(sub_nodes, 2.1)
    # Optional halo for visibility on top of backdrop
    try:
        import matplotlib.patheffects as pe  # type: ignore

        sub_nodes.set_path_effects(
            [pe.withStroke(linewidth=max(0.5, node_border_width), foreground="white", alpha=0.6)]
        )
    except Exception:
        pass

    labels = {
        n: feature_label_text(
            H, n, atom_symbols=atom_symbols_eff, show_current_degree=show_current_degree, show_residual=show_residual
        )
        for n in H.nodes
    }
    label_kwargs = dict(font_size=font_size)
    if label_box:
        label_kwargs["bbox"] = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=label_box_alpha)

    text_dict = nx.draw_networkx_labels(H, pos, labels=labels, ax=ax, **label_kwargs)
    for txt in (text_dict or {}).values():
        try:
            txt.set_zorder(3)
            import matplotlib.patheffects as pe  # type: ignore

            txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white", alpha=0.9)])
        except Exception:
            pass

    # 3) Highlight edges last (on H)
    if highlight_edges:
        edgelist = []
        for a, b in highlight_edges:
            if H.has_edge(a, b):
                edgelist.append((a, b))
            elif H.has_edge(b, a):
                edgelist.append((b, a))
        if edgelist:
            hi_edges = nx.draw_networkx_edges(
                H,
                pos,
                ax=ax,
                edgelist=edgelist,
                width=highlight_width,
                edge_color=highlight_color,
            )
            _set_zorder(hi_edges, 4)
            _set_capstyle(hi_edges, "round")

    _pad_axes_to_text(ax, pad=0.06)
    ax.axis("off")
    if label:
        ax.set_title(label, fontsize=12)
    if created_fig:
        plt.tight_layout()
    return ax


import matplotlib.pyplot as plt


def plot_logp_kde(
    dataset: str,
    lp: np.ndarray,  # dataset logP
    lg: np.ndarray,  # generated logP
    out: Path,  # output file path
    target: float | None = None,  # target logP (vertical line)
    description: str | None = None,
    *,
    bw_adjust: float = 0.9,
    figsize: tuple = (8, 5.5),
    evals_total: dict | None = None,  # e.g. {"validity_pct":"80%", "final_pct":"65%"}
    evals_valid: dict | None = None,  # e.g. {"mae_to_target":"0.41", "success_at_eps":"74%"}
    epsilon: float | None = None,  # if None -> 0.25 * std(dataset)
) -> None:
    """
    Plot KDEs of dataset vs generated logP; draw μ(dataset), μ/±σ(gen), target,
    and shade the band target±ε (ε = 0.25*σ(dataset) unless provided).
    Two metric panels are rendered: Total (left-bottom) and Valid (right-bottom).
    """
    # --- stats ---
    mean_ds, std_ds = lp.mean(), lp.std(ddof=1)
    mean_gen, std_gen = lg.mean(), lg.std(ddof=1)
    eps = epsilon if epsilon is not None else 0.25 * std_ds

    # --- style/axes ---
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=figsize)

    # KDEs
    sns.kdeplot(lp, fill=True, color="lightgrey", alpha=0.55, linewidth=0, bw_adjust=bw_adjust, ax=ax)
    sns.kdeplot(lg, fill=True, color="steelblue", alpha=0.85, linewidth=0, bw_adjust=bw_adjust, ax=ax)

    # μ(dataset) (no ±σ lines)
    ax.axvline(mean_ds, color="grey", linestyle="--", linewidth=1.4, zorder=5)

    # μ/±σ (gen)
    ax.axvline(mean_gen, color="steelblue", linestyle="--", linewidth=1.8, zorder=6)
    ax.axvline(mean_gen - std_gen, color="steelblue", linestyle=":", linewidth=1.2, zorder=6)
    ax.axvline(mean_gen + std_gen, color="steelblue", linestyle=":", linewidth=1.2, zorder=6)

    # target

    if target:
        # band around TARGET (for conditional eval parity)
        low, high = float(target) - eps, float(target) + eps
        pct_in_band = np.logical_and(lg >= low, lg <= high).mean() * 100.0
        # ε-band around TARGET
        ax.axvline(float(target), color="black", linestyle="-", linewidth=1.6, zorder=7)
        ax.axvspan(low, high, color="steelblue", alpha=0.18, zorder=4)

    # labels
    ax.set_xlabel("logP", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    sns.despine()
    fig.tight_layout()

    # --- legend ---
    legend_elements = [
        Patch(facecolor="lightgrey", edgecolor="none", alpha=0.55, label=f"{dataset} KDE (n={lp.size})"),
        Patch(facecolor="steelblue", edgecolor="none", alpha=0.85, label=f"Generated KDE (n={lg.size})"),
        Line2D(
            [0],
            [0],
            color="grey",
            linestyle="--",
            linewidth=1.4,
            label=f"μ {dataset} = {mean_ds:.2f} (±σ={std_ds:.2f})",
        ),
        Line2D([0], [0], color="steelblue", linestyle="--", linewidth=1.8, label=f"μ gen = {mean_gen:.2f}"),
        Line2D([0], [0], color="steelblue", linestyle=":", linewidth=1.2, label=f"±σ gen (σ={std_gen:.2f})"),
    ]

    if target:
        legend_elements.extend(
            [
                Line2D([0], [0], color="black", linestyle="-", linewidth=1.6, label=f"target = {float(target):.2f}"),
                Patch(
                    facecolor="steelblue",
                    edgecolor="none",
                    alpha=0.18,
                    label=f"target±ε (ε={eps:.2f}; 0.25·σ({dataset}))"
                    if epsilon is None
                    else f"target±ε (ε={eps:.2f})",
                ),
            ]
        )
    ax.legend(
        handles=legend_elements,
        frameon=False,
        ncol=1,
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0.5,
    )

    # --- title ---
    title = dataset if not description else f"{dataset} - {description}"
    ax.set_title(title, fontsize=13, pad=12)

    # --- metric panels: stacked bottom-left (TOTAL above, VALID below) ---
    if target:
        lines_total = ["TOTAL"]
        if evals_total:
            for k, v in evals_total.items():
                lines_total.append(f"{k}: {v}")

        lines_valid = ["VALID"]
        if evals_valid:
            for k, v in evals_valid.items():
                lines_valid.append(f"{k}: {v}")

        box_kwargs = {
            "transform": ax.transAxes,
            "ha": "left",
            "va": "bottom",
            "fontsize": 10,
            "color": "steelblue",
            "bbox": {"facecolor": "white", "edgecolor": "steelblue", "boxstyle": "round,pad=0.35", "alpha": 0.90},
        }

        # Draw VALID at the bottom
        y0 = 0.05
        ax.text(0.02, y0, "\n".join(lines_valid), **box_kwargs)

        # Draw TOTAL above VALID
        line_h = 0.045  # approx line height in axes coords
        y1 = y0 + line_h * max(3, len(lines_valid)) + 0.02  # small vertical gap
        ax.text(0.02, y1, "\n".join(lines_total), **box_kwargs)

    # save
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
