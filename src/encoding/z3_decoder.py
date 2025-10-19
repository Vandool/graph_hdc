import os
from collections import Counter, defaultdict

import z3

# Z3 Configuration
z3.set_param("parallel.enable", True)
z3.set_param("parallel.threads.max", os.cpu_count())
z3.set_param("auto_config", False)
z3.set_param("smt.random_seed", 42)  # any int; try a few if you re-run
z3.set_param("sat.phase", "random")  # diversify SAT polarity


def deg(node_type: tuple[int, int, int, int]) -> int:
    """
    Return the per-vertex target degree (undirected) for a given node type.
    """
    return node_type[1] + 1


def normalize_instance(
    nodes_multiset: list[tuple[int, int, int, int]],
    edges_multiset: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
) -> dict:
    """
    Normalize the user input:
    - Map each distinct node type (4-tuple) -> type_id in a deterministic order.
    - Produce a canonical node order: types sorted by (deg(type), count, type_tuple),
      and nodes grouped by type in that order.
    - Build:
        n: int
        T: int
        types: list[tuple] (type_id -> type_tuple)
        type_of: list[int] (vertex u -> type_id)
        deg_of_type: list[int] (type_id -> per-vertex degree)
        M: list[list[int]] (type-pair edge quotas; symmetric; M[i][i] allowed; no self-loops on vertices)
        allowed: list[list[bool]] (u,v pair allowed ⇔ M[type[u]][type[v]] > 0 and u!=v)
    """
    # Count nodes per type and compute per-type degrees
    node_type_counts = Counter(nodes_multiset)
    all_types = sorted(node_type_counts.keys())  # temporary order

    # per-type degree map (using your deg())
    deg_map = {t: deg(t) for t in all_types}

    # Canonical type order (stabilizes enumeration and helps symmetry breaking)
    types_sorted = sorted(all_types, key=lambda t: (deg_map[t], node_type_counts[t], t))
    type_id = {t: i for i, t in enumerate(types_sorted)}
    T = len(types_sorted)

    # Canonical vertex order: group by type in types_sorted order
    type_of = []
    for t in types_sorted:
        type_of.extend([type_id[t]] * node_type_counts[t])
    n = len(type_of)
    types = types_sorted
    deg_of_type = [deg_map[t] for t in types]

    # Build-directed counts of type-pair edges from edges_multiset
    dir_count = Counter((type_id[a], type_id[b]) for (a, b) in edges_multiset)

    # Convert to undirected quotas M (symmetric).
    # Assumption per your spec: for i != j, edges_multiset contains both (i,j) and (j,i) equally.
    # So M[i][j] = dir_count[(i, j)] for i!=j, and assert symmetry.
    # For i == j (intra-type edges between distinct vertices), each undirected edge
    # would appear twice as (t,t), so use half.
    M = [[0] * T for _ in range(T)]
    for i in range(T):
        # intra-type
        same = dir_count.get((i, i), 0)
        if same % 2 != 0:
            raise ValueError("edges_multiset has odd count for (t,t); expected even (both directions).")
        M[i][i] = same // 2
        for j in range(i + 1, T):
            a = dir_count.get((i, j), 0)
            b = dir_count.get((j, i), 0)
            if a != b:
                raise ValueError(f"edges_multiset not symmetric for type pair ({i},{j}): {a} vs {b}")
            M[i][j] = M[j][i] = a

    # Quick feasibility sanity (optional but useful):
    # Sum of degrees == 2 * total edges
    total_edges = sum(M[i][j] for i in range(T) for j in range(i, T))
    if sum(deg_of_type[type_of[u]] for u in range(n)) != 2 * total_edges:
        # TODO: Maybe we can apply some correction here and still get a feasible solution?
        raise ValueError("Handshake check failed: sum degrees != 2 * |E| implied by M.")

    # Allowed mask at vertex level: allowed[u][v] iff M[type[u]][type[v]] > 0 and u!=v
    allowed = [[False] * n for _ in range(n)]
    for u in range(n):
        tu = type_of[u]
        for v in range(n):
            if u == v:
                continue
            tv = type_of[v]
            allowed[u][v] = M[tu][tv] > 0

    # Optional capacity sanity (uncomment if needed):
    # - Intra-type: M[i][i] ≤ C(|T_i|, 2)
    # - Inter-type: M[i][j] ≤ |T_i| * |T_j|
    # These checks reject impossible quota requests early, saving solver time.

    return {"n": n, "T": T, "types": types, "type_of": type_of, "deg_of_type": deg_of_type, "M": M, "allowed": allowed}


def add_graph_variables(n: int):
    """
    Create Z3 variables:
    - x[u][v] Bool for u<v, mirrored for v>u. x[u][u] is False (no self loops).
    - f[u][v] Int flow variable (0..n-1) for connectivity on directed arcs (u!=v).
    """
    x = [[None] * n for _ in range(n)]
    f = [[None] * n for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u < v:
                x[u][v] = z3.Bool(f"x_{u}_{v}")
                x[v][u] = x[u][v]  # alias for symmetry
            elif u == v:
                x[u][v] = False
            # flow vars (directed)
            if u != v:
                f[u][v] = z3.Int(f"f_{u}_{v}")
    return x, f


def constrain_disallowed_pairs(s: z3.Solver, n: int, x, allowed: list[list[bool]]):
    """Force disallowed edge variables to False."""
    # Only the upper triangle (u < v) is iterated to avoid duplicate constraints on aliases.
    # If a type–pair (type(u), type(v)) admits zero remaining capacity in M, the vertex-level
    # edge variable is hard-fixed to False. This shrink-wraps the search space significantly.
    for u in range(n):
        for v in range(u + 1, n):
            if not allowed[u][v]:
                s.add(x[u][v] == False)


def constrain_degrees(s: z3.Solver, n: int, x, type_of: list[int], deg_of_type: list[int]):
    """Per-vertex degree equals the per-type degree (pseudo-Boolean equality)."""
    # A pseudo-Boolean equality is used instead of integer Sum(If(..)) to keep the problem in
    # the PB fragment. Each row’s literals are directly equated to the target degree.
    for u in range(n):
        deg_u = deg_of_type[type_of[u]]
        lits = [x[min(u, v)][max(u, v)] for v in range(n) if v != u]
        s.add(z3.PbEq([(lit, 1) for lit in lits], deg_u))


def constrain_type_pair_quotas(s: z3.Solver, n: int, x, type_of: list[int], T: int, M: list[list[int]]):
    """
    Exact type-pair quotas, robust to empty literal sets (PB equalities).
    """
    # Vertices are bucketed by type once to avoid repeated scans.
    idx = [[] for _ in range(T)]
    for u, ti in enumerate(type_of):
        idx[ti].append(u)

    # Intra-type quotas: sum of upper-triangle literals among vertices in T_i equals M[i][i].
    for i in range(T):
        Ui = idx[i]
        lits = []
        for a in range(len(Ui)):
            for b in range(a + 1, len(Ui)):
                u, v = Ui[a], Ui[b]
                lits.append(x[min(u, v)][max(u, v)])
        rhs = M[i][i]
        if lits:
            s.add(z3.PbEq([(lit, 1) for lit in lits], rhs))
        # No admissible intra-type pairs exist; a non-zero quota would be impossible.
        elif rhs != 0:
            s.add(z3.BoolVal(False))

    # Inter-type quotas: all cross pairs (u in T_i, v in T_j) are collected and equated to M[i][j].
    for i in range(T):
        for j in range(i + 1, T):
            Ui, Uj = idx[i], idx[j]
            lits = []
            for u in Ui:
                for v in Uj:
                    a, b = (u, v) if u < v else (v, u)
                    lits.append(x[a][b])
            rhs = M[i][j]
            if lits:
                s.add(z3.PbEq([(lit, 1) for lit in lits], rhs))
            elif rhs != 0:
                s.add(z3.BoolVal(False))


def constrain_connectivity_flow(s: z3.Solver, n: int, x, f, root: int = 0):
    """
    Enforce connectedness via a single-commodity flow model.
    """
    # ---------------------------------------------------------------------------
    # Capacities
    # ----------
    # Each directed flow variable is bounded by (n-1) * x[u][v]; this gates flow to
    # existing edges and prevents flow through absent edges. The upper bound (n-1) is
    # sufficient because at most (n-1) units need to cross any cut in a connected graph.
    #
    # Balances
    # --------
    # A single source at ``root`` injects (n-1) units of flow. Every other node demands 1.
    # This ensures reachability to all vertices and forbids disconnected components.
    #
    # Complexity note
    # ---------------
    # The model introduces O(n^2) Int variables but typically yields robust pruning behavior
    # compared to ad-hoc cut constraints. It also avoids the need for exponential families
    # of cuts or iterative separation.
    # ---------------------------------------------------------------------------
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            s.add(f[u][v] >= 0)
            s.add(f[u][v] <= (n - 1) * z3.If(x[min(u, v)][max(u, v)], 1, 0))

    def outflow(u):
        return z3.Sum([f[u][v] for v in range(n) if v != u])

    def inflow(u):
        return z3.Sum([f[v][u] for v in range(n) if v != u])

    s.add(outflow(root) - inflow(root) == n - 1)
    for w in range(n):
        if w == root:
            continue
        s.add(inflow(w) - outflow(w) == 1)


def constrain_within_type_lex(s: z3.Solver, n: int, x, type_of: list[int]):
    """
    Symmetry breaking within each type via lexicographic ordering of adjacency rows.
    """
    # ---------------------------------------------------------------------------
    # Rationale
    # ---------
    # Vertices within the same type are indistinguishable. Without symmetry breaking,
    # permutations of those vertices would generate isomorphic duplicates. A simple and
    # effective breaker is to impose lexicographically non-increasing order on the
    # adjacency rows of vertices within each type (columns are in fixed canonical order).
    #
    # Implementation
    # --------------
    # For a type bucket [u0, u1, ..., uk-1], enforce:
    #   Row(u0) >=_lex Row(u1) >=_lex ... >=_lex Row(uk-1)
    # where Row(u) is the 0/1 vector of incident edges in the fixed column order excluding u.
    # The lex comparator is expressed as a disjunction over the first differing coordinate.
    # ---------------------------------------------------------------------------
    cols = list(range(n))

    def row_bits(u):
        # Construct the row as a 0/1 Int vector using If-then-else on the Bool literals,
        # skipping the diagonal. Column order is fixed as range(n) to stabilize lex ordering.
        return [z3.If(x[min(u, v)][max(u, v)], 1, 0) for v in cols if v != u]

    def lex_geq(A, B):
        # Enforce A >=_lex B by expressing the “first difference” schema:
        #   (A0 > B0) OR (A0==B0 AND A1 > B1) OR ... OR (A0..A_{k-2}==B0..B_{k-2} AND A_{k-1} >= B_{k-1})
        k = len(A)
        disj = []
        for i in range(k):
            equal_prefix = [A[t] == B[t] for t in range(i)]
            if i < k - 1:
                disj.append(z3.And(*([*equal_prefix, A[i] > B[i]])))
            else:
                disj.append(z3.And(*([*equal_prefix, A[i] >= B[i]])))
        return z3.Or(*disj)

    verts_by_type = defaultdict(list)
    for u, t in enumerate(type_of):
        verts_by_type[t].append(u)

    for _, verts in verts_by_type.items():
        for i in range(len(verts) - 1):
            u, v = verts[i], verts[i + 1]
            s.add(lex_geq(row_bits(u), row_bits(v)))


def enumerate_graphs(
    nodes_multiset: list[tuple],
    edges_multiset: list[tuple[tuple, tuple]],
    max_solutions: int = 10,
) -> list[dict]:
    """
    Enumerate all possible graph realizations for a given node and edge multiset
    under the defined connectivity and symmetry constraints.
    :return: A list of dictionaries containing the node metadata and edge connectivity.
                The key "ordered_nodes" contains the node metadata in canonical order.
                The key "associated_edge_idxs" contains the edge connectivity in adjacency-list form.
    :rtype: List[dict]
    """
    # ---------------------------------------------------------------------------
    # Instance Normalization
    # ----------------------
    # The user-facing multisets of nodes and typed edges are first normalized into
    # canonical solver data structures. The normalization step assigns type IDs,
    # derives per-type degrees, computes the type–pair quota matrix M, and builds
    # the vertex-level "allowed" adjacency mask that defines which edge variables
    # are permitted to exist in the Boolean model.
    #
    # This step guarantees deterministic vertex ordering and isomorphism-invariant
    # input for the solver, ensuring that all enumerated solutions are canonical.
    # ---------------------------------------------------------------------------
    inst = normalize_instance(nodes_multiset, edges_multiset)
    n = inst["n"]  # Number of nodes
    T = inst["T"]  # Number of node types
    type_of = inst["type_of"]  # Node type index for each node
    deg_of_type = inst["deg_of_type"]  # Per-type degree for each node type
    M = inst["M"]  # Type-pair quota matrix
    allowed = inst["allowed"]  # Adjacency mask for allowed edges
    types = inst["types"]  # Node type index for each node in canonical order

    # ---------------------------------------------------------------------------
    # Solver and Variable Setup
    # -------------------------
    # A fresh Z3 solver is instantiated. The graph structure is described by:
    #   - Boolean edge variables x[u][v], aliased for u<v to represent an undirected edge.
    #   - Integer flow variables f[u][v], used exclusively by the connectivity model.
    #
    # Constraints are added modularly:
    #   1) Disallowed edge pairs are fixed to False.
    #   2) Per-vertex degree equality constraints (PB equalities).
    #   3) Type–pair edge quota equalities (PB equalities).
    #   4) Single-commodity flow constraints to ensure connectivity.
    #   5) Lexicographic symmetry-breaking constraints within each node type.
    #
    # The resulting SMT instance captures exactly the admissible undirected, connected,
    # isomorph-free graphs satisfying the given type–pair quotas and per-type degrees.
    # ---------------------------------------------------------------------------
    # Solver specific for our problem: QF-LIA(+PB)
    s = z3.SolverFor("QF_LIA")

    x, f = add_graph_variables(n)

    constrain_disallowed_pairs(s, n, x, allowed)
    constrain_degrees(s, n, x, type_of, deg_of_type)
    constrain_type_pair_quotas(s, n, x, type_of, T, M)
    constrain_connectivity_flow(s, n, x, f, root=0)
    constrain_within_type_lex(s, n, x, type_of)

    # ---------------------------------------------------------------------------
    # Model Enumeration Loop
    # ----------------------
    # The solver is called repeatedly to get distinct satisfying assignments.
    # Each assignment corresponds to one possible graph structure.
    #
    # For every model:
    #   - Extract the Boolean values of x[u][v] for u<v to build the edge list.
    #   - Duplicate each undirected edge as [u,v] and [v,u] for PyG compatibility.
    #   - Build a blocking clause that inverts at least one x[u][v] decision, ensuring
    #     the next model represents a different graph (standard SAT enumeration pattern).
    #
    # Enumeration stops after `max_solutions` solutions or once the solver returns unsat.
    # ---------------------------------------------------------------------------
    unique_final_graphs = []
    while len(unique_final_graphs) < max_solutions and s.check() == z3.sat:
        m = s.model()

        # Edge extraction: collect all edges marked True in the current model.
        # The canonical vertex order from normalization is used consistently.
        edges_idx = []
        block = []
        for u in range(n):
            for v in range(u + 1, n):
                val = z3.is_true(m[x[u][v]])
                if val:
                    # Both directions are included to simplify PyG edge_index construction.
                    edges_idx.append([u, v])
                    edges_idx.append([v, u])
                # Build the blocking literal: the negation pattern ensures the next model
                # flips at least one decision among the current edge bits.
                block.append(x[u][v] if not val else z3.Not(x[u][v]))

        # Vertex annotation: reconstruct ordered node tuples from type indices.
        # The order is canonical and independent of solver search.
        ordered_nodes = [types[type_of[u]] for u in range(n)]

        # Append the current model output to the result list. Each entry contains both
        # node metadata and edge connectivity in adjacency-list form.
        unique_final_graphs.append({"ordered_nodes": ordered_nodes, "associated_edge_idxs": edges_idx})

        # Blocking clause: prevents the solver from reproducing the same model.
        # Each disjunct corresponds to flipping one previously decided edge literal.
        s.add(z3.Or(block))

    # The accumulated list of graph realizations is returned. Each element represents
    # a unique, connected, isomorph-free configuration consistent with the given type
    # and quota constraints.
    if len(unique_final_graphs) == 0:
        print("No solution found.")
    return unique_final_graphs


if __name__ == "__main__":
    import random
    import time
    from pathlib import Path

    import torch
    import torchhd
    import z3
    from matplotlib import pyplot as plt
    from torch_geometric.data import Batch, Data

    from src.datasets.qm9_smiles_generation import QM9Smiles
    from src.datasets.zinc_smiles_generation import ZincSmiles
    from src.encoding.configs_and_constants import (
        QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG,
        ZINC_SMILES_HRR_6144_G1G4_CONFIG,
    )
    from src.encoding.graph_encoders import load_or_create_hypernet
    from src.utils.chem import draw_mol
    from src.utils.utils import DataTransformer
    from src.utils.visualisations import draw_nx_with_atom_colorings

    qm9 = QM9Smiles(split="train")
    zinc = ZincSmiles(split="train")

    sorted_qm9 = sorted(qm9, key=lambda x: len(x.x))
    sorted_zinc = sorted(zinc, key=lambda x: len(x.x))

    samples = {
        "debug1": (qm9[0], "QM9Smiles"),
        "debug2": (qm9[102], "QM9Smiles"),
        "debug3": (qm9[103], "QM9Smiles"),
        "small_1": (sorted_qm9[100], "QM9Smiles"),
        "small_2": (sorted_qm9[10000], "QM9Smiles"),
        "small_3": (sorted_qm9[-1], "QM9Smiles"),
        "large_1": (sorted_zinc[1000], "ZincSmiles"),
        "large_2": (zinc[10], "ZincSmiles"),
        "large_3": (sorted_zinc[-100000], "ZincSmiles"),
        "large_4": (sorted_zinc[-5000], "ZincSmiles"),
        "large_5": (sorted_zinc[-1000], "ZincSmiles"),
        "large_6": (ZincSmiles(split="test")[10], "ZincSmiles"),
        "large_7": (sorted_zinc[-1], "ZincSmiles"),
    }
    examples = []
    out = Path() / "plots"
    out.mkdir(exist_ok=True, parents=True)
    for i, (name, (data, dataset)) in enumerate(samples.items()):
        res = {}
        nx_g = DataTransformer.pyg_to_nx(data)
        draw_nx_with_atom_colorings(nx_g, dataset=dataset)
        plt.savefig(out / f"{name}_nx.png")
        plt.show()
        mol, _ = DataTransformer.nx_to_mol_v2(nx_g, dataset="qm9" if dataset == "QM9Smiles" else "zinc")
        draw_mol(mol=mol, save_path=str(out / f"{name}_mol.png"), fmt="png")
        node_tuples = [tuple(i) for i in data.x.int().tolist()]
        edge_idxs = [tuple(e) for e in data.edge_index.t().cpu().int().tolist()]
        edge_tuples = [(node_tuples[u], node_tuples[v]) for u, v in edge_idxs]

        # Input
        print(f"{name}: |V|:{nx_g.number_of_nodes()} |E|:{nx_g.number_of_edges()}============================")
        for max_sol in [1, 10, 100]:
            t0 = time.perf_counter()
            solutions = enumerate_graphs(
                nodes_multiset=[tuple(e) for e in random.sample(node_tuples, k=len(node_tuples))],
                edges_multiset=edge_tuples,
                max_solutions=max_sol,
            )
            t_enum = time.perf_counter() - t0

            def to_pyg_data(ordered_nodes, edge_indexes):
                return Data(
                    x=torch.tensor(ordered_nodes, dtype=torch.float),
                    edge_index=(torch.tensor(edge_indexes, dtype=torch.long).t().contiguous()),
                )

            pygs = [to_pyg_data(sol["ordered_nodes"], sol["associated_edge_idxs"]) for sol in solutions]

            ds_config = (
                QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG
                if dataset == "QM9Smiles"
                else ZINC_SMILES_HRR_6144_G1G4_CONFIG
            )
            hypernet = load_or_create_hypernet(cfg=ds_config, do_print=False).to(torch.device("cpu")).eval()

            target_hdc = hypernet.forward(Batch.from_data_list([data]))["graph_embedding"]
            decoded_hdc = hypernet.forward(Batch.from_data_list(pygs))["graph_embedding"]
            sims_t = torchhd.cos(target_hdc, decoded_hdc).flatten()

            # Top-k (largest first)
            k = min(3, sims_t.numel())
            topk = torch.topk(sims_t, k=k)
            top_idxs = topk.indices.tolist()
            top_vals = topk.values.tolist()
            print(f"Enumerated {len(solutions)} graphs in {t_enum:.2f}s - Best cos sim: {float(top_vals[0]):.2f}")
            # Plot
            fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), constrained_layout=True)
            if k == 1:
                axes = [axes]

            for ax, idx, val in zip(axes, top_idxs, top_vals, strict=False):
                pyg = pygs[idx]
                nx_g = DataTransformer.pyg_to_nx(pyg)

                plt.sca(ax)  # draw_* uses current axes
                draw_nx_with_atom_colorings(nx_g, dataset=dataset, label=f"[{name}-{idx}] sim {float(val):.2f}")
                ax.set_axis_off()

            plt.show()

# Debug
# nodes: [(0, 1, 0, 1), (0, 1, 0, 1), (0, 2, 0, 0), (2, 0, 0, 0), (2, 0, 0, 0), (2, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)]
# edgges: [((0, 1, 0, 1), (0, 2, 0, 0)), ((0, 2, 0, 0), (0, 1, 0, 1)), ((0, 1, 0, 1), (2, 0, 0, 0)), ((2, 0, 0, 0), (0, 1, 0, 1)), ((1, 0, 0, 0), (1, 0, 0, 0)), ((1, 0, 0, 0), (1, 0, 0, 0)), ((0, 1, 0, 1), (0, 2, 0, 0)), ((0, 2, 0, 0), (0, 1, 0, 1)), ((0, 1, 0, 0), (0, 2, 0, 0)), ((0, 2, 0, 0), (0, 1, 0, 0)), ((0, 1, 0, 0), (2, 0, 0, 0)), ((2, 0, 0, 0), (0, 1, 0, 0)), ((0, 1, 0, 1), (2, 0, 0, 0)), ((2, 0, 0, 0), (0, 1, 0, 1))]
