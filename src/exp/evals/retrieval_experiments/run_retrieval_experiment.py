"""
Retrieval Experiment: Encoding-Decoding Ablation Study

This script evaluates the encoding-decoding performance of HDC graph representations
with ablations over VSA models (HRR, MAP), dimensions, and depths.

Metrics:
- Edge decoding accuracy
- Correction percentage
- Final graph accuracy (exact match)
- Average cosine similarity
- Timing breakdown (encoding, edge decoding, graph decoding, total)

Usage:
    python run_retrieval_experiment.py --vsa HRR --hv_dim 1024 --depth 3 --dataset qm9
    python run_retrieval_experiment.py --vsa MAP --hv_dim 2048 --depth 4 --dataset zinc --iter_budget 10
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torchhd
from torch.utils.data import Subset
from torch_geometric import seed_everything
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import (
    QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG,
    ZINC_SMILES_HRR_1024_F64_5G1NG4_CONFIG,
    DSHDCConfig,
)
from src.encoding.graph_encoders import HyperNet
from src.encoding.the_types import VSAModel
from src.utils.utils import DataTransformer

seed_everything(seed=42)


def create_dynamic_config(base_dataset: str, vsa_model: str, hv_dim: int, depth: int) -> DSHDCConfig:
    """
    Create a dynamic DSHDCConfig for the given parameters.

    Parameters
    ----------
    base_dataset : str
        Either "qm9" or "zinc"
    vsa_model : str
        Either "HRR" or "MAP"
    hv_dim : int
        Hypervector dimension
    depth : int
        Message passing depth

    Returns
    -------
    DSHDCConfig
        Configuration object
    """
    if base_dataset == "qm9":
        # QM9 configuration: 4 features (atom_type, degree, formal_charge, total_num_Hs)
        ds_config = QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG
    else:  # zinc
        ds_config = ZINC_SMILES_HRR_1024_F64_5G1NG4_CONFIG

    ds_config.vsa = VSAModel(vsa_model)
    ds_config.hv_dim = hv_dim
    ds_config.hypernet_depth = depth
    ds_config.normalize = True
    return ds_config


def graphs_isomorphic(g1: nx.Graph, g2: nx.Graph) -> bool:
    """
    Check if two graphs are isomorphic with node feature matching.

    Handles different node attribute formats:
    - Format 1: {'feat': Feat(...), 'target_degree': ...}
    - Format 2: {'type': (atom_type, degree_idx, formal_charge_idx, explicit_hs)}
    """
    if g1.number_of_nodes() != g2.number_of_nodes() or g1.number_of_edges() != g2.number_of_edges():
        return False

    def normalize_node_features(node_attrs):
        """
        Normalize node attributes to tuple representation.

        Note: Feat.to_tuple() filters out falsy values (0, None), which is incorrect.
        We manually extract the tuple to preserve zeros.
        """
        if "feat" in node_attrs:
            # Format 1: Has Feat object
            feat = node_attrs["feat"]
            # Manual extraction to preserve zeros (to_tuple() filters them out incorrectly)
            if feat.is_in_ring is not None:
                # ZINC dataset (5 features)
                return (feat.atom_type, feat.degree_idx, feat.formal_charge_idx, feat.explicit_hs, feat.is_in_ring)
            # QM9 dataset (4 features)
            return (feat.atom_type, feat.degree_idx, feat.formal_charge_idx, feat.explicit_hs)
        if "type" in node_attrs:
            # Format 2: Has type tuple
            return node_attrs["type"]
        # Fallback: return None
        return None

    def node_match(n1, n2):
        """Match nodes by normalized feature tuples."""
        t1 = normalize_node_features(n1)
        t2 = normalize_node_features(n2)
        return t1 == t2 and t1 is not None

    try:
        return nx.is_isomorphic(g1, g2, node_match=node_match)
    except Exception:
        return False


def run_single_experiment(
    vsa_model: str,
    hv_dim: int,
    depth: int,
    dataset_name: str,
    iteration_budget: int,
    n_samples: int = 1000,
    output_dir: Path | None = None,
) -> dict:
    """
    Run a single retrieval experiment.

    Parameters
    ----------
    vsa_model : str
        VSA model name ("HRR" or "MAP")
    hv_dim : int
        Hypervector dimension
    depth : int
        Message passing depth
    dataset_name : str
        Dataset name ("qm9" or "zinc")
    iteration_budget : int
        Number of pattern matching iterations for decoding
    n_samples : int, optional
        Number of samples to evaluate
    output_dir : Path, optional
        Directory to save results

    Returns
    -------
    dict
        Dictionary containing all metrics
    """
    print(f"\n{'=' * 80}")
    print(
        f"Running experiment: VSA={vsa_model}, dim={hv_dim}, depth={depth}, dataset={dataset_name}, iter_budget={iteration_budget}"
    )
    print(f"{'=' * 80}\n")

    # Create configuration
    config = create_dynamic_config(dataset_name, vsa_model, hv_dim, depth)

    # Load dataset
    if dataset_name == "qm9":
        dataset = QM9Smiles(split="test")
    else:  # zinc
        dataset = ZincSmiles(split="test")

    # Initialize HyperNet
    hypernet = HyperNet(
        config=config,
        depth=depth,
        use_explain_away=True,
        use_edge_codebook=True,
    )
    hypernet.eval()
    hypernet.decoding_limit_for = config.base_dataset

    nodes_set = set(map(tuple, dataset.x.long().tolist()))
    hypernet.limit_nodes_codebook(limit_node_set=nodes_set)

    # Randomly sample n_samples from the dataset
    dataset_size = len(dataset)
    if n_samples > dataset_size:
        print(f"Warning: Requested n_samples ({n_samples}) > dataset size ({dataset_size}). Using full dataset.")
        sample_indices = list(range(dataset_size))
    else:
        # Random sampling without replacement
        sample_indices = np.random.choice(dataset_size, size=n_samples, replace=False).tolist()

    print(f"Sampled {len(sample_indices)} molecules from {dataset_size} total")

    # Create subset of dataset with sampled indices
    sampled_dataset = Subset(dataset, sample_indices)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hypernet = hypernet.to(device)

    # Decoder settings
    decoder_settings = {
        "iteration_budget": iteration_budget,
        "max_graphs_per_iter": 1024,
        "top_k": 1,  # We only need the best match
        "sim_eps": 0.0001,
        "early_stopping": False,  # No early stopping for fair comparison
    }

    # Phase 1: Batch Encoding
    # Use DataLoader for efficient batch encoding
    batch_size = 1024  # Adjust based on GPU memory
    dataloader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=False)

    print(f"Encoding {len(sampled_dataset)} samples in batches of {batch_size}...")

    encoded_results = []  # Store (pyg_data, edge_term, graph_term, encoding_time)

    for batch in tqdm(dataloader, desc="Encoding batches"):
        batch = batch.to(device)

        start_time = time.time()
        with torch.no_grad():
            encoding_output = hypernet.forward(batch)
        batch_encoding_time = time.time() - start_time

        # Store individual results from batch
        edge_terms = encoding_output["edge_terms"]
        graph_terms = encoding_output["graph_embedding"]

        # Split batch back to individual samples
        batch_list = batch.to_data_list()

        for i, pyg_data in enumerate(batch_list):
            encoded_results.append(
                {
                    "pyg_data": pyg_data,
                    "edge_term": edge_terms[i],
                    "graph_term": graph_terms[i],
                    "encoding_time": batch_encoding_time / len(batch_list),  # Approximate per-sample time
                }
            )

    print(f"Encoding complete. Processing {len(encoded_results)} samples for decoding...")

    # Metrics
    edge_accuracies = []
    correction_levels = []
    graph_accuracies = []
    cosine_similarities = []

    encoding_times = []
    edge_decoding_times = []
    graph_decoding_times = []
    total_times = []

    # Phase 2 & 3: Edge Decoding and Graph Decoding
    for result in tqdm(encoded_results, desc="Decoding samples"):
        pyg_data = result["pyg_data"]
        edge_term = result["edge_term"]
        graph_term = result["graph_term"]
        encoding_time = result["encoding_time"]

        encoding_times.append(encoding_time)

        # Convert PyG data to NetworkX for ground truth comparison
        nx_graph = DataTransformer.pyg_to_nx(pyg_data)

        # Phase 2: Edge Decoding
        start_time = time.time()
        with torch.no_grad():
            decoded_edges = hypernet.decode_order_one_no_node_terms(edge_term.clone())
        edge_decoding_time = time.time() - start_time
        edge_decoding_times.append(edge_decoding_time)

        # Compute edge accuracy using NetworkX graph
        # Real Data
        node_tuples = [tuple(i) for i in pyg_data.x.int().tolist()]
        original_edges = [(node_tuples[e[0]], node_tuples[e[1]]) for e in pyg_data.edge_index.t().int().cpu().tolist()]
        original_edges_counter = Counter(original_edges)
        decoded_edges_counter = Counter(decoded_edges)

        # Edge accuracy: intersection over union
        intersection = sum((original_edges_counter & decoded_edges_counter).values())
        union = sum((original_edges_counter | decoded_edges_counter).values())
        edge_accuracy = intersection / union if union > 0 else 0.0
        edge_accuracies.append(edge_accuracy)

        # Phase 3: Full Graph Decoding
        start_time = time.time()
        with torch.no_grad():
            decoding_result = hypernet.decode_graph(
                edge_term=edge_term,
                graph_term=graph_term,
                decoder_settings=decoder_settings,
            )
        graph_decoding_time = time.time() - start_time
        graph_decoding_times.append(graph_decoding_time)

        total_time = encoding_time + edge_decoding_time + graph_decoding_time
        total_times.append(total_time)

        # Record correction level
        correction_levels.append(decoding_result.correction_level.name)

        # Check graph accuracy (exact isomorphism)
        if len(decoding_result.nx_graphs) > 0:
            decoded_graph = decoding_result.nx_graphs[0]
            graph_match = graphs_isomorphic(nx_graph, decoded_graph)
            graph_accuracies.append(1.0 if graph_match else 0.0)

            # Compute cosine similarity by re-encoding the decoded graph
            pyg_decoded = DataTransformer.nx_to_pyg_with_type_attr(decoded_graph)
            batch_decoded = Batch.from_data_list([pyg_decoded]).to(device)
            with torch.no_grad():
                reencoded_output = hypernet.forward(batch_decoded)
            reencoded_graph_term = reencoded_output["graph_embedding"][0]

            cos_sim = torchhd.cos(graph_term, reencoded_graph_term).item()
            cosine_similarities.append(cos_sim)
        else:
            graph_accuracies.append(0.0)
            cosine_similarities.append(0.0)

    # Compute summary statistics
    correction_counter = Counter(correction_levels)
    correction_percentages = {k: v / len(correction_levels) * 100 for k, v in correction_counter.items()}

    results = {
        "vsa_model": vsa_model,
        "hv_dim": hv_dim,
        "depth": depth,
        "dataset": dataset_name,
        "iteration_budget": iteration_budget,
        "n_samples": len(sample_indices),
        # Accuracies
        "edge_accuracy_mean": np.mean(edge_accuracies),
        "edge_accuracy_std": np.std(edge_accuracies),
        "graph_accuracy_mean": np.mean(graph_accuracies),
        "graph_accuracy_std": np.std(graph_accuracies),
        # Correction statistics
        "correction_level_0_pct": correction_percentages.get("ZERO", 0.0),
        "correction_level_1_pct": correction_percentages.get("ONE", 0.0),
        "correction_level_2_pct": correction_percentages.get("TWO", 0.0),
        "correction_level_3_pct": correction_percentages.get("THREE", 0.0),
        "correction_level_fail_pct": correction_percentages.get("FAIL", 0.0),
        # Cosine similarity
        "cosine_similarity_mean": np.mean(cosine_similarities),
        "cosine_similarity_std": np.std(cosine_similarities),
        # Timing
        "encoding_time_mean": np.mean(encoding_times),
        "encoding_time_std": np.std(encoding_times),
        "edge_decoding_time_mean": np.mean(edge_decoding_times),
        "edge_decoding_time_std": np.std(edge_decoding_times),
        "graph_decoding_time_mean": np.mean(graph_decoding_times),
        "graph_decoding_time_std": np.std(graph_decoding_times),
        "total_time_mean": np.mean(total_times),
        "total_time_std": np.std(total_times),
    }

    # Save results
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        filename = f"{vsa_model}_{dataset_name}_dim{hv_dim}_depth{depth}_iter{iteration_budget}.json"
        with open(output_dir / filename, "w") as f:
            json.dump(results, f, indent=2)

        # Save detailed results
        detailed_results = pd.DataFrame(
            {
                "edge_accuracy": edge_accuracies,
                "graph_accuracy": graph_accuracies,
                "correction_level": correction_levels,
                "cosine_similarity": cosine_similarities,
                "encoding_time": encoding_times,
                "edge_decoding_time": edge_decoding_times,
                "graph_decoding_time": graph_decoding_times,
                "total_time": total_times,
            }
        )
        detailed_filename = f"{vsa_model}_{dataset_name}_dim{hv_dim}_depth{depth}_iter{iteration_budget}_detailed.csv"
        detailed_results.to_csv(output_dir / detailed_filename, index=False)

    # Print summary
    print("\nResults Summary:")
    print(f"  Edge Accuracy:        {results['edge_accuracy_mean']:.4f} ± {results['edge_accuracy_std']:.4f}")
    print(f"  Graph Accuracy:       {results['graph_accuracy_mean']:.4f} ± {results['graph_accuracy_std']:.4f}")
    print(f"  Cosine Similarity:    {results['cosine_similarity_mean']:.4f} ± {results['cosine_similarity_std']:.4f}")
    print(f"  Correction Level 0:   {results['correction_level_0_pct']:.2f}%")
    print(f"  Correction Level 1:   {results['correction_level_1_pct']:.2f}%")
    print(f"  Correction Level 2:   {results['correction_level_2_pct']:.2f}%")
    print(f"  Correction Level 3:   {results['correction_level_3_pct']:.2f}%")
    print(f"  Correction Level FAIL: {results['correction_level_fail_pct']:.2f}%")
    print(f"  Encoding Time:        {results['encoding_time_mean']:.4f} ± {results['encoding_time_std']:.4f} s")
    print(
        f"  Edge Decoding Time:   {results['edge_decoding_time_mean']:.4f} ± {results['edge_decoding_time_std']:.4f} s"
    )
    print(
        f"  Graph Decoding Time:  {results['graph_decoding_time_mean']:.4f} ± {results['graph_decoding_time_std']:.4f} s"
    )
    print(f"  Total Time:           {results['total_time_mean']:.4f} ± {results['total_time_std']:.4f} s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run retrieval experiment")
    parser.add_argument("--vsa", type=str, default="HRR", choices=["HRR", "MAP"], help="VSA model (default: HRR)")
    parser.add_argument("--hv_dim", type=int, default=1600, help="Hypervector dimension (default: 1600 for QM9)")
    parser.add_argument("--depth", type=int, default=3, help="Message passing depth (default: 3)")
    parser.add_argument(
        "--dataset", type=str, default="zinc", choices=["qm9", "zinc"], help="Dataset name (default: qm9)"
    )
    parser.add_argument("--iter_budget", type=int, default=1, help="Iteration budget for decoding (default: 1)")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to evaluate (default: 10)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory (default: ./results)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    results = run_single_experiment(
        vsa_model=args.vsa,
        hv_dim=args.hv_dim,
        depth=args.depth,
        dataset_name=args.dataset,
        iteration_budget=args.iter_budget,
        n_samples=args.n_samples,
        output_dir=output_dir,
    )

    # Append to summary CSV
    summary_csv = output_dir / "summary.csv"
    df = pd.DataFrame([results])
    if summary_csv.exists():
        df_existing = pd.read_csv(summary_csv)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(summary_csv, index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
