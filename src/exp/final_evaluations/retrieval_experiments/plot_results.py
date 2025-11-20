"""
Plot Results from Retrieval Experiments

This script generates publication-quality plots from the retrieval experiment results.

Usage:
    python plot_results.py --results_dir ./results --output_dir ./plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication-quality style
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def plot_accuracy_vs_dim_by_depth(df: pd.DataFrame, metric: str, output_path: Path, dataset: str, vsa: str, decoder: str):
    """
    Plot accuracy vs. hypervector dimension, with separate lines for each depth.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    metric : str
        Metric to plot (e.g., "graph_accuracy_mean")
    output_path : Path
        Output file path
    dataset : str
        Dataset name for title
    vsa : str
        VSA model name for title
    decoder : str
        Decoder type for filtering and title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter for specific dataset, VSA, and decoder
    df_filtered = df[(df["dataset"] == dataset) & (df["vsa_model"] == vsa) & (df["decoder"] == decoder)]

    # Get unique depths
    depths = sorted(df_filtered["depth"].unique())

    # Color palette
    colors = sns.color_palette("viridis", len(depths))

    # Plot each depth
    for i, depth in enumerate(depths):
        df_depth = df_filtered[df_filtered["depth"] == depth]
        df_depth = df_depth.sort_values("hv_dim")

        ax.plot(
            df_depth["hv_dim"],
            df_depth[metric],
            marker="o",
            markersize=8,
            linewidth=2,
            label=f"Depth {depth}",
            color=colors[i],
        )

        # Add error bars if std is available
        std_metric = metric.replace("_mean", "_std")
        if std_metric in df_depth.columns:
            ax.fill_between(
                df_depth["hv_dim"],
                df_depth[metric] - df_depth[std_metric],
                df_depth[metric] + df_depth[std_metric],
                alpha=0.2,
                color=colors[i],
            )

    ax.set_xlabel("Hypervector Dimension", fontsize=14, fontweight="bold")
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=14, fontweight="bold")
    ax.set_title(f"{dataset.upper()} - {vsa} - {decoder.replace('_', ' ').title()}", fontsize=16, fontweight="bold")
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_timing_breakdown(df: pd.DataFrame, output_path: Path, dataset: str, vsa: str, decoder: str):
    """
    Plot timing breakdown (stacked bar chart) for different configurations.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    output_path : Path
        Output file path
    dataset : str
        Dataset name for filtering
    vsa : str
        VSA model name for filtering
    decoder : str
        Decoder type for filtering
    """
    # Filter for specific dataset, VSA, and decoder
    df_filtered = df[(df["dataset"] == dataset) & (df["vsa_model"] == vsa) & (df["decoder"] == decoder)]

    # Select a few representative configurations
    df_filtered = df_filtered[df_filtered["depth"].isin([3, 4])]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    configs = []
    encoding_times = []
    edge_decoding_times = []
    graph_decoding_times = []

    for _, row in df_filtered.iterrows():
        config_label = f"D{row['hv_dim']}\nDepth{row['depth']}"
        configs.append(config_label)
        encoding_times.append(row["encoding_time_mean"])
        edge_decoding_times.append(row["edge_decoding_time_mean"])
        graph_decoding_times.append(row["graph_decoding_time_mean"])

    x = np.arange(len(configs))
    width = 0.6

    # Stacked bars
    p1 = ax.bar(x, encoding_times, width, label="Encoding", color="#1f77b4")
    p2 = ax.bar(x, edge_decoding_times, width, bottom=encoding_times, label="Edge Decoding", color="#ff7f0e")
    p3 = ax.bar(
        x,
        graph_decoding_times,
        width,
        bottom=np.array(encoding_times) + np.array(edge_decoding_times),
        label="Graph Decoding",
        color="#2ca02c",
    )

    ax.set_xlabel("Configuration", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time (seconds)", fontsize=14, fontweight="bold")
    ax.set_title(f"Timing Breakdown - {dataset.upper()} - {vsa} - {decoder.replace('_', ' ').title()}", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_correction_level_distribution(df: pd.DataFrame, output_path: Path, dataset: str, vsa: str, decoder: str):
    """
    Plot distribution of correction levels as a stacked bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    output_path : Path
        Output file path
    dataset : str
        Dataset name for filtering
    vsa : str
        VSA model name for filtering
    decoder : str
        Decoder type for filtering
    """
    # Filter for specific dataset, VSA, and decoder
    df_filtered = df[(df["dataset"] == dataset) & (df["vsa_model"] == vsa) & (df["decoder"] == decoder)]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    configs = []
    level_0 = []
    level_1 = []
    level_2 = []
    level_3 = []
    level_fail = []

    for _, row in df_filtered.iterrows():
        config_label = f"D{row['hv_dim']}\nDepth{row['depth']}"
        configs.append(config_label)
        level_0.append(row["correction_level_0_pct"])
        level_1.append(row["correction_level_1_pct"])
        level_2.append(row["correction_level_2_pct"])
        level_3.append(row["correction_level_3_pct"])
        level_fail.append(row["correction_level_fail_pct"])

    x = np.arange(len(configs))
    width = 0.6

    # Stacked bars
    p1 = ax.bar(x, level_0, width, label="Level 0 (No Correction)", color="#2ca02c")
    p2 = ax.bar(x, level_1, width, bottom=level_0, label="Level 1", color="#1f77b4")
    bottom = np.array(level_0) + np.array(level_1)
    p3 = ax.bar(x, level_2, width, bottom=bottom, label="Level 2", color="#ff7f0e")
    bottom += np.array(level_2)
    p4 = ax.bar(x, level_3, width, bottom=bottom, label="Level 3", color="#d62728")
    bottom += np.array(level_3)
    p5 = ax.bar(x, level_fail, width, bottom=bottom, label="FAIL", color="#7f7f7f")

    ax.set_xlabel("Configuration", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=14, fontweight="bold")
    ax.set_title(f"Correction Level Distribution - {dataset.upper()} - {vsa} - {decoder.replace('_', ' ').title()}", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10, rotation=45, ha="right")
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_heatmap_accuracy(df: pd.DataFrame, output_path: Path, dataset: str, vsa: str, decoder: str, metric: str):
    """
    Plot heatmap of accuracy with depth on y-axis and dimension on x-axis.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    output_path : Path
        Output file path
    dataset : str
        Dataset name for filtering
    vsa : str
        VSA model name for filtering
    decoder : str
        Decoder type for filtering
    metric : str
        Metric to plot
    """
    # Filter for specific dataset, VSA, and decoder
    df_filtered = df[(df["dataset"] == dataset) & (df["vsa_model"] == vsa) & (df["decoder"] == decoder)]

    # Pivot table for heatmap
    pivot_df = df_filtered.pivot_table(values=metric, index="depth", columns="hv_dim")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": metric.replace("_", " ").title()},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    ax.set_xlabel("Hypervector Dimension", fontsize=14, fontweight="bold")
    ax.set_ylabel("Depth", fontsize=14, fontweight="bold")
    ax.set_title(f"{dataset.upper()} - {vsa} - {decoder.replace('_', ' ').title()} - {metric.replace('_', ' ').title()}", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_iteration_budget_comparison(df: pd.DataFrame, output_path: Path, dataset: str, vsa: str, decoder: str):
    """
    Plot comparison of iteration budgets (for ZINC only).

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    output_path : Path
        Output file path
    dataset : str
        Dataset name (should be "zinc")
    vsa : str
        VSA model name for filtering
    decoder : str
        Decoder type for filtering
    """
    # Filter for specific dataset, VSA, and decoder
    df_filtered = df[(df["dataset"] == dataset) & (df["vsa_model"] == vsa) & (df["decoder"] == decoder)]

    # Get unique iteration budgets
    iter_budgets = sorted(df_filtered["iteration_budget"].unique())

    if len(iter_budgets) <= 1:
        print(f"Skipping iteration budget comparison (only {len(iter_budgets)} budget(s) found)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Graph accuracy vs dimension for different iteration budgets
    ax = axes[0]
    colors = sns.color_palette("Set2", len(iter_budgets))

    for i, iter_budget in enumerate(iter_budgets):
        df_iter = df_filtered[df_filtered["iteration_budget"] == iter_budget]
        df_iter = df_iter.sort_values("hv_dim")

        ax.plot(
            df_iter["hv_dim"],
            df_iter["graph_accuracy_mean"],
            marker="o",
            markersize=8,
            linewidth=2,
            label=f"Iter Budget {iter_budget}",
            color=colors[i],
        )

    ax.set_xlabel("Hypervector Dimension", fontsize=14, fontweight="bold")
    ax.set_ylabel("Graph Accuracy", fontsize=14, fontweight="bold")
    ax.set_title(f"Accuracy vs Iteration Budget - {dataset.upper()} - {vsa} - {decoder.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Plot 2: Timing vs dimension for different iteration budgets
    ax = axes[1]

    for i, iter_budget in enumerate(iter_budgets):
        df_iter = df_filtered[df_filtered["iteration_budget"] == iter_budget]
        df_iter = df_iter.sort_values("hv_dim")

        ax.plot(
            df_iter["hv_dim"],
            df_iter["graph_decoding_time_mean"],
            marker="s",
            markersize=8,
            linewidth=2,
            label=f"Iter Budget {iter_budget}",
            color=colors[i],
        )

    ax.set_xlabel("Hypervector Dimension", fontsize=14, fontweight="bold")
    ax.set_ylabel("Graph Decoding Time (s)", fontsize=14, fontweight="bold")
    ax.set_title(f"Timing vs Iteration Budget - {dataset.upper()} - {vsa} - {decoder.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_hit_rate_by_node_size_comparison(results_dir: Path, output_dir: Path, dataset: str, vsa: str, decoder: str):
    """
    Plot hit rate by node size for different configurations (dimensions and depths).

    Reads detailed CSV files and creates multi-line plots showing how hit rate varies
    with molecule size across different model configurations.

    Parameters
    ----------
    results_dir : Path
        Directory containing detailed CSV files
    output_dir : Path
        Directory to save plots
    dataset : str
        Dataset name ("qm9" or "zinc")
    vsa : str
        VSA model name ("HRR")
    decoder : str
        Decoder type ("pattern_matching" or "greedy")
    """
    # Find all detailed CSV files for this dataset/VSA/decoder combination
    pattern = f"{vsa}_{dataset}_*_{decoder}_detailed.csv"
    csv_files = list(results_dir.glob(pattern))

    if not csv_files:
        print(f"No detailed CSV files found for {dataset} - {vsa} - {decoder}")
        return

    # Parse configuration from filename and load data
    configs_data = []
    for csv_file in csv_files:
        # Extract dim, depth, iter from filename
        # Format: {vsa}_{dataset}_dim{dim}_depth{depth}_iter{iter}_{decoder}_detailed.csv
        parts = csv_file.stem.split("_")
        dim = None
        depth = None
        iter_budget = None
        found_decoder = None

        for i, part in enumerate(parts):
            if part.startswith("dim"):
                dim = int(part[3:])
            elif part.startswith("depth"):
                depth = int(part[5:])
            elif part.startswith("iter"):
                iter_budget = int(part[4:])
            elif part in ["pattern", "greedy"]:
                # Handle decoder name (might be split across parts for pattern_matching)
                if part == "pattern" and i + 1 < len(parts) and parts[i + 1] == "matching":
                    found_decoder = "pattern_matching"
                elif part == "greedy":
                    found_decoder = "greedy"

        if dim is None or depth is None or iter_budget is None or found_decoder != decoder:
            continue

        # Load detailed results
        df = pd.read_csv(csv_file)

        # Check if num_nodes column exists (backward compatibility)
        if "num_nodes" not in df.columns:
            print(f"Skipping {csv_file.name} - missing 'num_nodes' column (old result file)")
            continue

        # Compute hit rate by node size
        hit_rate_by_size = df.groupby("num_nodes")["graph_accuracy"].mean().reset_index()
        hit_rate_by_size.columns = ["num_nodes", "hit_rate"]

        configs_data.append(
            {
                "dim": dim,
                "depth": depth,
                "iter_budget": iter_budget,
                "hit_rate_by_size": hit_rate_by_size,
            }
        )

    if not configs_data:
        print(f"No valid data found for {dataset} - {vsa}")
        return

    # Group by iteration budget (for ZINC, which may have multiple iter budgets)
    iter_budgets = sorted(set(c["iter_budget"] for c in configs_data))

    for iter_budget in iter_budgets:
        configs_for_iter = [c for c in configs_data if c["iter_budget"] == iter_budget]

        # Sort by dim and depth for consistent coloring
        configs_for_iter.sort(key=lambda x: (x["dim"], x["depth"]))

        fig, ax = plt.subplots(figsize=(14, 7))

        # Generate colors
        colors = sns.color_palette("tab10", len(configs_for_iter))

        # Plot each configuration
        for i, config in enumerate(configs_for_iter):
            hit_rate_df = config["hit_rate_by_size"]
            label = f"D={config['dim']}, Depth={config['depth']}"

            ax.plot(
                hit_rate_df["num_nodes"],
                hit_rate_df["hit_rate"] * 100,  # Convert to percentage
                marker="o",
                markersize=6,
                linewidth=2,
                label=label,
                color=colors[i],
                alpha=0.8,
            )

        ax.set_xlabel("Number of Nodes (Molecule Size)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Hit Rate (%)", fontsize=14, fontweight="bold")
        title = f"Hit Rate by Molecular Size - {dataset.upper()} - {vsa} - {decoder.replace('_', ' ').title()}"
        if len(iter_budgets) > 1:
            title += f" (Iter Budget: {iter_budget})"
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.legend(loc="best", frameon=True, shadow=True, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        plt.tight_layout()

        # Save plot
        if len(iter_budgets) > 1:
            output_path = output_dir / f"{dataset}_{vsa}_{decoder}_hit_rate_by_node_size_iter{iter_budget}.pdf"
        else:
            output_path = output_dir / f"{dataset}_{vsa}_{decoder}_hit_rate_by_node_size.pdf"

        plt.savefig(output_path)
        plt.close()

        print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot retrieval experiment results")
    parser.add_argument("--results_dir", type=str, default="./to_report", help="Results directory")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Output directory for plots")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load summary CSV
    summary_csv = results_dir / "summary.csv"
    if not summary_csv.exists():
        print(f"Error: Summary CSV not found at {summary_csv}")
        return

    df = pd.read_csv(summary_csv)

    print(f"Loaded {len(df)} experiment results")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"VSA models: {df['vsa_model'].unique()}")
    print(f"Decoders: {df['decoder'].unique() if 'decoder' in df.columns else ['unknown']}")
    print(f"Dimensions: {sorted(df['hv_dim'].unique())}")
    print(f"Depths: {sorted(df['depth'].unique())}")

    # Generate plots for each dataset, VSA, and decoder combination
    for dataset in df["dataset"].unique():
        for vsa in df["vsa_model"].unique():
            # Check if decoder column exists (backward compatibility)
            decoders = df["decoder"].unique() if "decoder" in df.columns else ["pattern_matching"]

            for decoder in decoders:
                print(f"\nGenerating plots for {dataset.upper()} - {vsa} - {decoder}...")

                # Plot 1: Graph accuracy vs dimension by depth
                plot_accuracy_vs_dim_by_depth(
                    df,
                    metric="graph_accuracy_mean",
                    output_path=output_dir / f"{dataset}_{vsa}_{decoder}_graph_accuracy_vs_dim.pdf",
                    dataset=dataset,
                    vsa=vsa,
                    decoder=decoder,
                )

                # Plot 2: Edge accuracy vs dimension by depth
                plot_accuracy_vs_dim_by_depth(
                    df,
                    metric="edge_accuracy_mean",
                    output_path=output_dir / f"{dataset}_{vsa}_{decoder}_edge_accuracy_vs_dim.pdf",
                    dataset=dataset,
                    vsa=vsa,
                    decoder=decoder,
                )

                # Plot 3: Cosine similarity vs dimension by depth
                plot_accuracy_vs_dim_by_depth(
                    df,
                    metric="cosine_similarity_mean",
                    output_path=output_dir / f"{dataset}_{vsa}_{decoder}_cosine_similarity_vs_dim.pdf",
                    dataset=dataset,
                    vsa=vsa,
                    decoder=decoder,
                )

                # Plot 4: Timing breakdown
                plot_timing_breakdown(
                    df,
                    output_path=output_dir / f"{dataset}_{vsa}_{decoder}_timing_breakdown.pdf",
                    dataset=dataset,
                    vsa=vsa,
                    decoder=decoder,
                )

                # Plot 5: Correction level distribution
                plot_correction_level_distribution(
                    df,
                    output_path=output_dir / f"{dataset}_{vsa}_{decoder}_correction_levels.pdf",
                    dataset=dataset,
                    vsa=vsa,
                    decoder=decoder,
                )

                # Plot 6: Heatmap of graph accuracy
                plot_heatmap_accuracy(
                    df,
                    output_path=output_dir / f"{dataset}_{vsa}_{decoder}_heatmap_graph_accuracy.pdf",
                    dataset=dataset,
                    vsa=vsa,
                    decoder=decoder,
                    metric="graph_accuracy_mean",
                )

                # Plot 7: Iteration budget comparison (ZINC only)
                if dataset == "zinc":
                    plot_iteration_budget_comparison(
                        df,
                        output_path=output_dir / f"{dataset}_{vsa}_{decoder}_iteration_budget_comparison.pdf",
                        dataset=dataset,
                        vsa=vsa,
                        decoder=decoder,
                    )

                # Plot 8: Hit rate by node size comparison
                plot_hit_rate_by_node_size_comparison(
                    results_dir=results_dir,
                    output_dir=output_dir,
                    dataset=dataset,
                    vsa=vsa,
                    decoder=decoder,
                )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
