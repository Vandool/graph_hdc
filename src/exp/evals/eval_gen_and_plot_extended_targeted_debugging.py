import argparse
import os
import random
from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import seed_everything

from src.datasets.utils import get_split
from src.encoding.configs_and_constants import (
    SupportedDataset,
)
from src.encoding.corrections import correct
from src.encoding.graph_encoders import get_node_counter_corrective, get_node_counter_fp, target_reached
from src.generation.generation import HDCGenerator
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, pick_device

# --- scientific paper style ---
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.size": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

# keep it modest to avoid oversubscription; tune if needed
num = max(1, min(8, os.cpu_count() or 1))
torch.set_num_threads(num)
torch.set_num_interop_threads(max(1, min(2, num)))  # coordination threads

# (optional but often helps BLAS backends)
os.environ.setdefault("OMP_NUM_THREADS", str(num))
os.environ.setdefault("MKL_NUM_THREADS", str(num))

seed = 42
seed_everything(seed)
device = pick_device()
EVALUATOR = None

DTYPE = torch.float32


def eval_generation(
    ds: SupportedDataset, n_samples: int, gen_mod_hint: str, *, draw: bool, plot: bool, out_suffix: str = ""
) -> dict[str, Any]:
    global EVALUATOR
    base_dataset = ds.default_cfg.base_dataset
    generator = HDCGenerator(gen_model_hint=gen_mod_hint, ds_config=ds.default_cfg, device=device, dtype=DTYPE)
    # generator = HDCZ3Generator(gen_model_hint=gen_mod_hint, ds_config=QM9_SMILES_HRR_1600_CONFIG_F64, device=device)

    samples = generator.get_raw_samples(n_samples=n_samples)
    sampled_edge_terms = samples["edge_terms"].as_subclass(ds.default_cfg.vsa.tensor_class)

    training_edge_terms = get_split("train", ds_config=ds.default_cfg)
    random_samples_idxs = random.sample(range(len(training_edge_terms)), n_samples)
    random_edge_terms = (
        training_edge_terms.index_select(random_samples_idxs)
        .edge_terms.view(n_samples, ds.default_cfg.hv_dim)
        .to(device=device, dtype=DTYPE)
    )

    hypernet = generator.hypernet

    norms_edge_terms_data = []  # list of norms
    norms_edge_terms_sampled = []  # list of norms
    norms_progress_data = []  # list of list of norms
    norms_progress_sampled = []  # list of list of norms
    norms_progress_corrected_1 = []  # list of list of norms
    sims_progress_data = []
    sims_progress_samples = []
    sims_progress_corrected_1 = []
    number_of_not_complete_nodes = []
    number_of_not_complete_nodes_after_bad_correction = []
    number_of_not_complete_nodes_after_better_correction = []
    complete_count = 0
    after_correction_complete_count = 0
    decoded_sets_counter = Counter()
    for i in range(n_samples):
        # Data
        norms_edge_terms_data.append(random_edge_terms[i].norm().item())
        decoded_edges_d, norms_, sims = hypernet.decode_order_one_no_node_terms(
            random_edge_terms[i].clone(), debug=True
        )
        norms_progress_data.append(norms_)
        sims_progress_data.append(sims)
        node_counter_d = get_node_counter_corrective(decoded_edges_d)
        node_counter_d_ft = get_node_counter_fp(decoded_edges_d)

        # Sampled
        norms_edge_terms_sampled.append(sampled_edge_terms[i].norm().item())
        decoded_edges_s, norms_, sims = hypernet.decode_order_one_no_node_terms(
            sampled_edge_terms[i].clone(), debug=True
        )
        norms_progress_sampled.append(norms_)
        sims_progress_samples.append(sims)

        # Correction
        node_counter_s = get_node_counter_corrective(decoded_edges_s)
        node_counter_s_fp = get_node_counter_fp(decoded_edges_s)
        not_complete_node = sum([(v - float(int(v)) != 0.0) for v in node_counter_s_fp.values()])
        number_of_not_complete_nodes.append(not_complete_node)
        corrected_edge_sets = []
        if not target_reached(decoded_edges_s):
            # First try to add/remove from the initial decoding
            corrected_edge_sets = correct(node_counter_s_fp, decoded_edges_s)

            # if failed perform a corrective decoding and repeat corrections
            if len(corrected_edge_sets) == 0:
                # for method in ["ceil", "round", "max_round"]:
                node_counter_s = get_node_counter_corrective(decoded_edges_s, method=method)
                decoded_edges_c, norms_, sims = hypernet.decode_order_one(
                    edge_term=random_edge_terms[i].clone(), node_counter=node_counter_s, debug=True
                )
                norms_progress_corrected_1.append(norms_)
                sims_progress_corrected_1.append(sims)
                node_counter_c = get_node_counter_corrective(decoded_edges_c)
                node_counter_c_fp = get_node_counter_fp(decoded_edges_c)
                not_complete_node_c = sum([(v - float(int(v)) != 0.0) for v in node_counter_c_fp.values()])
                number_of_not_complete_nodes_after_bad_correction.append(not_complete_node_c)
                if target_reached(decoded_edges_c):
                    corrected_edge_sets = [decoded_edges_c]
                    complete_count += 1
                else:
                    corrected_edge_sets = correct(node_counter_c_fp, decoded_edges_c)
            after_correction_complete_count += 1 if len(corrected_edge_sets) > 0 else 0
        else:
            corrected_edge_sets = [decoded_edges_s]
            complete_count += 1
            after_correction_complete_count += 1

        if len(corrected_edge_sets) == 0:
            correct(node_counter_c_fp, decoded_edges_c)
            print("DEBUG")

        decoded_sets_counter[len(corrected_edge_sets)] += 1

    print(
        f"Complete: {(complete_count / n_samples):.2f}, after correction: {(after_correction_complete_count / n_samples):.2f}"
    )

    base_dir = GLOBAL_ARTEFACTS_PATH / "decoding_debugging_plots_3"
    base_dir.mkdir(parents=True, exist_ok=True)

    # ---- plotting ----
    from pathlib import Path

    import matplotlib.pyplot as plt

    def _savefig(fig, path: Path):
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.show()
        plt.close(fig)

    # 1) Compare the simple lists: norms_edge_terms_data vs norms_edge_terms_sampled
    fig, ax = plt.subplots()
    ax.plot(norms_edge_terms_data, label="data norms")
    ax.plot(norms_edge_terms_sampled, label="random-train norms")
    ax.set_xlabel("sample index")
    ax.set_ylabel("L2 norm")
    ax.set_title("Edge-term norms (data vs random train)")
    ax.legend()
    _savefig(fig, base_dir / f"edge_term_norms_compare{out_suffix}.png")

    # 2) Progress plots (list[list[float]]) — plot each sequence as a faint line
    def _plot_progress(progress: list[list[float]], title: str, fname: str):
        fig, ax = plt.subplots()
        for seq in progress:
            if not seq:
                continue
            ax.plot(seq, linewidth=0.8, alpha=0.6)
        ax.set_xlabel("decode step")
        ax.set_ylabel("L2 norm")
        ax.set_title(title)
        _savefig(fig, base_dir / f"{fname}{out_suffix}.png")

    _plot_progress(norms_progress_data, "Decode progress (data)", "decode_progress_data")
    _plot_progress(norms_progress_sampled, "Decode progress (random train)", "decode_progress_random")
    _plot_progress(
        norms_progress_corrected_1, "Decode progress (random train - correction)", "decode_progress_random_correction"
    )
    _plot_progress(sims_progress_data, "Cos Sim Progress (data)", "sims_progress_random")
    _plot_progress(sims_progress_samples, "Cos Sim Progress (random train)", "sims_progress_random")
    _plot_progress(
        sims_progress_samples, "Cos Sim Progress (random train - corrected)", "sims_progress_random_corrected"
    )

    def _plot_integer_distribution(values: list[int], title: str, fname: str):
        counts = Counter(values)
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]

        fig, ax = plt.subplots()
        ax.bar(xs, ys, width=0.8, align="center", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Number of incomplete nodes")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        _savefig(fig, base_dir / f"{fname}{out_suffix}.png")

    _plot_integer_distribution(
        number_of_not_complete_nodes,
        f"Distribution of incomplete nodes before correction (#samples: {len(number_of_not_complete_nodes)})",
        "to_be_corrected_before",
    )
    _plot_integer_distribution(
        number_of_not_complete_nodes_after_bad_correction,
        f"Distribution of incomplete nodes after correction (#samples: {len(number_of_not_complete_nodes_after_bad_correction)})",
        "to_be_corrected_after",
    )

    # Convert Counter to list for plotting (keys represent number of valid decoded sets)
    decoded_sets_list = [k for k, v in decoded_sets_counter.items() for _ in range(v)]
    _plot_integer_distribution(
        decoded_sets_list,
        f"Distribution of valid decoded edge set variants (#samples: {n_samples})",
        "decoded_sets_variants",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate samples from a trained model with plots.")
    p.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3.value,
        choices=[ds.value for ds in SupportedDataset],
    )
    p.add_argument("--n_samples", type=int, default=1000)
    args = p.parse_args()
    models = {
        "qm9": {
            "gen_models": ["R1_nvp_QM9SmilesHRR1600F64G1NG3_f16_lr0.000525421_wd0.0005_bs256_an"],
        },
    }
    n_samples = args.n_samples
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    ds = SupportedDataset(args.dataset)
    for g in models[ds.default_cfg.base_dataset]["gen_models"]:
        if ds.default_cfg.name in g:
            try:
                eval_generation(
                    ds=ds,
                    n_samples=n_samples,
                    gen_mod_hint=g,
                    draw=False,
                    plot=True,
                    out_suffix="_normalize_vs_not",
                )
            except Exception as e:
                print(f"Error for {g}: {e}")
                continue
