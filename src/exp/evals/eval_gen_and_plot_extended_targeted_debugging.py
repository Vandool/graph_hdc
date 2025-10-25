import argparse
import os
import random
from typing import Any

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import seed_everything

from src.datasets.utils import get_split
from src.encoding.configs_and_constants import (
    SupportedDataset,
)
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


def eval_generation(
    ds: SupportedDataset, n_samples: int, gen_mod_hint: str, *, draw: bool, plot: bool, out_suffix: str = ""
) -> dict[str, Any]:
    global EVALUATOR
    base_dataset = ds.default_cfg.base_dataset
    generator = HDCGenerator(gen_model_hint=gen_mod_hint, ds_config=ds.default_cfg, device=device)
    # generator = HDCZ3Generator(gen_model_hint=gen_mod_hint, ds_config=QM9_SMILES_HRR_1600_CONFIG_F64, device=device)

    samples = generator.get_raw_samples(n_samples=n_samples)
    sampled_edge_terms = samples["edge_terms"].as_subclass(ds.default_cfg.vsa.tensor_class)

    training_edge_terms = get_split("train", ds_config=ds.default_cfg)
    random_samples_idxs = random.sample(range(len(training_edge_terms)), n_samples)
    random_edge_terms = (
        training_edge_terms.index_select(random_samples_idxs)
        .edge_terms.view(n_samples, ds.default_cfg.hv_dim)
        .to(device)
    )

    hypernet = generator.hypernet

    norms_edge_terms_data = []  # list of norms
    norms_edge_terms_sampled = []  # list of norms
    norms_progress_data = []  # list of list of norms
    norms_progress_sampled = []  # list of list of norms
    sims_progress_data = []
    sims_progress_samples = []
    for i in range(n_samples):
        # Data
        norms_edge_terms_data.append(sampled_edge_terms[i].norm().item())
        decoded_edges, norms_, sims = hypernet.decode_order_one_no_node_terms(
            sampled_edge_terms[i].clone().to(torch.float64), debug=True
        )
        norms_progress_data.append(norms_)
        sims_progress_data.append(sims)

        # Sampled
        norms_edge_terms_sampled.append(random_edge_terms[i].norm().item())
        decoded_edges, norms_, sims = hypernet.decode_order_one_no_node_terms(
            random_edge_terms[i].clone().to(torch.float64), debug=True
        )
        norms_progress_sampled.append(norms_)
        sims_progress_samples.append(sims)

    base_dir = GLOBAL_ARTEFACTS_PATH / "decoding_debugging_plots_2"
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
    _plot_progress(sims_progress_data, "Cos Sim Progress (data)", "sims_progress_random")
    _plot_progress(sims_progress_samples, "Cos Sim Progress (random train)", "sims_progress_random")
    # ---- /plotting ----


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate samples from a trained model with plots.")
    p.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3.value,
        choices=[ds.value for ds in SupportedDataset],
    )
    p.add_argument("--n_samples", type=int, default=100)
    args = p.parse_args()
    models = {
        "qm9": {
            "gen_models": [
                "nvp_QM9SmilesHRR1600F64G1G3_f4_lr0.000862736_wd0.0001_bs192_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f4_lr8.69904e-5_wd0_bs288_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f7_lr9.4456e-5_wd0.0003_bs448_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f8_lr0.00057532_wd0.0003_bs32_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f8_lr6.69953e-5_wd0.0001_bs160_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f9_lr0.000179976_wd0.0003_bs288_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f14_lr0.000112721_wd0.0005_bs224_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f14_lr0.000132447_wd3e-6_bs160_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f15_hid800_s42_lr0.000160949_wd3e-6_bs224_conNone_datSupportedDataset.QM9_SMILES_HRR_1600_F64_G1G3_epo700_expNone_hv_2_hv_1600_is_0_res0_smf4_smi1_smw10_use1_vsaVSAModel.HRR_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f15_lr0.000160949_wd3e-6_bs224_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f15_lr6.29685e-5_wd3e-6_bs128_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f16_hid400_s42_lr0.000154612_wd3e-6_bs32_conNone_datSupportedDataset.QM9_SMILES_HRR_1600_F64_G1G3_epo700_expNone_hv_2_hv_1600_is_0_res0_smf4_smi1_smw10_use1_vsaVSAModel.HRR_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f16_hid800_s42_lr0.000430683_wd3e-6_bs512_conNone_datSupportedDataset.QM9_SMILES_HRR_1600_F64_G1G3_epo700_expNone_hv_2_hv_1600_is_0_res0_smf4_smi1_smw10_use1_vsaVSAModel.HRR_an",
                "nvp_QM9SmilesHRR1600F64G1G3_f16_lr0.000525421_wd0.0005_bs256_an",
                ##
                "nvp_QM9SmilesHRR1600F64G1NG3_f4_lr0.000862736_wd0.0001_bs192_an",
                "nvp_QM9SmilesHRR1600F64G1NG3_f4_lr8.69904e-5_wd0_bs288_an",
                "nvp_QM9SmilesHRR1600F64G1NG3_f7_lr9.4456e-5_wd0.0003_bs448_an",
                "nvp_QM9SmilesHRR1600F64G1NG3_f8_lr6.69953e-5_wd0.0001_bs160_an",
                "nvp_QM9SmilesHRR1600F64G1NG3_f14_lr0.000112721_wd0.0005_bs224_an",
                "nvp_QM9SmilesHRR1600F64G1NG3_f16_lr0.000525421_wd0.0005_bs256_an",
            ],
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
