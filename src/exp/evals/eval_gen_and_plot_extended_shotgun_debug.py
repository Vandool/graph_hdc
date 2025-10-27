import argparse
import json
import os
import time
from pprint import pprint
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pytorch_lightning import seed_everything
from rdkit import Chem
from rdkit.Chem import QED
from scipy.stats import gaussian_kde

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import (
    SupportedDataset,
)
from src.generation.evaluator import GenerationEvaluator, rdkit_logp
from src.generation.generation import HDCGenerator
from src.utils.chem import draw_mol
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, GLOBAL_BEST_MODEL_PATH, find_files, pick_device
from src.utils.visualisations import plot_logp_kde

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

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

seed = 42
seed_everything(seed)
device = pick_device()
EVALUATOR = None


def eval_generation(
    ds: SupportedDataset, n_samples: int, gen_mod_hint: str, *, draw: bool, plot: bool, out_suffix: str = ""
) -> dict[str, Any]:
    global EVALUATOR  # noqa: PLW0603
    base_dataset = ds.default_cfg.base_dataset
    generator = HDCGenerator(gen_model_hint=gen_mod_hint, ds_config=ds.default_cfg, device=device, dtype=DTYPE)
    # generator = HDCZ3Generator(gen_model_hint=gen_mod_hint, ds_config=ds.default_cfg, device=device)

    generator.decoder_settings = {
        "initial_limit": 2048,
        "limit": 1024,
        "beam_size": 768,
        "pruning_method": "cos_sim",
        "use_size_aware_pruning": True,
        "use_one_initial_population": True,
        "use_g3_instead_of_h3": True,
    }
    # generator.decoder_settings = {
    #     "max_soultions": 10000
    # }

    t0_gen = time.perf_counter()
    samples = generator.generate_most_similar(n_samples=n_samples)
    t_gen = time.perf_counter() - t0_gen

    nx_graphs = samples["graphs"]
    final_flags = samples["final_flags"]
    sims = samples["similarities"]

    results = {
        "gen_model": gen_mod_hint,
        "n_samples": n_samples,
        "t_gen_per_sample": t_gen / n_samples,
    }

    if EVALUATOR is None:
        EVALUATOR = GenerationEvaluator(base_dataset=base_dataset, device=device)

    evals = EVALUATOR.evaluate(samples=nx_graphs, final_flags=final_flags, sims=sims)
    results.update({f"eval_{k}": v for k, v in evals.items()})
    pprint(results)

    mols, valid_flags, sims = EVALUATOR.get_mols_and_valid_flags()

    dt = "f32" if torch.float32 == DTYPE else "f64"
    base_dir = (
        GLOBAL_ARTEFACTS_PATH
        / "generation_and_plots_best_models"
        / f"{base_dataset}_{gen_mod_hint}_hdc-decoder_{n_samples}-{dt}-samples{out_suffix}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save for later re-plotting
    logp_gen_list = np.array(
        [rdkit_logp(mol) for mol, valid in zip(mols, valid_flags, strict=False) if valid], dtype=float
    )
    np.save(base_dir / "logp.npy", logp_gen_list)
    (base_dir / "results.json").write_text(json.dumps(results, indent=4))

    if draw:
        base_dir.mkdir(parents=True, exist_ok=True)
        for i, (mol, valid, sim) in enumerate(zip(mols, valid_flags, sims, strict=False)):
            if valid:
                logp = rdkit_logp(mol)
                out = base_dir / f"Sim_{sim:.3f}__LogP_{logp:.3f}{i}.png"
                draw_mol(mol=mol, save_path=out, fmt="png")

    if plot:
        ds = QM9Smiles(split="train") if base_dataset == "qm9" else ZincSmiles(split="train")
        lp = np.array(ds.logp.tolist())

        plot_logp_kde(
            dataset=base_dataset,
            lp=lp,
            lg=logp_gen_list,
            out=(base_dir / "logp_overlay.png"),
            description="HDC-Decoder",
        )
    if plot:
        # --- dataset side ---
        ds = QM9Smiles(split="train") if base_dataset == "qm9" else ZincSmiles(split="train")
        ds_num_nodes, ds_num_edges, ds_logp, ds_qed = [], [], [], []
        for d in ds:
            m = Chem.MolFromSmiles(d.smiles)
            ds_num_nodes.append(int(m.GetNumAtoms()))
            ds_num_edges.append(int(m.GetNumBonds()))
            ds_logp.append(float(d.logp.detach().cpu().reshape(-1)[0]))
            ds_qed.append(float(d.qed.detach().cpu().reshape(-1)[0]))
            # ds_sa.append(float(d.sa_score.detach().cpu().reshape(-1)[0]))

        # --- generated side (valid molecules only) ---
        gen_num_nodes, gen_num_edges, gen_logp, gen_qed = [], [], [], []
        for mol, valid in zip(mols, valid_flags, strict=False):
            if valid:
                gen_num_nodes.append(int(mol.GetNumAtoms()))
                gen_num_edges.append(int(mol.GetNumBonds()))
                gen_logp.append(float(rdkit_logp(mol)))
                gen_qed.append(float(QED.qed(mol)))
                # gen_sa.append(float(calculate_sa_score(mol)))

        def kde_overlay(a, b, xlabel, title, path):
            a, b = np.array(a), np.array(b)
            lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
            x = np.linspace(lo, hi, 512)
            plt.figure(figsize=(5.5, 3.8))
            plt.plot(x, gaussian_kde(a)(x), label="Dataset", linewidth=2)
            plt.plot(x, gaussian_kde(b)(x), label="Generated", linewidth=2, linestyle="--")
            plt.xlabel(xlabel)
            plt.ylabel("Density")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        def step_overlay(a, b, xlabel, title, path):
            bins = np.arange(min(*a, *b) - 0.5, max(*a, *b) + 1.5, 1)
            plt.figure(figsize=(5.5, 3.8))
            plt.hist(a, bins=bins, histtype="step", linewidth=2, label="Dataset", density=True)
            plt.hist(b, bins=bins, histtype="step", linewidth=2, linestyle="--", label="Generated", density=True)
            plt.xlabel(xlabel)
            plt.ylabel("Density")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        step_overlay(ds_num_nodes, gen_num_nodes, "Num. atoms", "Node count", base_dir / "matplotlib_nodes.pdf")
        step_overlay(ds_num_edges, gen_num_edges, "Num. edges", "Edge count", base_dir / "matplotlib_edges.pdf")
        kde_overlay(ds_logp, gen_logp, "logP", "logP distribution", base_dir / "matplotlib_logp.pdf")
        kde_overlay(ds_qed, gen_qed, "QED", "QED distribution", base_dir / "matplotlib_qed.pdf")
        # kde_overlay(ds_sa, gen_sa, "SA score", "SA distribution", base_dir / "matplotlib_sa.pdf")

        sns.set_theme(style="whitegrid", context="talk")  # good defaults

        def sns_kde_overlay(a, b, xlabel, title, path):
            plt.figure(figsize=(5.5, 3.8))
            sns.kdeplot(a, label="Dataset", linewidth=2)
            sns.kdeplot(b, label="Generated", linewidth=2, linestyle="--")
            plt.xlabel(xlabel)
            plt.ylabel("Density")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        def sns_step_overlay(a, b, xlabel, title, path):
            plt.figure(figsize=(5.5, 3.8))
            sns.histplot(a, label="Dataset", bins=30, stat="density", element="step", fill=False, linewidth=2)
            sns.histplot(
                b, label="Generated", bins=30, stat="density", element="step", fill=False, linewidth=2, linestyle="--"
            )
            plt.xlabel(xlabel)
            plt.ylabel("Density")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        # Example calls
        sns_step_overlay(ds_num_nodes, gen_num_nodes, "Num. atoms", "Node count", base_dir / "seaborn_nodes.pdf")
        sns_step_overlay(ds_num_edges, gen_num_edges, "Num. edges", "Edge count", base_dir / "seaborn_edges.pdf")
        sns_kde_overlay(ds_logp, gen_logp, "logP", "logP distribution", base_dir / "seaborn_logp.pdf")
        sns_kde_overlay(ds_qed, gen_qed, "QED", "QED distribution", base_dir / "seaborn_qed.pdf")
        # sns_kde_overlay(ds_sa, gen_sa, "SA score", "SA distribution", base_dir / "seaborn_sa.pdf")

        # Sims distribution
        sims_valid = [float(s) for s, v in zip(sims, valid_flags, strict=False) if v]
        sims_valid = np.asarray(sims_valid, dtype=float)
        x = np.linspace(sims_valid.min(), sims_valid.max(), 512)

        # Matplotlib
        plt.figure(figsize=(5.5, 3.8))
        plt.plot(x, gaussian_kde(sims_valid)(x), linewidth=2, label="Generated (valid)")
        plt.xlabel("Similarity")
        plt.ylabel("Density")
        plt.title(f"{base_dataset} - Similarity distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(base_dir / "matplotlib_sims.pdf", bbox_inches="tight")
        plt.close()

        # Seaborn
        plt.figure(figsize=(5.5, 3.8))
        sns.kdeplot(sims_valid, label="Generated (valid)", linewidth=2)
        plt.xlabel("Similarity")
        plt.ylabel("Density")
        plt.title(f"{base_dataset} - Similarity distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(base_dir / "seaborn_sims.pdf", bbox_inches="tight")
        plt.close()

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate samples from a trained model with plots.")
    p.add_argument("--n_samples", type=int, default=2000)
    args = p.parse_args()
    n_samples = args.n_samples
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    datasests = [
        # SupportedDataset.QM9_SMILES_HRR_1600_F64_G1G3,
        SupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3,
        # SupportedDataset.ZINC_SMILES_HRR_6144_F64_G1G3,
    ]
    for p in find_files(
        start_dir=GLOBAL_BEST_MODEL_PATH / "0_real_nvp_v2", prefixes=("epoch",), desired_ending=".ckpt"
    ):
        name = p.parent.parent.name
        # if name not in "R1_nvp_QM9SmilesHRR1600F64G1NG3_f16_hid400_lr0.000345605_wd3e-6_bs160_smf6.5_smi2.2_smw16_an":
        #     continue

        if name not in [
            "R1_nvp_QM9SmilesHRR1600F64G1NG3_f15_hid1600_s42_lr0.0004818_wd0.0005_bs288",
            "R1_nvp_QM9SmilesHRR1600F64G1NG3_f16_hid400_lr0.000345605_wd3e-6_bs160_smf6.5_smi2.2_smw16_an",
            "R1_nvp_QM9SmilesHRR1600F64G1NG3_f16_hid1600_s42_lr0.000221865_wd0.0005_bs32",
            # "R1_nvp_QM9SmilesHRR1600F64G1NG3_f16_lr0.000525421_wd0.0005_bs256_an",
        ]:
            continue

        if (ds_config := next((d for d in datasests if d.default_cfg.name in name), None)) is None:
            print(f"[SKIPPED] {p}")
            continue
        try:
            eval_generation(
                ds=ds_config,
                n_samples=n_samples,
                gen_mod_hint=name,
                draw=False,
                plot=True,
                out_suffix="_ceil",
            )
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
            continue
