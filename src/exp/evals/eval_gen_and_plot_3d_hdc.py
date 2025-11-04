import argparse
import json
import os
import time
from pprint import pprint
from typing import Any

import numpy as np
import torch
from pytorch_lightning import seed_everything

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import (
    QM9_SMILES_HRR_1600_CONFIG_F64,
)
from src.generation.evaluator import GenerationEvaluator, rdkit_logp
from src.generation.generation import HDCGenerator
from src.utils.chem import draw_mol
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, pick_device
from src.utils.visualisations import plot_logp_kde

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
device = torch.device("cpu")
EVALUATOR = None


def eval_generation(
    base_dataset: str,
    n_samples: int,
    gen_mod_hint: str,
    *,
    draw: bool,
    plot: bool,
) -> dict[str, Any]:
    global EVALUATOR  # noqa: PLW0603

    generator = HDCGenerator(gen_model_hint=gen_mod_hint, ds_config=QM9_SMILES_HRR_1600_CONFIG_F64, device=device)

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

    mols, valid_flags, sims = EVALUATOR.get_mols_valid_flags_sims_and_correction_levels()

    base_dir = (
        GLOBAL_ARTEFACTS_PATH
        / "generation_and_plots"
        / f"{base_dataset}_{gen_mod_hint}_hdc-decoder_{n_samples}-samples"
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

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate samples from a trained model with plots.")
    p.add_argument("--dataset", type=str, default="qm9", choices=["zinc", "qm9"])
    p.add_argument("--n_samples", type=int, default=1000)
    args = p.parse_args()
    models = {
        "qm9": {
            "gen_models": [
                # "nvp-3d-f64_qm9_f8_hid1792_lr0.000747838_wd1e-5_bs384_smf5.9223_smi2.08013_smw16_an",
                # "nvp-3d-f64_qm9_f8_hid1536_lr0.000503983_wd1e-5_bs384_smf7.43606_smi1.94892_smw15_an",
                "nvp-3d-f64_qm9_f8_hid800_lr0.000373182_wd1e-5_bs384_smf6.54123_smi2.25695_smw16_an"
            ],
        },
    }
    n_samples = args.n_samples
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    dataset = args.dataset
    for g in models[dataset]["gen_models"]:
        res = eval_generation(
            base_dataset=dataset,
            n_samples=n_samples,
            gen_mod_hint=g,
            draw=False,
            plot=True,
        )
