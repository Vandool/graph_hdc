import argparse
import itertools
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import torch
from pytorch_lightning import seed_everything

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import QM9_SMILES_HRR_1600_CONFIG, ZINC_SMILES_HRR_7744_CONFIG
from src.encoding.oracles import Oracle
from src.generation.evaluator import GenerationEvaluator, rdkit_logp
from src.generation.generation import Generator
from src.utils import registery
from src.utils.chem import draw_mol
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, GLOBAL_MODEL_PATH, find_files
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
# device = pick_device()
device = torch.device("cpu")
EVALUATOR = None


def get_model_type(path: Path | str) -> registery.ModelType:
    res: registery.ModelType = "MLP"
    if "bah" in str(path):
        res = "BAH"
    elif "gin-c" in str(path):
        res = "GIN-C"
    elif "gin-f" in str(path):
        res = "GIN-F"
    elif "nvp" in str(path):
        res = "NVP"
    return res


def make_lambda_cosine_decay(*, steps: int, lam_hi: float = 1e-2, lam_lo: float = 1e-4):
    steps = max(1, int(steps))

    def sched(s: int) -> float:
        # s ∈ [0, steps-1]
        t = min(max(s / (steps - 1 if steps > 1 else 1), 0.0), 1.0)
        # starts at lam_hi, ends at lam_lo
        return lam_lo + 0.5 * (lam_hi - lam_lo) * (1.0 + math.cos(math.pi * t))

    return sched


def make_lambda_linear_decay(*, steps: int, lam_hi: float = 1e-2, lam_lo: float = 1e-4):
    steps = max(1, int(steps))

    def sched(s: int) -> float:
        t = min(max(s / (steps - 1 if steps > 1 else 1), 0.0), 1.0)
        return lam_hi + t * (lam_lo - lam_hi)

    return sched


def make_lambda_two_phase(
    *,
    steps: int,
    lam_hi: float = 1e-2,
    lam_lo: float = 1e-4,
    warm_frac: float = 0.2,
):
    steps = max(1, int(steps))
    warm = int(max(0.0, min(1.0, warm_frac)) * steps)

    def sched(s: int) -> float:
        if s < warm:
            return lam_hi
        # cosine decay to lam_lo over the remaining steps
        remain = max(1, steps - warm)
        t = min(max((s - warm) / remain, 0.0), 1.0)
        return lam_lo + 0.5 * (lam_hi - lam_lo) * (1.0 + math.cos(math.pi * t))

    return sched


schedulers: dict[str, Callable] = {
    "cosine": make_lambda_cosine_decay,
    "two-phase": make_lambda_two_phase,
    "linear": make_lambda_linear_decay,
}


def get_gen_model(hint: str) -> Path | None:
    gen_paths = find_files(
        start_dir=GLOBAL_MODEL_PATH / "0_real_nvp_v2",
        prefixes=("epoch",),
        desired_ending=".ckpt",
    )
    for p in gen_paths:
        if hint in str(p):
            return p
    return None


def get_classifier(hint: str) -> Path | None:
    paths = find_files(
        start_dir=GLOBAL_MODEL_PATH,
        prefixes=("epoch",),
        desired_ending=".ckpt",
        skip_substrings=("nvp", "lpr"),
    )
    for p in paths:
        if hint in str(p):
            return p
    return None


def eval_cond_gen(
    base_dataset: str,
    n_samples: int,
    gen_mod_hint: str,
    classifier_hint: str,
    decoder_setting: dict,
    *,
    draw: bool,
    plot: bool,
) -> dict[str, Any]:
    global EVALUATOR  # noqa: PLW0603

    gen_ckpt_path = get_gen_model(hint=gen_mod_hint)
    print(f"Generator Checkpoint: {gen_ckpt_path}")
    ## Retrieve the model
    model_type_gen = get_model_type(gen_ckpt_path)
    gen_model = (
        registery.retrieve_model(model_type_gen)
        .load_from_checkpoint(gen_ckpt_path, map_location=device, strict=True)
        .to(device)
    )

    # ---- Hyperparams ----
    generator = Generator(
        gen_model=gen_model,
        oracle=Oracle(model_path=get_classifier(hint=classifier_hint)),
        ds_config=QM9_SMILES_HRR_1600_CONFIG if dataset == "qm9" else ZINC_SMILES_HRR_7744_CONFIG,
        decoder_settings=decoder_setting,
        device=device,
    )

    t0_gen = time.perf_counter()
    nx_graphs, final_flags, sims = generator.generate_most_similar(n_samples=n_samples, only_final_graphs=False)
    t_gen = time.perf_counter() - t0_gen

    results = {
        "gen_model": str(gen_ckpt_path.parent.parent.stem),
        "n_samples": n_samples,
        "t_gen_per_sample": t_gen / n_samples,
        "classifier": classifier_hint,
        **decoder_setting,
    }

    if EVALUATOR is None:
        EVALUATOR = GenerationEvaluator(base_dataset=base_dataset, device=device)

    evals = EVALUATOR.evaluate(samples=nx_graphs, final_flags=final_flags, sims=sims)
    results.update({f"eval_{k}": v for k, v in evals.items()})
    pprint(results)

    mols, valid_flags, sims = EVALUATOR.get_mols_and_valid_flags()

    base_dir = (
        GLOBAL_ARTEFACTS_PATH
        / "generation_and_plots"
        / f"{base_dataset}_{os.getenv('GEN_MODEL')}_{os.getenv('CLASSIFIER')}_{n_samples}-samples"
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
            description=f"Classifier ({os.getenv('CLASSIFIER')}",
        )

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate samples from a trained model with plots.")
    p.add_argument("--dataset", type=str, default="zinc", choices=["zinc", "qm9"])
    p.add_argument("--n_samples", type=int, default=5)
    args = p.parse_args()
    decoder_settings = {
        "qm9": {
            "beam_size": 64,
            "use_pair_feasibility": True,
            "expand_on_n_anchors": 8,
            "trace_back_settings": {
                "beam_size_multiplier": 2,
                "trace_back_attempts": 3,
                "agitated_rounds": 1,  # how many rounds after applying trace back keep the beam size larger
            },
        },
        "zinc": {
            "beam_size": 4,
            "use_pair_feasibility": True,
            "expand_on_n_anchors": 8,
            "trace_back_settings": {
                "beam_size_multiplier": 2,
                "trace_back_attempts": 3,
                "agitated_rounds": 1,  # how many rounds after applying trace back keep the beam size larger
            },
        },
    }
    models = {
        "qm9": {
            "gen_models": [
                "nvp_qm9_h1600_f12_hid1024_s42_lr5e-4_wd0.0_an",
            ],
            "classifiers": ["gin-f_baseline_qm9_resume", "BAH_med_hardpool_qm9"],
        },
        "zinc": {
            "gen_models": [
                "nvp_zinc_f11_hid1152_lr0.000218409_wd0_bs64_smf5.99998_smi1",
                "nvp_zinc_f10_hid1152_lr7.61217e-5_wd0_bs64_smf6.00085_smi0",
                "nvp_zinc_h7744_f6_hid512_s42_lr1e-3_wd0",
                "nvp_zinc_h7744_f12_hid1280_s42_lr5e-4_wd0",
                "nvp_zinc_h7744_f12_hid768_s42_lr1e-3_wd1e-4_an",
                "nvp_zinc_h7744_f12_hid1024_s42_lr5e-4_wd1e-4_an",
                "nvp_zinc_h7744_f4_hid256_s42_lr1e-3_wd0",
                "nvp_zinc_h7744_f8_hid512_s42_lr5e-4_wd0",
                "nvp_zinc_h7744_f4_hid256_s42_lr1e-3_wd1e-4_an",
                "nvp_zinc_h7744_f12_hid1024_s42_lr5e-4_wd0",
                "nvp_zinc_h7744_f12_hid384_s42_lr1e-3_wd0",
            ],
            "classifiers": ["gin-f_baseline_zinc_resume_3", "BAH_med_zinc/models/epoch08-val0.3788.ckpt"],
        },
    }
    n_samples = args.n_samples
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    dataset = args.dataset
    for g, c in itertools.product(models[dataset]["gen_models"], models[dataset]["classifiers"]):
        decoder_setting = decoder_settings[dataset]
        res = eval_cond_gen(
            base_dataset=dataset,
            n_samples=n_samples,
            gen_mod_hint=g,
            classifier_hint=c,
            decoder_setting=decoder_setting,
            draw=False,
            plot=True,
        )
