import os
import time
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import seed_everything
from rdkit import Chem

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import SupportedDataset
from src.encoding.oracles import Oracle
from src.generation.generation import Generator
from src.utils import registery
from src.utils.chem import canonical_key, draw_mol, is_valid_molecule, reconstruct_for_eval
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, GLOBAL_MODEL_PATH, find_files


# keep it modest to avoid oversubscription; tune if needed
num = max(1, min(8, os.cpu_count() or 1))
torch.set_num_threads(num)
torch.set_num_interop_threads(max(1, min(2, num)))  # coordination threads

# (optional but often helps BLAS backends)
os.environ.setdefault("OMP_NUM_THREADS", str(num))
os.environ.setdefault("MKL_NUM_THREADS", str(num))


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


seed = 42
seed_everything(seed)
device = torch.device("cpu")
draw = True

T_canon_zinc = None
T_canon_qm9 = None

n_samples = 100
strict_decoder = False
beam_size = 48
expand_on_n_anchors = 12
generation = "generation7"

results: dict[str, str] = {}
# Iterate all the checkpoints
loop = 0
gen_paths = list(find_files(
        start_dir=GLOBAL_MODEL_PATH / "0_real_nvp_v2",
        prefixes=("epoch",),
        desired_ending=".ckpt",
        skip_substring="zinc",
    ))
print(f"Found {len(gen_paths)} generator checkpoints")
for gen_ckpt_path in gen_paths:
    # print(f"loop #{loop}")
    # if loop >= 1:
    #     break
    # loop += 1
    print(f"Generator Checkpoint: {gen_ckpt_path}")
    # Read the metrics from training
    evals_dir = gen_ckpt_path.parent.parent / "evaluations"
    epoch_metrics = pd.read_parquet(evals_dir / "metrics.parquet")
    # Find min val_loss
    idx = epoch_metrics["val_loss"].idxmin()
    min_val_loss = epoch_metrics.loc[idx, "val_loss"]
    best_epoch = epoch_metrics.loc[idx, "epoch"]

    print(f"Best epoch: {best_epoch}, min val_loss: {min_val_loss}")

    ## Retrieve the model
    model_type_gen = get_model_type(gen_ckpt_path)
    try:
        gen_model = registery.retrieve_model(model_type_gen).load_from_checkpoint(
            gen_ckpt_path, map_location=device, strict=True
        )
    except Exception as e:
        results[gen_ckpt_path] = f"Error: {e}"
        continue

    # We want evaluations against training set
    split = "train"
    ## Determine Dataset
    T_canon = None
    if "zinc" in str(gen_ckpt_path):
        ds = SupportedDataset.ZINC_SMILES_HRR_7744
        base_dataset = "zinc"
        dataset = ZincSmiles(split=split)
        if T_canon_zinc is None:
            T_canon_zinc = {d.eval_smiles for d in dataset}
        T_canon = T_canon_zinc
    else:  # Case qm9
        ds = SupportedDataset.QM9_SMILES_HRR_1600
        base_dataset = "qm9"
        dataset = QM9Smiles(split=split)
        if T_canon_qm9 is None:
            T_canon_qm9 = {d.eval_smiles for d in dataset}
        T_canon = T_canon_qm9

    # Best classifier checkpoint for
    if base_dataset == "zinc":
        ckpt_path = GLOBAL_MODEL_PATH / "1_gin/gin-f_baseline_zinc/models/epoch06-val0.2718.ckpt"
    else:
        ckpt_path = GLOBAL_MODEL_PATH / "1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt"
        # ckpt_path = GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_med_qm9/models/epoch19-val0.2648.ckpt"
    print(f"Classifier's checkpoint: {ckpt_path}")

    # Read the metrics from training
    evals_dir = ckpt_path.parent.parent / "evaluations"
    epoch_metrics = pd.read_parquet(evals_dir / "epoch_metrics.parquet")

    val_loss = "val_loss" if "val_loss" in epoch_metrics.columns else "val_loss_cb"
    best = epoch_metrics.loc[epoch_metrics[val_loss].idxmin()].add_suffix("_best")
    oracle_threshold = best["val_best_thr_best"]
    print(f"Oracle Threshold: {oracle_threshold}")
    last = epoch_metrics.iloc[-1].add_suffix("_last")

    ## Determine model type
    model_type: registery.ModelType = get_model_type(ckpt_path)

    ## Classifier and Oracle
    try:
        classifier = (
            registery.retrieve_model(name=model_type)
            .load_from_checkpoint(ckpt_path, map_location="cpu", strict=True)
            .to(device)
            .eval()
        )
    except Exception as e:
        results[ckpt_path] = f"Error: {e}"
        continue
    oracle = Oracle(model=classifier, model_type=model_type)

    decoder_settings = {
        "beam_size": beam_size,
        "oracle_threshold": oracle_threshold,
        # "strict": False,
        "use_pair_feasibility": True,
        "expand_on_n_anchors": expand_on_n_anchors,
    }
    generator = Generator(
        gen_model=gen_model, oracle=oracle, ds_config=ds.default_cfg, decoder_settings=decoder_settings, device=device
    )

    start_t = time.perf_counter()

    samples, final_flags, sims = generator.generate_most_similar(n_samples, only_final_graphs=strict_decoder)
    delta_t = time.perf_counter() - start_t
    assert len(samples) == n_samples

    # --- convert to RDKit mols (keep length n_samples) ---
    mols: list[Chem.Mol | None] = []
    for g in samples:
        mol: Chem.Mol | None = None
        try:
            mol = reconstruct_for_eval(g, dataset=base_dataset)
        except Exception as e:
            print(f"nx_to_mol error: {e}")
        mols.append(mol)
    assert len(mols) == n_samples

    # --- Validity ---
    valid_flags = [(m is not None and is_valid_molecule(m)) for m in mols]
    n_valid = sum(1 for f in valid_flags if f)
    validity = 100.0 * n_valid / n_samples if n_samples > 0 else 0.0

    # --- Uniqueness (among valid) ---
    valid_canon = [canonical_key(m) for m, f in zip(mols, valid_flags, strict=False) if f]
    valid_canon = [c for c in valid_canon if c is not None]
    unique_valid_smiles: set[str] = set(valid_canon)
    uniqueness = 100.0 * (len(unique_valid_smiles) / n_valid) if n_valid > 0 else 0.0

    novel_set = unique_valid_smiles - T_canon
    novelty = 100.0 * (len(novel_set) / n_valid) if n_valid > 0 else 0.0

    # --- N.U.V. (intersection ratio over all generated) ---
    nuv = 100.0 * (len(novel_set) / n_samples) if n_samples > 0 else 0.0

    ### Save metrics
    gen_path = "/".join(gen_ckpt_path.parts[-4:])
    run_metrics = {
        "gen_path": str(gen_path),
        "best_epoch": best_epoch,
        "min_val_loss": min_val_loss,
        "dataset": ds.value,
        "num_eval_samples": n_samples,
        "time_per_sample": delta_t / n_samples,
        "strict": strict_decoder,
        "final_flags": int(100 * sum(final_flags) / n_samples),
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "nuv": nuv,
        "classifier": str(ckpt_path),
        "model_type": model_type,
        **decoder_settings
    }

    if draw:
        base_dir = GLOBAL_ARTEFACTS_PATH / generation / f"drawings_valid_strict-decoder-{strict_decoder}"
        base_dir.mkdir(parents=True, exist_ok=True)
        for i, (mol, mask, final) in enumerate(zip(mols, valid_flags, final_flags, strict=False)):
            if mask:
                out = base_dir / f"{gen_path.replace('/', '-')!s}-{final}-{i}.png"
                draw_mol(mol=mol, save_path=out, fmt="png")

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = GLOBAL_ARTEFACTS_PATH / generation
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / f"{base_dataset}.parquet"
    csv_path = asset_dir / f"{base_dataset}.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([run_metrics])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)
