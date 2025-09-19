import time
from pathlib import Path

import networkx as nx
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from rdkit import Chem
from rdkit.Chem import Draw

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import SupportedDataset
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.encoding.oracles import Oracle
from src.generation.generation import Generator
from src.utils import registery
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, GLOBAL_MODEL_PATH, DataTransformer, find_files
from src.utils.visualisations import draw_nx_with_atom_colorings


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

def draw_rdkit_mol(smile: str, out_png: Path) -> None:
    try:
        mol = Chem.MolFromSmiles(smile)
        img = Draw.MolToImage(mol, size=(350, 250), kekulize=True)
        print(f"Saving {out_png}...")
        img.save(out_png)
    except Exception as e:
        print(f"Failed to save {out_png}. Error: {e}")
        pass


seed = 42
seed_everything(seed)
device = torch.device("cpu")
draw = False

results: dict[str, str] = {}
# Iterate all the checkpoints
for gen_ckpt_path in find_files(start_dir=GLOBAL_MODEL_PATH / "0_real_nvp_v2", prefixes=("epoch",), desired_ending=".ckpt"):
    if "qm9" in str(gen_ckpt_path):
        continue

    print(f"Generator Checkpoint: {gen_ckpt_path}")
    # Read the metrics from training
    evals_dir = gen_ckpt_path.parent.parent / "evaluations"
    epoch_metrics = pd.read_parquet(evals_dir / "metrics.parquet")
    min_val_loss = epoch_metrics["val_loss"].min()
    print(epoch_metrics.head(5))

    ## Retrieve the model
    model_type_gen = get_model_type(gen_ckpt_path)
    try:
        gen_model = registery.retrieve_model(model_type_gen).load_from_checkpoint(gen_ckpt_path, map_location=device, strict=True)
    except Exception as e:
        results[gen_ckpt_path] = f"Error: {e}"
        continue

    split = "train"
    ## Determine Dataset
    if "zinc" in str(gen_ckpt_path):
        ds = SupportedDataset.ZINC_SMILES_HRR_7744
        dataset = ZincSmiles(split=split)
    else:  # Case qm9
        ds = SupportedDataset.QM9_SMILES_HRR_1600
        dataset = QM9Smiles(split=split)

    ckpt_path = GLOBAL_MODEL_PATH/ "1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt"
    print(f"File Name: {ckpt_path}")

    # Read the metrics from training
    evals_dir = ckpt_path.parent.parent / "evaluations"
    epoch_metrics = pd.read_parquet(evals_dir / "epoch_metrics.parquet")
    print(epoch_metrics.head(5))

    val_loss = "val_loss" if "val_loss" in epoch_metrics.columns else "val_loss_cb"
    best = epoch_metrics.loc[epoch_metrics[val_loss].idxmin()].add_suffix("_best")
    last = epoch_metrics.iloc[-1].add_suffix("_last")

    ## Determine model type
    model_type: registery.ModelType = get_model_type(ckpt_path)

    ## Hyper net
    hypernet: HyperNet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds.default_cfg).to(device=device)
    assert not hypernet.use_edge_features()
    assert not hypernet.use_graph_features()

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
    oracle = Oracle(model=classifier, encoder=hypernet, model_type=model_type)

    oracle_setting = {
        "beam_size": 32,
        "oracle_threshold": best["val_best_thr_best"] if True else 0.5,
        "strict": False,
        "use_pair_feasibility": True,
        "expand_on_n_anchors": 8,
    }
    generator = Generator(
        gen_model=gen_model,
        oracle=oracle,
        ds_config=ds.default_cfg,
        oracle_settings=oracle_setting,
    )
    n_samples = 10
    start_t = time.perf_counter()
    samples: list[nx.Graph] = generator.generate(n_samples, most_similar=True)
    delta_t = time.perf_counter() - start_t
    assert len(samples) == n_samples

    # --- helper: canonical SMILES ---
    def _canon_smiles(m: Chem.Mol | None) -> str | None:
        if m is None:
            return None
        try:
            Chem.Kekulize(m, clearAromaticFlags=True)
        except Exception:
            pass  # best-effort; sanitize already done in nx_to_mol
        try:
            return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
        except Exception:
            return None

    # --- convert to RDKit mols (keep length n_samples) ---
    mols: list[Chem.Mol | None] = []
    for g in samples:
        mol: Chem.Mol | None = None
        try:
            mol = DataTransformer.nx_to_mol(g, sanitize=True, kekulize=True)
        except Exception as e:
            print(f"nx_to_mol error: {e}")
        mols.append(mol)
    assert len(mols) == n_samples

    # --- Validity ---
    valid_flags = [(m is not None and m.GetNumAtoms() > 0) for m, _ in mols]
    n_valid = sum(1 for f in valid_flags if f)
    validity = 100.0 * n_valid / n_samples if n_samples > 0 else 0.0

    # --- Uniqueness (among valid) ---
    valid_canon = [_canon_smiles(m) for m, f in zip(mols, valid_flags) if f]
    valid_canon = [c for c in valid_canon if c is not None]
    unique_valid_smiles: set[str] = set(valid_canon)
    uniqueness = 100.0 * (len(unique_valid_smiles) / n_valid) if n_valid > 0 else 0.0

    # --- Novelty (vs training canonical set) ---
    # Build canonical training set T_canon
    def _item_to_smiles(item) -> str | None:
        if hasattr(item, "smiles"):
            return str(item.smiles)
        if isinstance(item, dict) and "smiles" in item:
            return str(item["smiles"])
        if isinstance(item, str):
            return item
        return None

    T_raw = [_item_to_smiles(d) for d in dataset]
    T_raw = [s for s in T_raw if s]  # drop None/empty
    T_canon: set[str] = set()
    for s in T_raw:
        m = Chem.MolFromSmiles(s, sanitize=True) if s else None
        c = _canon_smiles(m)
        if c:
            T_canon.add(c)

    novel_set = unique_valid_smiles - T_canon
    novelty = 100.0 * (len(novel_set) / n_valid) if n_valid > 0 else 0.0

    # --- N.U.V. (intersection ratio over all generated) ---
    nuv = 100.0 * (len(novel_set) / n_samples) if n_samples > 0 else 0.0

    ### Save metrics
    gen_path = "/".join(gen_ckpt_path.parts[-4:])
    run_metrics = {
        "gen_path": str(gen_path),
        "clsfr_path": str(ckpt_path),
        "model_type": model_type,
        "dataset": ds.value,
        "num_eval_samples": n_samples,
        "time_per_sample": delta_t / n_samples,
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "nuv": nuv,
    }

    if draw:
        for i, s in enumerate(unique_valid_smiles):
            out = GLOBAL_ARTEFACTS_PATH / "generation" / "drawings" / f"{gen_path}-{i}.png"
            draw_rdkit_mol(smile=s, out_png=out)

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = GLOBAL_ARTEFACTS_PATH / "generation"
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / "zinc.parquet"
    csv_path = asset_dir / "zinc.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([run_metrics])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)
