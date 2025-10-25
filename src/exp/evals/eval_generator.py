import os
import time
from pathlib import Path
from pprint import pprint

import pandas as pd
import torch
from pytorch_lightning import seed_everything

from src.encoding.configs_and_constants import QM9_SMILES_HRR_1600_CONFIG, ZINC_SMILES_HRR_7744_CONFIG
from src.encoding.oracles import Oracle
from src.generation.evaluator import GenerationEvaluator
from src.generation.generation import OracleGenerator
from src.utils import registery
from src.utils.chem import draw_mol
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
draw = False

evaluator = None

n_samples = 100
strict_decoder = False
beam_size = 8
expand_on_n_anchors = 9
USE_PAIR_FEASIBILITY = True
generation = "generation12"

errors: dict[str, str] = {}
# Iterate all the checkpoints
loop = 0
gen_paths = list(
    find_files(
        start_dir=GLOBAL_MODEL_PATH / "0_real_nvp_v2",
        prefixes=("epoch",),
        desired_ending=".ckpt",
        skip_substrings=("qm9",),
    )
)
print(f"Found {len(gen_paths)} generator checkpoints")
for gen_ckpt_path in gen_paths:
    # print(f"loop #{loop}")
    # if loop >= 1:
    #     break
    # loop += 1
    do = {
        #     # QM9 TOP
        #     "0_real_nvp_v2/nvp_qm9_h1600_f12_hid1024_s42_lr5e-4_wd1e-4_an/models/epoch42-val-4172.5571.ckpt",
        #     "0_real_nvp_v2/nvp_qm9_h1600_f12_hid384_s42_lr1e-3_wd0.0_an/models/epoch26-val-2516.5386.ckpt",
        #     "0_real_nvp_v2/nvp_qm9_h1600_f12_hid1024_s42_lr5e-4_wd0.0_an/models/epoch50-val-4308.1338.ckpt",
        #     "0_real_nvp_v2/nvp_qm9_h1600_f8_hid512_s42_lr1e-3_wd1e-4_an/models/epoch12-val-1689.4788.ckpt",
        #     "0_real_nvp_v2/nvp_qm9_h1600_f12_hid768_s42_lr1e-3_wd0.0_an/models/epoch12-val-2198.1746.ckpt",
        #     "0_real_nvp_v2/nvp_qm9_h1600_f12_hid768_s42_lr1e-3_wd1e-4_an/models/epoch13-val-2190.4648.ckpt",
        #     "0_real_nvp_v2/nvp_qm9_h1600_f8_hid512_s42_lr5e-4_wd0.0_an/models/epoch41-val-3322.6777.ckpt",
        #     "0_real_nvp_v2/nvp_qm9_h1600_f12_hid512_s42_lr1e-3_wd0.0_an/models/epoch19-val-2442.2910.ckpt",
        #     "0_real_nvp_v2/nvp_qm9_h1600_f8_hid512_s42_lr1e-3_wd0.0_an/models/epoch25-val-2071.7695.ckpt",
        #     # ZINC TOP
        # "nvp_zinc_h7744_f4_hid256_s42_lr1e-3_wd1e-4_an",
        # "nvp_zinc_h7744_f12_hid384_s42_lr1e-3_wd0.0_an",
        # "nvp_zinc_h7744_f8_hid512_s42_lr5e-4_wd0.0_an",
        # "nvp_zinc_h7744_f8_hid512_s42_lr1e-3_wd0.0_noan",
        # "nvp_zinc_h7744_f6_hid512_s42_lr1e-3_wd0.0_an",
        # "nvp_zinc_h7744_f12_hid1024_s42_lr5e-4_wd0.0_an",
        # "nvp_zinc_h7744_f4_hid256_s42_lr1e-3_wd0.0_an",
        # "nvp_zinc_f11_hid1152_lr0.000218409_wd0_bs64_smf5.99998_smi1.00004_smw15_an",
        # "nvp_zinc_h7744_f12_hid1024_s42_lr5e-4_wd1e-4_an",
        # "nvp_zinc_h7744_f12_hid1280_s42_lr5e-4_wd0.0_an",
        # "nvp_zinc_f10_hid1152_lr7.61217e-5_wd0_bs64_smf6.00085_smi0.999514_smw15_an",
        # "nvp_zinc_h7744_f12_hid768_s42_lr1e-3_wd1e-4_an",
    }
    if not any(d in str(gen_ckpt_path) for d in do):
        print(f"Skipping {gen_ckpt_path}")
        continue

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
        errors[gen_ckpt_path] = f"Error: {e}"
        continue

    if "zinc" in str(gen_ckpt_path):
        base_dataset = "zinc"
        ckpt_path = GLOBAL_MODEL_PATH / "1_gin/gin-f_baseline_zinc/models/epoch06-val0.2718.ckpt"
        # ckpt_path = GLOBAL_MODEL_PATH / "1_gin/gin-f_baseline_zinc_resume/models/epoch10-val0.2387.ckpt"
        # ckpt_path = GLOBAL_MODEL_PATH / "1_gin/gin-f_baseline_zinc_resume_3/models/epoch18-val0.2083.ckpt"
    else:
        base_dataset = "qm9"
        evaluator = GenerationEvaluator(base_dataset=base_dataset)
        # ckpt_path = GLOBAL_MODEL_PATH / "1_gin/gin-f_baseline_qm9_resume/models/epoch10-val0.3359.ckpt"
        # ckpt_path = GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_large_qm9/models/epoch16-val0.2740.ckpt"
        # ckpt_path = GLOBAL_MODEL_PATH / "1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt"
        ckpt_path = GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_med_qm9/models/epoch19-val0.2648.ckpt"
        # ckpt_path = GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_base_qm9_v2/models/epoch23-val0.2772.ckpt"
        # ckpt_path = GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_larger_qm9/models/epoch16-val0.2949.ckpt"

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
        errors[ckpt_path] = f"Error: {e}"
        continue
    oracle = Oracle(model=classifier, model_type=model_type)

    generator = OracleGenerator(
        gen_model=gen_model,
        oracle=oracle,
        ds_config=ZINC_SMILES_HRR_7744_CONFIG if base_dataset == "zinc" else QM9_SMILES_HRR_1600_CONFIG,
        decoder_settings={
            "beam_size": beam_size,
            "oracle_threshold": oracle_threshold,
            # "strict": False,
            "use_pair_feasibility": USE_PAIR_FEASIBILITY,
            "expand_on_n_anchors": expand_on_n_anchors,
        },
        device=device,
    )

    start_t = time.perf_counter()

    samples, final_flags, sims = generator.generate_most_similar(n_samples, only_final_graphs=strict_decoder)
    delta_t = time.perf_counter() - start_t
    assert len(samples) == n_samples
    print(f"Sampled {len(samples)} graphs")

    if evaluator is None:
        evaluator = GenerationEvaluator(base_dataset=base_dataset)
    metrics = evaluator.evaluate(samples=samples, final_flags=final_flags, sims=sims)

    ### Save metrics
    gen_path = "/".join(gen_ckpt_path.parts[-4:])
    run_metrics = {
        "gen_path": str(gen_path),
        "best_epoch": best_epoch,
        "min_val_loss": min_val_loss,
        "num_eval_samples": n_samples,
        "time_per_sample": delta_t / n_samples,
        "strict": strict_decoder,
        **metrics,
        "classifier": str(ckpt_path),
        "model_type": model_type,
        "beam_size": beam_size,
        "oracle_threshold": oracle_threshold,
        # "strict": False,
        "use_pair_feasibility": USE_PAIR_FEASIBILITY,
        "expand_on_n_anchors": expand_on_n_anchors,
    }

    pprint(run_metrics)

    if draw:
        base_dir = GLOBAL_ARTEFACTS_PATH / generation / f"drawings_valid_strict-decoder-{strict_decoder}"
        base_dir.mkdir(parents=True, exist_ok=True)
        mols, valid_flags, sims = evaluator.get_mols_and_valid_flags()
        for i, (mol, mask, final, sim) in enumerate(zip(mols, valid_flags, final_flags, sims, strict=False)):
            if mask:
                out = base_dir / f"{gen_path.replace('/', '-')!s}-{final}-sim{sim:.3f}-{i}.png"
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

pprint(errors)
