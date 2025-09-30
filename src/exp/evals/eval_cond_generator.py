import math
import os
from pathlib import Path
from pprint import pprint

import normflows as nf
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

from src.encoding.configs_and_constants import QM9_SMILES_HRR_1600_CONFIG
from src.encoding.oracles import Oracle
from src.generation.evaluator import GenerationEvaluator, rdkit_logp
from src.generation.generation import Generator
from src.generation.logp_regressor import LogPRegressor
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
draw = True
base_dataset = "qm9"
evaluator = None

n_samples = 10
strict_decoder = False
beam_size = 64
expand_on_n_anchors = 9
USE_PAIR_FEASIBILITY = False
generation = "conditional_generation"

errors: dict[str, str] = {}
# Iterate all the checkpoints
loop = 0
gen_paths = list(
    find_files(
        start_dir=GLOBAL_MODEL_PATH / "0_real_nvp_v2",
        prefixes=("epoch",),
        desired_ending=".ckpt",
        skip_substrings=("zinc",),
    )
)
print(f"Found {len(gen_paths)} generator checkpoints")
for gen_ckpt_path in gen_paths:
    # print(f"loop #{loop}")
    # if loop >= 1:
    #     break
    # loop += 1
    do = {
        # QM9 TOP
        "0_real_nvp_v2/nvp_qm9_h1600_f12_hid1024_s42_lr5e-4_wd1e-4_an/models/epoch42-val-4172.5571.ckpt",
        "0_real_nvp_v2/nvp_qm9_h1600_f12_hid384_s42_lr1e-3_wd0.0_an/models/epoch26-val-2516.5386.ckpt",
        "0_real_nvp_v2/nvp_qm9_h1600_f12_hid1024_s42_lr5e-4_wd0.0_an/models/epoch50-val-4308.1338.ckpt",
        "0_real_nvp_v2/nvp_qm9_f10_hid1536_lr0.000513397_wd0.0001_bs55_smf5.54071_smi1.94859_smw15_an/models/epoch79-val-4466.5771.ckpt",
        "0_real_nvp_v2/nvp_qm9_f10_hid1472_lr0.000513196_wd0.0001_bs56_smf5.41974_smi1.39839_smw15_an/models/epoch75-val-4311.4238.ckpt",
        "0_real_nvp_v2/nvp_qm9_h1600_f8_hid512_s42_lr1e-3_wd1e-4_an/models/epoch12-val-1689.4788.ckpt", #TOP
        "0_real_nvp_v2/nvp_qm9_h1600_f12_hid768_s42_lr1e-3_wd0.0_an/models/epoch12-val-2198.1746.ckpt",
        "0_real_nvp_v2/nvp_qm9_h1600_f12_hid768_s42_lr1e-3_wd1e-4_an/models/epoch13-val-2190.4648.ckpt",
        "0_real_nvp_v2/nvp_qm9_h1600_f8_hid512_s42_lr5e-4_wd0.0_an/models/epoch41-val-3322.6777.ckpt",
        "0_real_nvp_v2/nvp_qm9_h1600_f12_hid512_s42_lr1e-3_wd0.0_an/models/epoch19-val-2442.2910.ckpt",  # 30 / 100
        "0_real_nvp_v2/nvp_qm9_h1600_f8_hid512_s42_lr1e-3_wd0.0_an/models/epoch25-val-2071.7695.ckpt",
        "0_real_nvp_v2/nvp_qm9_f11_hid1408_lr0.000512915_wd0.0001_bs53_smf7.2233_smi2.25368_smw15_an/models/epoch25-val-3257.9829.ckpt", #TOP
        "0_real_nvp_v2/nvp_qm9_f11_hid832_lr0.000518923_wd0.0001_bs64_smf5.99953_smi0.999355_smw15_an/models/epoch53-val-4126.5376.ckpt",
        "0_real_nvp_v2/nvp_qm9_f13_hid896_lr0.000513923_wd0.0001_bs56_smf6.10179_smi0.778424_smw18_an/models/epoch24-val-3298.2949.ckpt", #TOP
        "0_real_nvp_v2/nvp_qm9_f14_hid512_lr0.000514306_wd0.0001_bs58_smf5.76312_smi0.1_smw14_an/models/epoch36-val-3579.4844.ckpt",
        "0_real_nvp_v2/nvp_qm9_h1600_f6_hid1024_s42_lr1e-3_wd1e-4_an/models/epoch126-val-2239.6909.ckpt",
        "0_real_nvp_v2/nvp_qm9_f10_hid1472_lr0.000512529_wd0.0001_bs55_smf4.08773_smi0.880154_smw15_an/models/epoch41-val-3605.5576.ckpt", #TOP
        "0_real_nvp_v2/nvp_qm9_f10_hid1472_lr0.000509395_wd0.0001_bs57_smf5.74707_smi0.1_smw15_an/models/epoch27-val-3149.5359.ckpt", #TOP
        "0_real_nvp_v2/nvp_qm9_f10_hid1472_lr0.000512756_wd0.0001_bs54_smf5.60543_smi2.9344_smw15_an/models/epoch45-val-3927.9780.ckpt", #TOP
        "0_real_nvp_v2/nvp_qm9_f10_hid1344_lr0.00051444_wd0.0001_bs45_smf5.55412_smi0.730256_smw11_an/models/epoch19-val-2999.3181.ckpt", #TOP
        # ZINC TOP
        # "0_real_nvp_v2/nvp_zinc_h7744_f4_hid256_s42_lr1e-3_wd1e-4_an/models/epoch16-val-2940.8835.ckpt",
        # "0_real_nvp_v2/nvp_zinc_h7744_f12_hid384_s42_lr1e-3_wd0.0_an/models/epoch13-val-14633.5537.ckpt",
        # "0_real_nvp_v2/nvp_zinc_h7744_f8_hid512_s42_lr5e-4_wd0.0_an/models/epoch21-val-15153.5127.ckpt",
        # "0_real_nvp_v2/nvp_zinc_h7744_f8_hid512_s42_lr1e-3_wd0.0_noan/models/epoch10-val-10562.5449.ckpt",
        # "0_real_nvp_v2/nvp_zinc_h7744_f6_hid512_s42_lr1e-3_wd0.0_an/models/epoch12-val-9837.6328.ckpt",
        # "0_real_nvp_v2/nvp_zinc_h7744_f12_hid1024_s42_lr5e-4_wd0.0_an/models/epoch13-val-18494.6348.ckpt",
        # "0_real_nvp_v2/nvp_zinc_h7744_f4_hid256_s42_lr1e-3_wd0.0_an/models/epoch16-val-2688.6597.ckpt",
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

    lpr_path = "lpr/lpr-best-epoch126-val0.0207.ckpt"
    logp_regressor = LogPRegressor.load_from_checkpoint(GLOBAL_MODEL_PATH / lpr_path, map_location=device, strict=True)

    gen_model.eval()
    logp_regressor.eval()
    for p in gen_model.parameters():
        p.requires_grad = False
    for p in logp_regressor.parameters():
        p.requires_grad = False

    gen_model.to(device)
    logp_regressor.to(device)

    # ------------------ schedules (factories) ------------------

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

    # ---- Hyperparams ----
    B = n_samples
    latent_dim = gen_model.flat_dim
    target = 0.0
    # lambda_prior = 1e-2  # strength of N(0,I) prior pull on z
    lambda_hi = 1e-2 # [5e-3, 1e-2]
    lambda_lo = 1e-4 # [1e-4, 1e-3]
    steps = 50
    lr = 1e-3
    epsilon = 0.2  # success@epsilon threshold for logP

    # ---- Initialize latents ----
    # z = torch.randn(B, latent_dim, device=device, requires_grad=True)
    # z = nf.distributions.DiagGaussian((B, latent_dim), trainable=False)
    base = nf.distributions.DiagGaussian(latent_dim, trainable=False).to(device)
    z = base.sample(B)
    z = z.detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    # Pick one schedule (no lambda keyword)
    sched, sched_name = make_lambda_cosine_decay(steps=steps, lam_hi=lambda_hi, lam_lo=lambda_lo), "cosine"
    # sched = make_lambda_linear_decay(steps=steps, lam_hi=lambda_hi, lam_lo=lambda_lo)
    # sched = make_lambda_two_phase(steps=steps, lam_hi=lambda_hi, lam_lo=lambda_lo, warm_frac=0.2)

    min_loss = float("inf")
    # ---- Guidance loop ----
    pbar = tqdm(range(steps), desc="Steps", unit="step")
    for s in pbar:
        hdc = gen_model.decode_from_latent(z)  # needs grad
        y_hat = logp_regressor.gen_forward(hdc)

        # # Test y_hat of the normal sampling
        # hdc_normal, _ = gen_model.sample(n_samples)
        # y_hat_normal = logp_regressor.gen_forward(hdc_normal)

        lam = sched(s)  # <-- annealed prior weight
        target_tensor = torch.full_like(y_hat, float(target))
        loss = ((y_hat - target_tensor) ** 2).mean() + lam * z.pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        min_loss = min(min_loss, loss.item())

        # update tqdm display
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "λ_prior": f"{lam:.2e}"})
        # # optional: small jitter every so often to encourage diversity
        # if (s + 1) % 25 == 0:
        #     with torch.no_grad():
        #         z.mul_(0.98).add_(0.02 * torch.randn_like(z))

    # ---- Decode once more and take the results ----
    with torch.no_grad():
        hdc = gen_model.decode_from_latent(z)
        y_pred = logp_regressor.gen_forward(hdc)

    # ---- Success@epsilon (using model's prediction; replace with RDKit eval if you want) ----
    hits = (y_pred - target).abs() <= epsilon
    success_rate = hits.float().mean().item()
    mae = (y_pred - target).abs().mean().item()

    # Only evaluate the hits
    hits = [s for s, h in zip(hdc, hits, strict=True) if h]
    n, g = gen_model.split(torch.stack(hits))
    # n, g = gen_model.split(hdc)

    ## Actually decode and evaluate on rdkit
    ## Classifier and Oracle
    # classifier_ckpt = GLOBAL_MODEL_PATH / "1_gin/gin-f_baseline_qm9_resume/models/epoch10-val0.3359.ckpt"
    # classifier_ckpt = GLOBAL_MODEL_PATH / "1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt"
    # classifier_ckpt = GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_large_qm9/models/epoch16-val0.2740.ckpt"
    classifier_ckpt = GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_med_qm9/models/epoch19-val0.2648.ckpt"
    model_type = get_model_type(classifier_ckpt)
    classifier = (
        registery.retrieve_model(name=model_type)
        .load_from_checkpoint(classifier_ckpt, map_location="cpu", strict=True)
        .to(device)
        .eval()
    )

    # Read the metrics from training
    evals_dir = classifier_ckpt.parent.parent / "evaluations"
    epoch_metrics = pd.read_parquet(evals_dir / "epoch_metrics.parquet")

    val_loss = "val_loss" if "val_loss" in epoch_metrics.columns else "val_loss_cb"
    best = epoch_metrics.loc[epoch_metrics[val_loss].idxmin()].add_suffix("_best")
    oracle_threshold = best["val_best_thr_best"]
    print(f"Oracle Threshold: {oracle_threshold}")
    last = epoch_metrics.iloc[-1].add_suffix("_last")

    oracle = Oracle(model=classifier, model_type=model_type)

    decoder_settings = {
        "beam_size": beam_size,
        "oracle_threshold": oracle_threshold,
        # "strict": False,
        "use_pair_feasibility": USE_PAIR_FEASIBILITY,
        "expand_on_n_anchors": expand_on_n_anchors,
    }
    generator = Generator(
        gen_model=gen_model,
        oracle=oracle,
        ds_config=QM9_SMILES_HRR_1600_CONFIG,
        decoder_settings=decoder_settings,
        device=device,
    )

    nx_graphs, final_flag, sims = generator.decode(node_terms=n, graph_terms=g)
    # nx_graphs, final_flag, sims = generator.generate_most_similar(n_samples=n_samples, only_final_graphs=False)

    results = {
        "gen_model": str(gen_ckpt_path),
        "logp_regressor": str(lpr_path),
        "n_samples": n_samples,
        "target": 0.0,
        "min_gda_loss": min_loss,
        "lambda_scheduler": sched_name,
        "lambda_hi": lambda_hi,
        "lambda_low": lambda_lo,
        "steps": steps,
        "lr": lr,
        "epsilon": epsilon,
        "initial_success_rate": success_rate,
        "classifier": classifier_ckpt.parent.parent.stem,
        **decoder_settings,
    }
    if not evaluator:
        evaluator = GenerationEvaluator(base_dataset=base_dataset, device=device)

    # results.update(evaluator.evaluate(samples=nx_graphs, final_flags=final_flag, sims=sims))
    evals = evaluator.evaluate_conditional(samples=nx_graphs, target=target, final_flags=final_flag, eps=epsilon)
    results.update({f"eval_{k}": v for k, v in evals.items()})
    pprint(results)

    if draw:
        base_dir = GLOBAL_ARTEFACTS_PATH / generation / f"drawings_valid_strict-decoder-{strict_decoder}"
        base_dir.mkdir(parents=True, exist_ok=True)
        mols, valid_flags = evaluator.get_mols_and_valid_flags()
        for i, (mol, valid) in enumerate(zip(mols, valid_flags, strict=False)):
            if valid:
                logp = rdkit_logp(mol)
                out = base_dir / f"{gen_ckpt_path.parent.parent.stem}__{evals_dir.parent.stem}__logp{logp:.3f}{i}.png"
                draw_mol(mol=mol, save_path=out, fmt="png")

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = GLOBAL_ARTEFACTS_PATH / generation
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / f"{base_dataset}.parquet"
    csv_path = asset_dir / f"{base_dataset}.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([results])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)
