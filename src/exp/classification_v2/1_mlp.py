import contextlib
import json

import torch.multiprocessing as mp
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.datasets.zinc_pairs_v2 import ZincPairsV2
import contextlib
with contextlib.suppress(RuntimeError):
    mp.set_sharing_strategy("file_system")

import os
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import ZINC_SMILES_HRR_7744_CONFIG
from src.encoding.graph_encoders import AbstractGraphEncoder, load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.exp.classification_v2.classification_utils import (
    Config,
    PairsGraphsDataset,
    ParentH2Cache,
    atomic_save,
    collate_pairs,
    encode_batch,
    encode_g2_with_cache,
    get_args,
    gpu_mem,
)
from src.utils.utils import GLOBAL_MODEL_PATH, pick_device


# ---------- tiny logger ----------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


os.environ.setdefault("PYTHONUNBUFFERED", "1")

def setup_exp(dir_name: str | None = None) -> dict:
    script_path = Path(__file__).resolve()
    experiments_path = script_path.parent
    script_stem = script_path.stem

    base_dir = experiments_path / "results" / script_stem
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Setting up experiment in {base_dir}")
    now = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{''.join(random.choices(string.ascii_lowercase, k=4))}"
    exp_dir = base_dir / now if not dir_name else base_dir / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory created: {exp_dir}")

    dirs = {
        "exp_dir": exp_dir,
        "models_dir": exp_dir / "models",
        "evals_dir": exp_dir / "evaluations",
        "artefacts_dir": exp_dir / "artefacts",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(script_path, exp_dir / script_path.name)
        print(f"Saved a copy of the script to {exp_dir / script_path.name}")
    except Exception as e:
        print(f"Warning: Failed to save script copy: {e}")

    return dirs

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
# MLP classifier on concatenated (h1, h2) – no normalization, GELU, no dropout
class MLPClassifier(nn.Module):
    def __init__(self, hv_dim: int = 88 * 88, hidden_dims: list[int] | None = None):
        """
        hv_dim: dimension of each HRR vector (e.g., 7744)
        hidden_dims: e.g., [4096, 2048, 512, 128]
        """
        super().__init__()
        hidden_dims = hidden_dims or [2048, 1024, 512, 128]
        d_in = hv_dim * 2
        layers: list[nn.Module] = []
        last = d_in
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.GELU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        # h1,h2: [B, hv_dim]
        x = torch.cat([h1, h2], dim=-1)  # [B, 2*D]
        return self.net(x).squeeze(-1)  # [B]


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------


def save_training_checkpoint(
        path: Path,
        *,
        model_state: dict,
        optim_state: dict,
        config: Config,
        epoch: int,
        global_step: int,
        best_metric: float,
        rng_cpu: torch.ByteTensor,
        rng_cuda_all: list[torch.ByteTensor] | None = None,
        sched_state: dict | None = None,
        extra: dict | None = None,
) -> None:
    ckpt = {
        "created_at": datetime.now(tz=UTC).isoformat() + "Z",
        "type": "training_checkpoint",
        "model_state": model_state,
        "optimizer_state": optim_state,
        "scheduler_state": sched_state,
        "config": asdict(config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "rng_state_cpu": rng_cpu,
        "rng_state_cuda": rng_cuda_all,
        "extra": extra or {},
    }
    atomic_save(ckpt, path)


def load_training_checkpoint(path: Path, device: torch.device = torch.device("cpu")) -> dict:
    # map_location avoids surprise GPU alloc; weights_only reduces pickle surface.
    return torch.load(path, map_location=device, weights_only=False)


def save_inference_bundle(
        path: Path,
        *,
        model_state: dict,
        config: Config,
        metadata: dict | None = None,
) -> None:
    bundle = {
        "created_at": datetime.now(UTC).isoformat() + "Z",
        "type": "inference_bundle",
        "model_state": model_state,
        "config": asdict(config),
        "metadata": metadata or {},
    }
    atomic_save(bundle, path)


def make_loader(ds, batch_size, shuffle, cfg, collate_fn):
    kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "pin_memory": (torch.cuda.is_available() and cfg.pin_memory),
        "collate_fn": collate_fn,
        "persistent_workers": False,  # important for memory
        "worker_init_fn": lambda _: torch.set_num_threads(1),  # keep per-worker light
    }
    if cfg.num_workers > 0:  # only valid when workers > 0
        kwargs["prefetch_factor"] = max(1, cfg.prefetch_factor)
    return DataLoader(**kwargs)


def _sanitize_for_parquet(d: dict) -> dict:
    """Make dict Arrow-friendly (Path/Enum/etc → str, tensors → int/float)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, VSAModel):
            out[k] = v.value
        elif torch.is_tensor(v):
            out[k] = v.item() if v.numel() == 1 else v.detach().cpu().tolist()
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------
# Eval + Train
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, encoder, loader, device, criterion, cfg, h2_cache, *, return_details: bool = False,
             max_batches: int | None = 50):
    model.eval();
    encoder.eval()
    ys, ps, ls = [], [], []
    total_loss, total_n = 0.0, 0

    for bi, (g1_b, g2_b, y, parent_ids) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        y = y.to(device)
        h1 = encode_batch(encoder, g1_b, device=device, micro_bs=cfg.micro_bs)
        h2 = encode_g2_with_cache(encoder, g2_b, parent_ids, device, h2_cache, cfg.micro_bs)

        logits = model(h1, h2)
        loss = criterion(logits, y)

        prob = torch.sigmoid(logits).detach().cpu()
        ys.append(y.detach().cpu())
        ps.append(prob)
        if return_details:
            ls.append(logits.detach().cpu())

        total_loss += float(loss.item()) * y.size(0)
        total_n += y.size(0)

    if total_n == 0:
        return {"auc": float("nan"), "ap": float("nan"), "loss": float("nan")}

    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()

    # Safe metrics: small slices may be single-class
    unique = np.unique(y)
    if unique.size < 2:
        auc = float("nan");
        ap = float("nan")
    else:
        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)

    out = {"auc": auc, "ap": ap, "loss": total_loss / max(1, total_n)}
    if return_details:
        out["y"] = y
        out["p"] = p
        out["logits"] = torch.cat(ls).numpy()
    return out


def train(
        train_pairs,
        valid_pairs,
        encoder: AbstractGraphEncoder,
        models_dir: Path,
        evals_dir: Path,
        artefacts_dir: Path,
        cfg: Config,
        device: torch.device,
        resume_ckpt: dict | None = None,
        resume_retrain_last_epoch=True,
):
    log("In Training ... ")
    log("Setting up datasets …")
    train_ds = PairsGraphsDataset(train_pairs)
    valid_ds = PairsGraphsDataset(valid_pairs)
    log(f"Datasets ready. train_size={len(train_ds):,} valid_size={len(valid_ds):,}")

    log("Setting up dataloaders …")
    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg, collate_pairs)
    valid_loader = make_loader(valid_ds, cfg.batch_size, False, cfg, collate_pairs)
    log("In Training ... Data loaders ready.")

    encoder = encoder.to(device).eval()

    # ----- model + optim -----
    hv_dim = (resume_ckpt["config"]["hv_dim"] if resume_ckpt else cfg.hv_dim)
    hidden_dims = (resume_ckpt["config"]["hidden_dims"] if resume_ckpt else cfg.hidden_dims)
    model = MLPClassifier(hv_dim=hv_dim, hidden_dims=hidden_dims).to(device)
    # PairV2: [ALL] size=45749267 | positives=8266188 (18.1%) | negatives=37483079 (81.9%)
    ratio = 37483079 / 8266188
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([ratio])).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    log(f"Model on: {next(model.parameters()).device}")

    base_row = _sanitize_for_parquet(
        {
            **asdict(cfg),
            "train_size": len(train_pairs),
            "valid_size": len(valid_pairs),
            "model_params": sum(p.numel() for p in model.parameters()),
            "type": "MLPClassifier",
            "hidden_dims": cfg.hidden_dims,
            "activation": "GELU",
            "dropout": 0.0,
            "normalization": "None",
        }
    )

    rows, train_losses, valid_losses, valid_aucs = [], [], [], []
    best_auc, global_step = -float("inf"), 0

    start_epoch = 1
    if resume_ckpt is not None:
        # 1) model/optim
        model.load_state_dict(resume_ckpt["model_state"], strict=True)
        optim.load_state_dict(resume_ckpt["optimizer_state"])

        # 3) bookkeeping
        best_auc = float(resume_ckpt.get("best_metric", best_auc))
        global_step = int(resume_ckpt.get("global_step", 0))
        last_epoch = int(resume_ckpt.get("epoch", 0))

        # 4) RNG (optional but good for determinism)
        try:
            torch.set_rng_state(resume_ckpt["rng_state_cpu"])
            if torch.cuda.is_available() and resume_ckpt.get("rng_state_cuda") is not None:
                torch.cuda.set_rng_state_all(resume_ckpt["rng_state_cuda"])
        except Exception as e:
            log(f"Resume: RNG restore failed ({e}); continuing without exact determinism.")

        # 5) where to continue
        start_epoch = last_epoch if resume_retrain_last_epoch else (last_epoch + 1)
        last_epoch = int(resume_ckpt.get("epoch", 0))
        log(f"Resuming from epoch={last_epoch}, step={global_step}, best_auc={best_auc:.4f}. "
            f"Continuing at epoch {start_epoch}.")

    g2_cache_train = ParentH2Cache(max_items=50_000)
    g2_cache_valid = ParentH2Cache(max_items=5_000)

    # Save the run config for provenance
    (evals_dir / "run_config.json").write_text(json.dumps(base_row, indent=2))

    # Since training can take long, we save based on time
    last_save_t = time.time()

    log(f"Starting training on {device} {gpu_mem(device)}")
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        epoch_loss_sum, n_seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"train e{epoch}", dynamic_ncols=True)
        t0 = time.time()

        for g1_b, g2_b, y, parent_ids in pbar:
            n = y.size(0)
            n_seen += n
            y = y.to(device)

            # Precompute encodings (no grad) to keep training simple and fast
            with torch.no_grad():
                h1 = encode_batch(encoder, g1_b, device=device, micro_bs=cfg.micro_bs)
                h2 = encode_g2_with_cache(encoder, g2_b, parent_ids, device, g2_cache_train, cfg.micro_bs)

            logits = model(h1, h2)
            loss = criterion(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            epoch_loss_sum += float(loss.item()) * n
            global_step += 1
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}")

            should_save = cfg.save_every_seconds and (time.time() - last_save_t) >= cfg.save_every_seconds
            if should_save:
                snap_path = models_dir / f"autosnap_e{epoch:03d}_s{global_step:08d}.pt"
                save_training_checkpoint(
                    path=snap_path,
                    model_state=model.state_dict(),
                    optim_state=optim.state_dict(),
                    sched_state=None,
                    config=cfg,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_auc,
                    rng_cpu=torch.get_rng_state(),
                    rng_cuda_all=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    extra={"activation": "GELU", "normalization": "None"},
                )
                last_save_t = time.time()
                log(f"[autosave] saved {snap_path.name}")

                # --- periodic mid-epoch validation After saving also eval (Or before it?) so we can save the best?---
                mid = evaluate(model, encoder, valid_loader, device, criterion, cfg, g2_cache_valid,
                               return_details=False)
                valid_losses.append(mid["loss"])
                valid_aucs.append(mid["auc"])
                rows.append(_sanitize_for_parquet({
                    **base_row,
                    "epoch": epoch,
                    "train_loss": float(epoch_loss_sum / max(1, n_seen)),
                    "valid_loss": float(mid["loss"]),
                    "valid_auc": float(mid["auc"]),
                    "valid_ap": float(mid["ap"]),
                    "global_step": int(global_step),
                    "mid_epoch": True,
                }))
                log(f"[mid-eval step {global_step}] valid_loss={mid['loss']:.6f} valid_auc={mid['auc']} valid_ap={mid['ap']}")
                # optional: update best on mid-epoch too
                if (mid["auc"] == mid["auc"]) and (mid["auc"] > best_auc):  # NaN-safe compare
                    best_auc = mid["auc"]
                    save_inference_bundle(path=models_dir / "inf-best-auc.pt", model_state=model.state_dict(),
                                          config=cfg)
                    # Optional: inference bundle here using your save_inference_bundle(...)

        train_loss = epoch_loss_sum / max(1, n_seen)
        train_losses.append(train_loss)
        log(f"[epoch {epoch}] train_loss={train_loss:.6f} | time={time.time() - t0:.1f}s")

        metrics = evaluate(model, encoder, valid_loader, device, criterion, cfg, g2_cache_valid, max_batches=None)
        valid_losses.append(metrics["loss"])
        valid_aucs.append(metrics["auc"])
        log(
            f"[epoch {epoch}] valid_loss={metrics['loss']:.6f}  valid_auc={metrics['auc']:.4f}  valid_ap={metrics['ap']:.4f}"
        )

        rows.append(
            _sanitize_for_parquet(
                {
                    **base_row,
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "valid_loss": float(metrics["loss"]),
                    "valid_auc": float(metrics["auc"]),
                    "valid_ap": float(metrics["ap"]),
                    "global_step": int(global_step),
                }
            )
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), models_dir / "best.pt")
            log(f"[epoch {epoch}] saved best.pt (auc={best_auc:.4f})")

    # Always save a final checkpoint
    torch.save(model.state_dict(), models_dir / "last.pt")
    log("Saved last.pt")

    # ---- persist metrics history ----
    df = pd.DataFrame(rows)
    parquet_path = evals_dir / "metrics.parquet"
    if parquet_path.exists():
        try:
            old = pd.read_parquet(parquet_path)
            df = pd.concat([old, df], ignore_index=True)
        except Exception as e:
            log(f"Warning: failed to read/append existing parquet ({e}); writing fresh file.")
    df.to_parquet(parquet_path, index=False)
    log(f"Saved parquet → {parquet_path}")

    # ---- training curves ----
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.title("BCE loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(artefacts_dir / "loss.png")
    plt.close()

    plt.figure()
    plt.plot(valid_aucs)
    plt.title("Valid ROC-AUC")
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    plt.savefig(artefacts_dir / "auc.png")
    plt.close()

    # ---- detailed final eval on valid (for analysis artefacts) ----
    details = evaluate(model, encoder, valid_loader, device, criterion, cfg, g2_cache_valid, return_details=True)
    y = details["y"]
    p = details["p"]
    logits = details["logits"]

    # Save per-sample predictions for offline slicing
    valid_preds = pd.DataFrame(
        {
            "y": y.astype(int),
            "p": p.astype(float),
            "logit": logits.astype(float),
            # parent_ids — gather by re-running a light pass to collect ids only
        }
    )
    # Collect parent_ids aligned with valid_loader order
    pid_list = []
    for _, _, _, parent_ids in valid_loader:
        pid_list.extend(parent_ids.tolist())
    valid_preds["parent_id"] = pid_list[: len(valid_preds)]
    valid_preds.to_parquet(evals_dir / "valid_predictions.parquet", index=False)

    # ROC curve
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC (AUC={details['auc']:.4f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(artefacts_dir / "roc.png")
    plt.close()

    # PR curve
    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure()
    plt.plot(rec, prec)
    plt.title(f"PR (AP={details['ap']:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(artefacts_dir / "pr.png")
    plt.close()

    # Probability histogram
    plt.figure()
    plt.hist(p, bins=50)
    plt.title("Validation probability histogram")
    plt.xlabel("P(y=1)")
    plt.ylabel("count")
    plt.savefig(artefacts_dir / "prob_hist.png")
    plt.close()

    # Logit hist by class
    plt.figure()
    plt.hist(logits[y == 1], bins=50, alpha=0.6, label="pos")
    plt.hist(logits[y == 0], bins=50, alpha=0.6, label="neg")
    plt.title("Validation logit histogram")
    plt.xlabel("logit")
    plt.ylabel("count")
    plt.legend()
    plt.savefig(artefacts_dir / "logit_hist.png")
    plt.close()

    # Confusion matrix at 0.5
    y_hat = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    cm_txt = (
        f"Confusion Matrix @0.5\n"
        f"TN={tn}  FP={fp}\n"
        f"FN={fn}  TP={tp}\n"
        f"Accuracy={(tp + tn) / max(1, len(y)):.4f}  "
        f"Precision={tp / max(1, tp + fp):.4f}  "
        f"Recall={tp / max(1, tp + fn):.4f}\n"
    )
    (artefacts_dir / "confusion_matrix.txt").write_text(cm_txt)
    log(cm_txt)

    # Top confident errors (for quick inspection)
    err_mask = y_hat != y
    if err_mask.any():
        errs = valid_preds[err_mask]
        # confidence = distance from 0.5
        errs = errs.assign(confidence=(errs["p"] - 0.5).abs())
        errs = errs.sort_values("confidence", ascending=False).head(200)
        errs.to_csv(artefacts_dir / "top_confident_errors.csv", index=False)
        log(f"Saved top_confident_errors.csv ({len(errs)} rows)")

    log("Training finished.")


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def run_experiment(cfg: Config):
    log("Parsing args done. Starting run_experiment …")
    dirs = setup_exp(cfg.exp_dir_name)
    models_dir = Path(dirs["models_dir"])
    evals_dir = Path(dirs["evals_dir"])
    artefacts_dir = Path(dirs["artefacts_dir"])

    seed_everything(cfg.seed)

    # Dataset & Encoder (HRR @ 7744)
    ds_cfg = ZINC_SMILES_HRR_7744_CONFIG
    device = pick_device()
    log("Loading/creating hypernet …")
    hypernet = (
        load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds_cfg).to(device=device).eval()
    )
    log("Hypernet ready.")

    # Datasets
    log("Loading pair datasets …")
    train_full = ZincPairsV2(split="train", base_dataset=ZincSmiles(split="train"))
    valid_full = ZincPairsV2(split="valid", base_dataset=ZincSmiles(split="valid"))
    log(f"Pairs loaded. train_pairs={len(train_full)} valid_pairs={len(valid_full)}")

    # --- pick N parents & wrap as Subset ---
    train_indices = []
    if cfg.train_parents_start or cfg.train_parents_end:
        t_start = cfg.train_parents_start or 0
        t_end = cfg.train_parents_end or len(train_full)
        train_indices = range(t_start, t_end)
        log(f"Using training data from {t_start} to {t_end}.")

    valid_indices = []
    if cfg.valid_parents_start or cfg.valid_parents_end:
        v_start = cfg.valid_parents_start or 0
        v_end = cfg.valid_parents_end or len(valid_full)
        valid_indices = range(v_start, v_end)
        log(f"Using validation data from {v_start} to {v_end}.")

    train_small = torch.utils.data.Subset(train_full, train_indices) if train_indices else train_full
    valid_small = torch.utils.data.Subset(valid_full, valid_indices) if valid_indices else valid_full

    log(f"[subset] train_indices={len(train_small):,}  valid_indices={len(valid_small):,}")

    # --- resume if requested ---
    resume_ckpt = None
    if cfg.continue_from is not None:
        try:
            # map to CPU; we'll .to(device) after constructing modules
            resume_ckpt = load_training_checkpoint(cfg.continue_from, device=torch.device("cpu"))
            log(f"Loaded checkpoint: {cfg.continue_from.name}")
            # Optional: sanity-check shapes
            ck = resume_ckpt["config"]
            if (ck.get("hv_dim") != cfg.hv_dim) or (ck.get("hidden_dims") != cfg.hidden_dims):
                log("Warning: overriding hv_dim/hidden_dims from checkpoint to ensure compatibility.")
                cfg.hv_dim = ck["hv_dim"]
                cfg.hidden_dims = ck["hidden_dims"]
        except TypeError:
            # Fallback if your torch version doesn't support weights_only
            resume_ckpt = torch.load(cfg.continue_from, map_location="cpu")
        except Exception as e:
            log(f"Failed to load checkpoint {cfg.continue_from}: {e}")
            resume_ckpt = None

    train(
        train_pairs=train_small,
        valid_pairs=valid_small,
        encoder=hypernet,
        models_dir=models_dir,
        evals_dir=evals_dir,
        artefacts_dir=artefacts_dir,
        cfg=cfg,
        device=device,
        resume_ckpt=resume_ckpt,
        resume_retrain_last_epoch=cfg.resume_retrain_last_epoch,
    )


if __name__ == "__main__":
    log(f"Running {Path(__file__).resolve()}")
    cfg = get_args()
    pprint(asdict(cfg), indent=2)
    run_experiment(cfg)
