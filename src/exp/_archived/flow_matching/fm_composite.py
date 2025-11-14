import datetime
import enum
import json
import os
import random
import shutil
import string
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torchhd import HRRTensor

from src.datasets.utils import get_split
from src.encoding.configs_and_constants import Features, SupportedDataset
from src.encoding.correction_utilities import target_reached
from src.encoding.graph_encoders import CorrectionLevel, load_or_create_hypernet
from src.exp.real_nvp_hpo.hpo.folder_name import make_run_folder_name
from src.generation.analyze import analyze_terms_only
from src.normalizing_flow.models import FMConfig
from src.utils.registery import resolve_model, retrieve_model
from src.utils.utils import GLOBAL_MODEL_PATH, pick_device

DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
# ---------- tiny logger ----------
def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

if os.getenv("CLUSTER") == "local":
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"


def setup_exp(dir_name: str | None = None) -> dict:
    script_path = Path(__file__).resolve()
    experiments_path = script_path.parent
    script_stem = script_path.stem

    base_dir = experiments_path / "results" / script_stem
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Setting up experiment in {base_dir}")
    if dir_name:
        exp_dir = base_dir / dir_name
    else:
        slug = f"{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d_%H-%M-%S')}_{''.join(random.choices(string.ascii_lowercase, k=4))}"
        exp_dir = base_dir / slug
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


@torch.no_grad()
def fit_featurewise_standardization(
    model, loader, hv_count: int, hv_dim: int, max_batches: int | None = None, device="cpu"
):
    """
    Estimate per-feature mean and std (feature-wise) for the model's standardized space.
    Works safely under mixed-precision and supports [B, 3D] inputs.

    Parameters
    ----------
    model : FlowMatchingLightning
        Model whose `_flat_from_batch` defines the feature layout.
    loader : DataLoader
        Provides graph batches.
    hv_dim : int
        Hypervector dimension D.
    max_batches : int | None
        Limit number of batches for faster estimation.
    device : str | torch.device
        Device to perform accumulation on.
    """
    cnt = 0
    accum_dtype = DTYPE
    sum_vec = torch.zeros(hv_count * hv_dim, dtype=accum_dtype, device=device)
    sumsq_vec = torch.zeros(hv_count * hv_dim, dtype=accum_dtype, device=device)

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        batch = batch.to(device)
        x = model._flat_from_batch(batch).to(accum_dtype)  # [B, 3D]
        cnt += x.shape[0]
        sum_vec += x.sum(dim=0)
        sumsq_vec += (x * x).sum(dim=0)

    if cnt == 0:
        msg = "fit_featurewise_standardization(): empty loader or no samples."
        raise RuntimeError(msg)

    mu = sum_vec / cnt
    var = (sumsq_vec / cnt - mu**2).clamp_min_(0)
    sigma = var.sqrt().clamp_min_(1e-6)

    # Match model's dtype to avoid Double↔Half/Float mismatches
    tgt_dtype = model.mu.dtype if hasattr(model, "mu") else torch.get_default_dtype()
    model.set_standardization(mu.to(tgt_dtype), sigma.to(tgt_dtype))


@torch.no_grad()
def fit_per_term_standardization(model, loader, hv_dim: int, max_batches: int | None = None, device="cpu"):
    """
    Compute separate standardization for edge_terms and graph_terms.
    Each term gets its own mu and sigma vectors, then concatenated.

    Parameters
    ----------
    model : RealNVPV2Lightning
        Model whose buffers will be updated.
    loader : DataLoader
        Provides graph batches.
    hv_dim : int
        Hypervector dimension D (NOT 2*D).
    max_batches : int | None
        Limit number of batches for faster estimation.
    device : str | torch.device
        Device to perform accumulation on.
    """
    cnt = 0
    accum_dtype = DTYPE

    # Separate accumulators for edge and graph terms
    sum_edge = torch.zeros(hv_dim, dtype=accum_dtype, device=device)
    sumsq_edge = torch.zeros(hv_dim, dtype=accum_dtype, device=device)
    sum_graph = torch.zeros(hv_dim, dtype=accum_dtype, device=device)
    sumsq_graph = torch.zeros(hv_dim, dtype=accum_dtype, device=device)

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        batch = batch.to(device)
        edge = batch.edge_terms.to(accum_dtype)  # [B, D]
        graph = batch.graph_terms.to(accum_dtype)  # [B, D]

        cnt += edge.shape[0]

        sum_edge += edge.sum(dim=0)
        sumsq_edge += (edge * edge).sum(dim=0)
        sum_graph += graph.sum(dim=0)
        sumsq_graph += (graph * graph).sum(dim=0)

    if cnt == 0:
        msg = "fit_per_term_standardization(): empty loader or no samples."
        raise RuntimeError(msg)

    # Compute statistics for each term
    mu_edge = sum_edge / cnt
    var_edge = (sumsq_edge / cnt - mu_edge**2).clamp_min_(0)
    sigma_edge = var_edge.sqrt().clamp_min_(1e-6)

    mu_graph = sum_graph / cnt
    var_graph = (sumsq_graph / cnt - mu_graph**2).clamp_min_(0)
    sigma_graph = var_graph.sqrt().clamp_min_(1e-6)

    # Concatenate to match model's expected [2*D] format
    mu = torch.cat([mu_edge, mu_graph])
    sigma = torch.cat([sigma_edge, sigma_graph])

    # Set in model (using existing method)
    tgt_dtype = model.mu.dtype if hasattr(model, "mu") else torch.get_default_dtype()
    model.set_standardization(mu.to(tgt_dtype), sigma.to(tgt_dtype))

    # Store the split point for per-term transforms
    model._per_term_split = hv_dim
    log(f"Per-term standardization fitted: edge_terms [:{hv_dim}], graph_terms [{hv_dim}:]")


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


def plot_train_val_loss(
    df: pd.DataFrame,
    artefacts_dir: Path,
    *,
    skip_first: float = 0.1,  # int = epochs to skip; float in (0,1) = fraction to skip
    min_epoch: int | None = None,  # overrides skip_first if set
    smooth_window: int | None = None,  # rolling mean window (points)
    clip_q: tuple[float, float] = (0.02, 0.98),  # robust y-limits from quantiles
    logy: bool = False,
) -> None:
    """Plot train/val loss robustly (drops non-finite, clips outliers, averages per epoch)."""

    def _first_existing(cols: list[str]) -> str | None:
        for c in cols:
            if c in df.columns:
                return c
        return None

    epoch_col = _first_existing(["epoch", "step"])
    train_col = _first_existing(["train_loss_epoch", "train_loss", "epoch_train_loss"])
    val_col = _first_existing(["val_loss", "val/loss", "validation_loss"])

    if epoch_col is None or (train_col is None and val_col is None):
        print("No epoch/metric columns; skipping plot.")
        return

    # --- define cutoff ---
    uniq = pd.unique(df[epoch_col].dropna())
    try:
        uniq = np.sort(uniq.astype(int))
    except Exception:
        uniq = np.sort(uniq)
    if min_epoch is not None:
        cutoff = min_epoch
    else:
        if isinstance(skip_first, float) and 0 < skip_first < 1:
            k = int(round(skip_first * len(uniq)))
        else:
            k = int(skip_first)
        k = max(0, min(k, max(len(uniq) - 1, 0)))
        cutoff = uniq[k] if len(uniq) else 0

    df_f = df[df[epoch_col] >= cutoff].copy()

    def _series(col: str | None):
        if col is None or col not in df_f:
            return None
        s = (
            df_f[[epoch_col, col]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .groupby(epoch_col, as_index=False)[col]
            .mean()
            .rename(columns={col: "y"})
        )
        if s.empty:
            return None
        if smooth_window and smooth_window > 1:
            s["y"] = s["y"].rolling(smooth_window, min_periods=1).mean()
        return s

    train_s = _series(train_col)
    val_s = _series(val_col)

    if train_s is None and val_s is None:
        print("Nothing to plot after filtering; skipping.")
        return

    plt.figure(figsize=(10, 6))
    if train_s is not None:
        plt.plot(train_s[epoch_col].to_numpy(), train_s["y"].to_numpy(), label=f"{train_col} (≥{cutoff})")
    if val_s is not None:
        plt.plot(val_s[epoch_col].to_numpy(), val_s["y"].to_numpy(), label=f"{val_col} (≥{cutoff})")

    # --- robust y-limits (two-sided quantiles on combined series) ---
    ys = []
    if train_s is not None:
        ys.append(train_s["y"].to_numpy())
    if val_s is not None:
        ys.append(val_s["y"].to_numpy())
    if ys:
        y_all = np.concatenate(ys)
        if y_all.size:
            qlo, qhi = np.quantile(y_all, clip_q)
            if np.isfinite(qlo) and np.isfinite(qhi) and qhi > qlo:
                plt.ylim(qlo, qhi)

    # --- log scale only if strictly positive ---
    if logy:
        yb, yt = plt.gca().get_ylim()
        if yb > 0 and yt > 0:
            plt.yscale("log")
        else:
            print("logy requested but data ≤ 0; using linear scale.")

    plt.xlabel(epoch_col)
    plt.ylabel("loss")
    plt.title("Train vs. Validation Loss")
    plt.legend()
    plt.tight_layout()

    artefacts_dir.mkdir(parents=True, exist_ok=True)
    out = artefacts_dir / "loss_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved train/val loss plot to {out} (cutoff ≥ {cutoff})")


def on_a100() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        return False
    return "A100" in name  # simple & reliable enough


def pick_precision() -> int | str:
    """
    Choose training precision. BF16-mixed provides 2-3x speedup on A100/H100 GPUs
    while maintaining stability for most deep learning workloads.

    Override via PRECISION env var: PRECISION=64 for full FP64 if needed
    """
    # explicit override via env if you want: PRECISION in {"64","32","bf16-mixed","16-mixed"}
    p = os.getenv("PRECISION")
    if p:  # trust user override
        return int(p) if p.isdigit() else p

    # Default to BF16-mixed on modern GPUs for 2-3x speedup
    # Set PRECISION=64 if you need full FP64 precision
    if torch.cuda.is_available() and on_a100():
        return "bf16-mixed"  # 2-3x faster than FP64 on A100/H100
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16-mixed"  # local GPUs: fast & stable
    if torch.cuda.is_available():
        return "16-mixed"  # widest fallback
    return 32


def configure_tf32(precision):
    """Enable TF32 only when using bf16-mixed or 32-bit precision on Ampere+ GPUs."""
    if not torch.cuda.is_available():
        return

    name = torch.cuda.get_device_name(0)
    if "A100" in name or "H100" in name or "RTX 30" in name or "RTX 40" in name:
        if precision in ("bf16-mixed", 32, "32"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"[TF32] Enabled for {precision} on {name}")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print(f"[TF32] Disabled for {precision}")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(f"[TF32] Not supported on {name}")


class TimeLoggingCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        trainer.logger.log_metrics({"elapsed_time_sec": elapsed}, step=trainer.current_epoch)


def _summarize_arrays(arrs: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean, std, median, min, max, count for each key."""
    out = {}
    for k, v in arrs.items():
        if v.size == 0:  # skip empties
            continue
        out[f"{k}_mean"] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v))
        out[f"{k}_median"] = float(np.median(v))
        out[f"{k}_min"] = float(np.min(v))
        out[f"{k}_max"] = float(np.max(v))
        out[f"{k}_count"] = int(v.size)
    return out


def _hist(figpath: Path, data: np.ndarray, title: str, xlabel: str, bins: int = 80):
    arr = np.asarray(data).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        log(f"Skip hist {figpath.name}: empty/non-finite data")
        return
    vmin, vmax = float(arr.min()), float(arr.max())
    if np.isclose(vmin, vmax):
        span = 1.0 if vmin == 0.0 else 0.05 * abs(vmin)
        vmin, vmax, bins = vmin - span, vmax + span, 1
    plt.figure(figsize=(6, 4))
    plt.hist(arr, bins=bins, range=(vmin, vmax))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()


def run_experiment(cfg: FMConfig):
    pprint(cfg)
    # ----- setup dirs -----
    log("Parsing args done. Starting run_experiment …")
    dirs = setup_exp(cfg.exp_dir_name)
    exp_dir = Path(dirs["exp_dir"])
    models_dir = Path(dirs["models_dir"])
    evals_dir = Path(dirs["evals_dir"])
    artefacts_dir = Path(dirs["artefacts_dir"])

    # Save the config
    def _json_sanitize(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, enum.Enum):
            return obj.value
        return obj

    (evals_dir / "run_config.json").write_text(
        json.dumps({k: _json_sanitize(v) for k, v in asdict(cfg).items()}, indent=2)
    )

    seed_everything(cfg.seed)

    # Dataset & Encoder (HRR @ 7744)
    ds_cfg = cfg.dataset.default_cfg
    device = pick_device()
    log(f"Using device: {device!s}")

    # ----- datasets / loaders -----
    log(f"Loading {cfg.dataset.default_cfg.base_dataset} pair datasets.")
    train_dataset = get_split(split="train", ds_config=cfg.dataset.default_cfg)
    validation_dataset = get_split(split="valid", ds_config=cfg.dataset.default_cfg)
    log(
        f"Pairs loaded for {cfg.dataset.default_cfg.base_dataset}. train_pairs_full_size={len(train_dataset)} valid_pairs_full_size={len(validation_dataset)}"
    )
    # Ensure dtype compatibility
    train_dataset.data.edge_terms = train_dataset.data.edge_terms.to(DTYPE)
    train_dataset.data.graph_terms = train_dataset.data.graph_terms.to(DTYPE)
    validation_dataset.data.edge_terms = validation_dataset.data.edge_terms.to(DTYPE)
    validation_dataset.data.graph_terms = validation_dataset.data.graph_terms.to(DTYPE)

    log("Loading/creating hypernet …")
    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds_cfg).to(device=device, dtype=DTYPE).eval()
    log(f"Setting hypernet depth to {ds_cfg.hypernet_depth}!")
    hypernet.depth = ds_cfg.hypernet_depth
    log("Hypernet ready.")
    rtol, atol = (1e-4, 1e-6) if torch.float32 == DTYPE else (1e-5, 1e-8)
    assert torch.allclose(
        hypernet.forward(Batch.from_data_list([train_dataset[42]]))["edge_terms"],
        train_dataset[42].edge_terms.to(device=device),
        rtol=rtol,
        atol=atol,
    ), "edge terms are not equal"
    assert torch.allclose(
        hypernet.forward(Batch.from_data_list([train_dataset[0]]))["edge_terms"],
        train_dataset[0].edge_terms.to(device=device),
        rtol=rtol,
        atol=atol,
    ), "edge terms are not equal"
    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.NODE_FEATURES][0].codebook)
    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.NODE_FEATURES][0].codebook)
    assert hypernet.nodes_codebook.dtype == DTYPE
    assert hypernet.node_encoder_map[Features.NODE_FEATURES][0].codebook.dtype == DTYPE
    log("Hypernet ready.")

    # pick worker counts per GPU; tune for your cluster
    num_workers = 14 if os.getenv("CLUSTER") != "local" else 8

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=True,
        prefetch_factor=6,  # Increased from 6 for better GPU saturation
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
        prefetch_factor=4,  # Increased from 2 for better throughput
    )
    log(f"Datasets ready. train={len(train_dataset)} valid={len(validation_dataset)}")

    # ----- model / trainer -----
    model = resolve_model("FM", cfg=cfg).to(device=device)

    log(f"Model: {model!s}")
    log(f"Model device: {model.device}")
    log(f"Model hparams: {model.hparams}")

    # Choose standardization method
    if cfg.per_term_standardization:
        log("Using per-term standardization (separate for edge_terms and graph_terms)")
        fit_per_term_standardization(model, train_dataloader, hv_dim=cfg.hv_dim, device=device)
    else:
        log("Using global standardization (all dimensions together)")
        fit_featurewise_standardization(
            model, train_dataloader, hv_count=cfg.hv_count, hv_dim=cfg.hv_dim, device=device
        )

    csv_logger = CSVLogger(save_dir=str(evals_dir), name="logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=str(models_dir),
        auto_insert_metric_name=False,
        filename="epoch{epoch:02d}-val{val_loss:.4f}",
        save_last=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    time_logger = TimeLoggingCallback()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=50,
        min_delta=0.0,
        check_finite=True,  # stop if val becomes NaN/Inf
        verbose=True,
    )

    loggers = [csv_logger]

    precision = pick_precision()
    log(f"Using precision {precision!s}")
    configure_tf32(precision)

    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=loggers,
        callbacks=[checkpoint_callback, lr_monitor, time_logger, early_stopping],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=500,
        enable_progress_bar=True,
        deterministic=False,
        precision=precision,
        num_sanity_val_steps=0,
    )

    # ----- train -----
    t_start = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # ----- curves to parquet / png -----
    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    min_val_loss = float("inf")
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        ## Determine best val loos
        idx = df["val_loss"].idxmin()
        min_val_loss = df.loc[idx, "val_loss"]
        df.to_parquet(evals_dir / "metrics.parquet", index=False)
        plot_train_val_loss(df, artefacts_dir, logy=False)
        # Optional: print final numbers for quick scan
        train_last = df.loc[df["train_loss_epoch"].notna(), "train_loss_epoch"].tail(1)
        val_last = df.loc[df["val_loss"].notna(), "val_loss"].tail(1)
        if not train_last.empty or not val_last.empty:
            print(
                f"Final losses → train: {float(train_last.values[-1]) if not train_last.empty else 'n/a'} "
                f"| val: {float(val_last.values[-1]) if not val_last.empty else 'n/a'}"
            )

    # =================================================================
    # Post-training analysis: load best, sample & log
    # =================================================================
    best_path = checkpoint_callback.best_model_path
    if (not best_path) or ("nan" in Path(best_path).name) or (not Path(best_path).exists()):
        best_path = checkpoint_callback.last_model_path

    if not best_path or not Path(best_path).exists():
        log("No checkpoint found (best/last). Skipping post-training analysis.")
        return 0.0

    log(f"Loading best checkpoint: {best_path}")
    best_model = retrieve_model("FM").load_from_checkpoint(best_path)
    best_model.to(device).eval()
    best_model.to(dtype=DTYPE)

    # ---- sample from the flow ----
    with torch.no_grad():
        samples = best_model.sample_split(1000)  # each [K, D]
        analyze_terms_only(samples, name=ds_cfg.name, pdf_dir=artefacts_dir)
        # node_s = samples["node_terms"]
        edge_s = samples["edge_terms"]
        graph_s = samples["graph_terms"]

    edge_s = edge_s.as_subclass(HRRTensor)
    graph_s = graph_s.as_subclass(HRRTensor)

    # log(f"node_s device: {node_s.device!s}")
    log(f"graph_s device: {graph_s.device!s}")
    log(f"Hypernet node codebook device: {hypernet.nodes_codebook.device!s}")

    edge_np = edge_s.detach().cpu().numpy()
    graph_np = graph_s.detach().cpu().numpy()

    # per-branch norms and pairwise cosine samples
    edge_norm = np.linalg.norm(edge_np, axis=1)
    graph_norm = np.linalg.norm(graph_np, axis=1)

    def _pairwise_cosine(x: np.ndarray, m: int = 2000) -> np.ndarray:
        n = x.shape[0]
        if n < 2:
            return np.array([])
        idx = np.random.choice(n, size=(min(m, n - 1), 2), replace=True)
        a = x[idx[:, 0]]
        b = x[idx[:, 1]]
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.sum(an * bn, axis=1)

    # node_cos = _pairwise_cosine(node_np, m=4000)
    edge_cos = _pairwise_cosine(edge_np, m=4000)
    graph_cos = _pairwise_cosine(graph_np, m=4000)

    # plots
    # _hist(artefacts_dir / "sample_node_norm_hist.png", node_norm, "Sample node L2 norm", "||node||")
    _hist(artefacts_dir / "sample_node_edge_hist.png", edge_norm, "Sample edge L2 norm", "||edge||")
    _hist(artefacts_dir / "sample_graph_norm_hist.png", graph_norm, "Sample graph L2 norm", "||graph||")
    # if node_cos.size:
    #     _hist(artefacts_dir / "sample_node_cos_hist.png", node_cos, "Node pairwise cosine", "cos")
    if edge_cos.size:
        _hist(artefacts_dir / "sample_edge_cos_hist.png", edge_cos, "Edge pairwise cosine", "cos")
    if graph_cos.size:
        _hist(artefacts_dir / "sample_graph_cos_hist.png", graph_cos, "Graph pairwise cosine", "cos")

    # =================================================================
    # NEW: Decode samples and compute validity for composite metric
    # =================================================================
    log("=== Starting validity evaluation for composite metric ===")

    # Determine number of samples to decode based on dataset
    base_dataset = ds_cfg.base_dataset  # "qm9" or "zinc"
    n_decode = 1000

    log(f"Decoding {n_decode} samples for {base_dataset} dataset...")

    # Prepare samples for decoding
    edge_decode = edge_s[:n_decode]  # We only decode this as downstream task

    # Decode samples
    correction_levels = []

    log("Getting Hypernet ready for decoding ...")
    hypernet.base_dataset = base_dataset

    decode_start_time = time.time()
    for i in range(n_decode):
        if (i + 1) % 100 == 0 or i == 0:
            log(f"Decoding sample {i + 1}/{n_decode}...")

        try:
            initial_decoded_edges = hypernet.decode_order_one_no_node_terms(edge_decode[i].as_subclass(HRRTensor))

            if target_reached(initial_decoded_edges):
                correction_levels.append(CorrectionLevel.ZERO)
            else:
                correction_levels.append(CorrectionLevel.FAIL)

        except Exception as e:
            log(f"Warning: Failed to decode sample {i}: {e}")
            correction_levels.append(CorrectionLevel.FAIL)

    decode_elapsed = time.time() - decode_start_time
    log(f"Decoding completed in {decode_elapsed:.2f}s ({decode_elapsed / n_decode:.2f}s per sample)")

    # Compute CorrectionLevel.ZERO percentage (perfect decoding without corrections)
    n_zero_corrections = sum(1 for cl in correction_levels if cl == CorrectionLevel.ZERO)
    zero_correction_pct = 100.0 * n_zero_corrections / n_decode if n_decode else 0.0

    log("Correction Level Distribution:")
    for level in CorrectionLevel:
        count = sum(1 for cl in correction_levels if cl == level)
        pct = 100.0 * count / n_decode if n_decode else 0.0
        log(f"  {level.value}: {count}/{n_decode} ({pct:.2f}%)")
    log(f"CorrectionLevel.ZERO percentage: {zero_correction_pct:.2f}%")

    log("=== Composite Metric Computation ===")
    log(f"Min validation MSE: {min_val_loss:.4f}")
    log(f"CorrectionLevel.ZERO: {zero_correction_pct:.2f}%")

    # Save comprehensive metrics to JSON
    # Store correction level distribution
    correction_level_dist = {
        level.value: sum(1 for cl in correction_levels if cl == level) for level in CorrectionLevel
    }

    metrics_dict = {
        "exp_dir_name": cfg.exp_dir_name,
        "min_val_loss": float(min_val_loss),
        "zero_correction_pct": float(zero_correction_pct),
        "incorrect_pct": 100.0 - float(zero_correction_pct),
        "n_decoded": int(n_decode),
        "base_dataset": base_dataset,
        "decode_time_sec": float(decode_elapsed),
        "decode_time_per_sample_sec": float(decode_elapsed / n_decode),
        "correction_level_distribution": correction_level_dist,
        "training_time": (time.perf_counter() - t_start) // 60,
    }

    metrics_file = evals_dir / "hpo_metrics.json"
    metrics_file.write_text(json.dumps(metrics_dict, indent=2))
    log(f"Saved comprehensive metrics to {metrics_file}")

    log("Experiment completed.")
    log(f"Best val MSE: {min_val_loss:.4f}")
    log(f"CorrectionLevel.ZERO: {zero_correction_pct:.2f}%")
    return min_val_loss, 100.0 - zero_correction_pct


def get_hidden_channel_dist(dataset: SupportedDataset):
    low, high, step = 400, 1600, 400
    # 256-dim datasets (total input: 512 = 256 edge + 256 graph)
    if dataset in [SupportedDataset.QM9_SMILES_HRR_256_F64_G1NG3, SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4]:
        low, high, step = 512, 2048, 512
    elif dataset == SupportedDataset.ZINC_SMILES_HRR_1024_F64_5G1NG4:
        low, high, step = 512, 1024, 256
    elif dataset == SupportedDataset.ZINC_SMILES_HRR_2048_F64_5G1NG4:
        low, high, step = 1024, 2048, 512
    return low, high, step


def get_cfg(trial: optuna.Trial, dataset: SupportedDataset):
    low, high, step = get_hidden_channel_dist(dataset)
    cfg = {
        "batch_size": trial.suggest_int("batch_size", 128, 512, step=128),
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "hidden_dim": trial.suggest_int("hidden_dim", low, high, step=step),
        "num_hidden_layers": trial.suggest_int("num_hidden_layers", 3, 8),
        "time_emb_dim": trial.suggest_categorical("time_emb_dim", [32, 64, 128]),
    }
    flow_cfg = FMConfig()
    for k, v in cfg.items():
        setattr(flow_cfg, k, v)
    flow_cfg.dataset = dataset
    flow_cfg.hv_dim = dataset.default_cfg.hv_dim
    flow_cfg.hv_count = dataset.default_cfg.hv_count
    flow_cfg.vsa = dataset.default_cfg.vsa
    return flow_cfg


def run_zinc_trial(trial: optuna.Trial, dataset: SupportedDataset, norms_per: str):
    flow_cfg = get_cfg(trial, dataset=dataset)
    flow_cfg.per_term_standardization = norms_per == "term"
    flow_cfg.exp_dir_name = make_run_folder_name(
        {k: getattr(flow_cfg, k) for k in keys if k in flow_cfg.__dict__}, prefix=f"fm_comp_{dataset.default_cfg.name}"
    )
    return run_experiment(flow_cfg)


def run_qm9_trial(trial: optuna.Trial, dataset: SupportedDataset, norms_per: str):
    flow_cfg = get_cfg(trial, dataset=dataset)
    flow_cfg.per_term_standardization = norms_per == "term"
    flow_cfg.exp_dir_name = make_run_folder_name(
        {k: getattr(flow_cfg, k) for k in keys if k in flow_cfg.__dict__}, prefix=f"fm_comp_{dataset.default_cfg.name}"
    )
    return run_experiment(flow_cfg)


keys = {
    "batch_size",
    "lr",
    "time_emb_dim",
    "weight_decay",
    "hidden_dim",
    "num_hidden_layers",
    "seed",
    "per_term_standardization",
}
