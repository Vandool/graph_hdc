import datetime
import enum
import json
import os
import random
import shutil
import string
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import Features, SupportedDataset
from src.encoding.graph_encoders import load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.exp.logp_regressor.hpo.folder_name import make_run_folder_name
from src.generation.logp_regressor import ACTS, NORMS
from src.utils.registery import resolve_model, retrieve_model
from src.utils.utils import GLOBAL_MODEL_PATH, pick_device

LOCAL_DEV = "LOCAL_HDC_miss"

PROJECT_NAME = "real_nvp_v2"


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
# ---------- tiny logger ----------
def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


os.environ.setdefault("PYTHONUNBUFFERED", "1")


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


@dataclass
class Config:
    exp_dir_name: str | None = None
    seed: int = 42
    epochs: int = 50
    batch_size: int = 64
    is_dev: bool = os.getenv("IS_DEV", "0") == "1"

    # HDC / encoder
    hv_dim: int = 40 * 40  # 1600
    vsa: VSAModel = VSAModel.HRR
    dataset: SupportedDataset = SupportedDataset.QM9_SMILES_HRR_1600

    # Model
    lr: float = 1e-4
    weight_decay: float = 0.0
    hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256, 128, 64, 32])
    activation: str = "gelu"
    norm: str = "lay_norm"
    dropout: float = 0.0

    # Checkpointing
    continue_from: Path | None = None


def pick_precision():
    # Works on A100/H100 if BF16 is supported by the PyTorch/CUDA build.
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed"  # safest + fast on H100/A100
        return "16-mixed"  # widely supported fallback
    return 32  # CPU or MPS


class TimeLoggingCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):  # noqa: ARG002
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        trainer.logger.log_metrics({"elapsed_time_sec": elapsed}, step=trainer.current_epoch)


class LossCurveCallback(Callback):
    def __init__(self, artefacts_dir: Path):
        super().__init__()
        self.artefacts_dir = artefacts_dir

    def on_train_end(self, trainer, pl_module):
        # metrics.csv is already written by CSVLogger
        metrics_path = Path(trainer.logger.log_dir) / "metrics.csv"
        if not metrics_path.exists():
            return
        df = pd.read_csv(metrics_path)
        # select columns that exist
        cols = [c for c in df.columns if c in ("epoch", "train_loss", "val_loss")]
        if not {"train_loss", "val_loss"} <= set(cols):
            return
        plt.figure(figsize=(8, 6))
        plt.plot(df["epoch"], df["train_loss"], label="train")
        plt.plot(df["epoch"], df["val_loss"], label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss (MSE)")
        plt.legend()
        plt.title("Training vs Validation Loss")
        out_path = self.artefacts_dir / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"[LossCurveCallback] Saved loss curve to {out_path}")


def run_experiment(cfg: Config):
    local_dev = cfg.is_dev
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

    log("Loading/creating hypernet …")
    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds_cfg).to(device=device).eval()
    log("Hypernet ready.")
    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.ATOM_TYPE][0].codebook)
    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.ATOM_TYPE][0].codebook)
    log("Hypernet ready.")

    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.ATOM_TYPE][0].codebook)
    log("Hypernet ready.")

    # ----- datasets / loaders -----
    log(f"Loading {cfg.dataset.value} pair datasets.")
    if cfg.dataset == SupportedDataset.QM9_SMILES_HRR_1600:
        train_dataset = QM9Smiles(split="train", enc_suffix="HRR1600")
        validation_dataset = QM9Smiles(split="valid", enc_suffix="HRR1600")
    elif cfg.dataset == SupportedDataset.ZINC_SMILES_HRR_7744:
        train_dataset = ZincSmiles(split="train", enc_suffix="HRR7744")
        validation_dataset = ZincSmiles(split="valid", enc_suffix="HRR7744")
    log(
        f"Pairs loaded for {cfg.dataset.value}. train_pairs_full_size={len(train_dataset)} valid_pairs_full_size={len(validation_dataset)}"
    )

    # pick worker counts per GPU; tune for your cluster
    num_workers = 16 if torch.cuda.is_available() else 0
    if local_dev:
        train_dataset = train_dataset[: cfg.batch_size]
        validation_dataset = validation_dataset[: cfg.batch_size]
        num_workers = 0

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=True,
        prefetch_factor=None if local_dev else 6,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
        prefetch_factor=None if local_dev else 6,
    )
    log(f"Datasets ready. train={len(train_dataset)} valid={len(validation_dataset)}")

    # ----- model / trainer -----
    # model = RealNVPV2Lightning(cfg)
    model = resolve_model(
        "LPR",
        input_dim=2 * cfg.hv_dim,
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        dropout=cfg.dropout,
        norm=cfg.norm,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    ).to(device=device)

    log(f"Model: {model!s}")
    log(f"Model device: {model.device}")
    log(f"Model hparams: {model.hparams}")

    csv_logger = CSVLogger(save_dir=str(evals_dir), name="logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=str(models_dir),
        auto_insert_metric_name=False,
        filename="epoch{epoch:02d}-val{val_loss:.4f}",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    time_logger = TimeLoggingCallback()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=15,
        min_delta=0.0,
        check_finite=True,  # stop if val becomes NaN/Inf
        verbose=True,
    )

    loss_curve_cb = LossCurveCallback(artefacts_dir)

    # ----- W&B -----
    loggers = [csv_logger]

    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=loggers,
        callbacks=[checkpoint_callback, lr_monitor, time_logger, early_stopping, loss_curve_cb],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=500 if not local_dev else 1,
        enable_progress_bar=True,
        deterministic=False,
        precision=pick_precision(),
        num_sanity_val_steps=0,
    )

    # ----- train -----
    resume_path: Path | None = str(cfg.continue_from) if cfg.continue_from else None
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader, ckpt_path=resume_path)

    # ----- curves to parquet / png -----
    metrics_path_csv = Path(csv_logger.log_dir) / "metrics.csv"
    metrics_path_parquet = evals_dir / "metrics.parquet"
    if metrics_path_csv.exists():
        df = pd.read_csv(metrics_path_csv)
        df.to_parquet(metrics_path_parquet, index=False)

    best_path = checkpoint_callback.best_model_path
    print(f"Best model path: {best_path}")

    if not best_path or not Path(best_path).exists():
        log("No checkpoint found (best/last). Skipping post-training analysis.")
        return 0.0

    log(f"Loading best checkpoint: {best_path}")
    best_model = retrieve_model("LPR").load_from_checkpoint(best_path)
    best_model.to(device).eval()

    # get the best val_loss from CSV
    if metrics_path_csv.exists():
        df = pd.read_csv(metrics_path_csv)
        min_val_loss = df["val_loss"].min() if "val_loss" in df else float("nan")
    else:
        min_val_loss = float("nan")

    log("Experiment completed.")
    log(f"Best val loss: {min_val_loss:.4f}")
    log(f"Best checkpoint: {best_path}")
    return min_val_loss


def get_cfg(trial: optuna.Trial, dataset: str):
    cfg = {
        "batch_size": trial.suggest_int("batch_size", 32, 512, step=32),
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_categorical(
            "weight_decay",
            [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4],
        ),
        "depth": trial.suggest_categorical("depth", [1, 2, 3]),
        "h1": trial.suggest_int("h1", 256, 2048, step=64),
        "h2": trial.suggest_int("h2", 128, 1024, step=64),
        "h3": trial.suggest_int("h3", 64, 512, step=64),
        "activation": trial.suggest_categorical("activation", ACTS.keys()),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "norm": trial.suggest_categorical("norm", NORMS.keys()),
    }

    # build hidden_dims from depth
    hidden_dims = [cfg["h1"]]
    if cfg["depth"] >= 2:
        hidden_dims.append(cfg["h2"])
    if cfg["depth"] >= 3:
        hidden_dims.append(cfg["h3"])

    lpr_cfg = Config()
    lpr_cfg.batch_size = cfg["batch_size"]
    lpr_cfg.lr = cfg["lr"]
    lpr_cfg.weight_decay = cfg["weight_decay"]
    lpr_cfg.hidden_dims = hidden_dims
    lpr_cfg.activation = cfg["activation"]
    lpr_cfg.norm = None if cfg["norm"] == "none" else cfg["norm"]
    lpr_cfg.dropout = cfg["dropout"]

    lpr_cfg.exp_dir_name = make_run_folder_name(cfg, dataset=dataset)
    return lpr_cfg


def run_qm9_trial(trial: optuna.Trial):
    lpr_cfg = get_cfg(trial, dataset="qm9")
    lpr_cfg.dataset = SupportedDataset.QM9_SMILES_HRR_1600
    lpr_cfg.hv_dim = 40 * 40
    return run_experiment(lpr_cfg)
