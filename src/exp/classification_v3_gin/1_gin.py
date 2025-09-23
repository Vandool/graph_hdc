import argparse
import contextlib
import enum
import itertools
import json
import os
import random
import shutil
import string
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

# === BEGIN NEW ===
from torchhd import HRRTensor

from src.datasets.qm9_pairs import QM9Pairs
from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_pairs_v2 import ZincPairsV2
from src.datasets.zinc_pairs_v3 import ZincPairsV3
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import Features, SupportedDataset
from src.encoding.decoder import greedy_oracle_decoder, is_induced_subgraph_by_features
from src.encoding.graph_encoders import AbstractGraphEncoder, load_or_create_hypernet
from src.encoding.oracles import Oracle
from src.encoding.the_types import VSAModel
from src.utils.registery import ModelType, resolve_model
from src.utils.sampling import balanced_indices_for_validation, stratified_per_parent_indices_with_type_mix
from src.utils.utils import GLOBAL_MODEL_PATH, DataTransformer, pick_device, str2bool
from src.utils.visualisations import draw_nx_with_atom_colorings

with contextlib.suppress(RuntimeError):
    mp.set_sharing_strategy("file_system")


# ---------------------------------------------------------------------
# Utils & Config
# ---------------------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def enable_ampere_tensor_cores():
    if torch.cuda.is_available():
        # Faster float32 matmuls via TF32 on Ampere (A100/H100)
        with contextlib.suppress(AttributeError):
            torch.set_float32_matmul_precision("medium")  # or "high" for a bit more accuracy
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def pick_precision():
    if not torch.cuda.is_available():
        return 32
    # A100/H100 etc → bf16
    if torch.cuda.is_bf16_supported():
        return "bf16-mixed"
    # Volta/Turing/Ampere+ (sm >= 70) → fp16 mixed
    major, minor = torch.cuda.get_device_capability()
    if major >= 7:
        return "16-mixed"
    # Maxwell/Pascal and older → plain fp32
    return 32


@dataclass
class Config:
    # General
    project_dir: Path | None = None
    exp_dir_name: str | None = None
    seed: int = 42
    epochs: int = 5
    batch_size: int = 256
    is_dev: bool = False

    # Model (shared knobs)
    model_name: ModelType = "GIN-F"  # or GIN-C
    cond_units: list[int] = field(default_factory=lambda: [256, 128])
    cond_emb_dim: int = 128
    film_units: list[int] = field(default_factory=lambda: [128])
    conv_units: list[int] = field(default_factory=lambda: [64, 64, 64])
    pred_head_units: list[int] = field(default_factory=lambda: [256, 64, 1])

    # Evals
    oracle_num_evals: int = 1
    oracle_beam_size: int = 8

    # HDC / encoder
    hv_dim: int = 88 * 88  # 7744
    vsa: VSAModel = VSAModel.HRR
    dataset: SupportedDataset = SupportedDataset.QM9_SMILES_HRR_1600

    # Optim
    lr: float = 1e-4
    weight_decay: float = 0.0

    # Loader
    num_workers: int = 4
    prefetch_factor: int | None = 1
    pin_memory: bool = False

    # Checkpointing
    continue_from: Path | None = None
    resume_retrain_last_epoch: bool = False

    # Stratification
    p_per_parent: int = 20
    n_per_parent: int = 20
    exclude_negs: list[int] = field(default_factory=list)
    resample_training_data_on_batch: bool = False


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def setup_exp(dir_name: str | None = None) -> dict:
    script_path = Path(__file__).resolve()
    experiments_path = script_path.parent
    script_stem = script_path.stem

    base_dir = experiments_path / "results" / script_stem
    base_dir.mkdir(parents=True, exist_ok=True)

    log(f"Setting up experiment in {base_dir}")
    if dir_name:
        exp_dir = base_dir / dir_name
    else:
        slug = (
            f"{datetime.now(UTC).strftime('%Y-%m-%d_%H-%M-%S')}_{''.join(random.choices(string.ascii_lowercase, k=4))}"
        )
        exp_dir = base_dir / slug
    exp_dir.mkdir(parents=True, exist_ok=True)
    log(f"Experiment directory created: {exp_dir}")

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
        log(f"Saved a copy of the script to {exp_dir / script_path.name}")
    except Exception as e:
        log(f"Warning: Failed to save script copy: {e}")

    return dirs


# ---------------------------------------------------------------------
# Dataset and loaders
# ---------------------------------------------------------------------


class EpochResamplingSampler(Sampler[int]):
    def __init__(self, ds, *, p_per_parent, n_per_parent, exclude_neg_types, base_seed=42):
        self.ds = ds
        self.p = p_per_parent
        self.n = n_per_parent
        self.exclude = set(exclude_neg_types)
        self.base_seed = base_seed
        self._epoch = 0
        self._last_len = 0

    def __iter__(self):
        seed = self.base_seed + self._epoch
        idxs = stratified_per_parent_indices_with_type_mix(
            ds=self.ds,
            pos_per_parent=self.p,
            neg_per_parent=self.n,
            exclude_neg_types=self.exclude,
            seed=seed,
        )
        log(f"[epoch {self._epoch}] Resampled {len(idxs)} graphs.")
        self._last_len = len(idxs)
        self._epoch += 1
        return iter(idxs)

    def __len__(self):
        # just an estimate for progress bars; becomes exact after first epoch
        return self._last_len if self._last_len else len(self.ds)


class PairsGraphsEncodedDataset(Dataset):
    """
    Returns a single PyG Data per pair:
      data.x, data.edge_index, data.edge_attr(=1), data.edge_weights(=1),
      data.cond (encoded g2 -> [D]),
      data.y (float), data.parent_idx (long)
    Encoding is done on-the-fly with the provided encoder (no grads).
    """

    def __init__(
        self,
        pairs_ds: ZincPairsV2,
        *,
        encoder: AbstractGraphEncoder,
        device: torch.device,
        add_edge_attr: bool = True,
        add_edge_weights: bool = True,
    ):
        self.ds = pairs_ds
        self.encoder = encoder.eval()
        self.device = device
        self.add_edge_attr = add_edge_attr
        self.add_edge_weights = add_edge_weights
        for p in self.encoder.parameters():
            p.requires_grad = False

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def _ensure_graph_fields(g: Data, *, add_edge_attr: bool, add_edge_weights: bool) -> Data:
        E = g.edge_index.size(1)
        if add_edge_attr and getattr(g, "edge_attr", None) is None:
            g.edge_attr = torch.ones(E, 1, dtype=torch.float32)
        if add_edge_weights and getattr(g, "edge_weights", None) is None:
            g.edge_weights = torch.ones(E, dtype=torch.float32)
        return g

    @torch.no_grad()
    def __getitem__(self, idx):
        item = self.ds[idx]

        # g1 (candidate subgraph)
        g1 = Data(x=item.x1, edge_index=item.edge_index1)
        g1 = self._ensure_graph_fields(g1, add_edge_attr=self.add_edge_attr, add_edge_weights=self.add_edge_weights)

        # g2 (condition) -> encode to cond
        g2 = Data(x=item.x2, edge_index=item.edge_index2)

        # Encode a single graph safely
        batch_g2 = Batch.from_data_list([g2]).to(self.device)
        h2 = self.encoder.forward(batch_g2)["graph_embedding"]  # [1, D] on device
        cond = h2.detach().cpu()  # let PL move the whole Batch later
        cond = cond.as_subclass(torch.Tensor)

        # target/meta
        y = float(item.y.view(-1)[0].item())
        parent_idx = int(item.parent_idx.view(-1)[0].item()) if hasattr(item, "parent_idx") else -1

        # Attach fields to g1
        g1.cond = cond
        g1.y = torch.tensor(y, dtype=torch.float32)
        g1.parent_idx = torch.tensor(parent_idx, dtype=torch.long)
        return g1


# ---------------- DataModule with per-epoch resampling ----------------
class PairsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: Config,
        *,
        encoder: AbstractGraphEncoder,
        device: torch.device,
        is_dev: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.device = device
        self.is_dev = is_dev

        # set in setup()
        self.train_full = None
        self.valid_full = None
        self.valid_loader = None

    def setup(self, stage=None):
        log("Loading pair datasets …")
        if cfg.dataset == SupportedDataset.QM9_SMILES_HRR_1600:
            self.train_full = QM9Pairs(split="train", base_dataset=QM9Smiles(split="train"), dev=self.is_dev)
            self.valid_full = QM9Pairs(split="valid", base_dataset=QM9Smiles(split="valid"), dev=self.is_dev)
        elif cfg.dataset == SupportedDataset.ZINC_SMILES_HRR_7744:
            self.train_full = ZincPairsV3(split="train", base_dataset=ZincSmiles(split="train"), dev=self.is_dev)
            self.valid_full = ZincPairsV3(split="valid", base_dataset=ZincSmiles(split="valid"), dev=self.is_dev)
        log(
            f"Pairs loaded for {cfg.dataset.value}. train_pairs_full_size={len(self.train_full)} valid_pairs_full_size={len(self.valid_full)}"
        )

        # Precompute validation indices (fixed selection); loaders are built in *_dataloader()
        self._valid_indices = None

        self._valid_indices = balanced_indices_for_validation(ds=self.valid_full, seed=cfg.seed)
        log(f"Loaded {len(self._valid_indices)} validation pairs for validation")

    def train_dataloader(self):
        train_ds = PairsGraphsEncodedDataset(
            self.train_full, encoder=self.encoder, device=self.device, add_edge_attr=True, add_edge_weights=True
        )

        sampler = EpochResamplingSampler(
            self.train_full,
            p_per_parent=self.cfg.p_per_parent,
            n_per_parent=self.cfg.n_per_parent,
            exclude_neg_types=self.cfg.exclude_negs,
            base_seed=self.cfg.seed + 13,
        )

        return DataLoader(  # this is torch_geometric.loader.DataLoader
            train_ds,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=True,
            prefetch_factor=cfg.prefetch_factor,
        )

    def val_dataloader(self):
        valid_base = (
            torch.utils.data.Subset(self.valid_full, self._valid_indices)
            if self._valid_indices is not None
            else self.valid_full
        )
        valid_ds = PairsGraphsEncodedDataset(
            valid_base, encoder=self.encoder, device=self.device, add_edge_attr=True, add_edge_weights=True
        )
        return DataLoader(
            valid_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=True,
            prefetch_factor=cfg.prefetch_factor,
        )


# ---------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------


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


@torch.no_grad()
def evaluate_as_oracle(
    model,
    encoder,
    epoch: int,
    artifact_dir: Path,
    oracle_num_evals: int = 8,
    oracle_beam_size: int = 8,
    oracle_threshold: float = 0.5,
    dataset: SupportedDataset = SupportedDataset.ZINC_SMILES_HRR_7744,
):
    log(f"Evaluation classifier as oracle for {oracle_num_evals} examples @threshold:{oracle_threshold}...")

    # Helpers
    # Real Oracle

    model.eval()
    encoder.eval()
    ys = []

    ds = ZincSmiles(split="valid")[:oracle_num_evals]
    if dataset == SupportedDataset.QM9_SMILES_HRR_1600:
        ds = QM9Smiles(split="valid")[:oracle_num_evals]
    dataloader = DataLoader(dataset=ds, batch_size=oracle_num_evals, shuffle=False)
    batch = next(iter(dataloader))

    # Encode the whole graph in one HV
    graph_term = encoder.forward(batch)["graph_embedding"]
    graph_terms_hd = graph_term.as_subclass(HRRTensor)

    # Create Oracle
    oracle = Oracle(model=model, encoder=encoder, model_type="GIN-F")

    ground_truth_counters = {}
    datas = batch.to_data_list()
    for i in range(oracle_num_evals):
        full_graph_nx = DataTransformer.pyg_to_nx(data=datas[i])
        node_multiset = DataTransformer.get_node_counter_from_batch(batch=i, data=batch)

        nx_GS: list[nx.Graph] = greedy_oracle_decoder(
            node_multiset=node_multiset,
            oracle=oracle,
            full_g_h=graph_terms_hd[i],
            full_g_nx=full_graph_nx,
            beam_size=oracle_beam_size,
            expand_on_n_anchors=4,
            oracle_threshold=oracle_threshold,
            strict=False,
        )
        nx_GS = list(filter(None, nx_GS))
        if len(nx_GS) == 0:
            print("Nothing decoded!")
            ys.append(0)
            continue
        ps = []
        for j, g in enumerate(nx_GS):
            is_final = is_induced_subgraph_by_features(g1=g, g2=full_graph_nx, node_keys=["feat"])
            ps.append(int(is_final))
        correct_p = int(sum(ps) >= 1)
        if correct_p:
            log(f"Correct prediction for sample #{i} from ZincSmiles validation dataset.")
        ys.append(correct_p)
    acc = 0.0 if len(ys) == 0 else float(sum(ys) / len(ys))
    log(f"Oracle Accuracy within the graph decoder : {acc:.4f}")
    return acc


class MetricsPlotsAndOracleCallback(Callback):
    def __init__(
        self,
        *,
        encoder: AbstractGraphEncoder,
        cfg: Config,
        evals_dir: Path,
        artefacts_dir: Path,
        oracle_on_val_end: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.oracle_on_val_end = oracle_on_val_end
        self.evals_dir = Path(evals_dir)
        self.artefacts_dir = Path(artefacts_dir)
        # accumulators
        self._ys = []
        self._logits = []
        self._train_losses = []
        self._val_losses = []
        self._epoch_rows = []
        self._pr_rows = []  # list of dicts: epoch, thr, prec, rec
        self._roc_rows = []  # list of dicts: epoch, thr, tpr, fpr

    def _save_parquet_or_csv(self, df: pd.DataFrame, path: Path):
        try:
            df.to_parquet(path, index=False)
        except Exception:
            df.with_columns = None  # just to silence static analyzers
            df.to_csv(path.with_suffix(".csv"), index=False)

    def on_train_epoch_end(self, trainer, pl_module):
        # Lightning names can vary; try common keys
        cm = trainer.callback_metrics
        tr = None
        for k in ("loss_epoch", "train_loss_epoch", "loss"):
            if k in cm and torch.is_tensor(cm[k]):
                tr = float(cm[k].detach().cpu().item())
                break
        if tr is not None:
            self._train_losses.append(tr)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._ys.clear()
        self._logits.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            out = pl_module(batch)
            logits = out["graph_prediction"].squeeze(-1).detach().float().cpu()
            y = batch.y.detach().float().cpu()
        self._logits.append(logits)
        self._ys.append(y)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._ys:
            return

        # concat
        y = torch.cat(self._ys).numpy().astype(int)
        z = torch.cat(self._logits).numpy().astype(np.float32)
        p = 1.0 / (1.0 + np.exp(-z))  # sigmoid

        # unweighted val loss for comparability
        val_loss = float(
            F.binary_cross_entropy_with_logits(torch.from_numpy(z), torch.from_numpy(y.astype(np.float32)))
        )
        self._val_losses.append(val_loss)

        with contextlib.suppress(Exception):
            # Loss curves
            epochs = np.arange(len(self._val_losses))
            plt.figure()
            if self._train_losses:
                plt.plot(np.arange(len(self._train_losses)), self._train_losses, label="train_loss")
            plt.plot(epochs, self._val_losses, label="val_loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.artefacts_dir / "loss_curves.png")
            plt.close()


        # prevalence
        pi = float(y.mean())

        # robust metrics (handle single-class batches)
        if np.unique(y).size < 2:
            auc = ap = float("nan")
            prec = rec = thr_pr = None
            fpr = tpr = thr_roc = None
        else:
            auc = float(roc_auc_score(y, p))
            ap = float(average_precision_score(y, p))
            prec, rec, thr_pr = precision_recall_curve(y, p)
            fpr, tpr, thr_roc = roc_curve(y, p)

        # Brier
        brier = float(brier_score_loss(y, p)) if np.unique(y).size == 2 else float("nan")

        # @0.5 metrics
        yhat05 = (p >= 0.5).astype(int)
        acc05 = float((yhat05 == y).mean())
        f105 = float(f1_score(y, yhat05, zero_division=0))
        bal05 = float(balanced_accuracy_score(y, yhat05))
        try:
            mcc05 = float(matthews_corrcoef(y, yhat05))
        except Exception:
            mcc05 = 0.0

        # best-F1 from PR thresholds
        if prec is not None and len(prec) > 1:
            f1s = 2 * prec[1:] * rec[1:] / (prec[1:] + rec[1:] + 1e-12)
            best_i = int(np.nanargmax(f1s))
            best_thr = float(thr_pr[best_i])
            f1_best = float(f1s[best_i])
            # stash PR/ROC rows for parquet
            epoch = int(trainer.current_epoch)
            self._pr_rows.extend(
                [
                    {"epoch": epoch, "threshold": float(t), "precision": float(pr), "recall": float(rc)}
                    for pr, rc, t in zip(prec[1:], rec[1:], thr_pr, strict=False)
                ]
            )
            if fpr is not None:
                self._roc_rows.extend(
                    [
                        {"epoch": epoch, "threshold": float(t), "tpr": float(tp), "fpr": float(fp)}
                        for fp, tp, t in zip(fpr, tpr, thr_roc, strict=False)
                    ]
                )
        else:
            best_thr = float("nan")
            f1_best = float("nan")
            epoch = int(trainer.current_epoch)

        # --- Confusion matrix @best F1 threshold (if available) ---
        if np.isfinite(best_thr):
            yhat_best = (p >= best_thr).astype(int)
            cmb = confusion_matrix(y, yhat_best, labels=[0, 1])
            tnb, fpb, fnb, tpb = [int(v) for v in cmb.ravel()]
        else:
            tnb = fpb = fnb = tpb = np.nan

        # log to Lightning
        metrics = {
            "val_loss": val_loss,
            "val_auc": auc,
            "val_ap": ap,
            "val_brier": brier,
            "val_prevalence": pi,
            "val_acc@0.5": acc05,
            "val_f1@0.5": f105,
            "val_bal_acc@0.5": bal05,
            "val_mcc@0.5": mcc05,
            "val_best_f1": f1_best,
            "val_best_thr": best_thr,
            # confusion matrix at best threshold
            "val_tn@best": tnb,
            "val_fp@best": fpb,
            "val_fn@best": fnb,
            "val_tp@best": tpb,
        }
        pl_module.log_dict(metrics, prog_bar=True, logger=True)

        # persist epoch summary row now (so crashes don’t lose it)
        row = {"epoch": epoch, **metrics}
        self._epoch_rows.append(row)
        df_epoch = pd.DataFrame([row])
        self._save_parquet_or_csv(df_epoch, self.evals_dir / "epoch_metrics.parquet")

        # optional: Oracle eval with best_thr
        if self.oracle_on_val_end and np.isfinite(best_thr) and epoch > 0:
            with torch.no_grad():
                oracle_acc = evaluate_as_oracle(
                    model=pl_module,
                    encoder=self.encoder,
                    oracle_num_evals=self.cfg.oracle_num_evals,
                    oracle_beam_size=self.cfg.oracle_beam_size,
                    oracle_threshold=best_thr,
                    dataset=self.cfg.dataset,
                    artifact_dir=self.artefacts_dir,
                    epoch=epoch
                )
            pl_module.log("val_oracle_acc", float(oracle_acc), prog_bar=False, logger=True)
            # also store it
            self._epoch_rows[-1]["val_oracle_acc"] = float(oracle_acc)
            df_epoch = pd.DataFrame([self._epoch_rows[-1]])
            self._save_parquet_or_csv(df_epoch, self.evals_dir / "epoch_metrics.parquet")

        # store last-epoch arrays for plotting
        self._last_y = y
        self._last_p = p
        self._last_pr = (prec, rec, thr_pr) if prec is not None else None
        self._last_roc = (fpr, tpr, thr_roc) if fpr is not None else None

    def on_fit_end(self, trainer, pl_module):
        # Write full PR/ROC tables (all epochs) once
        if self._pr_rows:
            self._save_parquet_or_csv(pd.DataFrame(self._pr_rows), self.evals_dir / "pr_curve.parquet")
        if self._roc_rows:
            self._save_parquet_or_csv(pd.DataFrame(self._roc_rows), self.evals_dir / "roc_curve.parquet")

        # Write consolidated epoch metrics once (idempotent)
        if self._epoch_rows:
            df_all = pd.DataFrame(self._epoch_rows).drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
            self._save_parquet_or_csv(df_all, self.evals_dir / "epoch_metrics.parquet")

        # ---- Plots ----
        self.artefacts_dir.mkdir(parents=True, exist_ok=True)

        # 1) Loss curves
        epochs = np.arange(len(self._val_losses))
        plt.figure()
        if self._train_losses:
            plt.plot(np.arange(len(self._train_losses)), self._train_losses, label="train_loss")
        plt.plot(epochs, self._val_losses, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.artefacts_dir / "loss_curves.png")
        plt.close()

        # 2) AUC/AP over epochs (if available)
        if self._epoch_rows and "val_auc" in self._epoch_rows[0]:
            df_all = pd.DataFrame(self._epoch_rows).sort_values("epoch")
            if df_all["val_auc"].notna().any():
                plt.figure()
                plt.plot(df_all["epoch"], df_all["val_auc"])
                plt.xlabel("epoch")
                plt.ylabel("AUC")
                plt.tight_layout()
                plt.savefig(self.artefacts_dir / "auc_by_epoch.png")
                plt.close()
            if df_all["val_ap"].notna().any():
                plt.figure()
                plt.plot(df_all["epoch"], df_all["val_ap"])
                plt.xlabel("epoch")
                plt.ylabel("AP")
                plt.tight_layout()
                plt.savefig(self.artefacts_dir / "ap_by_epoch.png")
                plt.close()

        # 3) PR/ROC for last epoch
        if getattr(self, "_last_pr", None):
            prec, rec, _ = self._last_pr
            plt.figure()
            plt.plot(rec, prec)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.tight_layout()
            plt.savefig(self.artefacts_dir / "pr_curve_last.png")
            plt.close()
        if getattr(self, "_last_roc", None):
            fpr, tpr, _ = self._last_roc
            plt.figure()
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], "--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.tight_layout()
            plt.savefig(self.artefacts_dir / "roc_curve_last.png")
            plt.close()

        # 4) Calibration (reliability) for last epoch
        if getattr(self, "_last_p", None) is not None:
            p = self._last_p
            y = self._last_y
            bins = np.linspace(0, 1, 11)
            idx = np.digitize(p, bins) - 1
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            frac_pos = np.array([y[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(bin_centers))])
            plt.figure()
            plt.plot([0, 1], [0, 1], "--")
            plt.plot(bin_centers, frac_pos, marker="o")
            plt.xlabel("Predicted probability")
            plt.ylabel("Fraction positive")
            plt.tight_layout()
            plt.savefig(self.artefacts_dir / "calibration_last.png")
            plt.close()

            # 5) Confusion matrix heatmap (last epoch, @0.5)
        if getattr(self, "_last_p", None) is not None:
            yhat = (self._last_p >= 0.5).astype(int)
            cm = confusion_matrix(self._last_y, yhat, labels=[0, 1])

            plt.figure()
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix @0.5 (last epoch)")
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])
            for i, j in itertools.product(range(2), range(2)):
                plt.text(j, i, cm[i, j], ha="center", va="center")
            plt.tight_layout()
            plt.savefig(self.artefacts_dir / "confusion_matrix_last.png")
            plt.close()


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------


class TimeLoggingCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        trainer.logger.log_metrics({"elapsed_time_sec": elapsed}, step=trainer.current_epoch)


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def run_experiment(cfg: Config, is_dev: bool = False):
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

    device = pick_device()
    log(f"Using device: {device!s}")

    log("Loading/creating hypernet …")
    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=cfg.dataset.default_cfg).to(device=device).eval()
    log("Hypernet ready.")
    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.ATOM_TYPE][0].codebook)
    encoder = hypernet.to(device).eval()

    cpu_encoder = (
        load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=cfg.dataset.default_cfg)
        .to(device=torch.device("cpu"))
        .eval()
    )
    # datamodule with per-epoch resampling
    dm = PairsDataModule(cfg, encoder=cpu_encoder, device=torch.device("cpu"), is_dev=is_dev)
    model = resolve_model(
        name=cfg.model_name,
        input_dim=4,
        edge_dim=1,
        condition_dim=cfg.hv_dim,
        cond_units=cfg.cond_units,
        conv_units=cfg.conv_units,
        film_units=cfg.film_units,
        pred_units=cfg.pred_head_units,
        learning_rate=cfg.lr,
        cfg=cfg,
    ).to(device)

    log(f"Model: {model!s}")
    log(f"Model hparams: {model.hparams!s}")

    log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    log(f"Model on: {next(model.parameters()).device}")

    # ---- Callbacks
    csv_logger = CSVLogger(save_dir=str(evals_dir), name="logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=str(models_dir),
        auto_insert_metric_name=False,
        filename="epoch{epoch:02d}-val{val_loss:.4f}",
        save_last=True,
        save_on_train_epoch_end=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    time_logger = TimeLoggingCallback()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        min_delta=0.0,
        check_finite=True,  # stop if val becomes NaN/Inf
        verbose=True,
        check_on_train_epoch_end=False,
    )

    val_metrics_cb = MetricsPlotsAndOracleCallback(
        encoder=encoder, cfg=cfg, evals_dir=evals_dir, artefacts_dir=artefacts_dir, oracle_on_val_end=True
    )
    enable_ampere_tensor_cores()
    # Training
    resume_path: Path | None = str(cfg.continue_from) if cfg.continue_from else None
    last_epoch = 0
    if resume_path:
        log(f"Resuming from: {resume_path!s}")
        last_epoch = torch.load(resume_path)["epoch"]
        log(f"Checkpoint was at epoch {last_epoch}")
    trainer = Trainer(
        max_epochs=cfg.epochs + last_epoch + 1,
        logger=[csv_logger],
        callbacks=[checkpoint_callback, lr_monitor, time_logger, early_stopping, val_metrics_cb],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        # gradient_clip_val=1.0,  # Do we need this?
        log_every_n_steps=500 if not is_dev else 1,
        enable_progress_bar=True,
        deterministic=False,
        precision=pick_precision(),
        num_sanity_val_steps=0,
    )

    # --- Train
    resume_path: Path | None = str(cfg.continue_from) if cfg.continue_from else None
    if resume_path:
        log(f"Resuming from: {resume_path!s}")
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path)
    log("Finished training.")

    # Final plots


if __name__ == "__main__":
    # ----------------- CLI parsing that never clobbers defaults -----------------
    def _parse_int_list(s: str) -> list[int]:
        # accept "4096,2048,512,128" or with spaces
        return [int(tok) for tok in s.replace(" ", "").split(",") if tok]

    def _parse_vsa(s: str) -> VSAModel:
        return s if isinstance(s, VSAModel) else VSAModel(s)

    def _parse_supported_dataset(s: str) -> SupportedDataset:
        return s if isinstance(s, SupportedDataset) else SupportedDataset(s)

    def get_args(argv: list[str] | None = None) -> Config:
        """
        Build a Config by starting from dataclass defaults and then
        applying ONLY the CLI options the user actually provided.
        NOTE: For --vsa, pass a string like "HRR", not VSAModel.HRR.
        """
        cfg = Config()  # start with your defaults

        p = argparse.ArgumentParser(description="Experiment Config (unified)")

        # IMPORTANT: default=SUPPRESS so unspecified flags don't overwrite dataclass defaults
        p.add_argument(
            "--project_dir",
            "-pdir",
            type=Path,
            default=argparse.SUPPRESS,
            help="Project root (will be created if missing)",
        )
        p.add_argument("--exp_dir_name", type=str, default=argparse.SUPPRESS)

        p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
        p.add_argument("--epochs", "-e", type=int, default=argparse.SUPPRESS)
        p.add_argument("--batch_size", "-bs", type=int, default=argparse.SUPPRESS)
        p.add_argument("--is_dev", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)

        # Evals
        p.add_argument("--oracle_num_evals", type=int, default=argparse.SUPPRESS)
        p.add_argument("--oracle_beam_size", type=int, default=argparse.SUPPRESS)

        # Model knobs
        p.add_argument("--model_name", "-model", type=str, default=argparse.SUPPRESS)
        p.add_argument(
            "--film_units",
            type=_parse_int_list,
            default=argparse.SUPPRESS,
            help="Comma-separated: e.g. '128,64'",
        )
        p.add_argument(
            "--cond_units", type=_parse_int_list, default=argparse.SUPPRESS, help="Comma-separated: e.g. '256,128'"
        )
        p.add_argument(
            "--cond_emb_dim",
            type=int,
            default=argparse.SUPPRESS,
            help="If omitted but --cond_units is given, will default to last(cond_units)",
        )
        p.add_argument(
            "--conv_units", type=_parse_int_list, default=argparse.SUPPRESS, help="Comma-separated: e.g. '64,64,64'"
        )
        p.add_argument(
            "--pred_head_units",
            type=_parse_int_list,
            default=argparse.SUPPRESS,
            help="Comma-separated: e.g. '256,64,1'",
        )

        # HDC
        p.add_argument("--hv_dim", "-hd", type=int, default=argparse.SUPPRESS)
        p.add_argument("--vsa", "-v", type=_parse_vsa, default=argparse.SUPPRESS)
        p.add_argument("--dataset", "-ds", type=_parse_supported_dataset, default=argparse.SUPPRESS)

        # Optim
        p.add_argument("--lr", type=float, default=argparse.SUPPRESS)
        p.add_argument("--weight_decay", "-wd", type=float, default=argparse.SUPPRESS)

        # Loader
        p.add_argument("--num_workers", type=int, default=argparse.SUPPRESS)
        p.add_argument("--prefetch_factor", type=int, default=argparse.SUPPRESS)
        p.add_argument("--pin_memory", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)

        # Checkpointing
        p.add_argument("--continue_from", type=Path, default=argparse.SUPPRESS)
        p.add_argument("--resume_retrain_last_epoch", type=str2bool, default=argparse.SUPPRESS)

        # Stratification
        p.add_argument("--p_per_parent", type=int, default=argparse.SUPPRESS)
        p.add_argument("--n_per_parent", type=int, default=argparse.SUPPRESS)
        p.add_argument(
            "--exclude_negs", type=_parse_int_list, default=argparse.SUPPRESS, help="Comma-separated ints, e.g. '1,2,3'"
        )
        p.add_argument("--resample_training_data_on_batch", type=str2bool, default=argparse.SUPPRESS)

        ns = p.parse_args(argv)
        provided = vars(ns)  # only the keys the user actually passed

        # Apply only provided keys onto cfg
        for k, v in provided.items():
            # Make sure VSAModel parsed if user typed the enum value directly
            if k == "vsa" and isinstance(v, str):
                v = VSAModel(v)
            setattr(cfg, k, v)

        return cfg

    log(f"Running {Path(__file__).resolve()}")
    is_dev = os.getenv("LOCAL_HDC_", False)

    if is_dev:
        log("Running in local HDC (DEV) ...")
        cfg: Config = Config(
            exp_dir_name="overfitting_batch_norm",
            seed=42,
            epochs=1,
            batch_size=4,
            model_name="GIN-F",
            hv_dim=88 * 88,
            vsa=VSAModel.HRR,
            dataset=SupportedDataset.ZINC_SMILES_HRR_7744,
            lr=1e-4,
            weight_decay=0.0,
            num_workers=0,
            prefetch_factor=None,
            pin_memory=False,
            continue_from=None,
            resume_retrain_last_epoch=False,
            p_per_parent=2,
            n_per_parent=2,
            oracle_beam_size=8,
            oracle_num_evals=8,
            resample_training_data_on_batch=True,
        )
    else:
        log("Running in cluster ...")
        cfg = get_args()

    pprint(asdict(cfg), indent=2)
    run_experiment(cfg, is_dev=is_dev or cfg.is_dev)
