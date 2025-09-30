import argparse
import contextlib
import enum
import itertools
import json
import math
import os
import random
import shutil
import string
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
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
from torch import nn
from torch.nn import Module
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torchmetrics import AUROC, AveragePrecision

from src.datasets.qm9_pairs import QM9Pairs
from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_pairs_v3 import ZincPairsV3
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import Features, SupportedDataset
from src.encoding.graph_encoders import AbstractGraphEncoder, load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.utils.sampling import (
    balanced_indices_for_validation,
    stratified_per_parent_indices_with_type_mix,
)
from src.utils.utils import GLOBAL_MODEL_PATH, pick_device, str2bool

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
    model_name: str = "biaffine_head"
    is_dev: bool = False

    # Biaffine model knobs
    proj_dim: int = 1024
    n_heads: int = 8
    proj_hidden: int | None = None  # None => single Linear
    dropout: float = 0.0
    share_proj: bool = False
    norm: bool = True  # L2 normalize h1/h2 before projecting
    use_layernorm: bool = True
    use_temperature: bool = True
    pos_weight: float | None = None  # BCE pos_weight

    # HDC / encoder
    hv_dim: int = 40 * 40  # 1600
    vsa: VSAModel = VSAModel.HRR
    dataset: SupportedDataset = SupportedDataset.QM9_SMILES_HRR_1600

    # Optim
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Loader defaults for cuda
    num_workers: int = 16
    prefetch_factor: int | None = 6
    pin_memory: bool = True
    micro_bs: int = 64
    persistent_workers: bool = True

    # Checkpointing
    # for lightning resume
    continue_from: Path | None = None
    resume_retrain_last_epoch: bool = False

    # Stratification
    stratify: bool = True
    p_per_parent: int = 20
    n_per_parent: int = 20
    exclude_negs: set[int] = field(default_factory=list)
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
# Model
# ---------------------------------------------------------------------
class BiaffineHead(nn.Module):
    def __init__(
        self,
        hv_dim: int,
        proj_dim: int = 1024,
        n_heads: int = 8,
        proj_hidden: int | None = None,
        dropout: float = 0.0,
        tau_init: float = 8.0,  # large τ to tame early logits
        *,
        share_proj: bool = False,
        norm: bool = True,
        use_layernorm: bool = True,
        use_temperature: bool = True,
    ):
        super().__init__()
        self.norm = norm
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.n_heads = n_heads
        P = proj_dim

        def proj_mlp() -> Module:
            if proj_hidden is None:
                return nn.Linear(hv_dim, P, bias=False)
            return nn.Sequential(
                nn.Linear(hv_dim, proj_hidden, bias=True),
                nn.GELU(),
                nn.Linear(proj_hidden, P, bias=False),
            )

        self.p1 = proj_mlp()
        self.p2 = self.p1 if share_proj else proj_mlp()

        self.ln1 = nn.LayerNorm(P) if use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(P) if use_layernorm else nn.Identity()

        self.W = nn.Parameter(torch.empty(n_heads, P, P))
        nn.init.xavier_uniform_(self.W)

        self.gate = nn.Linear(3, n_heads, bias=True)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        self.diag_w = nn.Parameter(torch.zeros(P))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.zeros(()))

        self.use_temperature = use_temperature
        if use_temperature:
            # τ = softplus(log_tau)  with τ≈tau_init at start
            self.log_tau = nn.Parameter(torch.tensor(math.log(math.exp(tau_init) - 1.0)))

        # init: orthogonal for any Linear with bias=False (covers final layers)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is None:
                nn.init.orthogonal_(m.weight)

    def forward(self, h1, h2):
        # accept [B,1,D]
        if h1.ndim == 3 and h1.size(1) == 1:
            h1 = h1.squeeze(1)
        if h2.ndim == 3 and h2.size(1) == 1:
            h2 = h2.squeeze(1)

        if self.norm:
            h1 = F.normalize(h1, dim=-1)
            h2 = F.normalize(h2, dim=-1)

        u = self.ln1(self.p1(h1))
        v = self.ln2(self.p2(h2))
        u = self.dropout(u)
        v = self.dropout(v)

        # biaffine heads
        s_heads = torch.einsum("bp,hpq,bq->bh", u, self.W, v)  # [B,H]

        # tiny feature gate per head
        uv = u * v
        cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1) + 1e-12)
        feat = torch.stack([uv.mean(-1), (u - v).abs().mean(-1), cos], dim=-1)  # [B,3]
        gates = torch.softmax(self.gate(feat), dim=-1)  # [B,H]

        s_biaff = (gates * s_heads).sum(-1)  # [B]
        s_diag = (uv * self.diag_w).sum(-1)  # [B]
        s_dot = self.alpha * (u * v).sum(-1)  # [B]
        logits = s_biaff + s_diag + s_dot + self.bias

        if self.use_temperature:
            tau = F.softplus(self.log_tau) + 1e-6
            logits = logits / tau
        return logits


# ---------------------------------------------------------------------
# Lightning wrapper
# ---------------------------------------------------------------------
class LitBAHClassifier(pl.LightningModule):
    """
    Expect batches as (h1, h2, y) or {"h1":..., "h2":..., "y":...}, with h1/h2 ~ [B, D].
    If collate yields [B, 1, D], we squeeze the middle dim.
    """

    def __init__(
        self,
        *,
        hv_dim: int,
        proj_dim: int = 1024,  # 1024–1536 are good starting points
        n_heads: int = 8,  # 4–16; more heads => more expressiveness
        norm: bool = True,
        proj_hidden: int | None = None,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        pos_weight: float | None = None,  # None -> unweighted BCE
        share_proj: bool = False,  # or e.g. 2048 for 2-layer projections if you want even more capacity_
        use_layernorm: bool = True,
        use_temperature: bool = True,
    ) -> None:
        super().__init__()
        # Save EVERYTHING for checkpoint load
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = BiaffineHead(
            hv_dim=hv_dim,
            proj_dim=proj_dim,
            n_heads=n_heads,
            share_proj=share_proj,
            norm=norm,
            use_layernorm=use_layernorm,
            proj_hidden=proj_hidden,
            dropout=dropout,
            use_temperature=use_temperature,
        )

        # Register pos_weight as a buffer so device moves are handled automatically
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
        else:
            self.pos_weight = None  # type: ignore[assignment]

        # Metrics
        self.val_auc = AUROC(task="binary")
        self.val_ap = AveragePrecision(task="binary")

    # --------- helpers ---------
    @staticmethod
    def _fix_shapes(h1: torch.Tensor, h2: torch.Tensor, y: torch.Tensor):
        # squeeze [B,1,D] -> [B,D] emitted by your dataset/loader
        if h1.ndim == 3 and h1.size(1) == 1:
            h1 = h1.squeeze(1)
        if h2.ndim == 3 and h2.size(1) == 1:
            h2 = h2.squeeze(1)
        # y -> [B], float
        if y.ndim > 1:
            y = y.squeeze(-1)
        return h1, h2, y.float()

    # --------- Lightning API ---------
    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        return self.model(h1, h2)

    def training_step(self, batch, batch_idx: int):
        h1, h2, y, k, idx = batch
        h1, h2, y = self._fix_shapes(h1, h2, y)
        logits = self(h1, h2)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)

        # log loss as before
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.size(0), logger=True)

        # return rich outputs for callbacks (no grads)
        return {
            "loss": loss,
            "logits": logits.detach(),
            "y": y.detach(),
            "k": None if k is None else torch.as_tensor(k),
            "idx": None if idx is None else torch.as_tensor(idx),
        }

    def validation_step(self, batch, batch_idx: int):
        h1, h2, y, _, _ = batch
        h1, h2, y = self._fix_shapes(h1, h2, y)

        logits = self(h1, h2)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)

        # epoch-level val_loss for EarlyStopping
        sync = getattr(self.trainer, "world_size", 1) > 1
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=y.size(0),
            sync_dist=sync,
        )

        # metrics expect probabilities and int/bool targets
        probs = logits.sigmoid()
        self.val_auc.update(probs, y.int())
        self.val_ap.update(probs, y.int())

    def on_validation_epoch_end(self):
        # make sure these get logged too (optional)
        self.log("val_auc", self.val_auc.compute(), prog_bar=True)
        self.log("val_ap", self.val_ap.compute(), prog_bar=True)
        self.val_auc.reset()
        self.val_ap.reset()

    def test_step(self, batch, batch_idx: int):
        h1, h2, y = batch
        h1, h2, y = self._fix_shapes(h1, h2, y)
        logits = self(h1, h2)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        probs = logits.sigmoid()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_auc", self.val_auc(probs, y.int()), prog_bar=True)
        self.log("test_ap", self.val_ap(probs, y.int()), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# ---------------------------------------------------------------------
# Dataset and loaders
# ---------------------------------------------------------------------


class EpochResamplingSampler(Sampler[int]):
    """
    Epoch 0: your existing per-parent/type sampler.
    Epoch >=1: k-stratified, class-balanced sampling with a mixture
               of base indices and hard-pool indices controlled by alpha.
    """

    def __init__(
        self,
        ds,
        dm: "PairsDataModule",
        *,
        p_per_parent,
        n_per_parent,
        type_mix=None,
        exclude_neg_types=(),
        base_seed=42,
        balance_k2=True,
    ):
        self.ds = ds
        self.dm = dm
        self.p = p_per_parent
        self.n = n_per_parent
        self.type_mix = type_mix
        self.exclude = {int(t) for t in exclude_neg_types}
        self.base_seed = base_seed
        self.balance_k2 = balance_k2
        self._epoch = 0
        self._last_len = 0

    def _sample_k_stratified_indices(self, seed: int) -> list[int]:
        rng = random.Random(seed)
        p_target_k = self.dm.p_target_k or {}
        alpha = float(self.dm.alpha or 0.0)

        hard_by = getattr(self.dm.hard_pool, "by_key", {}) if self.dm.hard_pool is not None else {}
        base_by = self.dm.base_index_by_k_label

        keys_k = list(p_target_k.keys())
        if not keys_k:
            keys_k = sorted({k for (k, _y) in base_by.keys()})

        pk = np.array([p_target_k.get(k, 0.0) for k in keys_k], dtype=np.float64)
        pk = np.ones_like(pk) / len(pk) if pk.sum() <= 0 else pk / pk.sum()

        # keep epoch length close to epoch-0
        target_total = max(1, self._last_len)  # fallback guarded

        # helper: shuffled copy
        def shuffled(lst):
            lst = list(lst)
            rng.shuffle(lst)
            return lst

        # make pop-without-replacement pools for both sources
        from collections import deque

        def make_pool(source_dict):
            return {key: deque(shuffled(ids)) for key, ids in source_dict.items() if ids}

        pool_hard = make_pool(hard_by)
        pool_base = make_pool(base_by)

        def pop_from(pool, k, y):
            dq = pool.get((k, y))
            if dq:
                return dq.popleft()
            return None

        def pop_any_k_same_label(pool, y):
            # try same label, any k (weighted by pk)
            for _ in range(8):  # a few tries with k~pk
                k_try = int(rng.choices(keys_k, weights=pk, k=1)[0])
                v = pop_from(pool, k_try, y)
                if v is not None:
                    return v
            # brute fallback: scan keys
            for (kk, yy), dq in pool.items():
                if yy == y and dq:
                    return dq.popleft()
            return None

        def pop_any(pool):
            # last-resort: any (k,y)
            for key, dq in list(pool.items()):
                if dq:
                    return dq.popleft()
            return None

        # label schedule: randomized chunks to avoid starting skew
        labels = [0, 1] * ((target_total // 2) + 2)
        rng.shuffle(labels)
        labels = labels[:target_total]

        out = []
        for y in labels:
            if len(out) >= target_total:
                break
            k = int(rng.choices(keys_k, weights=pk, k=1)[0])
            use_hard = rng.random() < alpha

            # try preferred source first, then the other
            primary, secondary = (pool_hard, pool_base) if use_hard else (pool_base, pool_hard)

            v = pop_from(primary, k, y)
            if v is None:
                v = pop_any_k_same_label(primary, y)
            if v is None:
                v = pop_from(secondary, k, y)
            if v is None:
                v = pop_any_k_same_label(secondary, y)
            if v is None:
                v = pop_any(primary) or pop_any(secondary)

            if v is not None:
                out.append(v)
            else:
                # pools exhausted; stop early
                break

        # If we came up short, top up from whatever remains to hit target_total
        while len(out) < target_total:
            v = pop_any(pool_hard) or pop_any(pool_base)
            if v is None:
                break
            out.append(v)

        # keep order random; no sorting; also no de-dup needed since we pop w/o replacement
        return out

    def __iter__(self):
        seed = self.base_seed + self._epoch
        if self._epoch == 0 or self.dm.p_target_k is None:
            # epoch 0 (or before callback fires): your original per-parent policy
            idxs = stratified_per_parent_indices_with_type_mix(
                ds=self.ds,
                pos_per_parent=self.p,
                neg_per_parent=self.n,
                type_mix=self.type_mix,
                balance_k2=self.balance_k2,
                exclude_neg_types=self.exclude,
                seed=seed,
            )
        else:
            log(f"Resampling from hard poll with alpha: {self.dm.alpha} - prob. of sampling from the hard pool")
            idxs = self._sample_k_stratified_indices(seed=seed)
            pos = int(sum(self.dm.y_by_idx[i] for i in idxs))
            total = len(idxs)
            neg = total - pos
            log(
                f"[epoch {self._epoch}] selected={total}  pos={pos} ({pos / total:.1%})  neg={neg} ({neg / total:.1%})",
            )

        log(f"[epoch {self._epoch}] Resampled {len(idxs)} pairs.")
        self._last_len = len(idxs)
        self._epoch += 1
        return iter(idxs)

    def __len__(self):
        return self._last_len if self._last_len else len(self.ds)


class PairsEncodedDataset(Dataset):
    """
    Returns a single PyG Data per pair:
      data.x, data.edge_index, data.edge_attr(=1), data.edge_weights(=1),
      data.cond (encoded g2 -> [D]),
      data.y (float), data.parent_idx (long)
    Encoding is done on-the-fly with the provided encoder (no grads).
    """

    def __init__(
        self,
        pairs_ds: ZincPairsV3 | QM9Pairs,
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

    @torch.no_grad()
    def __getitem__(self, idx):
        item = self.ds[idx]

        # g1 (candidate subgraph)
        g1 = Data(x=item.x1, edge_index=item.edge_index1)
        # g2 (condition) -> encode to cond
        g2 = Data(x=item.x2, edge_index=item.edge_index2)

        # Encode a single graph safely
        batch_g1 = Batch.from_data_list([g1]).to(self.device)
        h1 = self.encoder.forward(batch_g1)["graph_embedding"]  # [1, D] on device

        # Encode a single graph safely
        batch_g2 = Batch.from_data_list([g2]).to(self.device)
        h2 = self.encoder.forward(batch_g2)["graph_embedding"]  # [1, D] on device

        h1 = h1.as_subclass(torch.Tensor)
        h2 = h2.as_subclass(torch.Tensor)

        # target/meta
        y = float(item.y.view(-1)[0].item())
        k = int(item.k.view(-1)[0].item())

        return h1, h2, y, k, idx


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
        self.alpha = None
        self.p_target_k = None
        self.base_index_by_k_label = None
        self.base_counts_k = None
        self.valid_indices = None
        self.hard_pool = None
        self.cfg = cfg
        self.encoder = encoder
        self.device = device
        self.is_dev = is_dev

        # set in setup()
        self.train_full = None
        self.valid_full = None
        self.valid_loader = None

    def setup(self, stage=None):
        log(f"Loading {cfg.dataset.value} pair datasets -- Dev:{self.is_dev}…")
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
        self.valid_indices = balanced_indices_for_validation(ds=self.valid_full, seed=cfg.seed)
        log(f"Loaded {len(self.valid_indices)} validation pairs for validation")

        # Precompute base k counts and (k,label)->indices for the *train* split:
        self.base_counts_k = Counter()
        self.base_index_by_k_label = defaultdict(list)  # (k,y)->[idx]

        # Iterate train_full once (no encoding) to record k and y cheaply
        for idx in range(len(self.train_full)):
            item = self.train_full[idx]
            k = int(item.k.view(-1)[0].item())
            y = int(item.y.view(-1)[0].item())
            self.base_counts_k[k] += 1
            self.base_index_by_k_label[(k, y)].append(idx)

        self.y_by_idx = np.zeros(len(self.train_full), dtype=np.int8)
        for idx in range(len(self.train_full)):
            item = self.train_full[idx]
            self.y_by_idx[idx] = int(item.y.view(-1)[0].item())

        # targets updated by HardMiningCallback
        self.hard_pool = None
        self.p_target_k = None
        self.alpha = 0.0  # start with base only

    # called by HardMiningCallback each epoch
    def set_k_sampling_targets(self, *, hard_pool: "HardPool", p_target_k: dict[int, float] | None, alpha: float):
        self.hard_pool = hard_pool
        self.p_target_k = {int(k): float(v) for k, v in p_target_k.items()} if p_target_k else None
        self.alpha = float(alpha)

    def train_dataloader(self):
        train_ds = PairsEncodedDataset(
            self.train_full, encoder=self.encoder, device=self.device, add_edge_attr=True, add_edge_weights=True
        )

        sampler = EpochResamplingSampler(
            ds=self.train_full,
            dm=self,
            p_per_parent=self.cfg.p_per_parent,
            n_per_parent=self.cfg.n_per_parent,
            exclude_neg_types=self.cfg.exclude_negs,
            base_seed=self.cfg.seed + hash(self.cfg.exp_dir_name) % 3,
        )

        return DataLoader(  # this is torch_geometric.loader.DataLoader
            train_ds,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor,
        )

    def val_dataloader(self):
        valid_base = (
            torch.utils.data.Subset(self.valid_full, self.valid_indices)
            if self.valid_indices is not None
            else self.valid_full
        )
        valid_ds = PairsEncodedDataset(
            valid_base, encoder=self.encoder, device=self.device, add_edge_attr=True, add_edge_weights=True
        )
        return DataLoader(
            valid_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor,
        )


class HardPool:
    def __init__(self, max_size=200_000, bins=None):
        self.max_size = max_size
        self.items = []  # list of (idx, k, y)
        self.by_key = defaultdict(list)  # (k, y) -> [idx]
        self.counts_k = Counter()

    def update(self, batch_indices, batch_k, batch_y, hard_mask):
        for i, k, y, is_hard in zip(batch_indices, batch_k, batch_y, hard_mask, strict=False):
            if not is_hard:
                continue
            self.items.append((i, k, y))
        if len(self.items) > self.max_size:
            # keep most recent
            self.items = self.items[-self.max_size :]
        # rebuild fast lookup
        self.by_key.clear()
        self.counts_k.clear()
        for i, k, y in self.items:
            self.by_key[(k, int(y))].append(i)
            self.counts_k[k] += 1


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


class MetricsPlotsAndOracleCallback(Callback):
    def __init__(
        self,
        *,
        encoder: AbstractGraphEncoder,
        cfg: Config,
        evals_dir: Path,
        artefacts_dir: Path,
    ):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
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
            # MINIMAL: unpack and reuse module’s shape fixer
            h1, h2, y = batch
            h1, h2, y = pl_module._fix_shapes(h1, h2, y)

            # call the LightningModule with h1,h2 (noit the whole batch)
            logits = pl_module(h1, h2)

            # stash for epoch metrics
            self._logits.append(logits.detach().float().cpu())
            self._ys.append(y.detach().float().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._ys:
            return

        # concat
        y = torch.cat(self._ys).numpy().astype(int)
        z = torch.cat(self._logits).numpy().astype(np.float32)
        p = torch.sigmoid(torch.from_numpy(z)).numpy()

        # unweighted val loss for comparability
        val_loss = float(
            F.binary_cross_entropy_with_logits(torch.from_numpy(z), torch.from_numpy(y.astype(np.float32)))
        )
        self._val_losses.append(val_loss)
        with contextlib.suppress(Exception):
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

        # --- Confusion matrix @0.5 ---
        cm05 = confusion_matrix(y, yhat05, labels=[0, 1])
        tn05, fp05, fn05, tp05 = [int(v) for v in cm05.ravel()]

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
            "val_loss_cb": val_loss,
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


class HardMiningCallback(pl.Callback):
    """
    Collects hard samples each epoch and updates the DataModule with:
      - hard_pool (indices with (k,y))
      - p_target_k: mixture of base k-distribution and hard k-distribution
    Also saves a histogram of hard-sample k to artefacts_dir.
    """

    def __init__(
        self,
        dm: "PairsDataModule",
        artefacts_dir: Path,
        alpha: float = 0.4,
        beta: float = 0.6,
        margin: float | None = None,
        cap_per_k_label: int = 50_000,
        warmup_epochs: int = 1,
    ):
        super().__init__()
        self.dm = dm
        self.artefacts_dir = Path(artefacts_dir)
        self.alpha = float(alpha)  # prob. to draw from hard pool
        self.beta = float(beta)  # weight of hard-k in target p(k)
        self.margin = margin  # if set, use |p-0.5| < margin as "semi-hard"
        self.pool = HardPool(max_size=1_000_000)
        self._stash = []  # per-batch accumulators
        self.cap_per_k_label = cap_per_k_label
        self.warmup_epochs = warmup_epochs

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Skip collecting during warm-up
        if trainer.current_epoch < self.warmup_epochs:
            return

        # outputs from training_step(...)
        logits = outputs.get("logits")
        y = outputs.get("y")
        k = outputs.get("k")
        idx = outputs.get("idx")
        if logits is None or y is None or idx is None or k is None:
            return
        probs = torch.sigmoid(logits.detach()).flatten()
        yb = y.detach().long().flatten()
        kb = k.detach().flatten().to(torch.long)
        ib = idx.detach().flatten().to(torch.long)

        if self.margin is None:  # noqa: SIM108
            hard_mask = probs.ge(0.5).long() != yb  # misclassified
        else:
            hard_mask = (probs - 0.5).abs() < float(self.margin)  # low-margin

        self._stash.append((ib.cpu(), kb.cpu(), yb.cpu(), hard_mask.cpu()))

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup_epochs:
            # still tell DM to use base-only sampling next epoch
            self.dm.set_k_sampling_targets(hard_pool=None, p_target_k=None, alpha=0.0)
            return

        # ramp schedules with raising epochs
        t = max(0, trainer.current_epoch - self.warmup_epochs + 1)
        T = 5  # ramp length in epochs
        ramp = min(1.0, t / T)
        self.alpha = 0.5 * ramp  # 0 → 0.5
        self.beta = 0.6 * ramp  # 0 → 0.6
        if self.margin is not None:
            self.margin = max(0.05, 0.15 * (1.0 - 0.8 * ramp))

        # flatten stashed batches
        if not self._stash:
            return
        idxs = torch.cat([t[0] for t in self._stash]).tolist()
        ks = torch.cat([t[1] for t in self._stash]).tolist()
        ys = torch.cat([t[2] for t in self._stash]).tolist()
        hard = torch.cat([t[3] for t in self._stash]).tolist()
        self._stash.clear()

        # update pool (bounded)
        self.pool.update(idxs, ks, ys, hard)

        # (optional) lightly cap per (k,y) to reduce label-noise amplification
        capped = defaultdict(list)
        for k, y in list(self.pool.by_key.keys()):
            lst = self.pool.by_key[(k, y)]
            if len(lst) > self.cap_per_k_label:
                random.shuffle(lst)
                capped[(k, y)] = lst[: self.cap_per_k_label]
            else:
                capped[(k, y)] = lst
        self.pool.by_key = capped  # replace

        # compute p_target(k) = (1-β) p_base(k) + β p_hard(k)
        base_counts = self.dm.base_counts_k  # precomputed in dm.setup()
        hard_counts = self.pool.counts_k
        ks_all = sorted(set(base_counts) | set(hard_counts))

        def _norm(counts):
            s = sum(counts.get(k, 0) for k in ks_all) or 1
            return {k: counts.get(k, 0) / s for k in ks_all}

        p_base = _norm(base_counts)
        p_hard = _norm(hard_counts)
        p_target_k = {k: (1.0 - self.beta) * p_base[k] + self.beta * p_hard[k] for k in ks_all}

        # hand new targets to DataModule (used by sampler next epoch)
        self.dm.set_k_sampling_targets(hard_pool=self.pool, p_target_k=p_target_k, alpha=self.alpha)

        # save a quick histogram of hard ks for inspection
        with contextlib.suppress(Exception):
            xs = ks_all
            ys = [hard_counts.get(k, 0) for k in xs]
            plt.figure()
            plt.bar(xs, ys)
            plt.xlabel("k (# nodes in g1)")
            plt.ylabel("# hard samples (this epoch cumulative)")
            self.artefacts_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(self.artefacts_dir / f"hard_k_hist_epoch{trainer.current_epoch:02d}.png")
            plt.close()


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

    # Dataset & Encoder (HRR @ 7744)
    ds_cfg = cfg.dataset.default_cfg
    device = pick_device()
    log(f"Using device: {device!s}")

    log("Loading/creating hypernet …")
    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds_cfg).to(device=device).eval()
    log("Hypernet ready.")
    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.ATOM_TYPE][0].codebook)
    encoder = hypernet.to(device).eval()

    cpu_encoder = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds_cfg).to(device=torch.device("cpu")).eval()
    # datamodule with per-epoch resampling
    dm = PairsDataModule(cfg, encoder=cpu_encoder, device=torch.device("cpu"), is_dev=is_dev)

    # ----- model + optim -----
    model = LitBAHClassifier(
        hv_dim=cfg.hv_dim,
        proj_dim=cfg.proj_dim,
        n_heads=cfg.n_heads,
        norm=cfg.norm,
        proj_hidden=cfg.proj_hidden,
        dropout=cfg.dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        pos_weight=(cfg.n_per_parent / cfg.p_per_parent) if cfg.p_per_parent != cfg.n_per_parent else None,
        share_proj=cfg.share_proj,
        use_layernorm=cfg.use_layernorm,
        use_temperature=cfg.use_temperature,
    ).to(device=device)

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
        patience=10,
        min_delta=0.0,
        check_finite=True,  # stop if val becomes NaN/Inf
        check_on_train_epoch_end=False,
        verbose=True,
    )

    val_metrics_cb = MetricsPlotsAndOracleCallback(
        encoder=encoder, cfg=cfg, evals_dir=evals_dir, artefacts_dir=artefacts_dir
    )

    hard_mining_cb = HardMiningCallback(
        dm=dm, artefacts_dir=artefacts_dir, alpha=0.4, beta=0.6, margin=None
    )  # set margin like 0.1 for semi-hard
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
        callbacks=[checkpoint_callback, lr_monitor, time_logger, early_stopping, val_metrics_cb, hard_mining_cb],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=1000 if not is_dev else 1,
        enable_progress_bar=True,
        deterministic=False,
        precision=pick_precision(),
        num_sanity_val_steps=0,
    )

    # --- Train
    resume_path: Path | None = str(cfg.continue_from) if cfg.continue_from else None
    if resume_path:
        log(f"Resuming from {resume_path!s}")
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path)
    log("Finished training.")

    # Final plots


if __name__ == "__main__":

    def _parse_int_list(s: str) -> list[int]:
        # accept "4096,2048,512,128" (spaces ignored)
        return [int(tok) for tok in s.replace(" ", "").split(",") if tok]

    def _parse_vsa(s: str) -> VSAModel:
        return s if isinstance(s, VSAModel) else VSAModel(s)

    def _parse_supported_dataset(s: str) -> SupportedDataset:
        return s if isinstance(s, SupportedDataset) else SupportedDataset(s)

    def _parse_maybe_int(s: str) -> int | None:
        sl = s.strip().lower()
        return None if sl in {"none", "null", "nil", ""} else int(s)

    def _parse_maybe_float(s: str) -> float | None:
        sl = s.strip().lower()
        return None if sl in {"none", "null", "nil", ""} else float(s)

    def get_args(argv: list[str] | None = None) -> Config:
        """
        Build a Config by starting from dataclass defaults and then
        applying ONLY the CLI options the user actually provided.
        NOTE: For --vsa, pass a string like "HRR", not VSAModel.HRR.
        """
        cfg = Config()  # start with your defaults

        p = argparse.ArgumentParser(description="Biaffine Head Experiment Config")

        # General
        p.add_argument("--project_dir", "-pdir", type=Path, default=argparse.SUPPRESS)
        p.add_argument("--exp_dir_name", type=str, default=argparse.SUPPRESS)
        p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
        p.add_argument("--epochs", "-e", type=int, default=argparse.SUPPRESS)
        p.add_argument("--batch_size", "-bs", type=int, default=argparse.SUPPRESS)
        p.add_argument("--model_name", type=str, default=argparse.SUPPRESS)
        p.add_argument("--is_dev", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)

        # Biaffine model knobs
        p.add_argument("--proj_dim", type=int, default=argparse.SUPPRESS)
        p.add_argument("--n_heads", type=int, default=argparse.SUPPRESS)
        p.add_argument("--proj_hidden", type=_parse_maybe_int, default=argparse.SUPPRESS)
        p.add_argument("--dropout", type=float, default=argparse.SUPPRESS)
        p.add_argument("--share_proj", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)
        p.add_argument("--norm", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)
        p.add_argument("--use_layernorm", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)
        p.add_argument("--use_temperature", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)
        p.add_argument("--pos_weight", type=_parse_maybe_float, default=argparse.SUPPRESS)

        # HDC / encoder
        p.add_argument("--hv_dim", "-hd", type=int, default=argparse.SUPPRESS)
        p.add_argument("--vsa", "-v", type=_parse_vsa, default=argparse.SUPPRESS)
        p.add_argument("--dataset", "-ds", type=_parse_supported_dataset, default=argparse.SUPPRESS)

        # Optim
        p.add_argument("--lr", type=float, default=argparse.SUPPRESS)
        p.add_argument("--weight_decay", "-wd", type=float, default=argparse.SUPPRESS)

        # Loader
        p.add_argument("--num_workers", type=int, default=argparse.SUPPRESS)
        p.add_argument("--prefetch_factor", type=_parse_maybe_int, default=argparse.SUPPRESS)
        p.add_argument("--pin_memory", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)
        p.add_argument("--micro_bs", type=int, default=argparse.SUPPRESS)
        p.add_argument("--persistent_workers", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)

        # Checkpointing
        p.add_argument("--continue_from", type=Path, default=argparse.SUPPRESS)
        p.add_argument("--resume_retrain_last_epoch", type=str2bool, default=argparse.SUPPRESS)

        # Stratification
        p.add_argument("--stratify", type=str2bool, default=argparse.SUPPRESS)
        p.add_argument("--p_per_parent", type=int, default=argparse.SUPPRESS)
        p.add_argument("--n_per_parent", type=int, default=argparse.SUPPRESS)
        p.add_argument(
            "--exclude_negs", type=_parse_int_list, default=argparse.SUPPRESS, help="Comma-separated ints, e.g. '1,2,3'"
        )
        p.add_argument("--resample_training_data_on_batch", type=str2bool, default=argparse.SUPPRESS)

        ns = p.parse_args(argv)
        provided = vars(ns)

        for k, v in provided.items():
            if k == "vsa" and isinstance(v, str):
                v = VSAModel(v)
            if k == "dataset" and isinstance(v, str):
                v = SupportedDataset(v)
            if k == "exclude_negs" and isinstance(v, list):
                v = set(v)
            setattr(cfg, k, v)

        return cfg

    log(f"Running {Path(__file__).resolve()}")
    is_dev = os.getenv("LOCAL_HDC", False)

    if is_dev:
        log("Running in local HDC (DEV) ...")
        cfg: Config = Config(
            exp_dir_name="overfitting_batch_norm",
            seed=42,
            epochs=10,
            batch_size=4,
            hv_dim=40 * 40,
            vsa=VSAModel.HRR,
            dataset=SupportedDataset.QM9_SMILES_HRR_1600,
            lr=1e-4,
            weight_decay=0.0,
            num_workers=0,
            prefetch_factor=None,
            persistent_workers=False,
            pin_memory=False,
            continue_from=None,
            resume_retrain_last_epoch=False,
            p_per_parent=5,
            n_per_parent=5,
            resample_training_data_on_batch=True,
            is_dev=True,
        )
    else:
        log("Running in cluster ...")
        cfg = get_args()

    pprint(asdict(cfg), indent=2)
    run_experiment(cfg, is_dev=is_dev or cfg.is_dev)
