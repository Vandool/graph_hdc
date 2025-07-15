import datetime
import os
import random
import shutil
import string
import time
from pathlib import Path
from pprint import pprint

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch
from torch_geometric.datasets import ZINC

from graph_hdc.utils import AbstractEncoder
from src.datasets import AddNodeDegree
from src.encoding.configs_and_constants import DatasetConfig, SupportedDataset
from src.encoding.graph_encoders import HyperNet
from src.encoding.the_types import VSAModel
from src.normalizing_flow.config import FlowConfig, get_flow_cli_args
from src.normalizing_flow.models import RealNVPLightning


def setup_exp(ds_value: str) -> dict:
    """
    Sets up experiment directories based on the current script location.

    Args:
        ds_value (str): Dataset name to use for global_dataset_dir.

    Returns:
        dict: Dictionary containing paths to various directories.
    """
    # Resolve script location
    script_path = Path(__file__).resolve()
    experiments_path = script_path.parent
    script_stem = script_path.stem  # without .py

    # Resolve base and project directories
    base_dir = experiments_path / "results" / script_stem
    base_dir.mkdir(parents=True, exist_ok=True)

    project_dir = script_path.parents[3]  # adjust as needed

    print(f"Setting up experiment in {base_dir}")
    now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{''.join(random.choices(string.ascii_lowercase, k=4))}"
    exp_dir = base_dir / now
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory created: {exp_dir}")

    dirs = {
        "exp_dir": exp_dir,
        "models_dir": exp_dir / "models",
        "evals_dir": exp_dir / "evaluations",
        "artefacts_dir": exp_dir / "artefacts",
        "global_model_dir": project_dir / "_models",
        "global_dataset_dir": project_dir / "_datasets" / ds_value,
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Save a copy of the script
    try:
        shutil.copy(script_path, exp_dir / script_path.name)
        print(f"Saved a copy of the script to {exp_dir / script_path.name}")
    except Exception as e:
        print(f"Warning: Failed to save script copy: {e}")

    return dirs


def plot_train_val_loss(df, artefacts_dir):
    train = df[df["train_loss_epoch"].notna()]
    val = df[df["val_loss"].notna()]

    plt.figure(figsize=(8, 5))
    plt.plot(train["epoch"], train["train_loss_epoch"], label="Train Loss")
    plt.plot(val["epoch"], val["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs. Validation Loss")
    plt.legend()
    plt.tight_layout()

    artefacts_dir.mkdir(exist_ok=True)
    plt.savefig(artefacts_dir / "train_val_loss.png")
    plt.close()
    print(f"Saved train/val loss plot to {artefacts_dir / 'train_val_loss.png'}")


def pca_encode(x: torch.Tensor, pca: PCA, norm: bool = False) -> torch.Tensor:
    """
    Encode data using a fitted PCA, with optional normalization.

    :param x: Input tensor of shape (..., features).
    :type x: torch.Tensor
    :param pca: Fitted PCA instance with attributes `mean_` and `std_`.
    :type pca: PCA
    :param norm: Whether to normalize input by mean and std before PCA.
    :type norm: bool
    :returns: Tensor of reduced dimensions, same dtype as input.
    :rtype: torch.Tensor

    The input is flattened over all but the last dimension, optionally normalized,
    transformed with the PCA, then returned as a tensor.
    """
    flat = x.view(-1, x.shape[-1]).cpu().numpy()
    if norm:
        flat = (flat - pca.mean_) / pca.std_
    reduced = pca.transform(flat)
    return torch.tensor(reduced, dtype=x.dtype)


def pca_decode(x_reduced: torch.Tensor, pca: PCA, denorm: bool = False) -> torch.Tensor:
    """
    Decode PCA-reduced data, with optional de-normalization.

    :param x_reduced: Reduced tensor of shape (..., reduced_features).
    :type x_reduced: torch.Tensor
    :param pca: Fitted PCA instance with attributes `mean_` and `std_`.
    :type pca: PCA
    :param denorm: Whether to reverse normalization after inverse PCA.
    :type denorm: bool
    :returns: Reconstructed tensor in original feature space.
    :rtype: torch.Tensor

    The reduced tensor is flattened, inverse-transformed by PCA, and then
    optionally scaled and shifted back.
    """
    flat = x_reduced.view(-1, x_reduced.shape[-1]).cpu().numpy()
    recon = pca.inverse_transform(flat)
    if denorm:
        recon = recon * pca.std_ + pca.mean_
    return torch.tensor(recon, dtype=x_reduced.dtype)


def load_or_fit_pca(
    train_dataset: Dataset, encoder: AbstractEncoder, pca_path: Path | None = None, n_components: float = 0.99999, n_fit: int = 20000
) -> PCA:
    """
    Load an existing PCA from disk or fit a new one and save it.

    :param train_dataset: Dataset for fitting PCA.
    :type train_dataset: Dataset
    :param encoder: Model or function returning a dict with keys \"node_terms\", \"edge_terms\", and \"graph_embedding\".
    :param pca_path: Path to load/save the PCA object.
    :type pca_path: Path
    :param n_components: Number of components or variance ratio for PCA.
    :type n_components: float
    :param n_fit: Maximum number of samples to fit PCA on.
    :type n_fit: int
    :returns: Fitted PCA instance with `mean_` and `std_` attributes.
    :rtype: PCA

    If a PCA exists at `pca_path`, it is loaded. Otherwise, embeddings
    are collected by applying `encoder` to dataset entries until `n_fit`
    samples, flattened, and used to fit a new PCA. The mean and std of the
    fit data are stored on the PCA for later normalization.
    """
    if pca_path is not None and pca_path.exists():
        print(f"Loading existing PCA from {pca_path}")
        pca = joblib.load(pca_path)
        print(f"Loaded PCA with {pca.n_components_} components")
        return pca

    print("Fitting PCA on training data...")
    n_fit = min(n_fit, len(train_dataset))
    X_fit = []
    for i in range(n_fit):
        data = train_dataset[i]
        batch_data = Batch.from_data_list([data])
        res = encoder.forward(data=batch_data)
        x = torch.stack(
            [res["node_terms"].squeeze(0), res["edge_terms"].squeeze(0), res["graph_embedding"].squeeze(0)], dim=0
        )  # [3, D]
        X_fit.append(x.cpu().numpy())
    X_fit = np.stack(X_fit)
    X_fit_flat = X_fit.reshape(-1, X_fit.shape[-1])

    # Compute mean and std for normalization
    mu = np.mean(X_fit_flat, axis=0)
    sigma = np.std(X_fit_flat, axis=0)

    # Fit PCA
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(X_fit_flat)

    # Attach normalization stats
    pca.mean_ = mu
    pca.std_ = sigma

    print(f"PCA reduced dimension: {pca.n_components_} from {X_fit.shape[-1]}")
    joblib.dump(pca, pca_path)
    print(f"Saved new PCA to {pca_path}")
    return pca


def load_or_create_hypernet(path: Path, cfg: DatasetConfig, depth: int) -> HyperNet:
    path = path / f"hypernet_{cfg.vsa.value}_d{cfg.hv_dim}_s{cfg.seed}_dpth{depth}.pt"
    if path.exists():
        print(f"Loading existing HyperNet from {path}")
        encoder = HyperNet(config=cfg, depth=depth)
        encoder.load(path)
    else:
        print("Creating new HyperNet instance.")
        encoder = HyperNet(config=cfg, depth=depth)
        encoder.populate_codebooks()
        encoder.save_to_path(path)
        print(f"Saved new HyperNet to {path}")
    return encoder


class EncodedPCADataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, encoder, pca: PCA | None = None, *, use_norm_pca: bool = False):
        self.base_dataset = base_dataset
        self.encoder = encoder
        self.pca = pca
        self.use_norm = use_norm_pca  # Whether to normalize the PCA

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        batch_data = Batch.from_data_list([data])
        res = self.encoder.forward(data=batch_data)
        x = torch.stack(
            [res["node_terms"].squeeze(0), res["edge_terms"].squeeze(0), res["graph_embedding"].squeeze(0)], dim=0
        )
        if self.pca is not None:
            return pca_encode(x, self.pca, self.use_norm)
        return x


def get_device():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"CUDA is available. Detected {count} GPU device{'s' if count != 1 else ''}.")
        return torch.device("cuda")
    print("CUDA is not available.")
    return torch.device("cpu")


class TimeLoggingCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        trainer.logger.log_metrics({"elapsed_time_sec": elapsed}, step=trainer.current_epoch)


def run_experiment(cfg: FlowConfig):
    print("Running experiment")
    pprint(cfg.__dict__, indent=2)

    dirs = setup_exp(cfg.dataset.value)
    exp_dir = dirs["exp_dir"]
    models_dir = dirs["models_dir"]
    evals_dir = dirs["evals_dir"]
    artefacts_dir = dirs["artefacts_dir"]
    global_model_dir = dirs["global_model_dir"]
    global_dataset_dir = dirs["global_dataset_dir"]

    # W&B Logging — use existing run (from sweep or manual init)
    run = wandb.run or wandb.init(project="realnvp-hdc", config=cfg.__dict__, name=f"run_{cfg.hv_dim}_{cfg.seed}", reinit=True)
    run.tags = [f"hv_dim={cfg.hv_dim}", f"vsa={cfg.vsa.value}", f"dataset={cfg.dataset.value}"]

    wandb_logger = WandbLogger(log_model=True, experiment=run)

    train_data = ZINC(root=str(global_dataset_dir), pre_transform=AddNodeDegree(), split="train", subset=True)[:1]
    train_dataset = Subset(train_data, indices=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    print(f"Train length = {len(train_dataset)}")  # → 4
    print(train_dataset[0])
    # validation_data = ZINC(root=str(global_dataset_dir), pre_transform=AddNodeDegree(), split="val", subset=True)[:1]
    validation_dataset = Subset(train_data, indices=[0, 0, 0, 0])
    print(f"{len(validation_dataset)=}")
    print(validation_dataset[0])


    device = get_device()
    ds = cfg.dataset
    ds.default_cfg.vsa = cfg.vsa
    ds.default_cfg.hv_dim = cfg.hv_dim
    ds.default_cfg.device = device
    ds.default_cfg.seed = cfg.seed
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}

    encoder = load_or_create_hypernet(path=global_model_dir, cfg=ds.default_cfg, depth=3)

    # Print decoded counters of one data point
    data_batch = Batch.from_data_list(train_data)
    encoded_data = encoder.forward(data_batch)
    lvl0_counter = encoder.decode_order_zero_counter(encoded_data['node_terms'])
    print(f"Decoded level zero counter = {lvl0_counter[0]}")
    print(f"Decoded level zero counter total = {lvl0_counter[0].total()}")
    lvl1_counter = encoder.decode_order_one_counter_explain_away_faster(encoded_data['edge_terms'])
    print(f"Decoded level one counter = {lvl1_counter[0]}")
    print(f"Decoded level one counter total = {lvl1_counter[0].total()}")


    n_components = 0.998
    pca_path = global_model_dir / f"hypervec_pca_{cfg.vsa.value}_d{cfg.hv_dim}_s{cfg.seed}_c{str(n_components)[2:]}.joblib"
    pca = load_or_fit_pca(
        train_dataset=ZINC(root=str(global_dataset_dir), pre_transform=AddNodeDegree(), split="train", subset=True),
        encoder=encoder,
        pca_path=pca_path,
        n_components=n_components,
        n_fit=20_000,
    )

    reduced_dim = int(pca.n_components_)
    cfg.num_input_channels = 3 * reduced_dim
    cfg.input_shape = (3, reduced_dim)

    train_dataloader = DataLoader(
        EncodedPCADataset(train_dataset, encoder, pca, use_norm_pca=True),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        EncodedPCADataset(validation_dataset, encoder, pca, use_norm_pca=True),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = RealNVPLightning(cfg)

    csv_logger = CSVLogger(save_dir=str(evals_dir), name="logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=2,
        mode="min",
        dirpath=str(models_dir),
        filename="epoch{epoch:02d}-val{val_loss:.2f}",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    time_logger = TimeLoggingCallback()

    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=[csv_logger, wandb_logger],
        callbacks=[checkpoint_callback, lr_monitor, time_logger],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        log_every_n_steps=20,
        enable_progress_bar=True,
        detect_anomaly=True,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df.to_parquet(evals_dir / "metrics.parquet")
        plot_train_val_loss(df, artefacts_dir)

    print("==== The Experiment is done! ====")


def sweep_entrypoint():
    wandb.init()
    args = wandb.config.as_dict()
    fixed_cfg = get_flow_cli_args()
    for k, v in args.items():
        setattr(fixed_cfg, k, v)

        # 🔧 Convert back to enums
    fixed_cfg.dataset = SupportedDataset(fixed_cfg.dataset)
    fixed_cfg.vsa = VSAModel(fixed_cfg.vsa)
    fixed_cfg.weight_decay = float(fixed_cfg.weight_decay)


    run_experiment(fixed_cfg)

if __name__ == "__main__":
    if "WANDB_SWEEP" in os.environ:
        sweep_entrypoint()
    else:
        run_experiment(get_flow_cli_args())

