import datetime
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
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.datasets import ZINC

from src.datasets import AddNodeDegree
from src.encoding.configs_and_constants import DatasetConfig
from src.encoding.graph_encoders import HyperNet
from src.normalizing_flow.config import SpiralFlowConfig, get_flow_cli_args
from src.normalizing_flow.neural_spiral_network import NeuralSplineLightning


def setup_exp(base_dir: Path, project_dir: Path, ds_value: str) -> dict:
    """
    Sets up and returns all directories as a dict.
    """
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

    # Save a copy of the current script
    try:
        script_file = Path(__file__).resolve()
        shutil.copy(script_file, exp_dir / script_file.name)
        print(f"Saved a copy of the script to {exp_dir / script_file.name}")
    except NameError:
        print("Warning: __file__ is not defined. Script not saved.")

    return dirs


def plot_train_val_loss(df, artefacts_dir):
    train = df[df["train_loss_epoch"].notnull()]
    val = df[df["val_loss"].notnull()]

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


def pca_encode(x, pca):
    flat = x.view(-1, x.shape[-1]).cpu().numpy()
    reduced = pca.transform(flat)
    return torch.tensor(reduced, dtype=x.dtype)


def pca_decode(x_reduced, pca):
    flat = x_reduced.view(-1, x_reduced.shape[-1]).cpu().numpy()
    recon = pca.inverse_transform(flat)
    return torch.tensor(recon, dtype=x_reduced.dtype)


def load_or_fit_pca(train_dataset: Dataset, encoder: HyperNet, pca_path: Path, n_components=0.99999, n_fit=20_000):
    if pca_path.exists():
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
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(X_fit_flat)
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
    def __init__(self, base_dataset, encoder, pca):
        self.base_dataset = base_dataset
        self.encoder = encoder
        self.pca = pca

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        batch_data = Batch.from_data_list([data])
        res = self.encoder.forward(data=batch_data)
        x = torch.stack(
            [res["node_terms"].squeeze(0), res["edge_terms"].squeeze(0), res["graph_embedding"].squeeze(0)], dim=0
        )
        x_reduced = pca_encode(x, self.pca)
        return x_reduced


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


def run_experiment(cfg: SpiralFlowConfig):
    print("Running experiment")
    print(pprint(cfg.__dict__, indent=2))

    dirs = setup_exp(cfg.base_dir, cfg.project_dir, cfg.dataset.value)
    exp_dir = dirs["exp_dir"]
    models_dir = dirs["models_dir"]
    evals_dir = dirs["evals_dir"]
    artefacts_dir = dirs["artefacts_dir"]
    global_model_dir = dirs["global_model_dir"]
    global_dataset_dir = dirs["global_dataset_dir"]

    # Datasets
    train_dataset = ZINC(root=str(global_dataset_dir), pre_transform=AddNodeDegree(), split="train", subset=True)
    print(f"Train dataset: {len(train_dataset)} samples")
    validation_dataset = ZINC(root=str(global_dataset_dir), pre_transform=AddNodeDegree(), split="val", subset=True)
    print(f"Validation dataset: {len(validation_dataset)} samples")

    # Update config
    device = get_device()
    vsa = cfg.vsa
    ds = cfg.dataset
    ds.default_cfg.vsa = vsa
    ds.default_cfg.hv_dim = cfg.hv_dim
    ds.default_cfg.device = device
    ds.default_cfg.seed = cfg.seed
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}

    # HyperNet: Load or create
    encoder = load_or_create_hypernet(
        path=global_model_dir, cfg=ds.default_cfg, depth=3
    )

    # PCA: Load or fit
    n_components = 0.99999
    pca_path = global_model_dir / f"hypervec_pca_{vsa.value}_d{cfg.hv_dim}_s{cfg.seed}_c{str(n_components)[2:]}.joblib"
    pca = load_or_fit_pca(
        train_dataset=train_dataset, encoder=encoder, pca_path=pca_path, n_components=n_components, n_fit=20_000
    )

    reduced_dim = int(pca.n_components_)
    cfg.num_input_channels = 3 * reduced_dim
    cfg.input_shape = (3, reduced_dim)

    # DataLoaders
    train_dataloader = DataLoader(
        EncodedPCADataset(train_dataset, encoder, pca),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        EncodedPCADataset(validation_dataset, encoder, pca),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Model
    model = NeuralSplineLightning(cfg)

    # Logging and callbacks
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
        logger=csv_logger,
        callbacks=[checkpoint_callback, lr_monitor, time_logger],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        log_every_n_steps=20,
        enable_progress_bar=True,
        detect_anomaly=True,  # Detect anomalies in training
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # Save final model weights (checkpoint already saves best/last)
    torch.save(model.state_dict(), models_dir / "final_model.pt")

    # Save metrics
    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df.to_parquet(evals_dir / "metrics.parquet")
        print(f"Saved training/validation metrics to {evals_dir / 'metrics.parquet'}")
        plot_train_val_loss(df, artefacts_dir)

    print("==== The Experiment is done! ====")


if __name__ == "__main__":
    run_experiment(get_flow_cli_args())
