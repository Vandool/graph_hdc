import datetime
import random
import shutil
import string
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.datasets import ZINC

from src.datasets import AddNodeDegree
from src.encoding.graph_encoders import HyperNet
from src.normalizing_flow.config import FlowConfig, get_flow_cli_args
from src.normalizing_flow.neural_spiral_network import NeuralSplineLightning


def plot_train_val_loss(df, exp_dir):
    # Filter for logged values
    train = df[df['train_loss_epoch'].notnull()]
    val = df[df['val_loss'].notnull()]

    plt.figure(figsize=(8,5))
    plt.plot(train['epoch'], train['train_loss_epoch'], label='Train Loss')
    plt.plot(val['epoch'], val['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs. Validation Loss')
    plt.legend()
    plt.tight_layout()

    artefacts_dir = exp_dir / "artefacts"
    artefacts_dir.mkdir(exist_ok=True)
    plt.savefig(artefacts_dir / "train_val_loss.png")
    plt.close()
    print(f"Saved train/val loss plot to {artefacts_dir / 'train_val_loss.png'}")


def setup_exp(base_dir: Path) -> tuple[Path, Path, Path]:
    print(f"Setting up experiment in {base_dir}")
    now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{''.join(random.choices(string.ascii_lowercase, k=4))}"
    base_dir = base_dir / now
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory created: {base_dir}")

    models = base_dir / "models"
    models.mkdir(parents=True, exist_ok=True)

    evals = base_dir / "evaluations"
    evals.mkdir(parents=True, exist_ok=True)

    # Save a copy of the current script
    try:
        script_file = Path(__file__).resolve()
        shutil.copy(script_file, base_dir / script_file.name)
        print(f"Saved a copy of the script to {base_dir / script_file.name}")
    except NameError:
        print("Warning: __file__ is not defined. Script not saved.")

    return base_dir, models, evals


class EncodedHypervectorDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, encoder):
        self.base_dataset = base_dataset
        self.encoder = encoder

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        # The encoder expects batched input, so wrap in a Batch

        batch_data = Batch.from_data_list([data])
        res = self.encoder.forward(data=batch_data)
        # [1, D] for each, squeeze out batch dim
        x = torch.stack(
            [res["node_terms"].squeeze(0), res["edge_terms"].squeeze(0), res["graph_embedding"].squeeze(0)], dim=0
        )  # shape [3, D]
        return x


def encode_hypervectors(batch, encoder):
    """
    batch: list of torch_geometric.data.Data objects (from DataLoader)
    encoder: HyperNet instance (callable)
    Returns: torch.Tensor [batch_size, 3, D]
    """
    # You may need to batch with torch_geometric's Batch if your encoder expects it

    batch_data = Batch.from_data_list(batch)
    res = encoder.forward(data=batch_data)
    # [batch_size, D] for each: node_terms, edge_terms, graph_embedding
    x = torch.stack([res["node_terms"], res["edge_terms"], res["graph_embedding"]], dim=1)
    return x  # shape [batch_size, 3, D]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"CUDA is available. Detected {count} GPU device{'s' if count != 1 else ''}.")
        return torch.device("cuda")

    print("CUDA is not available.")
    return torch.device("cpu")


def run_experiment(cfg: FlowConfig):
    print("Running experiment")
    print(pprint(cfg.__dict__, indent=2))
    exp_dir, models_dir, evals_dir = setup_exp(cfg.base_dir)
    print(f"Experiment directory: {exp_dir}")

    ## Apply configs
    device = get_device()
    vsa = cfg.vsa
    ds = cfg.dataset
    ds.default_cfg.vsa = vsa
    ds.default_cfg.hv_dim = cfg.hv_dim
    ds.default_cfg.device = device
    ds.default_cfg.seed = cfg.seed
    # Disable edge and graph features
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}

    ## Create Paths
    dataset_root = cfg.project_dir / "datasets" / ds.value
    dataset_root.mkdir(parents=True, exist_ok=True)
    train_dataset = ZINC(root=str(dataset_root), pre_transform=AddNodeDegree(), split="train", subset=True)
    print(f"Train dataset: {len(train_dataset)} samples")
    validation_dataset = ZINC(root=str(dataset_root), pre_transform=AddNodeDegree(), split="val", subset=True)
    print(f"Validation dataset: {len(validation_dataset)} samples")

    ### Initialize Hypernet and evals
    encoder = HyperNet(config=ds.default_cfg, hidden_dim=ds.default_cfg.hv_dim, depth=3)

    ## Initialize DataLoaders
    train_dataloader = DataLoader(
        EncodedHypervectorDataset(train_dataset, encoder),
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        EncodedHypervectorDataset(validation_dataset, encoder),
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Model
    model = NeuralSplineLightning(cfg)

    # Lightning logger and checkpointing
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

    trainer = Trainer(
        max_epochs=cfg.epochs if hasattr(cfg, "epochs") else 3,
        logger=csv_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        log_every_n_steps=20,
        enable_progress_bar=True,
    )

    # Training
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # Save final model weights (optional, as checkpoint already saves all)
    torch.save(model.state_dict(), models_dir / "final_model.pt")
    encoder.save_to_path(path=models_dir / f"encoder_{vsa.value}_d{cfg.hv_dim}_s{cfg.seed}.pt")

    # Convert Lightning logs to parquet (train/val losses)
    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df.to_parquet(evals_dir / "metrics.parquet")
        print(f"Saved training/validation metrics to {evals_dir / 'metrics.parquet'}")
        plot_train_val_loss(df, exp_dir)

if __name__ == "__main__":
    run_experiment(get_flow_cli_args())
