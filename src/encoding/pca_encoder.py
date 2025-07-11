import numpy as np
import joblib
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from typing import Callable, Optional, Union


class PCAEncoder:
    """
    PCA-based encoder/decoder with optional normalization and persistence.

    :param n_components: Number of components or fraction of variance to preserve.
    :param svd_solver: SVD solver for PCA (e.g. "full", "auto").
    :param pca_path: Optional filesystem path to load/save the PCA model.
    :param encoder_fn: Function mapping a dataset item to a 1D embedding array.
    """

    def __init__(
        self,
        n_components: Union[int, float] = 0.99,
        svd_solver: str = "full",
        pca_path: Optional[Union[str, Path]] = None,
        encoder_fn: Optional[Callable[[object], np.ndarray]] = None,
    ) -> None:
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.pca_path = Path(pca_path) if pca_path is not None else None
        self.encoder_fn = encoder_fn
        self.pca: Optional[PCA] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, dataset: Dataset, max_samples: int = 20000) -> PCA:
        """
        Fit the PCA model using a subset of data.

        :param dataset: Torch Dataset yielding items for `encoder_fn`.
        :param max_samples: Maximum number of samples to draw for fitting.
        :returns: Fitted PCA instance.
        """
        if self.pca_path and self.pca_path.exists():
            self.pca = joblib.load(self.pca_path)
            return self.pca

        if self.encoder_fn is None:
            raise ValueError("encoder_fn must be provided to fit the PCA")

        n_fit = min(len(dataset), max_samples)
        embeddings = []
        for idx in range(n_fit):
            emb = self.encoder_fn(dataset[idx])
            embeddings.append(np.asarray(emb).reshape(-1, emb.shape[-1]))
        X = np.vstack(embeddings)

        # Compute normalization
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12  # avoid divide-by-zero

        # Fit PCA
        self.pca = PCA(n_components=self.n_components, svd_solver=self.svd_solver)
        X_norm = (X - self.mean_) / self.std_
        self.pca.fit(X_norm)

        # Persist model
        if self.pca_path:
            joblib.dump(self.pca, self.pca_path)
        return self.pca

    def load(self) -> PCA:
        """
        Load a PCA model from disk.

        :returns: Loaded PCA instance.
        """
        if not self.pca_path or not self.pca_path.exists():
            raise FileNotFoundError("PCA model file not found at {}".format(self.pca_path))
        self.pca = joblib.load(self.pca_path)
        return self.pca

    def transform(
        self,
        x: Union[np.ndarray, torch.Tensor],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Project data into PCA space.

        :param x: Input array/tensor of shape (..., features).
        :param normalize: Apply preprocessing (x - mean) / std if True.
        :returns: Tensor of shape (..., n_components).
        """
        if self.pca is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("PCAEncoder is not fitted or loaded.")

        arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        flat = arr.reshape(-1, arr.shape[-1]).astype(float)
        if normalize:
            flat = (flat - self.mean_) / self.std_
        reduced = self.pca.transform(flat)

        tensor = torch.from_numpy(reduced).to(x if torch.is_tensor(x) else torch.float32)
        return tensor.view(*x.shape[:-1], reduced.shape[-1])

    def inverse_transform(
        self,
        z: Union[np.ndarray, torch.Tensor],
        denormalize: bool = True,
    ) -> torch.Tensor:
        """
        Reconstruct data from PCA space.

        :param z: PCA coefficients of shape (..., n_components).
        :param denormalize: Apply std * x + mean if True.
        :returns: Tensor in original feature space.
        """
        if self.pca is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("PCAEncoder is not fitted or loaded.")

        arr = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
        flat = arr.reshape(-1, arr.shape[-1])
        recon = self.pca.inverse_transform(flat)
        if denormalize:
            recon = recon * self.std_ + self.mean_

        tensor = torch.from_numpy(recon).to(z if torch.is_tensor(z) else torch.float32)
        return tensor.view(*z.shape[:-1], recon.shape[-1])

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the current PCA model to disk.

        :param path: Destination path for the PCA file.
        """
        if self.pca is None:
            raise RuntimeError("No PCA model to save.")
        joblib.dump(self.pca, Path(path))
