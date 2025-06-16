from typing import Optional

import pytest
from torch_geometric.data import Batch, Dataset
from torch_geometric.datasets import QM9, ZINC

from src.datasets import AddNodeDegree
from src.encoding.configs_and_constants import SupportedDataset, VSAModel
from src.encoding.feature_encoders import AbstractFeatureEncoder, CategoricalIntegerEncoder
from tests.utils import DATASET_TEST_PATH


@pytest.fixture(
    scope="session",
    params=[SupportedDataset.ZINC],
)
def dataset(request):
    """
    Load the first graph from the specified dataset (ZINC or QM9) using local assets.
    """
    root = DATASET_TEST_PATH / request.param.value
    data = ZINC(root=root / 'del')[0] if request.param == SupportedDataset.ZINC else QM9(root=root)[0]
    return data, request.param


@pytest.fixture
def encoder_factory():
    """
    Factory for creating CategoricalIntegerEncoder instances with custom dimensions and VSA models.

    Usage:
        encoder = encoder_factory(dim=32, vsa=VSAModel.MAP)
    """

    def _factory(
        dim: int,
        num_categories: int,
        encoder_cls: type[AbstractFeatureEncoder] = CategoricalIntegerEncoder,
        vsa: VSAModel = VSAModel.MAP,
        seed: Optional[int] = None,
    ) -> AbstractFeatureEncoder:
        return encoder_cls(dim=dim, num_categories=num_categories, vsa=vsa, seed=seed)

    return _factory


@pytest.fixture(scope="session")
def dataset_loader():
    """
    Returns a loader function for full PyG datasets by enum, using local assets.
    Usage:
        ds = dataset_loader(SupportedDataset.ZINC)
    """

    def _loader(ds_enum: SupportedDataset) -> Dataset:
        root = DATASET_TEST_PATH / ds_enum.value
        if ds_enum == SupportedDataset.ZINC:
            return ZINC(root=root)
        if ds_enum in {SupportedDataset.ZINC_NODE_DEGREE, SupportedDataset.ZINC_NODE_DEGREE_COMB}:
            return ZINC(root=root, pre_transform=AddNodeDegree())
        return QM9(root=root)

    return _loader


@pytest.fixture
def batch_data(dataset_loader):
    """
    Factory that returns a batched Data object for a given dataset enum and batch size.

    Usage in tests:
        batch = batch_data(SupportedDataset.ZINC, batch_size=4)
    """

    def _batch(ds_enum: SupportedDataset, batch_size: int) -> Batch:
        ds = dataset_loader(ds_enum)
        # collect first `batch_size` graphs
        data_list = [ds[i] for i in range(batch_size)]
        return Batch.from_data_list(data_list)

    return _batch
