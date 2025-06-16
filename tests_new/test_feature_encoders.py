from codeop import CommandCompiler

import pytest
import torch

from src.encoding.configs_and_constants import FeatureConfig
from src.encoding.feature_encoders import CategoricalIntegerEncoder, CombinatoricIntegerEncoder
from src.encoding.types import VSAModel


@pytest.mark.parametrize(
    ("dim", "vsa", "cfg"),
    [
        (16, VSAModel.MAP, FeatureConfig(count=28, encoder_cls=CategoricalIntegerEncoder)),
        (32, VSAModel.FHRR, FeatureConfig(count=28, encoder_cls=CategoricalIntegerEncoder)),
        (64, VSAModel.HRR, FeatureConfig(count=28, encoder_cls=CategoricalIntegerEncoder)),
    ],
)
def test_encode_shape_and_type(dataset, encoder_factory, dim, vsa, cfg):
    """
    The encoder should accept inputs of shape [..., N, 1] and return [..., N, D] for both datasets.
    """
    data, ds_enum = dataset
    encoder = encoder_factory(dim=dim, vsa=vsa, encoder_cls=cfg.encoder_cls, num_categories=cfg.count)
    x = data.x
    hv = encoder.encode(x)

    assert hv.shape == (x.shape[0], dim)


@pytest.mark.parametrize(
    "invalid_shape",
    [
        torch.randint(0, 10, (5,)),  # missing last singleton dim
        torch.randint(0, 10, (5, 2, 2)),  # last dim != 1
    ],
)
def test_encode_invalid_input_shape(encoder_factory, invalid_shape):
    """
    Passing a tensor without a singleton last dimension should raise ValueError.
    """

    encoder = encoder_factory(dim=16, vsa=VSAModel.MAP, num_categories=28, encoder_cls=CategoricalIntegerEncoder)
    with pytest.raises(ValueError, match="Expected"):
        encoder.encode(invalid_shape)


@pytest.mark.parametrize("vsa", [VSAModel.MAP, VSAModel.FHRR, VSAModel.HRR])
def test_decode_roundtrip_node_features(dataset, encoder_factory, vsa):
    """
    invariant x = decode(encode(x)) should hold for all vsa types.
    """
    data, ds_enum = dataset
    for cfg in ds_enum.default_cfg.node_feature_configs.values():
        sliced_data = data.x.narrow(dim=-1, start=cfg.index_range[0], length=cfg.index_range[1] - cfg.index_range[0])
        encoder = encoder_factory(dim=16, vsa=vsa, num_categories=cfg.count, encoder_cls=cfg.encoder_cls)
        hv = encoder.encode(sliced_data)
        decoded = encoder.decode(hv)

        torch.equal(decoded, sliced_data)

def test_decode_roundtrip_node_features_combinatorial():
    """
    invariant x = decode(encode(x)) should hold for all vsa types.
    """
    encoder = CombinatoricIntegerEncoder(dim=128, vsa=VSAModel.MAP, num_categories=28*6)
    features = torch.Tensor([(0,1), (5, 3)])
    encoded = encoder.encode(features)
    decoded = encoder.decode(encoded)
    assert torch.equal(decoded, features)

@pytest.mark.parametrize("vsa", [VSAModel.MAP, VSAModel.FHRR, VSAModel.HRR, ])
def test_decode_roundtrip_edge_features(dataset, encoder_factory, vsa):
    """
    invariant x = decode(encode(x)) should hold for all vsa types.
    """
    data, ds_enum = dataset
    for cfg in ds_enum.default_cfg.edge_feature_configs.values():
        edge_properties = data.edge_attr
        if edge_properties.dim() == 1:
            edge_properties = edge_properties.unsqueeze(1)
        sliced_data = edge_properties.narrow(dim=-1, start=cfg.index_range[0], length=cfg.index_range[1] - cfg.index_range[0])
        encoder = encoder_factory(dim=16, vsa=vsa, num_categories=cfg.count, encoder_cls=cfg.encoder_cls)
        hv = encoder.encode(sliced_data)
        decoded = encoder.decode(hv)

        torch.equal(decoded, sliced_data)


def test_get_codebook_properties(encoder_factory):
    """
    The codebook should have shape [C, D] and dtype float32 for both datasets.
    """
    dim = 8
    encoder = encoder_factory(dim=dim, vsa=VSAModel.MAP, num_categories=8)
    cb = encoder.get_codebook()
    assert cb.shape == (encoder.num_categories, dim)


def test_seeding_works(encoder_factory):
    """
    The codebooks created with one seed should be exactly the same.
    The codebooks created with different seeds should not be the same.
    """
    dim = 8
    cb1 = encoder_factory(dim=dim, vsa=VSAModel.MAP, seed=42, num_categories=3).get_codebook()
    cb2 = encoder_factory(dim=dim, vsa=VSAModel.MAP, seed=42, num_categories=3).get_codebook()
    assert torch.equal(cb1, cb2)

    cb3 = encoder_factory(dim=dim, vsa=VSAModel.MAP, seed=7, num_categories=3).get_codebook()
    assert not torch.equal(cb1, cb3)
    assert not torch.equal(cb2, cb3)
