import math

import pytest
import torch
import torchhd
from torchhd import BSCTensor, HRRTensor, MAPTensor

from src.utils.utils import TupleIndexer, cartesian_bind_tensor, cartesian_bind_tensor_2


def test_tuple_indexer_size():
    seq = [3, 2, 4]
    indexer = TupleIndexer(seq)
    assert indexer.size() == math.prod(seq)

def test_tuple_indexer_siz():
    seq = [3, 2, 0]
    indexer = TupleIndexer(seq)
    assert indexer.size() == math.prod([s for s in seq if s])



def test_empty_tuple_indexer():
    seq = []
    indexer = TupleIndexer(seq)
    assert indexer.size() == 0


@pytest.mark.parametrize("vsa_model, TensorClass", [
    ("HRR", HRRTensor),
])
def test_cartesian_bind_tensor_pair(vsa_model, TensorClass):
    """
    Verify that for two domains of size 2 each,
    cartesian_bind_tensor([a, b]) returns 4 rows:
      bind(a[0], b[0]), bind(a[0], b[1]),
      bind(a[1], b[0]), bind(a[1], b[1]).
    """
    torch.manual_seed(0)
    D = 6

    # Create two hypervector sets of shape [2, D]
    a = torchhd.random(2, D, vsa=vsa_model, device="cpu")
    b = torchhd.random(2, D, vsa=vsa_model, device="cpu")

    out = cartesian_bind_tensor([a, b])

    # 1) Check shape and class
    assert out.shape == (4, D)
    assert isinstance(out, TensorClass)
    # 2) Check that each row equals the bind of the corresponding pair
    #    Order of torch.cartesian_prod([range(2), range(2)]) is:
    #      (0, 0), (0, 1), (1, 0), (1, 1)
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            expected = torchhd.bind(a[i], b[j])
            assert torch.allclose(out[idx], expected, atol=1e-6), (
                f"Mismatch at idx={idx} for tuple ({i}, {j})")

@pytest.mark.parametrize("vsa_model, TensorClass", [
    ("HRR", HRRTensor),
])
def test_cartesian_bind_tensor_more_than_two(vsa_model, TensorClass):
    torch.manual_seed(0)
    D = 6

    # Create two hypervector sets of shape [2, D]
    a = torchhd.random(2, D, vsa=vsa_model, device="cpu")
    b = torchhd.random(2, D, vsa=vsa_model, device="cpu")
    c = torchhd.random(3, D, vsa=vsa_model, device="cpu")

    out = cartesian_bind_tensor([a, b, c])

    assert out.shape == (12, D)
    assert isinstance(out, TensorClass)

    for i in range(2):
        for j in range(2):
            for k in range(3):
                idx = i * (2 * 3) + j * 3 + k
                expected = torchhd.bind(a[i], b[j]).bind(c[k])
                assert torch.allclose(out[idx], expected, atol=1e-6)