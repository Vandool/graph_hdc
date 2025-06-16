import logging
from collections.abc import Callable
from typing import Literal

import pytest
import torch
import torchhd
from torch import Tensor

from src.encoding.types import VSAModel
from src.utils.utils import scatter_hd

logging.basicConfig()
logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


def scatter_manually(idx: Tensor, src: Tensor, red_fn: Callable):
    buckets = {0: [], 1: [], 2: []}
    for i, b in enumerate(idx.tolist()):
        buckets[b].append(src[i])
    expected = []
    for b in range(3):
        acc = buckets[b][0]
        for hv in buckets[b][1:]:
            acc = red_fn(acc, hv)
        expected.append(acc)
    return torch.stack(expected, dim=0)


def index_tensor():
    return torch.tensor([0, 1, 0, 1, 2], dtype=torch.long)


def get_op(op: Literal["bind", "bundle"]):
    if op == "bind":
        return torchhd.bind
    if op == "bundle":
        return torchhd.bundle
    raise ValueError(op)


@pytest.mark.parametrize(
    ("vsa", "red_op"),
    [
        (VSAModel.MAP, "bind"),
        (VSAModel.HRR, "bind"),
        # There's a problem with bundling of BSC, the torchhd implementation seems to be buggy
        # the bundle(a,b) != multibundle(stack(a,b)), which should be the same since none of the operations perform any
        # normalisation.
        # (VSAModel.BSC, "bundle"),
        (VSAModel.MAP, "bundle"),
        (VSAModel.HRR, "bundle"),
    ],
)
def test_scatter_hd_map_bundle(vsa, red_op):
    src = torchhd.random(5, 2, vsa=vsa.value)
    idx = index_tensor()

    # dim_size is 3
    expected = scatter_manually(idx, src, get_op(op=red_op))
    actual = scatter_hd(src, idx, op=red_op)

    ## Test scattering with the same dimensions as idx
    if vsa == VSAModel.HRR:
        assert torch.allclose(actual, expected, atol=1e-6)
    else:
        assert torch.equal(actual, expected)

    ## dim_size should stack an identity vector
    expected_2 = torch.cat([expected, torchhd.identity(1, src.shape[-1], vsa=vsa.value)], dim=0)
    actual_2 = scatter_hd(src, idx, op=red_op, dim_size=4)
    if vsa == VSAModel.HRR:
        assert torch.allclose(actual_2, expected_2, atol=1e-6)
    else:
        assert torch.equal(actual_2, expected_2)


### Picking the cartesian product


def cartesian_bind_tensor(list_tensors):
    """
    Args:
        list_tensors: List of P hypervector tensors, each [Nₚ, D].
    Returns:
        keys:  List of length B=∏Nₚ, each a tuple (i₁,…,iₚ).
        out:   Tensor of shape [B, D], where
               out[b] = torchhd.multibind([list_tensors[p][iₚ₍b₎] for p in 0..P-1]).
    """
    # Number of domains and feature-dim
    P = len(list_tensors)
    D = list_tensors[0].shape[1]

    # 1) build a meshgrid of indices [N₁,…,Nₚ] → P tensors each [N₁…Nₚ]
    grids = torch.meshgrid(*[torch.arange(t.shape[0], device=t.device) for t in list_tensors], indexing="ij")

    # 2) flatten each grid to shape [B]
    idxs = [g.reshape(-1) for g in grids]
    B = idxs[0].numel()

    # 3) gather each domain into [B, D]
    # list of [B,D]
    # 4) stack into [B, P, D] and bind across P → [B, D]
    # shape [B, P, D]
    # → [B, D]
    return torchhd.multibind(torch.stack([list_tensors[p][idxs[p]] for p in range(P)], dim=1))


# zipped = zip(ds.default_cfg.)


def cartesian_bind_tensor_3(sets: list[torch.Tensor]) -> torch.Tensor:
    """
    Fully vectorized: builds an index‐grid via torch.cartesian_prod,
    gathers each set, stacks along new dim=1 [P,D], and multibinds.
    Returns [N_prod, D].
    """
    if not sets:
        raise ValueError("Need at least one set")
    K = len(sets)
    D = sets[0].shape[-1]
    device = sets[0].device

    # 1) get the shapes and build the cartesian product of indices
    shapes = [s.shape[0] for s in sets]
    grids = torch.cartesian_prod(*[torch.arange(n, device=device) for n in shapes])  # → [N_prod, K]

    # 2) for each position k, gather hypervectors
    #    hv_k: [N_prod, D]
    hv_list = [sets[k][grids[:, k]] for k in range(K)]

    # 3) stack them into [N_prod, K, D]
    stacked = torch.stack(hv_list, dim=1)

    # 4) multibind along dim=1 → [N_prod, D]
    return torchhd.multibind(stacked)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("N", [10, 50, 100])
def test_meshgrid_vs_bcast_mesh(benchmark, N):
    D = 256
    A = torchhd.random(N, D, vsa="MAP", device="cpu")
    B = torchhd.random(N, D, vsa="MAP", device="cpu")
    sets = [A, B]

    def run_mesh():
        return cartesian_bind_tensor(sets)

    # warm-up + shape check
    out = run_mesh()
    assert out.shape == (N * N, D)

    # benchmark this one
    t_mesh = benchmark.pedantic(run_mesh, rounds=3, iterations=1)
    logger.info(t_mesh)
    # (optionally assert something about t_mesh)


@pytest.mark.parametrize("N", [10, 50, 100])
def test_meshgrid_vs_bcast_bcast(benchmark, N):
    D = 256
    A = torchhd.random(N, D, vsa="MAP", device="cpu")
    B = torchhd.random(N, D, vsa="MAP", device="cpu")
    sets = [A, B]

    def run_bcast():
        return cartesian_bind_tensor_3(sets)

    # warm-up + shape check
    out = run_bcast()
    assert out.shape == (N * N, D)

    # benchmark this one
    t_bcast = benchmark.pedantic(run_bcast, rounds=3, iterations=1)
    logger.info(t_bcast)
    # (optionally assert something about t_bcast)


@pytest.mark.parametrize("vsa", list(VSAModel))
def test_multibind_single_slot_identity(vsa):
    D = 64
    # generate a [1, D] tensor
    hv = torchhd.random(1, D, vsa=vsa.value, device="cpu")

    # functional API
    out_func = torchhd.multibind(hv)
    # method API
    out_method = hv.multibind()

    # what we expect: the inner vector, squeezed back to [D]
    expected = hv.squeeze(-2)

    assert out_func.shape == expected.shape
    assert out_method.shape == expected.shape
    assert torch.equal(out_func, expected)
    assert torch.equal(out_method, expected)


@pytest.mark.parametrize("vsa", list(VSAModel))
def test_multibundle_single_slot_identity(vsa):
    D = 64
    # generate a [1, D] tensor
    hv = torchhd.random(1, D, vsa=vsa.value, device="cpu")

    # functional API
    out_func = torchhd.multibundle(hv)
    # method API
    out_method = hv.multibundle()

    expected = hv.squeeze(-2)

    assert out_func.shape == expected.shape
    assert out_method.shape == expected.shape
    assert torch.equal(out_func, expected)
    assert torch.equal(out_method, expected)


def test_hrr_multibind_corrected():
    torch.manual_seed(0)

    a = torchhd.random(1, 6, vsa="HRR")
    b = torchhd.random(1, 6, vsa="HRR")

    a = a.squeeze(0)
    b = b.squeeze(0)

    # normal binds
    single_bind = torchhd.bind(a, b)
    # multibind
    s = torch.stack([a, b], dim=0)
    multibind = torchhd.multibind(s)
    assert torch.allclose(single_bind, multibind), f"single_bind:{single_bind}, multibind:{multibind}"
