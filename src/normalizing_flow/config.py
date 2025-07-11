import argparse
from pathlib import Path

import normflows as nf
import torch
from normflows.flows import (
    AutoregressiveRationalQuadraticSpline,
    CircularAutoregressiveRationalQuadraticSpline,
    CircularCoupledRationalQuadraticSpline,
    CoupledRationalQuadraticSpline,
)
from pydantic.dataclasses import dataclass
from torch import nn
from torch.nn import GELU, LeakyReLU, ReLU

from src.encoding.configs_and_constants import SupportedDataset
from src.encoding.the_types import VSAModel

PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/real_nvp_model"

@dataclass
class FlowConfig:
    """
    Configuration for constructing a Neural Spline Flow model.

    :param num_input_channels: Total dimension of the flattened input (e.g., 3 * D).
    :param num_flows: Number of coupling-spline blocks (and optional permutations).
    :param num_blocks: Number of hidden layers in each coupling NN.
    :param num_hidden_channels: Number of hidden units per layer in each coupling NN.
    :param num_context_channels: Dimension of optional conditioning/context vector.
    :param num_bins: Number of bins for the rational-quadratic spline.
    :param tail_bound: Threshold beyond which the spline is linear.
    :param activation: Activation module for coupling NNs.
    :param dropout_probability: Dropout rate in coupling NNs.
    :param permute: Whether to insert LU-permutations between blocks.
    :param init_identity: If True, initialize each spline as (approx.) identity.
    :param input_shape: Original tensor shape for reshaping samples back (e.g., (3, D)).
    """

    ## General Config
    project_dir: Path

    seed: int
    epochs: int
    batch_size: int

    ## HDC Config
    vsa: VSAModel
    hv_dim: int
    dataset: SupportedDataset

    ## Spiral Flow Config
    num_input_channels: int
    num_flows: int
    num_blocks: int
    num_hidden_channels: int
    num_context_channels: int | None = None
    num_bins: int = 8
    tail_bound: int = 3
    flow_type: type[
        CoupledRationalQuadraticSpline
        | CircularCoupledRationalQuadraticSpline
        | AutoregressiveRationalQuadraticSpline
        | CircularAutoregressiveRationalQuadraticSpline
    ] = nf.flows.AutoregressiveRationalQuadraticSpline
    activation: type[ReLU | GELU | LeakyReLU] = nn.ReLU
    dropout_probability: float = 0.0
    permute: bool = False
    init_identity: bool = True
    input_shape: tuple[int, ...] | None = None
    device: str = "cpu"
    lr: float = 1e-3  # learning rate for optimizer
    weight_decay: float = 0.0  # optimizer weight decay


# Helper for --input_shape (comma-separated integers, e.g., "3,512")
def parse_shape(s):
    return tuple(int(x) for x in s.split(","))


def get_activation(name: str) -> type[ReLU | GELU | LeakyReLU]:
    if name.lower() == "relu":
        return nn.ReLU
    if name.lower() == "gelu":
        return nn.GELU
    if name.lower() == "leakyrelu":
        return nn.LeakyReLU
    raise argparse.ArgumentTypeError(f"Unsupported activation: {name}")


def get_flow(
    name: str,
) -> type[CoupledRationalQuadraticSpline | AutoregressiveRationalQuadraticSpline]:
    key = name.lower()
    if key == "coupled":
        return nf.flows.CoupledRationalQuadraticSpline
    if key == "autoregressive":
        return nf.flows.AutoregressiveRationalQuadraticSpline
    raise argparse.ArgumentTypeError(
        f"Unsupported flow type: {name!r}. "
        "Choose from 'coupled', 'circular_coupled', "
        "'autoregressive', or 'circular_autoregressive'."
    )


def get_flow_cli_args() -> FlowConfig:
    parser = argparse.ArgumentParser(description="Neural Spline Flow CLI")

    ## General Config
    parser.add_argument(
        "--project_dir",
        "-pdir",
        type=Path,
        required=False,
        default=Path(PROJECT_DIR),
        help="The base directory, path to all the artefacts of the experiment",
    )

    parser.add_argument(
        "--seed",
        "-seed",
        type=int,
        default=42,
        help="The random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="The number of epochs to train the model (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=64,
        help="The batch size for training (default: 32)",
    )

    ## HDC Config
    parser.add_argument("--vsa", "-v", type=VSAModel, required=True, help="Hypervector Type")
    parser.add_argument("--hv_dim", "-hd", type=int, required=True, help="The dimension of hypervector space")
    parser.add_argument(
        "--dataset", "-ds", type=SupportedDataset, default="ZINC_ND_COMB", help="The dimension of hypervector space"
    )

    ## Spiral Flow Config
    parser.add_argument(
        "--num_input_channels",
        "-ic",
        type=int,
        required=True,
        help="Total dimension of the flattened input (e.g., 3*D)",
    )
    parser.add_argument("--num_flows", "-nf", type=int, default=8, help="Number of coupling-spline blocks")
    parser.add_argument("--num_blocks", "-nb", type=int, default=2, help="Number of hidden layers in coupling NN")
    parser.add_argument("--num_hidden_channels", "-nh", type=int, default=128, help="Number of hidden units per layer")
    parser.add_argument(
        "--num_context_channels", "-nc", type=int, default=None, help="Number of context/conditional channels"
    )
    parser.add_argument(
        "--num_bins", "-bins", type=int, default=8, help="Number of bins for the rational-quadratic spline"
    )
    parser.add_argument("--tail_bound", "-tb", type=int, default=3, help="Threshold beyond which the spline is linear")
    parser.add_argument(
        "--activation", "-a", type=get_activation, default="relu", help="Activation: relu, gelu, leakyrelu"
    )
    parser.add_argument(
        "--flow_type",
        "-ft",
        type=get_flow,
        default="autoregressive",
        help="Flow Type: coupled, autoregressive",
    )
    parser.add_argument(
        "--dropout_probability", "-dp", type=float, default=0.0, help="Dropout probability in coupling NNs"
    )
    parser.add_argument("--permute", "-p", action="store_true", help="Insert LU-permutations between blocks")
    parser.add_argument("--init_identity", "-ii", action="store_true", help="Initialize splines as identity")
    parser.add_argument(
        "--input_shape",
        "-is",
        type=parse_shape,
        default=None,
        help="Original tensor shape as comma-separated, e.g. '3,512'",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", "-wd", type=float, default=0.0, help="Optimizer weight decay")
    parser.add_argument(
        "--device",
        "-dev",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )

    args = parser.parse_args()
    flow_config = FlowConfig(**vars(parser.parse_args()))

    flow_config.activation = args.activation if isinstance(args.activation, type) else get_activation(args.activation)
    flow_config.flow_type = args.flow_type if isinstance(args.flow_type, type) else get_flow(args.flow_type)

    return flow_config
