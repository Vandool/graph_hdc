from typing import Optional, List, Tuple
from torch import nn, Tensor
import normflows as nf
from pydantic.dataclasses import dataclass


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
    num_input_channels: int
    num_flows: int
    num_blocks: int
    num_hidden_channels: int
    num_context_channels: Optional[int] = None
    num_bins: int = 8
    tail_bound: float = 3.0
    activation: nn.Module = nn.ReLU
    dropout_probability: float = 0.0
    permute: bool = False
    init_identity: bool = True
    input_shape: Optional[Tuple[int, ...]] = None


class NeuralSplineNetwork(nn.Module):
    """
    Neural Spline Flow model wrapping `normflows.NormalizingFlow`.

    Accepts inputs of shape
    ``(batch_size, *input_shape)`` → flattens to `(batch_size, num_input_channels)`.

    - `forward(x, context)` returns log-probabilities.
    - `sample(n, context)` returns samples shaped `(n, *input_shape)`.
    """

    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        latent_dim = cfg.num_input_channels
        flows: List[nf.Flow] = []

        for _ in range(cfg.num_flows):
            # Rational-Quadratic Spline coupling (autoregressive)
            spline = nf.flows.AutoregressiveRationalQuadraticSpline(
                features=latent_dim,
                hidden_features=cfg.num_hidden_channels,
                context_features=(cfg.num_context_channels or 0),
                num_bins=cfg.num_bins,
                tail_bound=cfg.tail_bound,
                hidden_layers=cfg.num_blocks,
                dropout_probability=cfg.dropout_probability,
                use_residual=cfg.init_identity,
                activation=cfg.activation(),
            )
            flows.append(spline)
            if cfg.permute:
                flows.append(nf.flows.LULinearPermute(latent_dim))

        base = nf.distributions.DiagGaussian(latent_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """
        Compute log-likelihood of `x` under the flow.

        :param x: Input tensor shaped (batch_size, *input_shape).
        :param context: Optional conditioning tensor.
        :returns: Log-probabilities of shape (batch_size,).
        """
        batch = x.shape[0]
        flat = x.view(batch, -1)
        if context is not None:
            return self.flow.log_prob(flat, context)
        return self.flow.log_prob(flat)

    def sample(self, num_samples: int, context: Optional[Tensor] = None) -> Tensor:
        """
        Draw `num_samples` from the flow and reshape to `input_shape`.

        :param num_samples: Number of samples to generate.
        :param context: Optional conditioning tensor.
        :returns: Samples shaped (num_samples, *input_shape).
        """
        z = self.flow.sample(num_samples, context)
        if self.cfg.input_shape is not None:
            return z.view(num_samples, *self.cfg.input_shape)
        return z


def get_model(cfg: FlowConfig) -> NeuralSplineNetwork:
    """
    Factory for constructing a NormalizingFlowModel from a FlowConfig.

    :param cfg: FlowConfig instance.
    :returns: Initialized NormalizingFlowModel.
    """
    return NeuralSplineNetwork(cfg)
