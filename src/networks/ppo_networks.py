from typing import Any, Callable, Optional, Sequence, List
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal, constant


def _default_gain_for_activation(activation: Optional[Callable]) -> float:
    # Heuristic: orthogonal gain sqrt(2) for ReLU-like; 1.0 for linear/unknown.
    if activation is None:
        return 1.0
    if activation in (nn.relu,):
        return np.sqrt(2)
    # If you use other activations (gelu/silu/tanh), either keep 1.0 or specialize here.
    return 1.0


def apply_layer(
    x,
    layer,
    train: bool,
    activation: Optional[Callable] = None,
    apply_bn: bool = False,
    apply_ln: bool = False,
    apply_brn: bool = False,  # kept for API compatibility (not implemented here)
):
    x = layer(x)

    assert not (apply_bn and apply_ln), "Batch Norm and Layer Norm together not supported"
    if activation is not None:
        if apply_ln:
            x = nn.LayerNorm(reduction_axes=-1)(x)
        if apply_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = activation(x)
    return x


class MLP(nn.Module):
    ffn_dims: List[int]
    activations: List[Optional[Callable]]
    batch_norm: bool = False
    layer_norm: bool = False
    batch_renorm: bool = False
    last_bias: bool = True

    # Base inits (used when not overridden per-layer)
    default_bias_init: Callable = constant(0.0)

    @nn.compact
    def __call__(self, x, train: bool = False):
        assert len(self.ffn_dims) == len(self.activations), "dims/activations length mismatch"
        L = len(self.ffn_dims)

        for i, (layer_dim, activation) in enumerate(zip(self.ffn_dims, self.activations)):
            use_bias = self.last_bias if (i == L - 1) else True

            gain = _default_gain_for_activation(activation)
            dense = nn.Dense(layer_dim, use_bias=use_bias,
                             kernel_init=orthogonal(gain), bias_init=self.default_bias_init)
            x = apply_layer(x, dense, train, activation=activation,
                            apply_bn=self.batch_norm, apply_brn=self.batch_renorm, apply_ln=self.layer_norm)
        return x


class ConvNet(nn.Module):
    conv_configs: list[dict]
    batch_norm: bool = False
    batch_renorm: bool = False
    layer_norm: bool = False

    default_bias_init: Callable = constant(0.0)

    @nn.compact
    def __call__(self, x, train: bool = False):
        for cfg in self.conv_configs:
            activation = cfg.get("activation", None)
            gain = cfg.get("gain", _default_gain_for_activation(activation))

            conv = nn.Conv(features=cfg["features"],
                           kernel_size=cfg["kernel_size"],
                           strides=cfg["strides"],
                           padding=cfg["padding"],
                           kernel_init=cfg.get("kernel_init", orthogonal(gain)),
                           bias_init=cfg.get("bias_init", self.default_bias_init))

            x = apply_layer(x, conv, train,
                            activation=activation,
                            apply_bn=self.batch_norm,
                            apply_brn=self.batch_renorm,
                            apply_ln=self.layer_norm)
        return x


def make_conv_configs() -> list[dict]:
    # EXACT DQN-style Atari conv trunk (VALID/strides/kernels), ReLU after each conv
    return [
        {"features": 32, "kernel_size": (8, 8), "strides": (4, 4), "padding": "VALID", "activation": nn.relu},
        {"features": 64, "kernel_size": (4, 4), "strides": (2, 2), "padding": "VALID", "activation": nn.relu},
        {"features": 64, "kernel_size": (3, 3), "strides": (1, 1), "padding": "VALID", "activation": nn.relu},
    ]


class AtariConvEnc(nn.Module):
    # --- user-configurable knobs ---
    use_batch_norm: bool = False
    use_layer_norm: bool = False

    mlp_dims: List[int] = (512,)
    mlp_activations: List[Optional[Callable]] = (nn.relu,)

    @nn.compact
    def __call__(self, x, train: bool = False):
        assert len(self.mlp_dims) == len(self.mlp_activations), "mlp_dims/mlp_activations length mismatch"
        assert not (self.use_batch_norm and self.use_layer_norm), "BN and LN together not supported"

        x = x / 255.0
        x = ConvNet(conv_configs=make_conv_configs(),
                    batch_norm=self.use_batch_norm, layer_norm=self.use_layer_norm)(x, train=train)
        x = x.reshape((x.shape[0], -1))

        x = MLP(ffn_dims=list(self.mlp_dims), activations=list(self.mlp_activations),
                batch_norm=self.use_batch_norm, layer_norm=self.use_layer_norm)(x, train=train)
        return x


class Critic(nn.Module):

    @nn.compact
    def __call__(self, x, train: bool = False):
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x, train: bool = False):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class QHead(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Standard Q-head scale. (You can keep orthogonal(1.0).)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)