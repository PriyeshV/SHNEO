import jax
from flax import linen as nn
from jax import numpy as jnp
from dataclasses import field
from functools import partial
from typing import Iterable, Callable, Optional, Sequence, List
from src.networks.layers import RMSNorm, BatchRenorm


Array = jnp.ndarray


def apply_layer(x, layer, train: bool, activation: Optional[Callable] = None,
                apply_bn: bool = False, apply_ln: bool = False, apply_brn: bool = False):
    x = layer(x)

    # assert not (apply_bn and apply_ln), "Batch Norm and Layer Norm together not supported"
    assert not (apply_bn and apply_brn), "Batch Norm and Batch ReNorm together not supported"
    if activation is not None:
        if apply_ln:
            x = nn.LayerNorm(reduction_axes=-1)(x)
        if apply_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        if apply_brn:
            x = BatchRenorm(use_running_average=not train)(x)
            # x = BatchRenorm(use_running_average=not train, momentum=0.99, warmup_steps=50_000, max_r=3.0, max_d=5.0)(x)
        x = activation(x)
    return x


class MLP(nn.Module):
    ffn_dims: list[int]
    activations: list[Optional[Callable]]
    batch_norm: bool = False
    layer_norm: bool = False
    batch_renorm: bool = False
    last_bias: bool = True

    @nn.compact
    def __call__(self, x, train: bool = False):
        assert len(self.ffn_dims) == len(self.activations), "dims/activations length mismatch"
        L = len(self.ffn_dims)

        for i, (layer_dim, activation) in enumerate(zip(self.ffn_dims, self.activations)):
            use_bias = self.last_bias if (i == L - 1) else True
            x = apply_layer(x, nn.Dense(layer_dim, use_bias=use_bias), train, activation=activation,
                            apply_bn=self.batch_norm, apply_brn=self.batch_renorm, apply_ln=self.layer_norm)
        return x


class PartialBatchNorm1D(nn.Module):
    exclude_idx: int
    stabilize_excluded: bool = True
    eps: float = 1e-5
    momentum: float = 0.9

    @nn.compact
    def __call__(self, x: Array, train: bool) -> Array:
        K = x.shape[-1]
        e = self.exclude_idx
        assert 0 <= e < K

        x_e = x[..., e:e+1]
        x_r = jnp.concatenate([x[..., :e], x[..., e+1:]], axis=-1)

        # BN on remaining dims
        x_r = nn.BatchNorm(
            use_running_average=not train,
            use_scale=False, use_bias=False,
            momentum=self.momentum, epsilon=self.eps,
        )(x_r)

        # Stabilize excluded dim without coupling (optional)
        if self.stabilize_excluded:
            # mean over all axes except last
            red_axes = tuple(range(x_e.ndim - 1))
            rms = jnp.sqrt(jnp.mean(x_e * x_e, axis=red_axes, keepdims=True) + self.eps)
            x_e = x_e / jax.lax.stop_gradient(rms)

        # Reinsert excluded dim
        if e == 0:
            return jnp.concatenate([x_e, x_r], axis=-1)
        if e == K - 1:
            return jnp.concatenate([x_r, x_e], axis=-1)
        left = x_r[..., :e]
        right = x_r[..., e:]
        return jnp.concatenate([left, x_e, right], axis=-1)


class SVD_Dense(nn.Module):
    layer_dims: list[int]
    activations: list[Optional[Callable]]
    layer_norm: bool = False
    batch_norm: bool = False
    op_batch_norm: bool = False
    laplacian: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        h = x
        if len(self.layer_dims) > 0:
            h = MLP(self.layer_dims, self.activations,
                    layer_norm=self.layer_norm,
                    batch_norm=self.batch_norm,
                    last_bias=False, name=self.name + '/MLP')(h, train)
        if self.op_batch_norm:
            K = h.shape[-1]
            exclude_idx = 0  #if self.laplacian else (K - 1)
            h = PartialBatchNorm1D(exclude_idx=exclude_idx)(h, train=train)
            # h = nn.BatchNorm(
            #     use_running_average=not train,
            #     use_scale=True,  # IMPORTANT: no gamma
            #     use_bias=False,  # IMPORTANT: no beta
            #     momentum=0.99,
            #     epsilon=1e-5,
            # )(h)
        return h


class ConvNet(nn.Module):
    conv_configs: list[dict]
    batch_norm: bool = False
    batch_renorm: bool = False
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Iterate over configurations and apply each convolution.
        for i, cfg in enumerate(self.conv_configs):
            conv_layer = nn.Conv(features=cfg["features"], kernel_size=cfg["kernel_size"],
                                 strides=cfg["strides"], padding=cfg["padding"])
            x = apply_layer(x, conv_layer, train, activation=cfg.get("activation", None),
                            apply_bn=self.batch_norm,
                            apply_brn=self.batch_renorm, apply_ln=self.layer_norm)
        return x


def make_conv_configs(H: int, W: int, ref_dim: int = 84,
                      ref_kernels: Sequence[int] = (8, 4, 3),
                      ref_strides: Sequence[int] = (4, 2, 1),
                      out_channels: Sequence[int] = (32, 64, 64),
                      width_mul: int = 1,
                      min_scale: float = 0.5
                      ) -> list[dict]:
    eff_ref = ref_dim if max(H, W) >= 42 else 42
    scale = max(min(H, W) / eff_ref, min_scale)
    ks = [max(1, int(round(k * scale))) for k in ref_kernels]
    ss = [max(1, int(round(s * scale))) for s in ref_strides]
    pads = ["VALID", "VALID", "SAME" if min(H, W) < eff_ref else "VALID"]
    return [{"features": int(out_channels[i] * width_mul),
             "kernel_size": (ks[i], ks[i]),
             "strides": (ss[i], ss[i]),
             "padding": pads[i],
             "activation": nn.relu} for i in range(3)]


class ImpalaConvBlock(nn.Module):
    batch_norm: bool = False
    layer_norm: bool = False

    out_channels: Sequence[int] = (16, 32, 32)
    kernel_init: Callable = nn.initializers.xavier_uniform()
    width_mul: int = 1

    class ResidualBlock(nn.Module):
        channel: int
        name: str
        batch_norm: bool
        layer_norm: bool
        kernel_init: Callable

        @nn.compact
        def __call__(self, x, train):
            h = x
            h = nn.relu(h)
            h = apply_layer(h,
                            nn.Conv(self.channel, kernel_size=(3, 3), strides=1, padding='SAME',
                                    kernel_init=self.kernel_init, name=self.name + '_0'),
                            train, activation=nn.relu, apply_bn=self.batch_norm, apply_ln=self.layer_norm)
            h = apply_layer(h,
                            nn.Conv(self.channel, kernel_size=(3, 3), strides=1, padding='SAME',
                                    kernel_init=self.kernel_init, name=self.name + '_1'),
                            train, activation=None, apply_bn=self.batch_norm, apply_ln=self.layer_norm)
            # no ReLU here in the original implementation
            return h + x

    @nn.compact
    def __call__(self, x, train: bool = False):
        for layer_id, channel in enumerate(self.out_channels):
            x = nn.Conv(features=int(channel * self.width_mul), kernel_size=(3, 3), strides=1, padding='SAME',
                        name='conv2d_' + str(layer_id), kernel_init=self.kernel_init)(x)
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
            x = self.ResidualBlock(name='res_block_conv_' + str(layer_id) + '_0',
                                   channel=int(channel * self.width_mul), kernel_init=self.kernel_init,
                                   batch_norm=self.batch_norm,
                                   layer_norm=self.layer_norm,
                                   )(x, train)
            x = self.ResidualBlock(name='res_block_conv_' + str(layer_id) + '_1',
                                   channel=int(channel * self.width_mul), kernel_init=self.kernel_init,
                                   batch_norm=self.batch_norm, layer_norm=self.layer_norm)(x, train)
        return x


def add_frame_diff(obs: jnp.ndarray) -> jnp.ndarray:
    """
    obs: [B, H, W, 4] -- 4 recent grayscale frames
    returns [B, H, W, 5] where channel 4 = frame4 - frame3
    """
    # assume frames are floats in [0,1] or [0,255], either way diff works
    last = obs[..., -1]
    prev = obs[..., -2]
    diff = last - prev  # [B, H, W]
    diff = diff[..., None]  # [B, H, W, 1]
    return jnp.concatenate([obs, diff], axis=-1)


class Conv_Encoder(nn.Module):
    layer_dims: list[int]
    activations: list[Optional[Callable]]
    img_norm: bool = True
    batch_norm: bool = False
    layer_norm: bool = False
    rms_norm: bool = False
    impala_cnn: bool = False
    add_frame_diff: bool = False
    min_scale: float = 1.0
    width_mul: int = 1

    @nn.compact
    def __call__(self, x, train: bool = False):
        # x is assumed to have shape (batch, H, W, C)
        B, H, W, _ = x.shape
        conv_configs = make_conv_configs(H, W, min_scale=self.min_scale, width_mul=self.width_mul)

        if self.img_norm:
            x = x / 255.0

        if self.add_frame_diff:
            x = add_frame_diff(x)

        if not self.impala_cnn:
            conv_net = ConvNet(name='CNN', conv_configs=conv_configs,
                               batch_norm=self.batch_norm,
                               layer_norm=self.layer_norm,
                               )
        else:
            conv_net = ImpalaConvBlock(name='ImpalaCNN', out_channels=(16, 32, 32),
                                       batch_norm=self.batch_norm,
                                       layer_norm=self.layer_norm,
                                       width_mul=self.width_mul)
        x = conv_net(x, train=train)
        x = x.reshape((x.shape[0], -1))

        # Optionally apply an MLP for further processing.
        h = x
        if len(self.layer_dims) > 0:
            h = MLP(self.layer_dims, self.activations,
                    layer_norm=self.layer_norm,
                    batch_norm=self.batch_norm,
                    last_bias=False, name=self.name + '/MLP')(h, train)

        if self.rms_norm:
            h = RMSNorm(num_features=self.layer_dims[-1], use_scale=True, use_bias=True)(h,
                                                                                         use_running_average=not train)

        return h


# class SA_LapNetwork(nn.Module):
#     n_actions: int
#     layer_dims: List[int]
#     activations: List[Optional[Callable]]
#     enc_class: Callable[..., nn.Module]
#     batch_norm: bool = False
#     layer_norm: bool = False
#     rms_norm: bool = False
#     get_state_emb: bool = False
#
#     one_hot: bool = True
#     act_emb_dims: int = 18
#     interaction: str = 'film'
#
#     @nn.compact
#     def __call__(self, obs, train: bool = False):
#         B, A = obs.shape[0], self.n_actions
#
#         # Action embeddings
#         if self.one_hot:
#             action_embed = jnp.eye(self.n_actions)                       # (A, A)
#         else:
#             action_embed = self.param(self.name + "/A_embed",
#                                       nn.initializers.normal(stddev=0.1),
#                                       (self.n_actions, self.act_emb_dims))  # (A, D_a)
#
#         # State encoder
#         phi_s = self.enc_class()(obs, train=train)  # (B, D)
#         D = phi_s.shape[-1]
#
#         # -------- Path 1: PROJECT (per-action linear map φ_s -> ℝ^O) --------
#         if self.interaction.lower() == "project":
#             out_dim = self.layer_dims[-1]
#             W = self.param(self.name + "/project/W",
#                            nn.initializers.he_normal(),
#                            (self.n_actions, D, out_dim))                  # (A, D, O)
#             b = self.param(self.name + "/project/b",
#                            nn.initializers.zeros,
#                            (self.n_actions, out_dim))
#
#             # φ_sa[b, a, o] = Σ_d φ_s[b, d] * W[a, d, o] + b[a, o]
#             # (einsum version)
#             phi_sa = jnp.einsum("bd,ado->bao", phi_s, W) + b[None, :, :]            # (B, A, O)
#
#             if self.rms_norm:
#                 phi_sa = RMSNorm(num_features=out_dim,
#                                  use_scale=True, use_bias=True)(phi_sa,
#                                                                use_running_average=not train)
#
#             if self.get_state_emb:
#                 return phi_s, phi_sa
#             return phi_sa
#
#         # -------- Path 2: FiLM / CONCAT --------
#         # Broadcast state and action embedding to (B, A, *)
#         phi_s_exp = jnp.broadcast_to(phi_s[:, None, :], (B, A, D))          # (B, A, D)
#         e_a = jnp.broadcast_to(
#             action_embed[None, :, :],
#             (B, A, action_embed.shape[-1])
#         )                                                                    # (B, A, A or D_a)
#
#         if self.interaction.lower() == "film":
#             # Map action embedding -> (B, A, D) then FiLM
#             gamma = nn.Dense(features=D, bias_init=nn.initializers.ones,
#                              name=self.name + '/FiLM/gamma')(e_a)            # (B, A, D)
#             beta = nn.Dense(features=D, bias_init=nn.initializers.zeros,
#                             name=self.name + '/FiLM/beta')(e_a)              # (B, A, D)
#             sa_input = gamma * phi_s_exp + beta                              # (B, A, D)
#         else:
#             # CONCAT then a small MLP to (B, A, D)
#             combined_feat = jnp.concatenate([phi_s_exp, e_a], axis=-1)       # (B, A, D + D_a)
#             sa_input = MLP(ffn_dims=[D], activations=[nn.relu],
#                            layer_norm=self.layer_norm,
#                            name=self.name + '/Concat/')(combined_feat)
#
#         # Shared MLP over the last dim (applied per (b,a))
#         mlp = MLP(ffn_dims=self.layer_dims,
#                   activations=self.activations,
#                   layer_norm=self.layer_norm,
#                   name=self.name + "/SA/MLP")
#         phi_sa = mlp(sa_input, train=train)                                   # (B, A, final_dim)
#
#         if self.rms_norm:
#             phi_sa = RMSNorm(num_features=self.layer_dims[-1],
#                              use_scale=True, use_bias=True)(
#                 phi_sa, use_running_average=not train
#             )
#
#         if self.get_state_emb:
#             return phi_s, phi_sa
#         return phi_sa



class SA_LapNetwork(nn.Module):
    """
    Shared trunk d0 + two heads:
      state_rep  = f_state(d0)
      action_rep = f_action(d0, a)

    combine="concat": phi_sa = [ state_rep(K_V) || action_rep(K_A) ] with K_V + K_A = K
    combine="sum":    phi_sa = state_rep(K) + (center_advantage ? A_c(K) : A(K))

    If get_state_emb=True:
      - "concat": returns (phi_s ∈ R^{K_V}, phi_sa ∈ R^{K})
      - "sum":    returns (phi_s ∈ R^{K},   phi_sa ∈ R^{K})
    """
    # Required
    n_actions: int
    layer_dims: List[int]                        # last element is K (total output dims)
    activations: List[Optional[Callable]]
    enc_class: Callable[..., nn.Module]

    # Combination mode
    combine: str = "concat"                      # "concat" or "sum"
    center_advantage: bool = False               # zero-mean A(s,·) before combine (sum/concat)

    # Normalization / behavior
    batch_norm: bool = False                     # (trunk BN handled in enc_class if needed)
    layer_norm: bool = False
    rms_norm: bool = False
    get_state_emb: bool = False

    # Action interaction for action head
    one_hot: bool = True
    act_emb_dims: int = 18
    interaction: str = 'project'                 # 'project' | 'film' | 'concat'

    # Branch capacities
    k_state: Optional[int] = None                # K_V for concat; ignored for sum
    state_hidden_dims: Optional[Sequence[int]] = None
    action_hidden_dims: Optional[Sequence[int]] = None

    # Sum-mode extra knobs
    action_gain: float = 1.0                     # scale A before sum (improves early SNR in sum mode)

    @staticmethod
    def _as_list(x: Optional[Sequence[int]]) -> List[int]:
        return [] if x is None else list(x)

    def _make_action_embed(self):
        if self.one_hot:
            return jnp.eye(self.n_actions)       # (A, A)
        else:
            return self.param(
                (self.name or "SA") + "/A_embed",
                nn.initializers.normal(stddev=0.1),
                (self.n_actions, self.act_emb_dims)
            )                                    # (A, D_a)

    @nn.compact
    def __call__(self, obs, train: bool = False):
        B, A = obs.shape[0], self.n_actions
        K = self.layer_dims[-1]
        final_act = self.activations[-1]  # usually None

        # -------- Shared trunk: d0 --------
        d0 = self.enc_class()(obs, train=train)                 # (B, D0)
        D0 = d0.shape[-1]

        # -------- Action embeddings --------
        action_embed = self._make_action_embed()

        # -------- Build STATE head --------
        if self.combine.lower() == "concat":
            K_V = self.k_state if (self.k_state is not None) else max(1, K // 4)
            K_A = K - K_V
            assert K_A > 0, f"k_state={K_V} must be < K={K}"

            st_hidden = self._as_list(self.state_hidden_dims)
            st_dims = st_hidden + [K_V]
            st_acts = [nn.relu] * len(st_hidden) + [final_act]

            phi_s = MLP(
                ffn_dims=st_dims,
                activations=st_acts,
                layer_norm=self.layer_norm,
                batch_norm=self.batch_norm,
                name=(self.name or "SA") + "/StateMLP",
            )(d0, train=train)                                     # (B, K_V)

            if self.rms_norm:
                phi_s = RMSNorm(num_features=K_V, use_scale=True, use_bias=True)(
                    phi_s, use_running_average=not train
                )
        else:
            # sum mode → state head outputs K
            K_V = K
            K_A = K  # for action head
            st_hidden = self._as_list(self.state_hidden_dims)
            st_dims = st_hidden + [K]
            st_acts = [nn.relu] * len(st_hidden) + [final_act]
            phi_s = MLP(
                ffn_dims=st_dims,
                activations=st_acts,
                layer_norm=self.layer_norm,
                batch_norm=self.batch_norm,
                name=(self.name or "SA") + "/StateMLP",
            )(d0, train=train)                                     # (B, K)

            # if self.rms_norm:
            #     phi_s = RMSNorm(num_features=K, use_scale=True, use_bias=True)(
            #         phi_s, use_running_average=not train
            #     )

        # -------- Build ACTION head (to K_A for concat, K for sum) --------
        inter = self.interaction.lower()
        target_A_dim = K_A

        if inter == "project":
            # Per-action linear map to target_A_dim
            W = self.param(
                (self.name or "SA") + "/project/W",
                nn.initializers.orthogonal(),
                (A, D0, target_A_dim),
            )
            b = self.param(
                (self.name or "SA") + "/project/b",
                nn.initializers.zeros,
                (A, target_A_dim),
            )
            action_rep = jnp.einsum("bd,adk->bak", d0, W) + b[None, :, :]     # (B,A,target_A_dim)

            act_hidden = self._as_list(self.action_hidden_dims)
            if len(act_hidden) > 0:
                post = MLP(
                    ffn_dims=act_hidden + [target_A_dim],
                    activations=[nn.relu] * len(act_hidden) + [final_act],
                    layer_norm=self.layer_norm,
                    batch_norm=self.batch_norm,
                    name=(self.name or "SA") + "/ActionPostMLP",
                )
                action_rep = post(action_rep, train=train)                    # (B,A,target_A_dim)

        else:
            # Build per-(b,a) inputs
            d0_ba = jnp.broadcast_to(d0[:, None, :], (B, A, D0))
            if inter == "film":
                e_a = jnp.broadcast_to(action_embed[None, :, :],
                                        (B, A, action_embed.shape[-1]))
                gamma = nn.Dense(
                    features=D0, kernel_init=nn.initializers.xavier_uniform(),
                    bias_init=nn.initializers.ones,
                    name=(self.name or "SA") + "/FiLM/gamma",
                )(e_a)
                beta  = nn.Dense(
                    features=D0, kernel_init=nn.initializers.xavier_uniform(),
                    bias_init=nn.initializers.zeros,
                    name=(self.name or "SA") + "/FiLM/beta",
                )(e_a)
                sa_in = gamma * d0_ba + beta                                  # (B,A,D0)
            elif inter == "concat":
                e_a = jnp.broadcast_to(action_embed[None, :, :],
                                        (B, A, action_embed.shape[-1]))
                sa_in = jnp.concatenate([d0_ba, e_a], axis=-1)                # (B,A,D0+De)
            else:
                raise NotImplementedError(f"interaction={self.interaction}")

            act_hidden = self._as_list(self.action_hidden_dims)
            act_dims = act_hidden + [target_A_dim]
            act_acts = [nn.relu] * len(act_hidden) + [final_act]
            action_rep = MLP(
                ffn_dims=act_dims,
                activations=act_acts,
                layer_norm=self.layer_norm,
                batch_norm=self.batch_norm,
                name=(self.name or "SA") + "/ActionMLP",
            )(sa_in, train=train)                                             # (B,A,target_A_dim)

        if self.center_advantage:
            action_rep = action_rep - action_rep.mean(axis=1, keepdims=True)  # advantage block

        # if self.rms_norm:
        #     action_rep = RMSNorm(num_features=target_A_dim, use_scale=True, use_bias=True)(
        #         action_rep, use_running_average=not train)

        # -------- Combine --------
        if self.combine.lower() == "concat":
            phi_s_b = jnp.broadcast_to(phi_s[:, None, :], (B, A, K_V))        # (B,A,K_V)
            phi_sa = jnp.concatenate([phi_s_b, action_rep], axis=-1)          # (B,A,K_V+K_A=K)
        elif self.combine.lower() == "sum":
            phi_s_b = jnp.broadcast_to(phi_s[:, None, :], (B, A, K))          # (B,A,K)
            phi_sa = phi_s_b + self.action_gain * action_rep                  # (B,A,K)
        else:
            raise ValueError(f"Unknown combine='{self.combine}', use 'concat' or 'sum'.")

        if self.rms_norm:
            phi_sa = RMSNorm(num_features=K, use_scale=True, use_bias=True)(phi_sa, use_running_average=not train)


        # -------- Outputs --------
        if self.get_state_emb:
            return phi_s, phi_sa
        return phi_sa


class SwiGLU(nn.Module):
    dim: int  # hidden dimension of the FFN

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1) two parallel projections
        x_a = nn.Dense(self.dim, name="proj_a")(x)  # [B, dim]
        x_b = nn.Dense(self.dim, name="proj_b")(x)  # [B, dim]
        # 2) swish gating
        return x_a * nn.swish(x_b)


def swiglu(x: jnp.ndarray) -> jnp.ndarray:
    """Splitx last dim in half, then a * swish(b)."""
    a, b = jnp.split(x, 2, axis=-1)
    return a * nn.silu(b)


class QAdapter(nn.Module):
    d: int
    @nn.compact
    def __call__(self, x):
        # identity init (rotation-friendly)
        def eye_init(key, shape, dtype=jnp.float32):
            return jnp.eye(shape[0], dtype=dtype)
        W = self.param('W', eye_init, (self.d, self.d))
        return x @ W


class FFNetwork(nn.Module):
    layer_dims: list[int]
    activations: list[Optional[Callable]]
    stop_grad: bool = False
    batch_norm: bool = False
    layer_norm: bool = False
    ip_batch_norm: bool = False
    ip_layer_norm: bool = False
    ip_l2_norm: bool = False
    n_actions: Optional[int] = None

    use_second_embedding: bool = False
    use_second_bn: bool = False
    use_eigen_transform: bool = False
    eigen_transform_config: dict = field(default_factory=dict)
    comb_type: str = 'concat'

    class SwiGLUBlock(nn.Module):
        dim: int  # output dimension

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            x2 = nn.Dense(self.dim * 2, name="swiglu_proj")(x)
            return swiglu(x2)

    @nn.compact
    def __call__(self, embeddings, embeddings_2=None, eigen_values=None, train: bool = False):
        h, emb_2 = embeddings, embeddings_2 if self.use_second_embedding else None
        if self.stop_grad:
            h = jax.lax.stop_gradient(h)

        # gamma = self.param("q_affine/gamma", nn.initializers.ones, (embeddings.shape[-1],))
        # beta = self.param("q_affine/beta", nn.initializers.zeros, (embeddings.shape[-1],))
        # h = h * gamma + beta

        # Optionally apply the eigenvalue transformation and scale the embeddings.
        if self.use_eigen_transform:
            eigen_scaling = EigenTransformTriple(gamma=self.eigen_transform_config.gamma,
                                                 inv_power=self.eigen_transform_config.inv_pow,
                                                 poly_degree=self.eigen_transform_config.poly_deg,
                                                 share_poly_weights=not self.eigen_transform_config.non_param,
                                                 name=self.name + '/EigVal')(eigen_values, train)
            eigen_scaling = nn.LayerNorm()(eigen_scaling)
            h = jnp.multiply(eigen_scaling, h)

        if self.ip_l2_norm:
            h = h / (jnp.linalg.norm(h, ord=2, axis=-1, keepdims=True) + 1e-6)

        if self.ip_layer_norm:
            h = nn.LayerNorm(name=self.name + '/iLN_Lap')(h)
            # if self.use_second_embedding:
            #     emb_2 = nn.LayerNorm(name=self.name+'/iLN_Q')(emb_2)

        if self.ip_batch_norm:
            # h = nn.BatchNorm(name=self.name + '/iBN_Lap', momentum=0.99, use_running_average=not train)(h)
            # norm_axis = (-2, -1) if h.ndim == 3 else -1
            norm_axis = -1
            h = BatchRenorm(name=self.name + '/iBReN_Lap', axis=norm_axis, use_running_average=not train)(h)
            # if self.use_second_embedding:
            #     emb_2 = BatchRenorm(name=self.name + '/iBReN_Q', use_running_average=not train)(h)

        if self.use_second_embedding:
            emb_1 = h
            # Expects emb_2 will be layer_normed and will have passed through ReLU activation

            if emb_1.ndim > emb_2.ndim:  # if lap emb is state-action and pqn is state-only
                assert emb_2.ndim == 2
                emb_2 = jnp.repeat(emb_2[:, None, :], emb_1.shape[1], axis=1)

            x = jnp.concatenate([emb_1, emb_2], axis=-1)
            if self.comb_type.lower() == 'concat':
                # h = x
                h = emb_2
            elif self.comb_type.lower() == 'film':
                x = nn.LayerNorm()(x)
                pqn = emb_2
                gamma = nn.Dense(pqn.shape[-1],
                                 kernel_init=nn.initializers.zeros,
                                 bias_init=nn.initializers.zeros,
                                 name=self.name + '/film_gamma')(x)
                beta = nn.Dense(pqn.shape[-1],
                                kernel_init=nn.initializers.zeros,
                                bias_init=nn.initializers.zeros,
                                name=self.name + '/film_beta')(x)
                h = (1.0 + gamma) * pqn + beta  # starts as identity on PQN
                h = nn.LayerNorm(name=self.name + '/post_film_ln')(h)
            else:  # Adaptive Summation
                x = nn.LayerNorm()(x)
                if self.comb_type == 's_gate':  # scalar gating
                    gate = self.SwiGLUBlock(dim=2, name='Gate')(x)
                    h = gate[..., 0:1] * emb_1 + gate[..., 1:] * emb_2
                else:  # vector gating
                    g1 = self.SwiGLUBlock(dim=emb_1.shape[-1], name="gate1_swiglu")(x)  # [B, D]
                    g2 = self.SwiGLUBlock(dim=emb_2.shape[-1], name="gate2_swiglu")(x)  # [B, D]
                    h = g1 * emb_1 + g2 * emb_2
            # if self.layer_norm and len(self.layer_dims) > 1:
            #     h = nn.LayerNorm(name=self.name + '/iLN')(h)

        # h = QAdapter(d=embeddings.shape[-1])(h)

        h = MLP(self.layer_dims, self.activations,
                # batch_norm=self.batch_norm,
                batch_renorm=self.batch_norm,
                layer_norm=self.layer_norm, name=self.name + '/QMLP')(h, train)
        if self.layer_dims[-1] == 1:
            h = jnp.squeeze(h, axis=-1)
        return h


class AttentiveFusion(nn.Module):
    op_dims: int
    n_heads: int = 8

    @nn.compact
    def __call__(self, emb_1, emb_2, train: bool = False):
        D = emb_1.shape[-1]
        seq = jnp.stack([emb_1, emb_2], axis=1)  # [B, 2, D]
        assert self.op_dims % self.n_heads == 0
        mha_1 = nn.MultiHeadDotProductAttention(num_heads=self.n_heads, qkv_features=max(64, D // self.n_heads),
                                                out_features=self.op_dims, name=self.name + '/Fusion/MHA_1')
        query = emb_2[:, None, :]
        out = mha_1(inputs_q=query, inputs_k=seq, inputs_v=seq, deterministic=not train)[:, 0, :]
        return out


class EigenTransformTriple(nn.Module):
    inv_power: int = 1  #
    gamma: float = 0.99
    poly_degree: int = 1  # Degree for the polynomial branch (>=1).
    share_poly_weights: Optional[bool] = None  # If True, share weights across eigenvalues.

    @nn.compact
    def __call__(self, eigenvalues, train):
        branches = []
        d = eigenvalues.shape[-1]

        # Fixed transformation with inverse: (eigenvalues^power) and then inverse sqrt.
        if self.inv_power != 0:
            fixed_inv = jnp.power(eigenvalues, self.inv_power)
            fixed_inv = jnp.nan_to_num(1 / jnp.sqrt(fixed_inv), nan=0, posinf=0, neginf=0)
            branches.append(fixed_inv)

        # Polynomial parameterization.
        if self.poly_degree > 0:
            if self.poly_degree == 1:
                shape = (1, 1) if self.share_poly_weights else (1, d)
                poly_transform = eigenvalues * self.param("poly_coeff", nn.initializers.ones, shape)
            else:
                # poly_terms = jnp.stack([(self.gamma**i) * jnp.power(eigenvalues, i) for i in range(1, self.poly_degree+1)], axis=-1)
                poly_terms = jnp.stack([jnp.power(eigenvalues, i) for i in range(1, self.poly_degree + 1)], axis=-1)
                if self.share_poly_weights is None:
                    poly_transform = eigenvalues + jnp.sum(poly_terms, axis=-1)
                else:
                    shape = (1, 1, poly_terms.shape[-1]) if self.share_poly_weights else (1, poly_terms.shape[-2],
                                                                                          poly_terms.shape[-1])
                    poly_coeff = self.param("poly_coeff", nn.initializers.normal(stddev=0.1), shape)
                    poly_transform = eigenvalues + jnp.sum(poly_coeff[0] * poly_terms, axis=-1)
            branches.append(poly_transform)

        if not branches:
            raise ValueError("At least one branch must be enabled in EigenTransformTriple.")

        # If only one branch is active, simply return its result.
        if len(branches) == 1:
            return branches[0]

        # Otherwise, blend the branches with learned gating weights.
        num_branches = len(branches)
        gate_params = self.param("gates", nn.initializers.ones, (1, num_branches))
        gates = nn.softmax(gate_params, axis=-1)  # Normalize to sum to 1.

        # Blend the branches using the gates.
        bias = self.param("bias", nn.initializers.zeros, (1, d))
        combined = sum(g * branch for g, branch in zip(jnp.split(gates, num_branches, axis=-1), branches))
        return combined + bias


class ScannedRNN(nn.Module):

    @partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        hidden_size = rnn_state[0].shape[-1]

        init_rnn_state = self.initialize_carry(hidden_size, *resets.shape)
        rnn_state = jax.tree_util.tree_map(lambda init, old: jnp.where(resets[:, jnp.newaxis], init, old),
                                           init_rnn_state, rnn_state)
        new_rnn_state, y = nn.OptimizedLSTMCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return (nn.OptimizedLSTMCell(hidden_size, parent=None)
                .initialize_carry(jax.random.PRNGKey(0), (*batch_size, hidden_size)))


class RNNEncoder(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    num_rnn_layers: int = 1
    add_last_action: bool = True

    layer_norm: bool = False
    ip_batch_norm: bool = True
    layer_dims: list[int] = field(default_factory=list)
    activations: list[Optional[Callable]] = field(default_factory=list)

    @nn.compact
    def __call__(self, hidden, x, done, last_action, train: bool = False):

        if self.ip_batch_norm:
            x = BatchRenorm(use_running_average=not train)(x)
        x = MLP([self.hidden_size] * self.num_layers, [nn.relu] * self.num_layers,
                batch_norm=False, layer_norm=self.layer_norm,
                name=self.name + '/MLP_Pre-RNN')(x)

        # add last action to the input of the rnn
        if self.add_last_action:
            last_action = jax.nn.one_hot(last_action, self.action_dim)
            x = jnp.concatenate([x, last_action], axis=-1)

        new_hidden = []
        for i in range(self.num_rnn_layers):
            rnn_in = (x, done)
            hidden_aux, x = ScannedRNN()(hidden[i], rnn_in)
            new_hidden.append(hidden_aux)

        if len(self.layer_dims) > 0:
            x = MLP(self.layer_dims, self.activations, batch_norm=False, layer_norm=self.layer_norm,
                    name=self.name + '/MLP_Post-RNN')(x, train)
            # x = jnp.concatenate([x, last_action], axis=-1)

        return new_hidden, x

    def initialize_carry(self, *batch_size):
        return [ScannedRNN.initialize_carry(self.hidden_size, *batch_size) for _ in range(self.num_rnn_layers)]


class RNN_SANetwork(nn.Module):
    n_actions: int
    enc_class: Callable[..., nn.Module]
    batch_norm: bool = False
    layer_norm: bool = False
    rms_norm: bool = False

    one_hot: bool = True
    act_emb_dims: int = 16
    interaction: str = 'film'

    layer_dims: list[int] = field(default_factory=list)
    activations: list[Optional[Callable]] = field(default_factory=list)

    @nn.compact
    def __call__(self, hidden, x, done, last_action, train: bool = False):
        # Encode the state
        T, B, A = x.shape[0], x.shape[1], self.n_actions
        if self.one_hot:
            action_embed = jnp.eye(self.n_actions)  # (A, A)
        else:
            action_embed = self.param(self.name + "/A_embed", nn.initializers.normal(stddev=0.1),
                                      (self.n_actions, self.act_emb_dims))  # (A, D)
            # Augment with additional projection for larger action embeddings as with continuous action space

        new_hs, phi_s = self.enc_class()(hidden, x, done, last_action, train=train)
        D = phi_s.shape[-1]

        # Broadcast state and action embedding to shape (B, A, D)
        phi_s_exp = jnp.broadcast_to(phi_s[:, :, None, :], (T, B, A, D))  # (B, A, D)
        e_a = action_embed[None, None, :, :]
        e_a = jnp.broadcast_to(e_a, (T, B, A, A if self.one_hot else self.act_emb_dims))  # (B, A, D)

        if self.interaction.lower() == "film":
            # With one-hot encoding this translates to a gating with action-based look-up table
            gamma = MLP(ffn_dims=[D // 2, D], activations=[nn.relu, None],
                        layer_norm=self.layer_norm, name=self.name + '/FiLM/gamma')(e_a)
            beta = MLP(ffn_dims=[D // 2, D], activations=[nn.relu, None],
                       layer_norm=self.layer_norm, name=self.name + '/FiLM/beta')(e_a)
            sa_input = gamma * phi_s_exp + beta
        else:
            combined_feat = jnp.concatenate([phi_s_exp, e_a], axis=-1)
            sa_input = MLP(ffn_dims=[D], activations=[nn.relu], layer_norm=self.layer_norm,
                           name=self.name + '/Concat/')(combined_feat)

        phi_sa = sa_input
        if len(self.layer_dims) > 0:
            mlp = MLP(ffn_dims=self.layer_dims, activations=self.activations,
                      layer_norm=self.layer_norm, name=self.name + "/SA/MLP")
            phi_sa = mlp(sa_input, train=train)  # (B, A, final_dim)

        if self.rms_norm:
            phi_sa = (RMSNorm(num_features=self.layer_dims[-1], use_scale=True, use_bias=True)
                      (phi_sa, use_running_average=not train))

        return new_hs, phi_sa


class Craft_FFNetwork(nn.Module):
    layer_dims: list[int]
    activations: list[Optional[Callable]]
    stop_grad: bool = False
    batch_norm: bool = False
    layer_norm: bool = False
    ip_batch_norm: bool = False
    ip_layer_norm: bool = False
    ip_l2_norm: bool = False
    n_actions: Optional[int] = None

    use_second_embedding: bool = False
    use_second_bn: bool = False
    use_last_action: bool = False
    use_eigen_transform: bool = False
    eigen_transform_config: dict = field(default_factory=dict)
    comb_type: str = 'concat'

    class SwiGLUBlock(nn.Module):
        dim: int  # output dimension

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            x2 = nn.Dense(self.dim * 2, name="swiglu_proj")(x)
            return swiglu(x2)

    @nn.compact
    def __call__(self, embeddings, embeddings_2=None, eigen_values=None, last_action=None, train: bool = False):
        x, emb_2 = embeddings, embeddings_2 if self.use_second_embedding else None
        if self.stop_grad:
            x = jax.lax.stop_gradient(x)

        # Optionally apply the eigenvalue transformation and scale the embeddings.
        if self.use_eigen_transform:
            eigen_scaling = EigenTransformTriple(gamma=self.eigen_transform_config.gamma,
                                                 inv_power=self.eigen_transform_config.inv_pow,
                                                 poly_degree=self.eigen_transform_config.poly_deg,
                                                 share_poly_weights=not self.eigen_transform_config.non_param,
                                                 name=self.name + '/EigVal')(eigen_values, train)
            eigen_scaling = nn.LayerNorm()(eigen_scaling)
            x = jnp.multiply(eigen_scaling, x)

        if self.ip_l2_norm:
            x = x / (jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-6)

        if self.ip_batch_norm:
            x = nn.BatchNorm(name=self.name + '/iBN_Lap', momentum=0.99, use_running_average=not train)(x)

        if self.ip_layer_norm and not len(self.layer_dims) > 1:
            x = nn.LayerNorm(name=self.name + '/iLN_Lap')(x)

        h = x
        if len(self.layer_dims) > 1:
            h = MLP(self.layer_dims[:-1], self.activations[:-1], self.batch_norm, self.layer_norm,
                    name=self.name + '/LapQ_MLP')(x, train)
            h = h + x

        if self.use_second_embedding:
            emb_1 = h
            emb_1, emb_2 = nn.LayerNorm(name=self.name + 'iLN-Emb_1')(emb_1), nn.LayerNorm(
                name=self.name + 'iLN-Emb_2')(emb_2)
            x = jnp.concatenate([emb_1, emb_2], axis=-1)
            h = x
            h = nn.LayerNorm(name=self.name + '/Joint')(h)
            # if self.ip_layer_norm:
            #     h = nn.LayerNorm(name=self.name+'/iLN')(h)

        h = MLP(self.layer_dims[-1:], self.activations[-1:], name=self.name + '/QMLP')(h, train)
        if self.layer_dims[-1] == 1:
            h = jnp.squeeze(h, axis=-1)
        return h