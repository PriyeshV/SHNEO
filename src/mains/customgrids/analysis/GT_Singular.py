import time
import math
import random
import numpy as np
from functools import partial
from scipy.optimize import linear_sum_assignment
from typing import Any, Union, Optional, List, Tuple

import jax
import optax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict
from flax.training.train_state import TrainState

# --- your project imports ---
from src.config.eigsr.customgrids.laplacian_gt import *

from src.utils.utils import *
from src.losses.singular_2 import *
from src.utils.jax_utils import *
from src.utils.grid_agent import GridAgent
from src.networks.networks import Conv_Encoder, SVD_Dense


class SVD_State(TrainState):
    running_sigma_diag: Any
    batch_stats: Union[FrozenDict, None]


class SVD_Encoder(nn.Module):
    Head_network: nn.Module
    U_network: nn.Module
    V_network: nn.Module

    def encode_head(self, obs, train: bool):
        return self.Head_network(obs, train=train)

    def encode_u(self, h, train: bool):
        return self.U_network(h, train=train)

    def encode_v(self, h, train: bool):
        return self.V_network(h, train=train)

    @nn.compact
    def __call__(self, obs, train: bool = False):
        h = self.Head_network(obs, train=train)
        f = self.U_network(h, train=train)
        g = self.V_network(h, train=train)
        return h, f, g


def _gather_images(env, S):
    """
    Collect one canonical image per *state*.
    env.env.unwrapped.images has shape (maze_h, maze_w, H_img, W_img, C).
    Returns: jnp.array with shape (S, H_img, W_img, C).
    """
    imgs = env.env.unwrapped.images
    H_img, W_img, C = imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]

    valid_positions = env.env.unwrapped.maze.all_empty_grids()
    assert len(valid_positions) == S, f"Expected {S} states, got {len(valid_positions)} positions."

    out = np.zeros((S, H_img, W_img, C), dtype=imgs.dtype)
    for s_id, (r, c) in enumerate(valid_positions):
        out[s_id] = imgs[r, c, :, :, :]  # (H_img, W_img, C)
    return jnp.array(out)


@partial(jax.jit, static_argnums=2)
def _get_all_state_embeddings(svd_state, images, use_lap_bn):
    vars_in = {"params": svd_state.params}
    if use_lap_bn:
        vars_in["batch_stats"] = svd_state.batch_stats

        # Use batch statistics, but DO NOT commit updates.
        (h, f_s, g_s), _ = svd_state.apply_fn(
            vars_in,
            images,
            train=True,
            mutable=["batch_stats"],   # required so BN won’t error
        )
        return f_s, g_s

    # no BN case
    h, f_s, g_s = svd_state.apply_fn(vars_in, images, train=False)
    return f_s, g_s


@jax.jit
def _cosine_sim_columns(E: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """
    Column-wise |cosine| similarity between GT singular vectors E [N,d]
    and learned embeddings X [N,d]. (Quick diagonal alignment like your helper.)
    """
    Ey = E / (jnp.linalg.norm(E, axis=0, keepdims=True) + 1e-10)
    Xy = X / (jnp.linalg.norm(X, axis=0, keepdims=True) + 1e-10)
    return jnp.abs(jnp.sum(Ey * Xy, axis=0))  # [d]


def _compute_gt_sing_sorted(L: jnp.ndarray, d: int, laplacian: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    is_sym = bool(jnp.allclose(L, L.T, atol=1e-6))

    if is_sym:
        U_vecs, S_vals, V_vecs = jnp.linalg.svd(L, full_matrices=True, compute_uv=True, hermitian=True)
    else:
        U_vecs, S_vals, V_vecs = jnp.linalg.svd(L, full_matrices=True, compute_uv=True, hermitian=False)

    V_vecs = V_vecs.T
    if laplacian:
        idx = jnp.argsort(S_vals)[:d]
    else:
        idx = jnp.argsort(S_vals)[-d:]
    return U_vecs[:, idx], S_vals[idx], V_vecs[:, idx]


def setup(env, args):
    key = jax.random.PRNGKey(args.exp.seed)

    Head_network = Conv_Encoder(name="Head", min_scale=1, width_mul=1, impala_cnn=False,
                                batch_norm=args.laplacian.batch_norm,
                                layer_norm=args.laplacian.layer_norm,
                                layer_dims=list(args.net.c_dims),
                                activations=[nn.relu] * len(args.net.c_dims))

    U_network = SVD_Dense(name="U",
                          op_batch_norm=True,
                          batch_norm=args.laplacian.batch_norm,
                          layer_norm=args.laplacian.layer_norm,
                          layer_dims=[args.laplacian.dims, args.laplacian.dims],
                          activations=[nn.relu, None],
                          laplacian=args.graph.laplacian)

    V_network = SVD_Dense(name="V",
                          op_batch_norm=True,
                          batch_norm=args.laplacian.batch_norm,
                          layer_norm=args.laplacian.layer_norm,
                          layer_dims=[args.laplacian.dims, args.laplacian.dims],
                          activations=[nn.relu, None],
                          laplacian=args.graph.laplacian)

    SVD_network = SVD_Encoder(name="SVD:Enc",
                              Head_network=Head_network,
                              U_network=U_network,
                              V_network=V_network)

    first_img = env.env.unwrapped.images[0][0]
    variables = SVD_network.init(key, np.expand_dims(first_img, 0), train=False)

    svd_state = SVD_State.create(apply_fn=SVD_network.apply,
                                 params=variables['params'],
                                 batch_stats=variables['batch_stats'],
                                 tx=optax.adam(learning_rate=args.laplacian.lr, eps=1.5e-4),
                                 running_sigma_diag=jnp.zeros((args.laplacian.dims,), dtype=jnp.float32))

    return key, svd_state


@jax.jit
def sample_pairs_from_P(key, P):
    """
    Sample a batch of indices (i -> j) from a row-stochastic matrix P [N,N].
    Returns:
      i: [B], j: [B]
    """
    B = P.shape[0]  # full-batch over all rows for determinism in GT runs
    i = jnp.arange(B)                       # [0..N-1]
    subkey = jax.random.split(key, 1)[0]
    j = jax.vmap(lambda r, k: jax.random.categorical(k, jnp.log(r + 1e-12)))(
        P, jax.random.split(subkey, B)
    )
    return i, j  # [N], [N]


@jax.jit
def sample_pairs_double(key, P):
    N = P.shape[0]
    i = jnp.arange(N)

    key1, key2 = jax.random.split(key, 2)
    keys1 = jax.random.split(key1, N)
    keys2 = jax.random.split(key2, N)

    j1 = jax.vmap(lambda r, k: jax.random.categorical(k, jnp.log(r + 1e-12)))(P, keys1)
    j2 = jax.vmap(lambda r, k: jax.random.categorical(k, jnp.log(r + 1e-12)))(P, keys2)
    return i, j1, j2


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def make_update_step(key, svd_state, images, Pk, A, gamma, use_laplacian, use_bn, lambda_r):
    # ------------------------------------------------------------
    # Double-sampling: same x, two independent next-states y1,y2
    # ------------------------------------------------------------
    i_idx, j1_idx, j2_idx = sample_pairs_double(key, Pk)
    x = images[i_idx]
    y1 = images[j1_idx]
    y2 = images[j2_idx]
    x1, y1 = x, y1
    x2, y2 = x, y2

    # # ----- sample one-step pairs (x -> y) from Pk -----
    # i_idx, j_idx = sample_pairs_from_P(key, Pk)  # [N], [N]
    # x_all = images[i_idx]  # [N,H,W,C]
    # y_all = images[j_idx]  # [N,H,W,C]
    #
    # # ----- split into two independent halves for double-sample estimator -----
    # B = x_all.shape[0]
    # B2 = B // 2
    # B_use = 2 * B2
    # x_all = x_all[:B_use]
    # y_all = y_all[:B_use]
    #
    # key, k_perm = jax.random.split(key)
    # perm = jax.random.permutation(k_perm, B_use)
    # idx1 = perm[:B2]
    # idx2 = perm[B2:2 * B2]
    #
    # x1, y1 = x_all[idx1], y_all[idx1]
    # x2, y2 = x_all[idx2], y_all[idx2]

    ema_alpha = 0.1

    def loss_fn(params):
        bs = svd_state.batch_stats

        # ------------------------------------------------------------
        # Helpers: (a) BN update pass (keeps batch_stats updates)
        #          (b) "batch-stat eval" pass (BN uses batch stats but
        #              we DISCARD updates, so batch_stats stays frozen)
        # ------------------------------------------------------------
        def _apply_head_bn_update(params, batch_stats, obs):
            vars_in = {"params": params}
            if use_bn:
                vars_in["batch_stats"] = batch_stats
                h, up = svd_state.apply_fn(
                    vars_in,
                    obs,
                    train=True,                 # force batch-stat BN
                    mutable=["batch_stats"],    # allow BN internals to update
                    method=SVD_Encoder.encode_head,
                )
                return h, up["batch_stats"]
            else:
                h = svd_state.apply_fn(vars_in, obs, train=True, method=SVD_Encoder.encode_head)
                return h, batch_stats

        def _apply_u_bn_update(params, batch_stats, h):
            vars_in = {"params": params}
            if use_bn:
                vars_in["batch_stats"] = batch_stats
                f, up = svd_state.apply_fn(
                    vars_in,
                    h,
                    train=True,
                    mutable=["batch_stats"],
                    method=SVD_Encoder.encode_u,
                )
                return f, up["batch_stats"]
            else:
                f = svd_state.apply_fn(vars_in, h, train=True, method=SVD_Encoder.encode_u)
                return f, batch_stats

        def _apply_v_bn_update(params, batch_stats, h):
            vars_in = {"params": params}
            if use_bn:
                vars_in["batch_stats"] = batch_stats
                g, up = svd_state.apply_fn(
                    vars_in,
                    h,
                    train=True,
                    mutable=["batch_stats"],
                    method=SVD_Encoder.encode_v,
                )
                return g, up["batch_stats"]
            else:
                g = svd_state.apply_fn(vars_in, h, train=True, method=SVD_Encoder.encode_v)
                return g, batch_stats

        def _apply_head_batchstat_eval(params, batch_stats, obs):
            """Use batch-stat BN (train=True) but discard batch_stats updates."""
            vars_in = {"params": params}
            if use_bn:
                vars_in["batch_stats"] = batch_stats
                h, _up = svd_state.apply_fn(
                    vars_in,
                    obs,
                    train=True,                 # force batch-stat BN
                    mutable=["batch_stats"],    # must be mutable or BN errors
                    method=SVD_Encoder.encode_head,
                )
                return h
            else:
                return svd_state.apply_fn(vars_in, obs, train=False, method=SVD_Encoder.encode_head)

        def _apply_u_batchstat_eval(params, batch_stats, h):
            vars_in = {"params": params}
            if use_bn:
                vars_in["batch_stats"] = batch_stats
                f, _up = svd_state.apply_fn(
                    vars_in,
                    h,
                    train=True,
                    mutable=["batch_stats"],
                    method=SVD_Encoder.encode_u,
                )
                return f
            else:
                return svd_state.apply_fn(vars_in, h, train=False, method=SVD_Encoder.encode_u)

        def _apply_v_batchstat_eval(params, batch_stats, h):
            vars_in = {"params": params}
            if use_bn:
                vars_in["batch_stats"] = batch_stats
                g, _up = svd_state.apply_fn(
                    vars_in,
                    h,
                    train=True,
                    mutable=["batch_stats"],
                    method=SVD_Encoder.encode_v,
                )
                return g
            else:
                return svd_state.apply_fn(vars_in, h, train=False, method=SVD_Encoder.encode_v)

        # =========================================================
        # (A) BN UPDATE PASS: update running stats in batch_stats
        # =========================================================
        h_x1_up, bs = _apply_head_bn_update(params, bs, x1)
        h_y1_up, bs = _apply_head_bn_update(params, bs, y1)

        # U always on x
        _, bs = _apply_u_bn_update(params, bs, h_x1_up)

        if use_laplacian:
            # V on both x and y
            _, bs = _apply_v_bn_update(params, bs, h_x1_up)
            _, bs = _apply_v_bn_update(params, bs, h_y1_up)
        else:
            # V only on y
            _, bs = _apply_v_bn_update(params, bs, h_y1_up)

        # Freeze BN stats for feature computation
        bs_frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, bs)

        # =========================================================
        # (B) FEATURE PASS: batch-stat BN but DISCARD BN updates
        # =========================================================
        # Batch-1
        h_x1e = _apply_head_batchstat_eval(params, bs_frozen, x1)
        h_y1e = _apply_head_batchstat_eval(params, bs_frozen, y1)

        f_x1 = _apply_u_batchstat_eval(params, bs_frozen, h_x1e)
        if not use_laplacian:
            g_x1 = _apply_v_batchstat_eval(params, bs_frozen, h_x1e)
            g_y1  = _apply_v_batchstat_eval(params, bs_frozen, h_y1e)
        else:
            h_xy = jnp.concatenate([h_x1e, h_y1e], axis=0)  # [2B, ...]
            g_xy = _apply_v_batchstat_eval(params, bs_frozen, h_xy)
            g_x1, g_y1 = jnp.split(g_xy, 2, axis=0)

        g1 = (g_x1 - g_y1) if use_laplacian else g_y1

        # Batch-2
        h_x2e = _apply_head_batchstat_eval(params, bs_frozen, x2)
        h_y2e = _apply_head_batchstat_eval(params, bs_frozen, y2)

        f_x2 = _apply_u_batchstat_eval(params, bs_frozen, h_x2e)
        if not use_laplacian:
            g_x2 = _apply_v_batchstat_eval(params, bs_frozen, h_x2e)
            g_y2  = _apply_v_batchstat_eval(params, bs_frozen, h_y2e)
        else:
            h_xy2 = jnp.concatenate([h_x2e, h_y2e], axis=0)  # [2B, ...]
            g_xy2 = _apply_v_batchstat_eval(params, bs_frozen, h_xy2)
            g_x2, g_y2 = jnp.split(g_xy2, 2, axis=0)

        g2 = (g_x2 - g_y2) if use_laplacian else g_y2

        # ---------------------------------------------------------
        # Loss + running sigma diag
        # ---------------------------------------------------------
        loss, metrics = svd_loss_fn(
            f_x1, g1, f_x2, g2,
            lambda_r=lambda_r,
            laplacian=use_laplacian,
            skip_idx=0,
            # eps=1e-4,
        )
        sigma1 = compute_sigma_hat(f_x1, g1)
        new_running = update_running_sigma_diag(
            svd_state.running_sigma_diag, sigma1, alpha=ema_alpha
        )

        # ---------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------
        def _cos(u, v, eps=1e-8):
            un = u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + eps)
            vn = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)
            return (un * vn).sum(-1)

        perm2 = jax.random.permutation(key, g_y1.shape[0])
        pos_cos = _cos(g_x1, g_y1).mean()
        neg_cos = _cos(g_x1, g_y1[perm2]).mean()
        v_norm = jnp.linalg.norm(g_x1, ord=2, axis=1).mean()

        return loss, (bs, metrics, new_running, pos_cos, neg_cos, v_norm)

    (loss, (new_bs, metrics, new_running_diag, pos_cos, neg_cos, v_norm)), grads = \
        jax.value_and_grad(loss_fn, has_aux=True)(svd_state.params)

    new_svd_state = svd_state.apply_gradients(grads=grads)
    if use_bn:
        new_svd_state = new_svd_state.replace(batch_stats=new_bs)
    new_svd_state = new_svd_state.replace(running_sigma_diag=new_running_diag)

    gn_total = pytree_l2_norm(grads)
    gn_head = l2_norm_by_prefix(grads, ("Head_network",))
    gn_u = l2_norm_by_prefix(grads, ("U_network",))
    gn_v = l2_norm_by_prefix(grads, ("V_network",))

    return new_svd_state, (loss, metrics, pos_cos, neg_cos, v_norm,
                           gn_total, gn_head, gn_u, gn_v)


def pytree_l2_norm(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0, jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.stack([jnp.sum(x*x) for x in leaves])))


def l2_norm_by_prefix(params_or_grads, prefix: tuple[str, ...]) -> jnp.ndarray:
    flat = flatten_dict(params_or_grads)  # keys are tuples
    picked = [v for k, v in flat.items() if k[:len(prefix)] == prefix]
    if not picked:
        return jnp.array(0.0, jnp.float32)
    return pytree_l2_norm(picked)


def match_columns_abs_cos(U_gt, U_hat, eps=1e-10):
    U_gt  = U_gt  / (jnp.linalg.norm(U_gt,  axis=0, keepdims=True) + eps)
    U_hat = U_hat / (jnp.linalg.norm(U_hat, axis=0, keepdims=True) + eps)

    C = jnp.abs(U_gt.T @ U_hat)  # true cosine matrix now

    d = C.shape[0]
    used = set()
    perm = []
    for i in range(d):
        j = int(jnp.argmax(C[i]))
        while j in used:
            C = C.at[i, j].set(-1.0)
            j = int(jnp.argmax(C[i]))
        used.add(j)
        perm.append(j)
    perm = jnp.array(perm)

    U_hat_m = U_hat[:, perm]
    signs = jnp.sign(jnp.sum(U_gt * U_hat_m, axis=0) + 1e-12)
    U_hat_m = U_hat_m * signs[None, :]
    return U_hat_m, perm, signs


def match_columns_hungarian(U_gt, U_hat, eps=1e-10):
    Ug = np.array(U_gt / (jnp.linalg.norm(U_gt, axis=0, keepdims=True) + eps))
    Uh = np.array(U_hat / (jnp.linalg.norm(U_hat, axis=0, keepdims=True) + eps))
    C = np.abs(Ug.T @ Uh)                 # [d,d]
    r, c = linear_sum_assignment(-C)      # maximize C
    Uh_m = Uh[:, c]
    signs = np.sign((Ug * Uh_m).sum(axis=0) + 1e-12)
    Uh_m = Uh_m * signs[None, :]
    return jnp.array(Uh_m), jnp.array(c), jnp.array(signs), jnp.array(C[r, c])


def learn_embeddings(svd_state, key, env, args, logger):
    # Gather one canonical image per state for encoding
    A = env.n_actions
    S = env.n_states if not args.laplacian.sa else (env.n_states)  # images per *state*
    images = _gather_images(env, env.n_states)  # [S,H,W,C]

    # ---------- build discounted K-step matrix once ----------
    P_base = jnp.array(env.graph.P, dtype=jnp.float32)
    K = int(max(1, args.laplacian.n_step))

    if K == 1:
        Pk = P_base
    else:
        # Pk = sum_{k=1..K} gamma^{k-1} P^k
        S = P_base.shape[0]
        P_power = jnp.eye(S, dtype=jnp.float32)  # P^0
        Pk_acc = jnp.zeros_like(P_base)
        for k in range(1, K + 1):
            P_power = P_power @ P_base           # P^k
            Pk_acc = Pk_acc + (args.env.gamma ** (k - 1)) * P_power
        Pk = Pk_acc

    # # Row-normalize to make it a proper categorical per row
    eps = 1e-12
    Pk = jnp.maximum(Pk, 0.0)                     # guard tiny negatives from numerics
    row_sums = jnp.sum(Pk, axis=1, keepdims=True)
    Pk = jnp.where(row_sums > 0, Pk / (row_sums + eps), jnp.full_like(Pk, 1.0 / Pk.shape[1]))

    d = int(args.laplacian.dims)
    if not args.graph.laplacian:
        N = Pk.shape[0]
        D = jnp.eye(N, dtype=Pk.dtype) / N
        M_gt = D @ Pk
    else:
        M_gt = jnp.eye(S) - Pk

    # flat = flatten_dict(svd_state.params)
    # print(sorted(set([k[0] for k in flat.keys()])))
    # # and optionally:
    # print(sorted(set([k[:2] for k in flat.keys()])))

    with jax.default_device(jax.devices('cpu')[0]):
        gt_u_vecs, gt_sing_vals, gt_v_vecs = _compute_gt_sing_sorted(M_gt, d, laplacian=args.graph.laplacian)

    start = time.time()
    for global_step in range(args.algo.tot_steps + 1):
        key, sub = jax.random.split(key)
        svd_state, (loss, svd_metrics, pos_cos, neg_cos, v_norm,
                    gn_total, gn_head, gn_u, gn_v) = make_update_step(sub, svd_state, images=images, Pk=Pk, A=A,
                                                                       gamma=args.env.gamma,
                                                                       use_laplacian=args.graph.laplacian,
                                                                       use_bn=args.laplacian.batch_norm,
                                                                       lambda_r=args.laplacian.barrier.value)

        if (global_step % 1000 == 0) and args.wb.on:
            phi_u, phi_v = _get_all_state_embeddings(svd_state, images, use_lap_bn=True)

            print(global_step, '-------------------------------------------')
            # print('Norm(U): ', jnp.linalg.norm(phi_u, ord=2, axis=0).mean())
            # print('Norm(V): ', jnp.linalg.norm(phi_v, ord=2, axis=0).mean())

            S = phi_u.shape[0]
            Gu = (phi_u.T @ phi_u) / S
            Gv = (phi_v.T @ phi_v) / S

            print("U diag mean:", jnp.diag(Gu).mean(), "offdiag mean abs:",
                  jnp.mean(jnp.abs(Gu - jnp.diag(jnp.diag(Gu)))))
            print("V diag mean:", jnp.diag(Gv).mean(), "offdiag mean abs:",
                  jnp.mean(jnp.abs(Gv - jnp.diag(jnp.diag(Gv)))))

            def subspace_score(U_gt, U_hat, eps=1e-10):
                U_gt = U_gt / (jnp.linalg.norm(U_gt, axis=0, keepdims=True) + eps)
                U_hat = U_hat / (jnp.linalg.norm(U_hat, axis=0, keepdims=True) + eps)
                M = U_gt.T @ U_hat
                return jnp.sum(M ** 2) / M.shape[0]  # == ||M||_F^2 / d

            print('subspace U: ', subspace_score(gt_u_vecs, phi_u))
            print('subspace V: ', subspace_score(gt_v_vecs, phi_v))

            # phi_u, _, _ = match_columns_abs_cos(gt_u_vecs, phi_u)
            # phi_v, _, _ = match_columns_abs_cos(gt_v_vecs, phi_v)
            # u_vecs_cos_sim = _cosine_sim_columns(gt_u_vecs, phi_u)
            # v_vecs_cos_sim = _cosine_sim_columns(gt_v_vecs, phi_v)

            phi_u, _, _, u_vecs_cos_sim = match_columns_hungarian(gt_u_vecs, phi_u)
            phi_v, _, _, v_vecs_cos_sim = match_columns_hungarian(gt_v_vecs, phi_v)

            print('U:--------')
            print(gt_u_vecs.mean(axis=0))
            print(phi_u.mean(axis=0))
            print('V:--------')
            print(gt_v_vecs.mean(axis=0))
            print(phi_v.mean(axis=0))

            sing_vals = svd_state.running_sigma_diag / env.n_states

            print(global_step, ' U Vec Accuracies: ', u_vecs_cos_sim)
            print(global_step, ' V Vec Accuracies: ', v_vecs_cos_sim)
            # print(jnp.min(gt_sing_vals), jnp.mean(gt_sing_vals), jnp.max(gt_sing_vals))
            print(jnp.round(gt_sing_vals, decimals=4))
            print(jnp.round(sing_vals, decimals=4))
            # print(jnp.round(jnp.abs(gt_sing_vals - sing_vals), decimals=3))
            print('--------------------------------------------------------')

            logs = {
                "loss/svd_loss": float(loss),
                "metrics/U/avg_cos_sim": float(jnp.mean(u_vecs_cos_sim)),
                "metrics/V/avg_cos_sim": float(jnp.mean(v_vecs_cos_sim)),
                # "metrics/U/max_cos_sim": float(jnp.max(evecs_cos_sim)),
                # "metrics/U/min_cos_sim": float(jnp.min(evecs_cos_sim)),
                "diag/pos_cos": float(pos_cos),
                "diag/neg_cos": float(neg_cos),
                "diag/norm": float(v_norm),
                "metrics/SPS": int(global_step / (time.time() - start) + 1e-9),

                "SVD/diag_mean": svd_metrics["svd/diag_mean"].mean(),
                "SVD/off_main": svd_metrics["svd/off_main"].mean(),
                "SVD/off_sb": svd_metrics["svd/off_sb"].mean(),
                "SVD/order_term": svd_metrics["svd/order_term"].mean(),
                "SVD/skip_idx": svd_metrics["svd/skip_idx"].mean(),

                "grads/total": float(gn_total),
                "grads/head": float(gn_head),
                "grads/u": float(gn_u),
                "grads/v": float(gn_v),
            }

            if args.wb.on:
                logger.log(logs, step=global_step)

    return svd_state, key


def main(args, logger):
    env = GridAgent(args)
    args.env.n_actions = env.n_actions
    print('Env:', args.env.maze_type, '| SA:', args.laplacian.sa,
          '| #States:', env.n_states, '| #Nodes:', env.graph.P.shape[0])

    with jax.default_device(jax.devices('gpu')[0]):
        key, svd_state = setup(env, args)
        svd_state, key = learn_embeddings(svd_state, key, env, args, logger)

    print('Env:', args.env.maze_type, '| SA:', args.laplacian.sa,
          '| #States:', env.n_states, '| #Nodes:', env.graph.P.shape[0])


if __name__ == "__main__":
    Args = DotDict(convert_args_to_dict(parse_args()))
    # env.graph will respect these flags in GridAgent
    Args.graph.laplacian    = True
    Args.graph.undirected   = False
    Args.graph.state_action = True if Args.laplacian.sa else False
    Args.graph.normalized_P = False

    if Args.wb.on:
        logger = create_wb_logger(Args)
        wandb.save('src/networks/networks.py', policy='now')
        wandb.save('src/losses/singular.py', policy='now')
        wandb.save('src/losses/singular_2.py', policy='now')

    main(Args, logger)

