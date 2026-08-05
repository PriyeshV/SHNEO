import time
import math
import random
import numpy as np
from functools import partial
from typing import Any, Union, Optional, List, Tuple

import jax
import jax.numpy as jnp
import optax
import flax
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.core import FrozenDict
import chex

# --- your project imports ---
from src.config.eigsr.customgrids.laplacian_gt import *

from src.utils.utils import *
from src.utils.jax_utils import *
from src.utils.grid_agent import GridAgent
import src.fbx_buffers.item_buffer as item_buffer
from src.networks.networks import Conv_Encoder, SA_LapNetwork
from src.losses.laplacian import *

# -------------------------------
# Train state for Laplacian model
# -------------------------------
class Lap_State(TrainState):
    betas: Any
    errors: Any
    quadratic_errors: Any
    barrier_coeffs: Any
    batch_stats: Optional[FrozenDict] = None


# -------------------------------
# Helpers
# -------------------------------
def _pair_index_to_sa(idx: jnp.ndarray, A: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert flattened SA index -> (s, a)."""
    s = idx // A
    a = idx % A
    return s, a


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

# def _gather_images(env, S):
#     """
#     Collect one canonical image per *state* and add a 1-channel marker that encodes (row,col).
#     Output shape: [S, H_img, W_img, C+1]
#     """
#     imgs = env.env.unwrapped.images  # [Hmaze, Wmaze, H, W, C]
#     H_img, W_img, C = imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]
#
#     maze = env.env.unwrapped.maze
#     H_maze, W_maze = maze.height, maze.width
#
#     valid_positions = maze.all_empty_grids()
#     assert len(valid_positions) == S, f"Expected {S} states, got {len(valid_positions)} positions."
#
#     out = np.zeros((S, H_img, W_img, C + 1), dtype=np.float32)
#     for s_id, (r, c) in enumerate(valid_positions):
#         base = imgs[r, c]  # (H_img, W_img, C)
#         # 1-pixel marker carrying (r,c) info; simple, non-invasive
#         marker = np.zeros((H_img, W_img, 1), dtype=np.float32)
#         marker[0, 0, 0] = (r + 1) / (H_maze + 1)
#         marker[0, 1, 0] = (c + 1) / (W_maze + 1)
#
#         out[s_id] = np.concatenate([base.astype(np.float32) / 255.0, marker], axis=-1)
#     return jnp.array(out)


def _get_all_state_embeddings(apply_fn, params, batch_stats, images) -> jnp.ndarray:
    """Encode all states once and return Phi_s [S, D]. Works with/without BN."""
    if batch_stats is None:
        out = apply_fn({'params': params}, images, train=False)
    else:
        out = apply_fn({'params': params, 'batch_stats': batch_stats},
                       images, train=False)  # use_running_average=True inside your module

    # Conv_Encoder returns [S, D]; SA_LapNetwork may return (phi_s, phi_sa) — keep first 2D tensor if present
    if isinstance(out, tuple):
        for x in out:
            if hasattr(x, "ndim") and x.ndim == 2:
                return x
        raise ValueError(f"Expected a [S,D] tensor in tuple, got {[getattr(x, 'shape', None) for x in out]}")
    return out  # [S, D]


def _extract_sa_tensor(out) -> jnp.ndarray:
    """
    Extract [B, A, D] SA embeddings from SA_LapNetwork outputs.
    Accepts (phi_s, phi_sa) or bare [B,A,D].
    """
    if isinstance(out, tuple):
        assert len(out) == 2, f"Expected (phi_s, phi_sa), got tuple of len {len(out)}"
        phi_sa = out[1]
        if not (hasattr(phi_sa, "ndim") and phi_sa.ndim == 3):
            raise ValueError(f"Second item must be [B,A,D], got {getattr(phi_sa, 'shape', None)}")
        return phi_sa
    if hasattr(out, "ndim") and out.ndim == 3:
        return out
    raise ValueError(f"Expected [B,A,D], got {getattr(out, 'shape', None)}")


def _get_state_and_sa_embeddings_for_states(
    apply_fn, params, batch_stats, images
) -> Tuple[Optional[jnp.ndarray], jnp.ndarray]:
    """
    Single forward pass over all states to fetch (phi_s, phi_sa).
    phi_s may be None if the network doesn't return it.
    """
    if batch_stats is None:
        out = apply_fn({'params': params}, images, train=False)
    else:
        out = apply_fn({'params': params, 'batch_stats': batch_stats}, images, train=False)

    # SA_LapNetwork(get_state_emb=True) → (phi_s, phi_sa)
    # SA_LapNetwork(get_state_emb=False) → phi_sa
    phi_sa = _extract_sa_tensor(out)             # [S, A, D]
    phi_s = None
    if isinstance(out, tuple) and hasattr(out[0], "ndim") and out[0].ndim == 2:
        phi_s = out[0]                           # [S, D]
    return phi_s, phi_sa

@jax.jit
def _cosine_sim_columns(E: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """
    Column-wise |cosine| similarity between GT eigenvectors E [N,d]
    and learned embeddings X [N,d]. (Quick diagonal alignment like your helper.)
    """
    Ey = E / (jnp.linalg.norm(E, axis=0, keepdims=True) + 1e-10)
    Xy = X / (jnp.linalg.norm(X, axis=0, keepdims=True) + 1e-10)
    return jnp.abs(jnp.sum(Ey * Xy, axis=0))  # [d]


def _center_actions(phi_sa: jnp.ndarray) -> jnp.ndarray:
    """Dueling-style centering across actions (per state)."""
    return phi_sa - phi_sa.mean(axis=1, keepdims=True)


# -------------------------------
# Setup model
# -------------------------------
def setup(env, args):
    key = jax.random.PRNGKey(args.exp.seed)

    # Build network (state or SA)
    if not args.laplacian.sa:
        lap_network = Conv_Encoder(
            name="Lap:Enc",
            min_scale=1.0,
            layer_dims=list(args.net.c_dims) + [args.laplacian.dims],
            activations=[nn.relu] * len(args.net.c_dims) + [None],
            batch_norm=args.laplacian.batch_norm,
        )
    else:
        # State encoder used inside SA network
        Enc = partial(
            Conv_Encoder,
            # layer_dims=list(args.net.c_dims) + [args.laplacian.dims],
            layer_dims=args.net.c_dims,
            batch_norm=args.laplacian.batch_norm,
            layer_norm=False,
            min_scale=1.0,
            activations=[nn.relu] * len(args.net.c_dims),
        )
        lap_network = SA_LapNetwork(
            n_actions=env.n_actions,
            enc_class=Enc,
            interaction='project',
            layer_dims=[args.laplacian.dims],  # K
            activations=[None],
            get_state_emb=True,
            combine="concat",
            center_advantage=False,  # optional; zero-mean across actions before the sum
            action_gain=1.0,  # you can start >1.0 for better early gradients if needed
            state_hidden_dims=[512],
            action_hidden_dims=[512],
            name="Lap:Enc",
        )

    # Init with one image
    first_img = env.env.unwrapped.images[0][0]
    # first_img = np.concatenate([first_img, np.zeros((*first_img.shape[:2], 1), np.float32)], axis=-1)  # (H,W,C+1)
    variables = lap_network.init(key, np.expand_dims(first_img, 0), train=False)
    params = variables['params']
    batch_stats = variables.get('batch_stats')

    opt = optax.adam(learning_rate=args.laplacian.lr, eps=1.5e-4)

    lap_state = Lap_State.create(
        apply_fn=lap_network.apply,
        params=params,
        tx=opt,
        batch_stats=batch_stats,
        barrier_coeffs=jnp.tril(jnp.ones((1, 1)), k=0) * args.laplacian.barrier.value,
        betas=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
        errors=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
        quadratic_errors=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
    )
    return key, lap_state


# -------------------------------
# Matrix-driven batch sampler
# -------------------------------
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


def make_update_step(images, Pk, A, gamma, lap_args):
    """
    Return a jitted step() that samples i->j from the K-step discounted matrix
    P_K = sum_{k=1..K} gamma^{k-1} P^k (state or state-action depending on env.graph.P),
    and runs multi_step_ALLO (n_steps=1) with your current Laplacian code.
    """
    adv_center = True  # dueling-style centering without changing the network

    @jax.jit
    def step(key, lap_state: Lap_State):
        # ---- sample pairs i->j from the K-step matrix ----
        i_idx, j_idx = sample_pairs_from_P(key, Pk)  # [N], [N]

        # Build observations (and action ids if SA)
        if not lap_args.sa:
            s, sp = i_idx, j_idx
            obs, nobs = images[s], images[sp]                     # [N,H,W,C]
            a = ap = None
        else:
            s, a   = _pair_index_to_sa(i_idx, A)                  # each [N]
            sp, ap = _pair_index_to_sa(j_idx, A)
            obs, nobs = images[s], images[sp]                     # [N,H,W,C]

        def forward(params, batch_stats):
            """
            Encode obs & nobs with BN threaded; returns (phi, nphi, new_batch_stats)
            where:
              - state case: phi=[B,D], nphi=[B,D]
              - SA case:    phi=[B,D] for action a, nphi=[B,A,D] for all next-actions
                              (with optional action-centering)
            """
            apply_fn = lap_state.apply_fn
            new_bs = batch_stats

            if batch_stats is None:
                out  = apply_fn({'params': params}, obs,  train=True)
                nout = apply_fn({'params': params}, nobs, train=True)
            else:
                out, upd1 = apply_fn({'params': params, 'batch_stats': batch_stats},
                                     obs, train=True, mutable=['batch_stats'])
                bs1 = upd1['batch_stats']
                nout, upd2 = apply_fn({'params': params, 'batch_stats': bs1},
                                      nobs, train=True, mutable=['batch_stats'])
                new_bs = upd2['batch_stats']

            if not lap_args.sa:
                # Conv_Encoder → [B,D] (or tuple with first [B,D])
                if isinstance(out, tuple):
                    # pick first 2D tensor
                    phi = None
                    for x in out:
                        if hasattr(x, "ndim") and x.ndim == 2:
                            phi = x; break
                    if phi is None:
                        raise ValueError(f"Expected [B,D] from state encoder, got {[getattr(x, 'shape', None) for x in out]}")
                    nphi = None
                    for x in nout:
                        if hasattr(x, "ndim") and x.ndim == 2:
                            nphi = x; break
                    if nphi is None:
                        raise ValueError(f"Expected [B,D] from state encoder (next), got {[getattr(x, 'shape', None) for x in nout]}")
                else:
                    phi, nphi = out, nout
                return phi, nphi, a, new_bs

            # SA case: take [B,A,D] then index by sampled actions
            phi = _extract_sa_tensor(out)    # [B,A,D]
            nphi = _extract_sa_tensor(nout)   # [B,A,D]
            return phi, nphi, a[:, None], new_bs

        # masks (no terminals here)
        B = i_idx.shape[0]
        mask        = jnp.ones((B,),   dtype=bool)   # [B]
        future_mask = jnp.ones((B, 1), dtype=bool)   # [B,1]

        def loss_fn(params, batch_stats):
            phi, nphi, actions, new_bs = forward(params, batch_stats)
            loss, (pos_loss, ortho_loss, dual_loss), (avg_w, err_mat) = multi_step_ALLO(
                key=key,
                phis_all=phi,
                future_phis_all=nphi[:, None, ...],  # [B,1,D] or [B,1,A,D]
                actions=actions,
                sample_mask=mask,
                masks=future_mask,
                barrier_coeffs=lap_state.barrier_coeffs,
                betas=lap_state.betas,
                gamma=gamma,
                state_action=lap_args.sa,
                uni_policy=lap_args.sa,     # SA P already includes next-action mixing
                n_steps=1,
                use_dual=lap_args.dual.on,
                use_weights=False,
                weights=None,
                average_loss=lap_args.average_loss,
                rms_norm=False,
                barrier_dims=1
            )

            batch_ids = jnp.arange(B)
            if lap_args.sa:
                phi = phi[batch_ids, actions[:, 0], :]  # [B, D]

                if uniform_pi:
                    sampled_fut = jax.random.randint(key, (B, N - 1), 0, A)  # [B, N]
                    nphi = jnp.take_along_axis(nphi, sampled_fut[:, :, None, None], axis=2).squeeze(2)  # [B, N, D]
                else:
                    idx = actions[:, 1:][:, :, None, None]  # [B, N, 1, 1]
                    nphi = jnp.take_along_axis(nphi, idx, axis=2).squeeze(2)  # [B, N, D]

            perm = jax.random.permutation(key, phi.shape[0])  # [B] a shuffled 0..B-1
            neg_phis = nphi[perm, :]  # still shape [B, D]

            def _cos(u, v, eps=1e-8):
                un = u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + eps)
                vn = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)
                return (un * vn).sum(-1)

            pos_cos = _cos(phi, nphi).mean()
            neg_cos = _cos(phi, neg_phis).mean()
            v_norm = jnp.linalg.norm(phi, ord=2, axis=1).mean()

            # NOTE: No auxiliary penalties. Pure ALLO + orthonormality only.
            return loss, (new_bs, err_mat, pos_loss, ortho_loss, dual_loss,
                          pos_cos, neg_cos, v_norm)

        (loss, (new_bs, err_mat, pos_loss, ortho_loss, dual_loss,
                pos_cos, neg_cos, v_norm)), grads = \
            jax.value_and_grad(loss_fn, has_aux=True)(lap_state.params, lap_state.batch_stats)

        new_state = lap_state.apply_gradients(grads=grads)
        if new_state.batch_stats is not None:
            new_state = new_state.replace(batch_stats=new_bs)

        new_state = update_ALLO_parameters(
            new_state, err_mat,
            update_rates=lap_args.rate,
            barrier_args=lap_args.barrier,
            dual_args=lap_args.dual,
            average_loss=lap_args.average_loss,
            barrier_dims=1
        )
        return new_state, (loss, pos_loss, ortho_loss, dual_loss, pos_cos, neg_cos, v_norm)

    return step


# -------------------------------
# Evaluation against GT eigenvectors
# -------------------------------
def _compute_gt_eigs_sorted(L: jnp.ndarray, d: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute top-d GT eigenpairs of L sorted from smallest to largest eigenvalue.
    Uses eigh if L is (numerically) symmetric; otherwise falls back to eig (real parts).
    Returns (evals[:d], evecs[:, :d]) with ascending evals.
    """
    is_sym = bool(jnp.allclose(L, L.T, atol=1e-6))
    if is_sym:
        evals, evecs = jnp.linalg.eigh(L)
    else:
        evals, evecs = jnp.linalg.eig(L)
        evals, evecs = jnp.real(evals), jnp.real(evecs)
    idx = jnp.argsort(evals)[:d]
    return evals[idx], evecs[:, idx]


@partial(jax.jit, static_argnums=(2))
def _get_all_embeddings_for_eval(lap_state: Lap_State, images, state_action) -> jnp.ndarray:
    """Return Phi with shape [N, d] (N=S for state-only; N=S*A for SA)."""
    if not state_action:
        return _get_all_state_embeddings(lap_state.apply_fn, lap_state.params, lap_state.batch_stats, images)  # [S,d]
    _, Phi_sa = _get_state_and_sa_embeddings_for_states(lap_state.apply_fn, lap_state.params, lap_state.batch_stats, images)  # [S,A,d]
    return Phi_sa.reshape(-1, Phi_sa.shape[-1])  # [S*A, d]


@jax.jit
def _compute_algebraic_eigs_from_Phi(Phi: jnp.ndarray, L: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Algebraic eigenvalues via Rayleigh quotient for each learned column φ_k:
      λ_k = (φ_k^T L φ_k) / (φ_k^T φ_k)
    Phi: [N, d], L: [N, N]  -> returns [d]
    """
    LPhi = L @ Phi                      # [N, d]
    num  = jnp.sum(Phi * LPhi, axis=0)  # diag(Phi^T L Phi)  -> [d]
    den  = jnp.sum(Phi * Phi, axis=0)   # diag(Phi^T Phi)    -> [d]
    return num / (den + eps)

# -------------------------------
# Training loop (matrix-driven)
# -------------------------------
def learn_embeddings(lap_state, key, env, args, logger):
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

    # Row-normalize to make it a proper categorical per row
    eps = 1e-12
    Pk = jnp.maximum(Pk, 0.0)                     # guard tiny negatives from numerics
    row_sums = jnp.sum(Pk, axis=1, keepdims=True)
    Pk = jnp.where(row_sums > 0, Pk / (row_sums + eps), jnp.full_like(Pk, 1.0 / Pk.shape[1]))

    # GT eigenpairs (ascending)
    L = jnp.eye(S) - Pk
    d = int(args.laplacian.dims)
    with jax.default_device(jax.devices('cpu')[0]):
        evals_gt, evecs_gt = _compute_gt_eigs_sorted(L, d)

    step_fn = make_update_step(images=images, Pk=Pk, A=A, gamma=args.env.gamma, lap_args=args.laplacian)

    start = time.time()
    for global_step in range(args.algo.tot_steps + 1):
        key, sub = jax.random.split(key)
        lap_state, (loss, pos, ortho, dual, pos_cos, neg_cos, v_norm) = step_fn(sub, lap_state)

        if (global_step % 1000 == 0) and args.wb.on:
            phi = _get_all_embeddings_for_eval(lap_state, images, args.laplacian.sa)
            evecs_cos_sim = _cosine_sim_columns(evecs_gt, phi)
            eig_vals = -1 * jnp.diag(lap_state.betas)
            r_eig_vals = _compute_algebraic_eigs_from_Phi(Phi=phi, L=L)
            print(global_step, ': ', evecs_cos_sim)
            print(jnp.min(evals_gt), jnp.mean(evals_gt), jnp.max(evals_gt))
            print(jnp.round(evals_gt, decimals=3))
            print(jnp.round(eig_vals, decimals=3))
            print(jnp.round(jnp.abs(evals_gt - eig_vals), decimals=3))
            logs = {
                "loss/total_loss": float(loss),
                "loss/pos_loss": float(pos),
                "loss/ortho_loss": float(ortho),
                "loss/dual_loss": float(dual),
                "metrics/avg_cos_sim": float(jnp.mean(evecs_cos_sim)),
                "metrics/max_cos_sim": float(jnp.max(evecs_cos_sim)),
                "metrics/min_cos_sim": float(jnp.min(evecs_cos_sim)),
                "diag/pos_cos": float(pos_cos),
                "diag/neg_cos": float(neg_cos),
                "diag/norm": float(v_norm),
                "metrics/SPS": int(global_step / (time.time() - start) + 1e-9),
                "lap/barrier": float(lap_state.barrier_coeffs[0, 0]),
                "lap/Max_EigVals": float(jnp.max(eig_vals)),
                "lap/Min_EigVals": float(jnp.min(eig_vals)),
                "lap/Mean_EigVals": float(jnp.mean(eig_vals)),
                "lap/Max_R_EigVals": float(jnp.max(r_eig_vals)),
                "lap/Min_R_EigVals": float(jnp.min(r_eig_vals)),
                "lap/Mean_R_EigVals": float(jnp.mean(r_eig_vals)),
            }
            logger.log(logs, step=global_step)
    return lap_state, key


# -------------------------------
# Main
# -------------------------------
def main(args, logger):
    env = GridAgent(args)
    args.env.n_actions = env.n_actions
    print('Env:', args.env.maze_type, '| SA:', args.laplacian.sa,
          '| #States:', env.n_states, '| #Nodes:', env.graph.L.shape[0])

    with jax.default_device(jax.devices('gpu')[0]):
        key, lap_state = setup(env, args)
        lap_state, key = learn_embeddings(lap_state, key, env, args, logger)

    print('Env:', args.env.maze_type, '| SA:', args.laplacian.sa,
          '| #States:', env.n_states, '| #Nodes:', env.graph.L.shape[0])


if __name__ == "__main__":
    Args = DotDict(convert_args_to_dict(parse_args()))
    # env.graph will respect these flags in GridAgent
    Args.graph.laplacian    = True
    Args.graph.undirected   = False
    Args.graph.state_action = True if Args.laplacian.sa else False
    Args.graph.normalized_P = False

    logger = create_wb_logger(Args)
    main(Args, logger)