import os
import time
import math
import random
import traceback
from copy import deepcopy
from functools import partial
from typing import Any, Union, NamedTuple, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from etils import epath

from src.config.customgrids.laplacian import *
args = convert_args_to_dict(parse_args())

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import chex
import flax
import optax
import gymnasium as gym
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

import src.fbx_buffers.trajectory_buffer as trajectory_buffer
import src.fbx_buffers.prioritised_trajectory_buffer as prioritised_trajectory_buffer

from src.utils.jax_utils import *
from src.utils.grid_agent import GridAgent
from src.networks.networks import FFNetwork, Conv_Encoder
from src.mains.algorithms.RND import (
    RNDState,
    rnd_setup,
    rnd_intrinsic_reward,
    rnd_update_obs_rms,
    rnd_update_rew_rms,
    rnd_update_once_per_epoch,
)
from src.utils.jax_utils import qema_update
from src.mains.algorithms.NEO import (
    NEOScaleState,
    neo_scale_from_state_calibrated,
)
from src.mains.algorithms.Laplacian import compute_batch_eigenvalues_seq
from src.losses.laplacian import multi_step_ALLO, update_ALLO_parameters, update_eigenvalues

print("Available JAX devices:", jax.devices())


# ============================================================
# Constants for the diagnostics in this file
# ============================================================
DIAG_FRACTIONS = tuple(np.linspace(0.1, 1.0, 10))
N_VIZ_EIGS = 4
N_BAR_EIGS = None   # None => use args.options.n_options
TRAJ_HORIZON = 25
TRAJ_STALL_PATIENCE = 4
TRAJ_STALL_TOL = 1e-6
HYBRID_RND_BETA = 10.0
ESCAPE_RADII = (2, 3, 4, 5)
PRIMARY_ESCAPE_RADIUS = 3
SUCCESS_NOVELTY_Q = 0.85
MAX_TRAJ_STARTS_TO_OVERLAY = 32


# ============================================================
# Helper dataclasses
# ============================================================
class DQN_Enc_State(TrainState):
    target_params: flax.core.FrozenDict
    batch_stats: Union[None, flax.core.FrozenDict] = None
    target_batch_stats: Union[None, flax.core.FrozenDict] = None


class DQN_FFN_State(TrainState):
    target_params: flax.core.FrozenDict
    batch_stats: Union[None, flax.core.FrozenDict] = None
    target_batch_stats: Union[None, flax.core.FrozenDict] = None


@struct.dataclass
class Lap_State(TrainState):
    betas: Any
    errors: Any
    quadratic_errors: Any
    barrier_coeffs: Any
    eigen_values: Any
    batch_stats: Union[FrozenDict, None]


@struct.dataclass
class Train_State:
    enc_state: DQN_Enc_State
    q_state: DQN_FFN_State
    lap_state: Lap_State
    rnd_state: RNDState
    neo_state: NEOScaleState
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


@chex.dataclass(frozen=True)
class TimeStep:
    observation: chex.Array     # [1,H,W,1]
    next_observation: chex.Array
    action: chex.Array          # [1,1]
    reward: chex.Array          # [1,1]
    done: chex.Array            # [1,1]
    rnd_bonus: chex.Array       # [1,1]


class LapBatch(NamedTuple):
    obs: jnp.ndarray
    dones: jnp.ndarray
    actions: jnp.ndarray
    novelty: jnp.ndarray
    counts: jnp.ndarray


class FlatStorage(struct.PyTreeNode):
    rnd_frame: chex.Array


# ============================================================
# Small utilities
# ============================================================

def rgb_to_gray_uint8(obs_rgb: np.ndarray) -> np.ndarray:
    obs = obs_rgb.astype(np.float32)
    gray = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray[..., None]


def make_eval_points(total_steps: int) -> List[int]:
    pts = sorted({max(1, int(round(frac * total_steps))) for frac in DIAG_FRACTIONS})
    return pts


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# Replay Buffer
# ============================================================
def setup_rb(args, screen_size, seq_len):
    buffer_args = {
        "max_size": args.buffer.size,
        "period": 1,
        "sample_sequence_length": seq_len,
        "add_batch_size": 1,
        "min_length_time_axis": args.algo.starts,
        "sample_batch_size": max(args.buffer.q_batch_size, args.buffer.lap_batch_size),
    }
    if args.buffer.prioritized:
        buffer_args["priority_exponent"] = 0.6
        fbx_buffer = prioritised_trajectory_buffer.make_prioritised_trajectory_buffer
    else:
        fbx_buffer = trajectory_buffer.make_trajectory_buffer

    rb = fbx_buffer(**buffer_args)
    rb = rb.replace(
        init=jax.jit(rb.init),
        can_sample=jax.jit(rb.can_sample),
        add=jax.jit(rb.add, donate_argnums=0),
        sample=jax.jit(rb.sample),
    )
    if args.buffer.prioritized:
        rb = rb.replace(set_priorities=jax.jit(rb.set_priorities, donate_argnums=0))

    dummy_timestep = TimeStep(
        observation=jnp.ones((screen_size[0], screen_size[1], 1), dtype=jnp.uint8),
        next_observation=jnp.ones((screen_size[0], screen_size[1], 1), dtype=jnp.uint8),
        action=jnp.int32(0),
        reward=jnp.float32(0.0),
        done=jnp.bool_(True),
        rnd_bonus=jnp.float32(0.0),
    )
    rb_state = rb.init(dummy_timestep)
    return rb, rb_state


@jax.jit
def get_buffer_key(key):
    key, sample_key = jax.random.split(key)
    return key, sample_key


@jax.jit
def compute_is_weights(probs, running_index, capacity, seq_len, beta):
    n_valid = jnp.clip(running_index - (seq_len - 1), a_min=1)
    n_valid = jnp.minimum(n_valid, capacity)
    w = (1.0 / (n_valid * probs + 1e-8)) ** beta
    return w / (jnp.max(w) + 1e-8)


# ============================================================
# Network setup
# ============================================================
def setup_train_state(key, env, args, n_actions):
    key, q_key, lap_key, rnd_key = jax.random.split(key, 4)

    enc_network = Conv_Encoder(
        name="DQN:Enc",
        min_scale=1.0,
        batch_norm=False,
        layer_norm=args.net.layer_norm,
        layer_dims=list(args.net.c_dims),
        activations=[nn.relu] * len(args.net.c_dims),
    )
    q_network = FFNetwork(
        name="DQN:Q",
        stop_grad=False,
        ip_batch_norm=False,
        batch_norm=False,
        layer_norm=args.net.layer_norm,
        layer_dims=list(args.net.q_dims) + [n_actions],
        activations=[nn.relu] * len(args.net.q_dims) + [None],
    )

    lap_network = Conv_Encoder(
        name="Lap:Enc",
        min_scale=1.0,
        batch_norm=args.laplacian.batch_norm,
        layer_norm=args.laplacian.layer_norm,
        layer_dims=list(args.net.c_dims) + [args.laplacian.dims],
        activations=[nn.relu] * len(args.net.c_dims) + [None],
    )

    obs, _ = env.reset(seed=args.exp.seed)
    obs = rgb_to_gray_uint8(obs)
    obs_b = np.expand_dims(obs, axis=0)
    dummy_embedding = np.ones((1, args.net.c_dims[-1]), dtype=np.float32)

    enc_variables = enc_network.init(q_key, obs_b, train=False)
    enc_lr_schedule = learning_rate_fn(
        init_lr=args.net.lr.enc if args.net.lr.enc is not None else args.net.lr.ffn,
        alpha=args.algo.lr_alpha,
        warmup_epochs=int(args.algo.tot_steps / 4),
        num_epochs=args.algo.tot_steps,
        steps_per_epoch=max(args.algo.train_freq, 1),
    )
    enc_optimizer = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adam(learning_rate=enc_lr_schedule, eps=1.5e-4),
    )
    enc_state = DQN_Enc_State.create(
        apply_fn=enc_network.apply,
        tx=enc_optimizer,
        params=enc_variables["params"],
        batch_stats=None,
        target_params=enc_variables["params"],
        target_batch_stats=None,
    )

    q_variables = q_network.init(q_key, dummy_embedding, train=False)
    q_lr_schedule = learning_rate_fn(
        init_lr=args.net.lr.ffn,
        alpha=args.algo.lr_alpha,
        warmup_epochs=int(args.algo.tot_steps / 4),
        num_epochs=args.algo.tot_steps,
        steps_per_epoch=max(args.algo.train_freq, 1),
    )
    q_optimizer = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adam(learning_rate=q_lr_schedule, eps=1.5e-4),
    )
    q_state = DQN_FFN_State.create(
        apply_fn=q_network.apply,
        tx=q_optimizer,
        params=q_variables["params"],
        batch_stats=None,
        target_params=q_variables["params"],
        target_batch_stats=None,
    )

    lap_variables = lap_network.init(lap_key, obs_b, train=False)
    lap_lr_schedule = learning_rate_fn(
        init_lr=args.laplacian.lr,
        alpha=args.algo.lr_alpha,
        warmup_epochs=int(args.algo.tot_steps / 4),
        num_epochs=args.algo.tot_steps,
        steps_per_epoch=max(args.laplacian.upd_freq if args.laplacian.upd_freq is not None else 1, 1),
    )
    lap_optimizer = optax.chain(
        optax.clip_by_global_norm(args.laplacian.max_grad_norm),
        optax.adamw(learning_rate=lap_lr_schedule, eps=1.5e-4),
    )
    lap_batch_stats = lap_variables["batch_stats"] if args.laplacian.batch_norm else None
    lap_state = Lap_State.create(
        apply_fn=lap_network.apply,
        tx=lap_optimizer,
        params=lap_variables["params"],
        batch_stats=lap_batch_stats,
        barrier_coeffs=jnp.tril(jnp.ones((1, 1)), k=0) * args.laplacian.barrier.value,
        betas=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
        eigen_values=jnp.ones((args.laplacian.dims,), dtype=jnp.float32) * 1e-3,
        errors=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
        quadratic_errors=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
    )

    rnd_state = rnd_setup(
        key=rnd_key,
        lr=args.net.lr.ffn,
        max_grad_norm=args.net.max_grad_norm,
        n_envs=1,
        int_gamma=args.env.gamma,
        obs_shape=obs.shape,
    )
    neo_state = NEOScaleState(
        max_inv_ema=jnp.array(1.0, dtype=jnp.float32),
        mean_scale_ema=jnp.array(1.0, dtype=jnp.float32),
    )

    return key, Train_State(
        enc_state=enc_state,
        q_state=q_state,
        lap_state=lap_state,
        rnd_state=rnd_state,
        neo_state=neo_state,
        timesteps=0,
        n_updates=0,
        grad_steps=0,
    )


# ============================================================
# Q functions
# ============================================================
@partial(jax.jit, static_argnums=(3,))
def get_q_values(obs, enc_state, q_state, use_target: bool = False):
    enc_p = enc_state.target_params if use_target else enc_state.params
    q_p   = q_state.target_params   if use_target else q_state.params

    embeddings = enc_state.apply_fn({'params': enc_p}, obs, train=False)
    q_values = q_state.apply_fn({'params': q_p}, embeddings, train=False)
    return q_values


@partial(jax.jit, static_argnums=(5,))
def sample_action(key, s_tau, obs, enc_state, q_state, sample=False):
    q_values = get_q_values(obs, enc_state, q_state, use_target=False)
    q_probs = jax.lax.cond(
        sample,
        lambda _: jax.nn.softmax(q_values / (s_tau + 1e-8), axis=-1),
        lambda _: jax.nn.softmax(q_values, axis=-1),
        operand=None,
    )
    action = jax.lax.cond(
        sample,
        lambda _: jax.random.categorical(key, logits=jnp.log(q_probs + 1e-8)),
        lambda _: jnp.argmax(q_probs, axis=-1),
        operand=None,
    )
    return action


@partial(jax.jit, static_argnums=0)
def update_target_network(tau, enc_state, q_state):
    enc_state = enc_state.replace(
        target_params=optax.incremental_update(enc_state.params, enc_state.target_params, tau)
    )
    q_state = q_state.replace(
        target_params=optax.incremental_update(q_state.params, q_state.target_params, tau)
    )
    return enc_state, q_state


@partial(jax.jit, static_argnums=(4,))
def update_q_network(data, priorities, enc_state, q_state, use_ddqn):
    q_next_target_all = get_q_values(data.next_observation[:, -1, ...], enc_state, q_state, use_target=True)
    if use_ddqn:
        q_next_online = get_q_values(data.next_observation[:, -1, ...], enc_state, q_state, use_target=False)
        next_action_id = jnp.argmax(jax.lax.stop_gradient(q_next_online), axis=-1)
        q_next_target = q_next_target_all[jnp.arange(next_action_id.shape[0]), next_action_id]
    else:
        q_next_target = jnp.max(q_next_target_all, axis=-1)

    n = args.algo.n_step
    r = data.reward[:, :n]
    d = data.done[:, :n]

    gamma_n = args.env.gamma ** n
    discounts = jnp.power(args.env.gamma, jnp.arange(n))[None, :]
    not_done = 1.0 - d.astype(jnp.float32)
    prefix = jnp.concatenate([jnp.ones((d.shape[0], 1), jnp.float32), not_done[:, :-1]], axis=1)
    mask = jnp.cumprod(prefix, axis=1)
    n_step_return = jnp.sum(r * discounts * mask, axis=1)
    boot_mask = 1.0 - jnp.any(d, axis=1).astype(jnp.float32)

    def td_loss_fn(enc_params, q_params):
        embeddings = enc_state.apply_fn({"params": enc_params}, data.observation[:, 0, ...], train=True)
        q_values = q_state.apply_fn({"params": q_params}, embeddings, train=True)
        q_pred = q_values[jnp.arange(q_values.shape[0]), data.action[:, 0]]

        if args.algo.clip_q_pred:
            target = n_step_return + boot_mask * gamma_n * jnp.clip(q_next_target, jnp.min(q_pred), jnp.max(q_pred))
        else:
            target = n_step_return + boot_mask * gamma_n * q_next_target

        td_error = q_pred - jax.lax.stop_gradient(target)
        upd_priorities = jnp.abs(td_error) + 1e-8
        loss = td_error ** 2
        wt_loss = jnp.mean(priorities * loss) if args.buffer.prioritized else jnp.mean(loss)
        return wt_loss, (jnp.mean(loss), q_pred.mean(), q_pred.min(), q_pred.max(), upd_priorities)

    (wt_td_loss, (td_loss, q_mean, q_min, q_max, new_priorities)), grads = jax.value_and_grad(
        td_loss_fn, has_aux=True, argnums=(0, 1)
    )(enc_state.params, q_state.params)
    grads_enc, grads_q = grads
    enc_state = enc_state.apply_gradients(grads=grads_enc)
    q_state = q_state.apply_gradients(grads=grads_q)
    return enc_state, (q_state, td_loss, q_mean, q_min, q_max, new_priorities)


# ============================================================
# Laplacian update using NEO-style novelty + inverse-count proxy
# ============================================================
@jax.jit
def compute_laplace_neo_targets_from_obs_seq(
    train_state: Train_State,
    obs_seq: jnp.ndarray,
):
    B, T = obs_seq.shape[:2]
    flat = obs_seq.reshape((B * T,) + obs_seq.shape[2:]).astype(jnp.float32)

    raw_bonus_flat = rnd_intrinsic_reward(train_state.rnd_state, flat)
    raw_bonus = raw_bonus_flat.reshape(B, T)

    rnd_bonus = raw_bonus / jnp.sqrt(train_state.rnd_state.rew_rms.var + 1e-8)
    rnd_bonus_n = rnd_bonus / (train_state.rnd_state.rnd_bonus_max_ema + 1e-8)

    inv_count = 1.0 / (rnd_bonus_n + 1e-6)
    inv_cap = 1e4
    inv_used = jnp.minimum(inv_count, inv_cap)

    return raw_bonus, rnd_bonus, rnd_bonus_n, inv_count, inv_used


@partial(jax.jit, static_argnums=(3,))
def update_grid_neo_laplacian_network(key, train_state, lap_batch, n_actions):
    gamma = args.env.gamma
    use_bn = args.laplacian.batch_norm
    use_barrier = args.laplacian.barrier.on
    use_dual = args.laplacian.dual.on
    lap_state = train_state.lap_state

    obs = lap_batch.obs  # [B,T,H,W,C]
    B, T, H, W, C = obs.shape
    obs_flat = obs.reshape(B * T, H, W, C)

    dones = lap_batch.dones
    valid_mask = jnp.logical_not(dones)
    mask = valid_mask[:, 0]
    future_masks = valid_mask[:, 1:]
    valid_trans = (valid_mask[:, :-1] & valid_mask[:, 1:]).astype(jnp.float32)

    novelty_all = lap_batch.novelty.astype(jnp.float32)
    counts_all = lap_batch.counts.astype(jnp.float32)
    actions = lap_batch.actions

    def laplacian_loss_fn(lap_params):
        updates = None
        lap_vars = {"params": lap_params}
        mutable = []
        if use_bn:
            lap_vars["batch_stats"] = lap_state.batch_stats
            mutable.append("batch_stats")

        if mutable:
            phis_flat, updates = lap_state.apply_fn(lap_vars, obs_flat, train=True, mutable=mutable)
        else:
            phis_flat = lap_state.apply_fn(lap_vars, obs_flat, train=True)

        phis = phis_flat.reshape(B, T, args.laplacian.dims)
        ph0 = phis[:, 0, :]
        fut = phis[:, 1:, :]

        loss, (pos_loss, ortho_loss, dual_loss), (avg_weight, err_mat) = multi_step_ALLO(
            key=key,
            phis_all=ph0,
            future_phis_all=fut,
            actions=actions,
            sample_mask=mask,
            masks=future_masks,
            barrier_coeffs=lap_state.barrier_coeffs,
            betas=lap_state.betas,
            weights=None,
            use_future_novelty=True,
            use_sample_novelty=True,
            novelty_values=novelty_all[:, 1:],
            zeta_values=counts_all,
            use_zeta=True,
            normalize_zeta=False,
            gamma=gamma,
            state_action=False,
            uni_policy=False,
            n_steps=future_masks.shape[1],
            use_dual=use_dual,
            use_weights=False,
            average_loss=args.laplacian.average_loss,
            rms_norm=getattr(args.laplacian, "rms_norm", False),
            barrier_dims=1,
        )

        lambda_batch = compute_batch_eigenvalues_seq(phis, valid_trans)
        return loss, (updates, pos_loss, ortho_loss, dual_loss, err_mat, lambda_batch)

    (eigen_loss, (updates, pos_loss, ortho_loss, dual_loss, curr_err_mat, lambda_batch)), grads = \
        jax.value_and_grad(laplacian_loss_fn, has_aux=True)(lap_state.params)

    new_lap_state = lap_state.apply_gradients(grads=grads)
    if use_bn:
        new_lap_state = new_lap_state.replace(batch_stats=updates["batch_stats"])

    if use_barrier or use_dual:
        new_lap_state = update_ALLO_parameters(
            new_lap_state,
            curr_err_mat,
            update_rates=args.laplacian.rate,
            barrier_args=args.laplacian.barrier,
            dual_args=args.laplacian.dual,
            average_loss=args.laplacian.average_loss,
        )

    ev_old = jnp.reshape(new_lap_state.eigen_values, (-1,))
    ev_new = update_eigenvalues(ev_old, lambda_batch, alpha=0.1)
    ev_new = jnp.reshape(ev_new, new_lap_state.eigen_values.shape)
    new_lap_state = new_lap_state.replace(eigen_values=ev_new)

    train_state = train_state.replace(lap_state=new_lap_state)
    return train_state, (eigen_loss, pos_loss, ortho_loss, dual_loss)


# ============================================================
# Full-state extraction for diagnostics
# ============================================================
def get_all_state_pixels(env) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], int], Dict[int, Tuple[int, int]]]:
    base_env = env.unwrapped
    maze = base_env.maze
    all_empty = maze.all_empty_grids()
    ij_to_i = {tuple(pos): i for i, pos in enumerate(all_empty)}
    i_to_ij = {i: tuple(pos) for i, pos in enumerate(all_empty)}

    imgs = []
    coords = []
    robot_backup = maze.robot.copy()
    for pos in all_empty:
        pos = np.asarray(pos, dtype=int)
        maze.reset_robot(pos)
        obs = base_env.get_state(pos)
        imgs.append(rgb_to_gray_uint8(obs))
        coords.append(pos.copy())
    maze.reset_robot(robot_backup)

    return np.stack(imgs, axis=0), np.stack(coords, axis=0), ij_to_i, i_to_ij


@jax.jit
def compute_all_state_embeddings(lap_state: Lap_State, state_imgs: jnp.ndarray):
    vars_l = {"params": lap_state.params}
    if args.laplacian.batch_norm:
        vars_l["batch_stats"] = lap_state.batch_stats
    return lap_state.apply_fn(vars_l, state_imgs, train=False)


@jax.jit
def compute_all_state_novelty(rnd_state: RNDState, state_imgs: jnp.ndarray):
    return rnd_intrinsic_reward(rnd_state, state_imgs.astype(jnp.float32))


# ============================================================
# Grid / graph helpers for diagnostics
# ============================================================
def get_valid_neighbors_from_grid(env, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
    maze = env.unwrapped.maze
    neigh = []
    for a in maze.ACTIONS:
        delta = maze.COMPASS[a]
        nxt = (coord[0] + delta[0], coord[1] + delta[1])
        if maze.is_empty(np.asarray(nxt)):
            neigh.append(nxt)
    return neigh


def build_neighbor_index(env, ij_to_i: Dict[Tuple[int, int], int]) -> Dict[int, List[int]]:
    nmap = {}
    for coord, idx in ij_to_i.items():
        neigh_coords = get_valid_neighbors_from_grid(env, coord)
        nmap[idx] = [ij_to_i[nc] for nc in neigh_coords]
    return nmap


# ============================================================
# Manual eigen-climb diagnostics
# ============================================================
def option_score_vanilla(phi: np.ndarray, s: int, sp: int, dim: int) -> float:
    return float(phi[sp, dim] - phi[s, dim])


def option_score_hybrid(phi: np.ndarray, novelty_n: np.ndarray, s: int, sp: int, dim: int, beta: float) -> float:
    return float(phi[sp, dim] - phi[s, dim] + beta * novelty_n[sp])


def greedy_climb_rollout(
    start_idx: int,
    dim: int,
    phi: np.ndarray,
    novelty_n: np.ndarray,
    neighbors: Dict[int, List[int]],
    coords: np.ndarray,
    use_hybrid: bool,
    beta: float,
    horizon: int = TRAJ_HORIZON,
    stall_patience: int = TRAJ_STALL_PATIENCE,
    stall_tol: float = TRAJ_STALL_TOL,
):
    path = [start_idx]
    unique = {start_idx}
    stall_ctr = 0
    curr = start_idx
    cum_novelty = 0.0
    phi_start = float(phi[start_idx, dim])
    best_delta = -1e18

    for _ in range(horizon):
        cand = neighbors[curr]
        if len(cand) == 0:
            break

        if use_hybrid:
            scores = np.array([option_score_hybrid(phi, novelty_n, curr, sp, dim, beta) for sp in cand], dtype=np.float32)
        else:
            scores = np.array([option_score_vanilla(phi, curr, sp, dim) for sp in cand], dtype=np.float32)

        best_local = int(np.argmax(scores))
        nxt = cand[best_local]
        gain = float(scores[best_local])

        if gain <= stall_tol:
            stall_ctr += 1
        else:
            stall_ctr = 0

        curr = nxt
        path.append(curr)
        unique.add(curr)
        cum_novelty += float(novelty_n[curr])
        best_delta = max(best_delta, float(phi[curr, dim] - phi_start))

        if stall_ctr >= stall_patience:
            break

    start_xy = coords[start_idx]
    end_xy = coords[path[-1]]
    l1 = int(np.abs(end_xy - start_xy).sum())

    return {
        "path": path,
        "unique_states": len(unique),
        "cum_novelty": cum_novelty,
        "stall_time": len(path) - 1,
        "end_l1_dist": l1,
        "best_phi_gain": best_delta,
    }


# ============================================================
# Visualization helpers
# ============================================================
def annotate_start_goal(env, ax, start_marker="S", goal_marker="G"):
    maze = env.unwrapped.maze
    s = maze.entrance
    g = maze.goal

    # x = col, y = row
    ax.text(
        s[1], s[0], start_marker,
        ha="center", va="center",
        fontsize=14, fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="green", linewidth=2, alpha=0.95),
        zorder=10,
    )
    ax.text(
        g[1], g[0], goal_marker,
        ha="center", va="center",
        fontsize=14, fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="red", linewidth=2, alpha=0.95),
        zorder=10,
    )


def render_scalar_on_grid(env, values, coords, title, save_path, cmap="viridis"):
    maze = env.unwrapped.maze
    grid = np.full((maze.height, maze.width), np.nan, dtype=np.float32)
    for idx, (r, c) in enumerate(coords):
        grid[r, c] = float(values[idx])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.invert_yaxis()

    annotate_start_goal(env, ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def render_success_map(env, values, coords, title, save_path, vmin=0.0, vmax=1.0, cmap="magma"):
    maze = env.unwrapped.maze
    grid = np.full((maze.height, maze.width), np.nan, dtype=np.float32)
    for idx, (r, c) in enumerate(coords):
        grid[r, c] = float(values[idx])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.invert_yaxis()

    annotate_start_goal(env, ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)

def render_rollout_overlay(env, coords, paths_vanilla, paths_hybrid, title, save_path):
    maze = env.unwrapped.maze
    fig, ax = plt.subplots(figsize=(7, 7))

    wall_mask = np.ones((maze.height, maze.width), dtype=np.float32)
    for r, c in coords:
        wall_mask[r, c] = 0.0
    ax.imshow(wall_mask, cmap="gray_r", alpha=0.25)

    for path in paths_vanilla:
        xy = coords[np.asarray(path)]
        ax.plot(xy[:, 1], xy[:, 0], alpha=0.35, linewidth=1.5)

    for path in paths_hybrid:
        xy = coords[np.asarray(path)]
        ax.plot(xy[:, 1], xy[:, 0], alpha=0.65, linewidth=2.0)

    annotate_start_goal(env, ax)

    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def render_metric_curves(history: List[Dict[str, float]], metric_keys: List[str], title: str, save_path: str):
    xs = [h["global_step"] for h in history]
    plt.figure(figsize=(8, 5))
    for k in metric_keys:
        ys = [h[k] for h in history]
        plt.plot(xs, ys, label=k)
    plt.xlabel("global_step")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def render_grouped_improvement_bars(
    diag_rows: List[Dict[str, float]],
    metric_base: str,
    radii: Tuple[int, ...],
    title: str,
    save_path: str,
):
    if len(diag_rows) == 0:
        return

    rows = sorted(diag_rows, key=lambda x: x["dim"])
    dims = [int(r["dim"]) for r in rows]
    x = np.arange(len(dims), dtype=np.float32)

    n_groups = len(radii)
    total_width = 0.8
    bar_w = total_width / n_groups
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_w

    fig_w = max(8, 0.9 * len(dims))
    plt.figure(figsize=(fig_w, 5))

    for i, radius in enumerate(radii):
        vals = []
        for r in rows:
            v_key = f"vanilla_{metric_base}_r{radius}_mean"
            h_key = f"hybrid_{metric_base}_r{radius}_mean"
            vals.append(float(r[h_key] - r[v_key]))

        vals = np.asarray(vals, dtype=np.float32)
        plt.bar(x + offsets[i], vals, width=bar_w, label=f"r={radius}")

    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(x, [f"dim {d}" for d in dims], rotation=45, ha="right")
    plt.ylabel(f"Hybrid - Vanilla {metric_base}")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ============================================================
# Core diagnostics run at checkpoints
# ============================================================
def run_full_diagnostics(global_step: int, train_state: Train_State, env, out_dir: str):
    ensure_dir(out_dir)
    state_imgs_u8, coords, ij_to_i, i_to_ij = get_all_state_pixels(env)
    state_imgs = state_imgs_u8.astype(np.float32)

    phi = np.asarray(jax.device_get(compute_all_state_embeddings(train_state.lap_state, jnp.asarray(state_imgs))))
    novelty = np.asarray(jax.device_get(compute_all_state_novelty(train_state.rnd_state, jnp.asarray(state_imgs))))
    novelty_n = novelty / (np.max(novelty) + 1e-8)

    neigh = build_neighbor_index(env, ij_to_i)
    # ------------------------------------------------------------
    # Small subset for image plots
    # ------------------------------------------------------------
    viz_eig_ids = list(range(min(N_VIZ_EIGS + 1, phi.shape[1])))
    if len(viz_eig_ids) > 1:
        viz_eig_ids = viz_eig_ids[1:]
    viz_eig_ids = viz_eig_ids[:N_VIZ_EIGS]

    # ------------------------------------------------------------
    # All dims for stacked-bar diagnostics
    # ------------------------------------------------------------
    if N_BAR_EIGS is None:
        n_bar_eigs = args.algo.n_options
        print('n_bar_eigs:', n_bar_eigs)
    else:
        n_bar_eigs = N_BAR_EIGS

    bar_eig_ids = list(range(min(n_bar_eigs + 1, phi.shape[1])))
    if len(bar_eig_ids) > 1:
        bar_eig_ids = bar_eig_ids[1:]
    bar_eig_ids = bar_eig_ids[:n_bar_eigs]

    novelty_thr = float(np.quantile(novelty_n, SUCCESS_NOVELTY_Q))
    diag_rows = []

    render_scalar_on_grid(
        env,
        novelty_n,
        coords,
        f"RND novelty map @ step {global_step}",
        os.path.join(out_dir, f"step_{global_step:07d}_rnd_map.png"),
        cmap="magma",
    )

    history_metrics = {
        "global_step": global_step,
    }

    for dim in bar_eig_ids:
        do_render = dim in viz_eig_ids

        eig_vals = phi[:, dim]
        eig_norm = (eig_vals - eig_vals.min()) / (eig_vals.max() - eig_vals.min() + 1e-8)

        if do_render:
            render_scalar_on_grid(
                env,
                eig_norm,
                coords,
                f"Eigenvector dim {dim} @ step {global_step}",
                os.path.join(out_dir, f"step_{global_step:07d}_eig_{dim}.png"),
                cmap="coolwarm",
            )

        # ------------------------------------------------------------
        # Edge scores
        # ------------------------------------------------------------
        vanilla_edge = np.zeros((len(coords),), dtype=np.float32)
        hybrid_edge = np.zeros((len(coords),), dtype=np.float32)
        for s in range(len(coords)):
            nbrs = neigh[s]
            if len(nbrs) == 0:
                continue
            vanilla_edge[s] = max(option_score_vanilla(phi, s, sp, dim) for sp in nbrs)
            hybrid_edge[s] = max(option_score_hybrid(phi, novelty_n, s, sp, dim, HYBRID_RND_BETA) for sp in nbrs)

        if do_render:
            render_scalar_on_grid(
                env,
                vanilla_edge,
                coords,
                f"Vanilla edge potential dim {dim} @ step {global_step}",
                os.path.join(out_dir, f"step_{global_step:07d}_edge_vanilla_{dim}.png"),
                cmap="viridis",
            )
            render_scalar_on_grid(
                env,
                hybrid_edge,
                coords,
                f"Hybrid edge+RND dim {dim} @ step {global_step}",
                os.path.join(out_dir, f"step_{global_step:07d}_edge_hybrid_{dim}.png"),
                cmap="viridis",
            )

        # ------------------------------------------------------------
        # Per-start diagnostics
        # ------------------------------------------------------------
        per_start_success_v = {r: np.zeros((len(coords),), dtype=np.float32) for r in ESCAPE_RADII}
        per_start_success_h = {r: np.zeros((len(coords),), dtype=np.float32) for r in ESCAPE_RADII}
        per_start_escape_v = {r: np.zeros((len(coords),), dtype=np.float32) for r in ESCAPE_RADII}
        per_start_escape_h = {r: np.zeros((len(coords),), dtype=np.float32) for r in ESCAPE_RADII}

        overlay_v = []
        overlay_h = []

        uniq_v, uniq_h = [], []
        nov_v, nov_h = [], []
        stall_v, stall_h = [], []
        esc_v, esc_h = [], []

        for s in range(len(coords)):
            rv = greedy_climb_rollout(
                start_idx=s,
                dim=dim,
                phi=phi,
                novelty_n=novelty_n,
                neighbors=neigh,
                coords=coords,
                use_hybrid=False,
                beta=HYBRID_RND_BETA,
            )
            rh = greedy_climb_rollout(
                start_idx=s,
                dim=dim,
                phi=phi,
                novelty_n=novelty_n,
                neighbors=neigh,
                coords=coords,
                use_hybrid=True,
                beta=HYBRID_RND_BETA,
            )

            for radius in ESCAPE_RADII:
                escv = float(rv["end_l1_dist"] >= radius)
                esch = float(rh["end_l1_dist"] >= radius)

                succ_v = float((escv > 0.5) and (rv["cum_novelty"] >= novelty_thr))
                succ_h = float((esch > 0.5) and (rh["cum_novelty"] >= novelty_thr))

                per_start_escape_v[radius][s] = escv
                per_start_escape_h[radius][s] = esch
                per_start_success_v[radius][s] = succ_v
                per_start_success_h[radius][s] = succ_h

            uniq_v.append(rv["unique_states"])
            uniq_h.append(rh["unique_states"])
            nov_v.append(rv["cum_novelty"])
            nov_h.append(rh["cum_novelty"])
            stall_v.append(rv["stall_time"])
            stall_h.append(rh["stall_time"])
            esc_v.append(float(rv["end_l1_dist"] >= PRIMARY_ESCAPE_RADIUS))
            esc_h.append(float(rh["end_l1_dist"] >= PRIMARY_ESCAPE_RADIUS))

            if do_render and s < MAX_TRAJ_STARTS_TO_OVERLAY:
                overlay_v.append(rv["path"])
                overlay_h.append(rh["path"])

        if do_render:
            render_success_map(
                env,
                per_start_success_v[PRIMARY_ESCAPE_RADIUS],
                coords,
                f"Vanilla success map dim {dim} @ step {global_step}",
                os.path.join(out_dir, f"step_{global_step:07d}_success_vanilla_{dim}.png"),
                cmap="magma",
            )
            render_success_map(
                env,
                per_start_success_h[PRIMARY_ESCAPE_RADIUS],
                coords,
                f"Hybrid success map dim {dim} @ step {global_step}",
                os.path.join(out_dir, f"step_{global_step:07d}_success_hybrid_{dim}.png"),
                cmap="magma",
            )
            render_rollout_overlay(
                env,
                coords,
                overlay_v,
                overlay_h,
                f"Trajectories dim {dim}: vanilla + hybrid @ step {global_step}",
                os.path.join(out_dir, f"step_{global_step:07d}_traj_overlay_{dim}.png"),
            )

        row = {
            "global_step": global_step,
            "dim": int(dim),
            "vanilla_unique_mean": float(np.mean(uniq_v)),
            "hybrid_unique_mean": float(np.mean(uniq_h)),
            "vanilla_novelty_mean": float(np.mean(nov_v)),
            "hybrid_novelty_mean": float(np.mean(nov_h)),
            "vanilla_stall_mean": float(np.mean(stall_v)),
            "hybrid_stall_mean": float(np.mean(stall_h)),
            "vanilla_escape_mean": float(np.mean(esc_v)),
            "hybrid_escape_mean": float(np.mean(esc_h)),
            "vanilla_success_mean": float(np.mean(per_start_success_v[PRIMARY_ESCAPE_RADIUS])),
            "hybrid_success_mean": float(np.mean(per_start_success_h[PRIMARY_ESCAPE_RADIUS])),
        }

        for radius in ESCAPE_RADII:
            row[f"vanilla_escape_r{radius}_mean"] = float(np.mean(per_start_escape_v[radius]))
            row[f"hybrid_escape_r{radius}_mean"] = float(np.mean(per_start_escape_h[radius]))
            row[f"vanilla_success_r{radius}_mean"] = float(np.mean(per_start_success_v[radius]))
            row[f"hybrid_success_r{radius}_mean"] = float(np.mean(per_start_success_h[radius]))

        diag_rows.append(row)

        history_metrics[f"dim{dim}/vanilla_success"] = row["vanilla_success_mean"]
        history_metrics[f"dim{dim}/hybrid_success"] = row["hybrid_success_mean"]
        history_metrics[f"dim{dim}/vanilla_escape"] = row["vanilla_escape_mean"]
        history_metrics[f"dim{dim}/hybrid_escape"] = row["hybrid_escape_mean"]

        for radius in ESCAPE_RADII:
            history_metrics[f"dim{dim}/vanilla_success_r{radius}"] = row[f"vanilla_success_r{radius}_mean"]
            history_metrics[f"dim{dim}/hybrid_success_r{radius}"] = row[f"hybrid_success_r{radius}_mean"]
            history_metrics[f"dim{dim}/vanilla_escape_r{radius}"] = row[f"vanilla_escape_r{radius}_mean"]
            history_metrics[f"dim{dim}/hybrid_escape_r{radius}"] = row[f"hybrid_escape_r{radius}_mean"]

    # np.save(os.path.join(out_dir, f"step_{global_step:07d}_phi.npy"), phi)
    # np.save(os.path.join(out_dir, f"step_{global_step:07d}_rnd.npy"), novelty)
    # np.save(os.path.join(out_dir, f"step_{global_step:07d}_coords.npy"), coords)

    render_grouped_improvement_bars(
        diag_rows,
        metric_base="success",
        radii=ESCAPE_RADII,
        title=f"Hybrid improvement in success @ step {global_step}",
        save_path=os.path.join(out_dir, f"step_{global_step:07d}_grouped_success_improvement.png"),
    )

    render_grouped_improvement_bars(
        diag_rows,
        metric_base="escape",
        radii=ESCAPE_RADII,
        title=f"Hybrid improvement in escape @ step {global_step}",
        save_path=os.path.join(out_dir, f"step_{global_step:07d}_grouped_escape_improvement.png"),
    )
    return history_metrics, diag_rows


# ============================================================
# Evaluation policy
# ============================================================
def evaluate_policy(train_state: Train_State, env, global_step: int):
    eval_env = deepcopy(env)
    obs, _ = eval_env.reset(seed=args.exp.seed)
    eps_return, terminated = 0.0, False

    for eps_step in range(args.algo.max_eps_len + 1):
        obs_g = rgb_to_gray_uint8(obs)
        obs_b = np.expand_dims(obs_g, axis=0)
        q_values = get_q_values(jnp.asarray(obs_b), train_state.enc_state, train_state.q_state, use_target=False)
        action = int(np.asarray(jax.device_get(jnp.argmax(q_values, axis=-1)))[0])
        obs, reward, terminated, _, _ = eval_env.step(action)
        eps_return += reward
        if terminated:
            break

    if args.wb.on:
        wandb.log({
            "test/episodic_return": eps_return,
            "test/episodic_length": eps_step,
        }, step=global_step)
    return terminated, eps_return, eps_step


# ============================================================
# Main training loop
# ============================================================
def q_learning(train_state: Train_State, rb_state, key, n_actions, env, rb, out_root: str):
    buffer_device = jax.devices(args.buffer.device)[0]
    total_timesteps = args.algo.tot_steps
    obs, _ = env.reset(seed=args.exp.seed)

    eval_points = set(make_eval_points(total_timesteps))
    metric_history = []
    diag_rows_all = []
    test_results = []

    eps_step, eps_return = 1, 0.0
    log_batch = []
    start_time = None
    seq_len = max(args.algo.n_step, args.laplacian.n_step) + 1

    for global_step in range(total_timesteps + 1):
        obs_gray = rgb_to_gray_uint8(obs)
        obs_b = np.expand_dims(obs_gray, axis=0)

        key, epsilon_key, action_key = jax.random.split(key, 3)
        epsilon = linear_schedule(
            args.algo.eps.start,
            args.algo.eps.end,
            int(args.algo.eps.frac * (total_timesteps + args.algo.starts)),
            global_step,
        )
        s_tau = linear_schedule(
            args.algo.s_tau.start,
            args.algo.s_tau.end,
            int(args.algo.s_tau.frac * (total_timesteps + args.algo.starts)),
            global_step,
        )

        if jax.random.uniform(epsilon_key) < epsilon:
            action = int(jax.random.randint(action_key, (), 0, n_actions))
        else:
            action = sample_action(action_key, s_tau, jnp.asarray(obs_b), train_state.enc_state, train_state.q_state, args.algo.sample_action)
            action = int(jax.device_get(action)[0])

        next_obs, reward, terminated, _, _ = env.step(action)
        next_obs_gray = rgb_to_gray_uint8(next_obs)
        next_obs_b = np.expand_dims(next_obs_gray, axis=0)

        rnd_bonus = float(np.asarray(jax.device_get(rnd_intrinsic_reward(train_state.rnd_state, jnp.asarray(next_obs_b.astype(np.float32)))))[0])

        with jax.default_device(buffer_device):
            timestep = TimeStep(
                observation=jnp.asarray(obs_b[:, None, ...]),  # [1,1,H,W,C]
                next_observation=jnp.asarray(next_obs_b[:, None, ...]),  # [1,1,H,W,C]
                action=jnp.expand_dims(jnp.asarray(action, dtype=jnp.int32), (0, 1)),
                reward=jnp.expand_dims(jnp.asarray(reward, dtype=jnp.float32), (0, 1)),
                done=jnp.expand_dims(
                    jnp.asarray(terminated or eps_step >= args.algo.max_eps_len, dtype=jnp.bool_), (0, 1)
                ),
                rnd_bonus=jnp.expand_dims(jnp.asarray(rnd_bonus, dtype=jnp.float32), (0, 1)),
            )
            rb_state = rb.add(rb_state, timestep)

        eps_return += reward
        if terminated or eps_step >= args.algo.max_eps_len:
            if args.wb.on and global_step >= args.algo.starts:
                wandb.log({
                    "values/episodic_return": eps_return,
                    "values/episodic_length": eps_step,
                    "values/epsilon": epsilon,
                }, step=global_step)
            log_batch.append(f"Global Step: {global_step} | Episodic Steps: {eps_step} | Return: {eps_return}")
            next_obs, _ = env.reset()
            eps_step, eps_return = 0, 0.0
        eps_step += 1
        obs = next_obs

        if global_step % 5000 == 0 and len(log_batch) > 0:
            print("\n".join(log_batch))
            log_batch = []

        if global_step == args.algo.starts:
            assert rb.can_sample(rb_state)
            print("Learning Starts")
            start_time = time.time()

        if global_step >= args.algo.starts:
            # -----------------------------------------------------
            # Update RND stats from latest frame every step
            # -----------------------------------------------------
            train_state = train_state.replace(
                rnd_state=rnd_update_obs_rms(train_state.rnd_state, jnp.asarray(next_obs_b.astype(np.float32)))
            )

            rnd_state_new, rnd_bonus_norm = rnd_update_rew_rms(
                train_state.rnd_state,
                jnp.asarray([[rnd_bonus]], dtype=jnp.float32),
            )

            bonus_max_ema = qema_update(
                prev=rnd_state_new.rnd_bonus_max_ema,
                x=rnd_bonus_norm,
                tau=1e-3,
                q=0.99,
                monotone=True,
            )

            rnd_state_new = rnd_state_new.replace(rnd_bonus_max_ema=bonus_max_ema)
            train_state = train_state.replace(rnd_state=rnd_state_new)

            if global_step % args.algo.train_freq == 0:
                beta = linear_schedule(args.buffer.beta, 0.1, total_timesteps, global_step - args.algo.starts)
                with jax.default_device(buffer_device):
                    key, sample_key = get_buffer_key(key)
                    batch = rb.sample(rb_state, sample_key)

                if args.buffer.prioritized:
                    priorities = compute_is_weights(
                        batch.probabilities,
                        rb_state.running_index,
                        rb_state.sum_tree_state.capacity,
                        seq_len,
                        beta,
                    )
                else:
                    priorities = None

                q_data = batch.experience

                q_batch = q_data.replace(
                    observation=q_data.observation[:, : args.algo.n_step + 1, ...],
                    next_observation=q_data.next_observation[:, : args.algo.n_step + 1, ...],
                    action=q_data.action[:, : args.algo.n_step + 1],
                    reward=q_data.reward[:, : args.algo.n_step + 1],
                    done=q_data.done[:, : args.algo.n_step + 1],
                    rnd_bonus=q_data.rnd_bonus[:, : args.algo.n_step + 1],
                )

                enc_state, (q_state, td_loss, q_mean, q_min, q_max, new_priorities) = update_q_network(
                    q_batch,
                    priorities,
                    train_state.enc_state,
                    train_state.q_state,
                    args.algo.ddqn,
                )
                train_state = train_state.replace(
                    enc_state=enc_state,
                    q_state=q_state,
                    grad_steps=train_state.grad_steps + 1,
                )

                if args.buffer.prioritized:
                    with jax.default_device(buffer_device):
                        rb_state = rb.set_priorities(rb_state, batch.indices, new_priorities)

                # Laplacian update at its own frequency
                if args.laplacian.upd_freq is not None and global_step % args.laplacian.upd_freq == 0:
                    lap_obs = q_data.observation[:, : args.laplacian.n_step + 1, ...]
                    lap_dones = q_data.done[:, : args.laplacian.n_step + 1]
                    lap_actions = q_data.action[:, : args.laplacian.n_step + 1]

                    raw_bonus_seq, rnd_bonus_seq, rnd_bonus_n_seq, inv_count_seq, inv_used_seq = \
                        compute_laplace_neo_targets_from_obs_seq(train_state, lap_obs.astype(jnp.float32))

                    neo_max = qema_update(
                        prev=train_state.neo_state.max_inv_ema,
                        x=inv_used_seq,
                        tau=1e-3,
                        q=0.99,
                        monotone=True,
                    )

                    neo_state_pre = train_state.neo_state.replace(max_inv_ema=neo_max)

                    zeta_seq, neo_state_new = neo_scale_from_state_calibrated(
                        inv_count=inv_used_seq,
                        neo_state=neo_state_pre,
                        delta=0.5,
                        k=1,
                        target_mean=0.1,
                        tau_mean=1e-3,
                    )

                    train_state = train_state.replace(neo_state=neo_state_new)

                    neo_log_stats = {
                        "raw_bonus_mean": jnp.mean(raw_bonus_seq),
                        "raw_bonus_std": jnp.std(raw_bonus_seq),
                        "raw_bonus_min": jnp.min(raw_bonus_seq),
                        "raw_bonus_max": jnp.max(raw_bonus_seq),

                        "rnd_bonus_mean": jnp.mean(rnd_bonus_seq),
                        "rnd_bonus_std": jnp.std(rnd_bonus_seq),
                        "rnd_bonus_min": jnp.min(rnd_bonus_seq),
                        "rnd_bonus_max": jnp.max(rnd_bonus_seq),

                        "rnd_bonus_n_mean": jnp.mean(rnd_bonus_n_seq),
                        "rnd_bonus_n_std": jnp.std(rnd_bonus_n_seq),
                        "rnd_bonus_n_min": jnp.min(rnd_bonus_n_seq),
                        "rnd_bonus_n_max": jnp.max(rnd_bonus_n_seq),

                        "inv_count_mean": jnp.mean(inv_count_seq),
                        "inv_count_std": jnp.std(inv_count_seq),
                        "inv_count_min": jnp.min(inv_count_seq),
                        "inv_count_max": jnp.max(inv_count_seq),

                        "inv_used_mean": jnp.mean(inv_used_seq),
                        "inv_used_std": jnp.std(inv_used_seq),
                        "inv_used_min": jnp.min(inv_used_seq),
                        "inv_used_max": jnp.max(inv_used_seq),

                        "zeta_mean": jnp.mean(zeta_seq),
                        "zeta_std": jnp.std(zeta_seq),
                        "zeta_min": jnp.min(zeta_seq),
                        "zeta_max": jnp.max(zeta_seq),
                    }

                    lap_batch = LapBatch(
                        obs=lap_obs.astype(jnp.float32),
                        dones=lap_dones,
                        actions=lap_actions,
                        novelty=rnd_bonus_seq,
                        counts=zeta_seq,
                    )

                    key, lap_key = jax.random.split(key)
                    train_state, (eig_loss, pos_loss, ortho_loss, dual_loss) = update_grid_neo_laplacian_network(
                        lap_key, train_state, lap_batch, n_actions
                    )
                else:
                    eig_loss = pos_loss = ortho_loss = dual_loss = jnp.array(0.0, dtype=jnp.float32)

                # RND predictor one update per train step using sampled next frames
                flat_storage = FlatStorage(
                    rnd_frame=q_data.next_observation[:, 0, ...].astype(jnp.float32)
                )
                if global_step % 8 == 0:
                    rnd_state2, rnd_fwd_loss = rnd_update_once_per_epoch(
                        train_state.rnd_state,
                        flat_storage,
                        update_proportion=0.01,
                        burnin=False,
                    )
                else:
                    rnd_state2 = train_state.rnd_state
                    rnd_fwd_loss = jnp.array(0.0, dtype=jnp.float32)

                train_state = train_state.replace(rnd_state=rnd_state2)

                if global_step % 1000 == 0 and args.wb.on:
                    wandb.log({
                        "losses/td_loss": jax.device_get(td_loss),
                        "values/q_values": jax.device_get(q_mean),
                        "values/q_min": jax.device_get(q_min),
                        "values/q_max": jax.device_get(q_max),
                        "values/SPS": int((global_step - args.algo.starts) / max(time.time() - start_time, 1e-6)),
                        "rnd/fwd_loss": jax.device_get(rnd_fwd_loss),
                        "eigen/total_loss": jax.device_get(eig_loss),
                        "eigen/pos_loss": jax.device_get(pos_loss),
                        "eigen/ortho_loss": jax.device_get(ortho_loss),
                        "eigen/dual_loss": jax.device_get(dual_loss),
                        "eigen/max_eig": jax.device_get(jnp.max(train_state.lap_state.eigen_values)),

                        "neo/rnd_bonus_max_ema": jax.device_get(train_state.rnd_state.rnd_bonus_max_ema),
                        "neo/max_inv_ema": jax.device_get(train_state.neo_state.max_inv_ema),

                        "neo/raw_bonus_mean": jax.device_get(
                            neo_log_stats["raw_bonus_mean"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/raw_bonus_std": jax.device_get(
                            neo_log_stats["raw_bonus_std"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/raw_bonus_min": jax.device_get(
                            neo_log_stats["raw_bonus_min"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/raw_bonus_max": jax.device_get(
                            neo_log_stats["raw_bonus_max"]) if 'neo_log_stats' in locals() else 0.0,

                        "neo/rnd_bonus_mean": jax.device_get(
                            neo_log_stats["rnd_bonus_mean"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/rnd_bonus_std": jax.device_get(
                            neo_log_stats["rnd_bonus_std"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/rnd_bonus_min": jax.device_get(
                            neo_log_stats["rnd_bonus_min"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/rnd_bonus_max": jax.device_get(
                            neo_log_stats["rnd_bonus_max"]) if 'neo_log_stats' in locals() else 0.0,

                        "neo/rnd_bonus_n_mean": jax.device_get(
                            neo_log_stats["rnd_bonus_n_mean"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/rnd_bonus_n_std": jax.device_get(
                            neo_log_stats["rnd_bonus_n_std"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/rnd_bonus_n_min": jax.device_get(
                            neo_log_stats["rnd_bonus_n_min"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/rnd_bonus_n_max": jax.device_get(
                            neo_log_stats["rnd_bonus_n_max"]) if 'neo_log_stats' in locals() else 0.0,

                        "neo/inv_count_mean": jax.device_get(
                            neo_log_stats["inv_count_mean"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/inv_count_std": jax.device_get(
                            neo_log_stats["inv_count_std"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/inv_count_min": jax.device_get(
                            neo_log_stats["inv_count_min"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/inv_count_max": jax.device_get(
                            neo_log_stats["inv_count_max"]) if 'neo_log_stats' in locals() else 0.0,

                        "neo/inv_used_mean": jax.device_get(
                            neo_log_stats["inv_used_mean"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/inv_used_std": jax.device_get(
                            neo_log_stats["inv_used_std"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/inv_used_min": jax.device_get(
                            neo_log_stats["inv_used_min"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/inv_used_max": jax.device_get(
                            neo_log_stats["inv_used_max"]) if 'neo_log_stats' in locals() else 0.0,

                        "neo/zeta_mean": jax.device_get(
                            neo_log_stats["zeta_mean"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/zeta_std": jax.device_get(
                            neo_log_stats["zeta_std"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/zeta_min": jax.device_get(
                            neo_log_stats["zeta_min"]) if 'neo_log_stats' in locals() else 0.0,
                        "neo/zeta_max": jax.device_get(
                            neo_log_stats["zeta_max"]) if 'neo_log_stats' in locals() else 0.0,
                    }, step=global_step)

            if (args.algo.tau == 1 and global_step % args.algo.tgt_upd_freq == 0) or (args.algo.tau < 1):
                enc_state, q_state = update_target_network(args.algo.tau, train_state.enc_state, train_state.q_state)
                train_state = train_state.replace(enc_state=enc_state, q_state=q_state)

            # -----------------------------------------------------
            # Policy evaluation
            # -----------------------------------------------------
            if (global_step >= total_timesteps * 0.5) and global_step % 5000 == 0:
                reached_goal, test_ret, test_len = evaluate_policy(train_state, env, global_step)
                test_results.append(float(reached_goal))
                n_success = int(np.asarray(test_results, np.int32).sum())
                success_rate = n_success / len(test_results)
                print("Success Rate:", f"{n_success}/{len(test_results)} = {success_rate:.3f}")
                if args.wb.on:
                    wandb.run.summary["Success Rate"] = success_rate

        train_state = train_state.replace(timesteps=global_step, n_updates=train_state.n_updates + int(global_step >= args.algo.starts))

        # ---------------------------------------------------------
        # Diagnostics every 10% of training
        # ---------------------------------------------------------
        if global_step in eval_points:
            import shutil

            diag_dir = os.path.join(out_root, f"step_{global_step:07d}")
            if os.path.exists(diag_dir):
                shutil.rmtree(diag_dir)
            os.makedirs(diag_dir, exist_ok=True)

            history_metrics, diag_rows = run_full_diagnostics(global_step, train_state, env, diag_dir)
            metric_history.append(history_metrics)
            diag_rows_all.extend(diag_rows)

            # np.save(os.path.join(out_root, "metric_history.npy"), metric_history, allow_pickle=True)
            # np.save(os.path.join(out_root, "diag_rows.npy"), diag_rows_all, allow_pickle=True)

            # aggregate curves accumulated so far
            if len(metric_history) >= 2:
                selected_dims = list(range(1, 1 + args.algo.n_options))

                for dim in selected_dims:
                    for metric_name in [
                        "vanilla_success",
                        "hybrid_success",
                        "vanilla_escape",
                        "hybrid_escape",
                    ]:
                        key_name = f"dim{dim}/{metric_name}"
                        if key_name in metric_history[0]:
                            safe_name = key_name.replace("/", "_")
                            render_metric_curves(
                                metric_history,
                                [key_name],
                                key_name,
                                os.path.join(out_root, f"curve_{safe_name}.png"),
                            )

            if args.wb.on:
                flat = {f"diag/{k}": v for k, v in history_metrics.items() if k != "global_step"}
                wandb.log(flat, step=global_step)
                step_tag = f"step_{global_step:07d}"

                for fname in sorted(os.listdir(diag_dir)):
                    if fname.endswith(".png"):
                        image_key = os.path.splitext(fname)[0]
                        prefix = f"{step_tag}_"
                        if image_key.startswith(prefix):
                            image_key = image_key[len(prefix):]

                        wandb.log(
                            {f"{step_tag}/{image_key}": wandb.Image(os.path.join(diag_dir, fname))},
                            step=global_step,
                        )

    end_time = time.time() if start_time is not None else None
    print("Training ended.")
    if start_time is not None:
        print("The training session started at", start_time, "and ended at", end_time)
        print("Took", end_time - start_time)
    return train_state, rb_state, key, metric_history, diag_rows_all


# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":

    grid = GridAgent(args)
    screen_size = args.env.screen_size
    if screen_size is None:
        screen_size = (grid.height, grid.width)
    else:
        screen_size = (args.env.screen_size, args.env.screen_size)

    Env = grid.env
    n_actions = Env.action_space.n
    seq_len = max(args.algo.n_step, args.laplacian.n_step) + 1

    run_tag = f"{args.env.maze_type}_seed{args.exp.seed}_steps{args.algo.tot_steps}_{int(time.time())}"
    out_root = os.path.join("artifacts", "grid_dqn_rnd_neo", run_tag)
    ensure_dir(out_root)

    if args.wb.on:
        create_wb_logger(args)
        wandb.save("src/networks/networks.py", policy="now")
        wandb.save("src/losses/laplacian.py", policy="now")
        wandb.save("src/mains/algorithms/Laplacian.py", policy="now")
        wandb.save("src/mains/algorithms/RND.py", policy="now")

    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    key = jax.random.PRNGKey(args.exp.seed)

    with jax.default_device(jax.devices(args.buffer.device)[0]):
        RB, RB_state = setup_rb(args, screen_size, seq_len)

    with jax.default_device(jax.devices("gpu")[0]):
        try:
            key, train_state = setup_train_state(key, Env, args, n_actions)
            train_state, RB_state, Key, metric_history, diag_rows = q_learning(
                train_state, RB_state, key, n_actions, Env, RB, out_root
            )
        except Exception:
            traceback.print_exc()
            if args.wb.on:
                wandb.finish(exit_code=1)
        finally:
            Env.close()
            if args.wb.on:
                close_logger()
            jax.clear_backends()
