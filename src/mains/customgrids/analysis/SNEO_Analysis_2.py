# true_source_count_progress_diagnostic_no_matching.py
#
# Purpose:
#   Train online DQN + RND + novelty-weighted ALLO/source-Laplacian
#   representation learning, and run a count-based mechanistic diagnostic
#   using finite-horizon tabular value iteration to solve the option-control
#   problem induced by each option reward:
#
#       Does count-augmented SNEO make more progress under the exact
#       count-source-Laplacian eigenfunction than NEO?
#
# Important design choice in this version:
#   - The TRUE source Laplacian used by the diagnostic is built from actual
#     tabular visitation counts N(s), not RND.
#   - The SNEO diagnostic rollout uses a count-based novelty bonus, not RND:
#         n_count(s) = 1 / sqrt(N(s)+1), normalized to [0,1].
#   - Dimension comparison is fixed order: learned dim d -> true dim d.
#     There is no dimension matching and no sign fixing.
#   - Quantitative metrics still evaluate all valid start states.
#
# Main mechanistic metrics:
#   method_true_progress:
#       phi_true_d(s_final_method) - phi_true_d(s_start). Higher is better.
#
#   method_true_step_agreement:
#       fraction of method rollout steps with positive progress under
#       the true source-Laplacian eigenfunction. Higher is better.
#
# Endpoint L1 distance is still saved as a debug column, but it is not used
# as the headline metric because it can penalize overshooting in the correct
# eigen-direction.

import os
import time
import random
import traceback
import argparse
import copy
from copy import deepcopy
from functools import partial
from typing import Any, Union, NamedTuple, Dict, List, Tuple
from dataclasses import replace, is_dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config.customgrids.laplacian import *
base_args = convert_args_to_dict(parse_args())

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import chex
import flax
import optax
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


# ============================================================
# Experiment-level constants
# ============================================================

DIAG_FRACTIONS = tuple(np.linspace(0.1, 1.0, 10))
PRIMARY_K = 3

TRAJ_HORIZON = 25
TRAJ_STALL_PATIENCE = 4
TRAJ_STALL_TOL = 1e-6

# Diagnostic option control. The diagnostic does not learn option policies;
# it solves the finite-horizon tabular option-control problem exactly by
# value iteration over the known grid graph.
USE_VALUE_ITERATION_ROLLOUTS = True
OPTION_VI_HORIZON = TRAJ_HORIZON
OPTION_VI_GAMMA = 0.99

RELIABLE_RND_BETA = 10.0
N_EVAL_DIMS = None
SKIP_DIM0 = True

# If True, the exact tabular source option climbs |phi|, matching the original
# tabular NEO paper. If False, it climbs signed phi, matching the learned
# intrinsic reward r_d(s,s') = phi_d(s') - phi_d(s).
USE_ABS_TRUE_OPTION = False

# Number of panel examples saved per snapshot.
N_VIS_DIMS = 3
N_VIS_STARTS = 3

# Numerical tolerance for source-Laplacian eigenvectors.
EIG_IMAG_TOL = 1e-6

# Snapshot source-Laplacian zeta scaling.
SOURCE_ZETA_DELTA = 0.5
SOURCE_ZETA_K = 1.0
SOURCE_ZETA_EPS = 1e-6
SOURCE_ZETA_CAP = 1e4


# ============================================================
# Dataclasses
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
    observation: chex.Array
    next_observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    rnd_bonus: chex.Array


class LapBatch(NamedTuple):
    obs: jnp.ndarray
    dones: jnp.ndarray
    actions: jnp.ndarray
    novelty: jnp.ndarray
    counts: jnp.ndarray


class FlatStorage(struct.PyTreeNode):
    rnd_frame: chex.Array


# ============================================================
# General utilities
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rgb_to_gray_uint8(obs_rgb: np.ndarray) -> np.ndarray:
    obs = obs_rgb.astype(np.float32)
    gray = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray[..., None]


def make_eval_points(total_steps: int) -> List[int]:
    return sorted({max(1, int(round(frac * total_steps))) for frac in DIAG_FRACTIONS})


def set_seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def copy_args_with_seed(args, seed: int):
    new_args = copy.deepcopy(args)

    if is_dataclass(new_args.exp):
        new_exp = replace(new_args.exp, seed=int(seed))
    else:
        new_exp = copy.deepcopy(new_args.exp)
        new_exp.seed = int(seed)

    if is_dataclass(new_args):
        new_args = replace(new_args, exp=new_exp)
    else:
        try:
            new_args.exp = new_exp
        except Exception:
            object.__setattr__(new_args, "exp", new_exp)

    return new_args


def get_robot_coord(env) -> Tuple[int, int]:
    """Returns current robot/grid coordinate from the custom GridAgent env."""
    return tuple(np.asarray(env.unwrapped.maze.robot, dtype=int))


def safe_count_current_state(env, ij_to_i: Dict[Tuple[int, int], int], visit_counts: np.ndarray):
    """Tracks actual tabular counts for count-based diagnostics."""
    try:
        coord = get_robot_coord(env)
        if coord in ij_to_i:
            visit_counts[ij_to_i[coord]] += 1.0
    except Exception:
        pass


# ============================================================
# Replay buffer
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
# Q functions and updates
# ============================================================

@partial(jax.jit, static_argnums=(3,))
def get_q_values(obs, enc_state, q_state, use_target: bool = False):
    enc_p = enc_state.target_params if use_target else enc_state.params
    q_p = q_state.target_params if use_target else q_state.params
    embeddings = enc_state.apply_fn({"params": enc_p}, obs, train=False)
    q_values = q_state.apply_fn({"params": q_p}, embeddings, train=False)
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
    enc_state = enc_state.replace(target_params=optax.incremental_update(enc_state.params, enc_state.target_params, tau))
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, tau))
    return enc_state, q_state


def make_update_q_network(args):

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

        (_, aux), grads = jax.value_and_grad(td_loss_fn, has_aux=True, argnums=(0, 1))(enc_state.params, q_state.params)
        td_loss, q_mean, q_min, q_max, new_priorities = aux
        grads_enc, grads_q = grads

        enc_state = enc_state.apply_gradients(grads=grads_enc)
        q_state = q_state.apply_gradients(grads=grads_q)

        return enc_state, (q_state, td_loss, q_mean, q_min, q_max, new_priorities)

    return update_q_network


# ============================================================
# Laplacian / RND update
# ============================================================

@jax.jit
def compute_laplace_neo_targets_from_obs_seq(train_state: Train_State, obs_seq: jnp.ndarray):
    B, T = obs_seq.shape[:2]
    flat = obs_seq.reshape((B * T,) + obs_seq.shape[2:]).astype(jnp.float32)

    raw_bonus_flat = rnd_intrinsic_reward(train_state.rnd_state, flat)
    raw_bonus = raw_bonus_flat.reshape(B, T)

    rnd_bonus = raw_bonus / jnp.sqrt(train_state.rnd_state.rew_rms.var + 1e-8)
    rnd_bonus_n = rnd_bonus / (train_state.rnd_state.rnd_bonus_max_ema + 1e-8)

    inv_count = 1.0 / (rnd_bonus_n + 1e-6)
    inv_used = jnp.minimum(inv_count, 1e4)

    return raw_bonus, rnd_bonus, rnd_bonus_n, inv_count, inv_used


def make_update_grid_neo_laplacian_network(args):

    @partial(jax.jit, static_argnums=(3,))
    def update_grid_neo_laplacian_network(key, train_state, lap_batch, n_actions):
        gamma = args.env.gamma
        use_bn = args.laplacian.batch_norm
        use_barrier = args.laplacian.barrier.on
        use_dual = args.laplacian.dual.on
        lap_state = train_state.lap_state

        obs = lap_batch.obs
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

            loss, (pos_loss, ortho_loss, dual_loss), (_, err_mat) = multi_step_ALLO(
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

        (eigen_loss, aux), grads = jax.value_and_grad(laplacian_loss_fn, has_aux=True)(lap_state.params)
        updates, pos_loss, ortho_loss, dual_loss, curr_err_mat, lambda_batch = aux

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

    return update_grid_neo_laplacian_network


# ============================================================
# Full state extraction
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


def make_compute_all_state_embeddings(args):

    @jax.jit
    def compute_all_state_embeddings(lap_state: Lap_State, state_imgs: jnp.ndarray):
        vars_l = {"params": lap_state.params}
        if args.laplacian.batch_norm:
            vars_l["batch_stats"] = lap_state.batch_stats
        return lap_state.apply_fn(vars_l, state_imgs, train=False)

    return compute_all_state_embeddings


@jax.jit
def compute_all_state_novelty(rnd_state: RNDState, state_imgs: jnp.ndarray):
    return rnd_intrinsic_reward(rnd_state, state_imgs.astype(jnp.float32))


# ============================================================
# Exact source-Laplacian diagnostics
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


def build_transition_matrix_from_neighbors(neighbors: Dict[int, List[int]], n_states: int) -> np.ndarray:
    P = np.zeros((n_states, n_states), dtype=np.float64)

    for i in range(n_states):
        neigh = neighbors[i]
        if len(neigh) == 0:
            P[i, i] = 1.0
        else:
            p = 1.0 / float(len(neigh))
            for j in neigh:
                P[i, j] = p

    return P


def normalize_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - np.mean(x, axis=0, keepdims=True)) / (np.std(x, axis=0, keepdims=True) + eps)


def compute_snapshot_zeta_from_rnd(
    novelty_raw: np.ndarray,
    neo_state: NEOScaleState = None,
    delta: float = SOURCE_ZETA_DELTA,
    k: float = SOURCE_ZETA_K,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    novelty_raw = np.asarray(novelty_raw, dtype=np.float64)
    novelty_n = novelty_raw / (np.max(novelty_raw) + SOURCE_ZETA_EPS)

    inv_count = 1.0 / (novelty_n + SOURCE_ZETA_EPS)
    inv_count = np.minimum(inv_count, SOURCE_ZETA_CAP)

    if neo_state is not None:
        try:
            max_inv = float(jax.device_get(neo_state.max_inv_ema))
            max_inv = max(max_inv, SOURCE_ZETA_EPS)
        except Exception:
            max_inv = float(np.max(inv_count) + SOURCE_ZETA_EPS)
    else:
        max_inv = float(np.max(inv_count) + SOURCE_ZETA_EPS)

    zeta = delta * np.power(np.clip(inv_count / max_inv, 0.0, 1.0), 1.0 / k)
    return zeta.astype(np.float64), novelty_n.astype(np.float64), inv_count.astype(np.float64)


def compute_count_novelty_and_zeta(
    visit_counts: np.ndarray,
    n_states: int,
    delta: float = SOURCE_ZETA_DELTA,
    k: float = SOURCE_ZETA_K,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds the exact count-based quantities used by the diagnostic.

    Source-Laplacian weights:
        zeta(s) = delta * (N(s) / max_s N(s))^(1/k)

    Count novelty for the SNEO diagnostic bonus:
        n_count(s) = 1 / sqrt(N(s)+1), normalized by its maximum.

    With this convention, unvisited or rarely visited states have high
    n_count, while frequently visited states receive high zeta.
    """
    if visit_counts is None:
        counts = np.zeros((n_states,), dtype=np.float64)
    else:
        counts = np.asarray(visit_counts, dtype=np.float64).reshape(-1)
        if counts.shape[0] != n_states:
            raise ValueError(
                f"visit_counts has length {counts.shape[0]}, expected {n_states}. "
                "The state enumeration used for counts and diagnostics must match."
            )

    max_count = float(np.max(counts))
    if max_count <= 0.0:
        # Before any meaningful count information exists, use a neutral source
        # Laplacian and uniform novelty.
        zeta = np.zeros((n_states,), dtype=np.float64)
        count_novelty = np.ones((n_states,), dtype=np.float64)
        count_norm = np.zeros((n_states,), dtype=np.float64)
        return zeta, count_novelty, count_norm

    count_norm = np.clip(counts / (max_count + eps), 0.0, 1.0)
    zeta = delta * np.power(count_norm, 1.0 / k)

    count_novelty = 1.0 / np.sqrt(counts + 1.0)
    count_novelty = count_novelty / (np.max(count_novelty) + eps)

    return zeta.astype(np.float64), count_novelty.astype(np.float64), count_norm.astype(np.float64)


def compute_true_source_laplacian_eigenvectors(P: np.ndarray, zeta: np.ndarray, n_eigs: int) -> Tuple[np.ndarray, np.ndarray]:
    n = P.shape[0]
    L = np.eye(n, dtype=np.float64) + np.diag(zeta.astype(np.float64)) - P.astype(np.float64)

    eigvals, eigvecs = np.linalg.eig(L)
    order = np.argsort(np.real(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    real_eigvals = []
    real_eigvecs = []

    for i in range(eigvals.shape[0]):
        val = eigvals[i]
        vec = eigvecs[:, i]

        if abs(np.imag(val)) <= EIG_IMAG_TOL and np.max(np.abs(np.imag(vec))) <= 1e-4:
            real_eigvals.append(float(np.real(val)))
            real_eigvecs.append(np.real(vec))
        else:
            # Fallback for mildly non-normal/directed cases.
            real_eigvals.append(float(np.real(val)))
            real_eigvecs.append(np.real(vec))

        if len(real_eigvals) >= n_eigs:
            break

    true_vals = np.asarray(real_eigvals, dtype=np.float64)
    true_phi = np.stack(real_eigvecs, axis=1).astype(np.float64)
    true_phi = normalize_columns(true_phi)
    return true_vals, true_phi


def choose_eval_dims(phi_dim: int, n_options: int) -> List[int]:
    n_eval = n_options if N_EVAL_DIMS is None else N_EVAL_DIMS
    if SKIP_DIM0:
        return list(range(1, min(n_eval + 1, phi_dim)))
    return list(range(min(n_eval, phi_dim)))


def build_fixed_dim_map_no_alignment(
    true_phi: np.ndarray,
    learned_phi: np.ndarray,
    eval_dims: List[int],
) -> Tuple[Dict[int, int], Dict[int, float], np.ndarray]:
    """
    Fixed-order comparison with no matching and no sign fixing.

    learned dimension d is compared directly to true dimension d.
    The returned true_phi is the original true_phi, unchanged.

    The correlation is logged only as a diagnostic; it is not used to
    permute dimensions or flip signs.
    """
    true_norm = normalize_columns(true_phi)
    learned_norm = normalize_columns(learned_phi)

    dim_to_true_dim = {}
    dim_to_corr = {}

    max_dim = min(true_phi.shape[1], learned_phi.shape[1])
    for d in eval_dims:
        if d >= max_dim:
            continue
        corr = float(np.mean(learned_norm[:, d] * true_norm[:, d]))
        dim_to_true_dim[int(d)] = int(d)
        dim_to_corr[int(d)] = float(corr)

    return dim_to_true_dim, dim_to_corr, true_phi


def option_score_vanilla(phi: np.ndarray, s: int, sp: int, dim: int) -> float:
    return float(phi[sp, dim] - phi[s, dim])


def option_score_reliable(phi: np.ndarray, novelty_n: np.ndarray, s: int, sp: int, dim: int, beta: float) -> float:
    return float(phi[sp, dim] - phi[s, dim] + beta * novelty_n[sp])


def option_score_true(phi_true: np.ndarray, s: int, sp: int, dim: int, use_abs: bool = USE_ABS_TRUE_OPTION) -> float:
    if use_abs:
        return float(abs(phi_true[sp, dim]) - abs(phi_true[s, dim]))
    return float(phi_true[sp, dim] - phi_true[s, dim])


def greedy_option_rollout(
    start_idx: int,
    dim: int,
    phi: np.ndarray,
    novelty_n: np.ndarray,
    neighbors: Dict[int, List[int]],
    coords: np.ndarray,
    method: str,
    beta: float,
    horizon: int = TRAJ_HORIZON,
    stall_patience: int = TRAJ_STALL_PATIENCE,
    stall_tol: float = TRAJ_STALL_TOL,
):
    assert method in ("TRUE", "NEO", "SNEO")

    path = [int(start_idx)]
    unique = {int(start_idx)}
    stall_ctr = 0
    curr = int(start_idx)

    for _ in range(horizon):
        cand = neighbors[curr]
        if len(cand) == 0:
            break

        if method == "TRUE":
            scores = np.asarray([option_score_true(phi, curr, sp, dim) for sp in cand], dtype=np.float64)
        elif method == "SNEO":
            scores = np.asarray([option_score_reliable(phi, novelty_n, curr, sp, dim, beta) for sp in cand], dtype=np.float64)
        else:
            scores = np.asarray([option_score_vanilla(phi, curr, sp, dim) for sp in cand], dtype=np.float64)

        best_local = int(np.argmax(scores))
        nxt = int(cand[best_local])
        gain = float(scores[best_local])

        if gain <= stall_tol:
            stall_ctr += 1
        else:
            stall_ctr = 0

        curr = nxt
        path.append(curr)
        unique.add(curr)

        if stall_ctr >= stall_patience:
            break

    start_xy = coords[path[0]]
    max_l1_dist = 0
    for idx in path:
        max_l1_dist = max(max_l1_dist, int(np.abs(coords[idx] - start_xy).sum()))

    return {
        "path": path,
        "duration": len(path) - 1,
        "unique_states": len(unique),
        "end_idx": int(path[-1]),
        "end_l1_from_start": int(np.abs(coords[path[-1]] - start_xy).sum()),
        "max_l1_from_start": int(max_l1_dist),
    }




def option_edge_reward(
    phi: np.ndarray,
    novelty_n: np.ndarray,
    s: int,
    sp: int,
    dim: int,
    method: str,
    beta: float,
) -> float:
    """
    One-step option reward used by the diagnostic tabular planner.

    TRUE/NEO:
        phi(sp, d) - phi(s, d)

    SNEO:
        phi(sp, d) - phi(s, d) + beta * novelty(sp)

    TRUE uses this same function with the exact source eigenfunction and
    method="TRUE"; SNEO uses the learned eigenfunction and count novelty.
    """
    if method == "TRUE":
        return option_score_true(phi, s, sp, dim)

    base = float(phi[int(sp), dim] - phi[int(s), dim])
    if method == "SNEO":
        return base + beta * float(novelty_n[int(sp)])
    if method == "NEO":
        return base
    raise ValueError(f"Unknown method: {method}")


def finite_horizon_option_value_iteration(
    phi: np.ndarray,
    novelty_n: np.ndarray,
    neighbors: Dict[int, List[int]],
    dim: int,
    method: str,
    beta: float,
    horizon: int = OPTION_VI_HORIZON,
    gamma: float = OPTION_VI_GAMMA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finite-horizon tabular value iteration for an option reward.

    V[h, s] is the optimal h-step value from state s.
    policy[h, s] is the best next state to take with h steps remaining.
    q_best[h, s] is the selected one-step Q value.

    This is not a learned option policy. It is an oracle planner used only
    for a mechanistic diagnostic: it asks what a well-trained option policy
    would do if optimized for the supplied option reward on the known graph.
    """
    assert method in ("TRUE", "NEO", "SNEO")

    n_states = phi.shape[0]
    horizon = int(horizon)
    gamma = float(gamma)

    V = np.zeros((horizon + 1, n_states), dtype=np.float64)
    policy = np.tile(np.arange(n_states, dtype=np.int64)[None, :], (horizon + 1, 1))
    q_best = np.zeros((horizon + 1, n_states), dtype=np.float64)

    for h in range(1, horizon + 1):
        for s in range(n_states):
            cand = neighbors.get(int(s), [])
            if len(cand) == 0:
                policy[h, s] = int(s)
                V[h, s] = 0.0
                q_best[h, s] = 0.0
                continue

            q_vals = []
            for sp in cand:
                r = option_edge_reward(
                    phi=phi,
                    novelty_n=novelty_n,
                    s=int(s),
                    sp=int(sp),
                    dim=dim,
                    method=method,
                    beta=beta,
                )
                q_vals.append(r + gamma * V[h - 1, int(sp)])

            q_vals = np.asarray(q_vals, dtype=np.float64)
            best_j = int(np.argmax(q_vals))
            best_sp = int(cand[best_j])
            policy[h, s] = best_sp
            V[h, s] = float(q_vals[best_j])
            q_best[h, s] = float(q_vals[best_j])

    return V, policy, q_best


def vi_option_rollout(
    start_idx: int,
    dim: int,
    phi: np.ndarray,
    novelty_n: np.ndarray,
    neighbors: Dict[int, List[int]],
    coords: np.ndarray,
    method: str,
    beta: float,
    horizon: int = OPTION_VI_HORIZON,
    gamma: float = OPTION_VI_GAMMA,
    precomputed: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
):
    """
    Roll out the finite-horizon value-iteration policy.

    At time t, the action is selected from policy[remaining_steps, s_t], so
    future novelty rewards are backed up into earlier decisions.
    """
    if precomputed is None:
        V, policy, q_best = finite_horizon_option_value_iteration(
            phi=phi,
            novelty_n=novelty_n,
            neighbors=neighbors,
            dim=dim,
            method=method,
            beta=beta,
            horizon=horizon,
            gamma=gamma,
        )
    else:
        V, policy, q_best = precomputed

    path = [int(start_idx)]
    curr = int(start_idx)
    unique = {curr}

    for remaining in range(int(horizon), 0, -1):
        nxt = int(policy[remaining, curr])
        if nxt == curr:
            break
        path.append(nxt)
        unique.add(nxt)
        curr = nxt

    start_xy = coords[path[0]]
    max_l1_dist = 0
    for idx in path:
        max_l1_dist = max(max_l1_dist, int(np.abs(coords[int(idx)] - start_xy).sum()))

    return {
        "path": path,
        "duration": len(path) - 1,
        "unique_states": len(unique),
        "end_idx": int(path[-1]),
        "end_l1_from_start": int(np.abs(coords[path[-1]] - start_xy).sum()),
        "max_l1_from_start": int(max_l1_dist),
        "planned_value": float(V[int(horizon), int(start_idx)]),
    }


def rollout_option_policy(
    start_idx: int,
    dim: int,
    phi: np.ndarray,
    novelty_n: np.ndarray,
    neighbors: Dict[int, List[int]],
    coords: np.ndarray,
    method: str,
    beta: float,
    horizon: int = OPTION_VI_HORIZON,
    gamma: float = OPTION_VI_GAMMA,
    precomputed: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
):
    """
    Dispatches between oracle value-iteration rollouts and the old local
    greedy rollout. The default is finite-horizon value iteration.
    """
    if USE_VALUE_ITERATION_ROLLOUTS:
        return vi_option_rollout(
            start_idx=start_idx,
            dim=dim,
            phi=phi,
            novelty_n=novelty_n,
            neighbors=neighbors,
            coords=coords,
            method=method,
            beta=beta,
            horizon=horizon,
            gamma=gamma,
            precomputed=precomputed,
        )

    out = greedy_option_rollout(
        start_idx=start_idx,
        dim=dim,
        phi=phi,
        novelty_n=novelty_n,
        neighbors=neighbors,
        coords=coords,
        method=method,
        beta=beta,
        horizon=horizon,
    )
    out["planned_value"] = np.nan
    return out

def l1_coord_distance(coords: np.ndarray, i: int, j: int) -> int:
    return int(np.abs(coords[int(i)] - coords[int(j)]).sum())


def rollout_true_potential_stats(
    path: List[int],
    true_phi: np.ndarray,
    true_dim: int,
) -> Dict[str, float]:
    """
    Scores a rollout path under the true source-Laplacian eigenfunction.

    Two main metrics are returned:
      1. true_progress: endpoint progress in true eigen-potential.
      2. true_step_agreement: fraction of rollout transitions that move in
         positive true-eigenfunction direction.

    Endpoint progress is not an endpoint imitation metric; it rewards moving
    farther in the true eigen-direction even if the endpoint differs from the
    exact greedy endpoint.
    """
    if len(path) == 0:
        return {
            "true_progress": 0.0,
            "true_pos_progress": 0.0,
            "true_neg_progress": 0.0,
            "true_step_agreement": 0.0,
            "n_steps_scored": 0,
            "true_start_value": 0.0,
            "true_end_value": 0.0,
        }

    s0 = int(path[0])
    sT = int(path[-1])

    if len(path) <= 1:
        val = float(true_phi[s0, true_dim])
        return {
            "true_progress": 0.0,
            "true_pos_progress": 0.0,
            "true_neg_progress": 0.0,
            "true_step_agreement": 0.0,
            "n_steps_scored": 0,
            "true_start_value": val,
            "true_end_value": val,
        }

    deltas = np.asarray(
        [float(true_phi[int(b), true_dim] - true_phi[int(a), true_dim])
         for a, b in zip(path[:-1], path[1:])],
        dtype=np.float64,
    )

    return {
        "true_progress": float(true_phi[sT, true_dim] - true_phi[s0, true_dim]),
        "true_pos_progress": float(np.maximum(deltas, 0.0).sum()),
        "true_neg_progress": float(np.maximum(-deltas, 0.0).sum()),
        "true_step_agreement": float(np.mean(deltas > 0.0)),
        "n_steps_scored": int(len(deltas)),
        "true_start_value": float(true_phi[s0, true_dim]),
        "true_end_value": float(true_phi[sT, true_dim]),
    }


# ============================================================
# Plotting utilities
# ============================================================

def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


PLOT_FONTS = {"title": 20, "axis_label": 18, "tick": 15, "legend": 15}


def apply_plot_style():
    plt.rcParams.update({
        "font.size": PLOT_FONTS["tick"],
        "axes.titlesize": PLOT_FONTS["title"],
        "axes.labelsize": PLOT_FONTS["axis_label"],
        "xtick.labelsize": PLOT_FONTS["tick"],
        "ytick.labelsize": PLOT_FONTS["tick"],
        "legend.fontsize": PLOT_FONTS["legend"],
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })


def values_to_grid(coords: np.ndarray, values: np.ndarray, fill_value=np.nan) -> Tuple[np.ndarray, int, int]:
    coords = np.asarray(coords, dtype=int)
    values = np.asarray(values)

    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    grid = np.full((h, w), fill_value, dtype=np.float64)

    for idx, (r, c) in enumerate(coords):
        grid[int(r), int(c)] = float(values[idx])

    return grid, h, w


def path_to_xy(coords: np.ndarray, path: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    pts = coords[np.asarray(path, dtype=int)]
    y = pts[:, 0]
    x = pts[:, 1]
    return x, y


def set_imshow_cell_limits(ax, grid: np.ndarray):
    """Avoids apparent edge clipping from imshow + tight bbox/colorbars."""
    h, w = grid.shape
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")


def plot_heatmap_with_optional_path(
    coords: np.ndarray,
    values: np.ndarray,
    save_path: str,
    title: str,
    path: List[int] = None,
    start_idx: int = None,
    end_idx: int = None,
    target_idx: int = None,
    cmap: str = "viridis",
):
    grid, _, _ = values_to_grid(coords, values)

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    im = ax.imshow(grid, origin="upper", cmap=cmap, interpolation="nearest")
    set_imshow_cell_limits(ax, grid)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if path is not None and len(path) > 0:
        x, y = path_to_xy(coords, path)
        ax.plot(x, y, linewidth=3.0, marker="o", markersize=4, label="rollout")

    if start_idx is not None:
        sx, sy = path_to_xy(coords, [start_idx])
        ax.scatter(sx, sy, s=100, marker="o", edgecolors="black", linewidths=1.5, label="start")

    if end_idx is not None:
        ex, ey = path_to_xy(coords, [end_idx])
        ax.scatter(ex, ey, s=130, marker="X", edgecolors="black", linewidths=1.5, label="end")

    if target_idx is not None:
        tx, ty = path_to_xy(coords, [target_idx])
        ax.scatter(tx, ty, s=160, marker="*", edgecolors="black", linewidths=1.5, label="true end")

    ax.set_title(title, fontsize=PLOT_FONTS["title"], pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="best", fontsize=PLOT_FONTS["legend"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_mechanism_panel(
    coords: np.ndarray,
    true_values: np.ndarray,
    learned_values: np.ndarray,
    neo_path: List[int],
    sneo_path: List[int],
    true_path: List[int],
    start_idx: int,
    save_path: str,
    title: str,
):
    true_grid, _, _ = values_to_grid(coords, true_values)
    learned_grid, _, _ = values_to_grid(coords, learned_values)

    true_end = true_path[-1]
    neo_end = neo_path[-1]
    sneo_end = sneo_path[-1]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), constrained_layout=False)

    panels = [
        ("True source eigenvector", true_grid, None, None),
        ("Learned pixel eigenvector", learned_grid, None, None),
        ("NEO on learned eigenvector", learned_grid, neo_path, neo_end),
        ("SNEO on learned eigenvector", learned_grid, sneo_path, sneo_end),
    ]

    for ax, (panel_title, grid, path, end_idx) in zip(axes, panels):
        im = ax.imshow(grid, origin="upper", cmap="viridis", interpolation="nearest")
        set_imshow_cell_limits(ax, grid)

        sx, sy = path_to_xy(coords, [start_idx])
        tx, ty = path_to_xy(coords, [true_end])
        ax.scatter(sx, sy, s=90, marker="o", edgecolors="black", linewidths=1.2, label="start")
        ax.scatter(tx, ty, s=150, marker="*", edgecolors="black", linewidths=1.2, label="true end")

        if path is not None:
            x, y = path_to_xy(coords, path)
            ax.plot(x, y, linewidth=3.0, marker="o", markersize=4)
            ex, ey = path_to_xy(coords, [end_idx])
            ax.scatter(ex, ey, s=120, marker="X", edgecolors="black", linewidths=1.2, label="method end")

        ax.set_title(panel_title, fontsize=PLOT_FONTS["title"])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=PLOT_FONTS["title"] + 2, y=1.04)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.015)
    cbar.ax.tick_params(labelsize=PLOT_FONTS["tick"])

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, fontsize=PLOT_FONTS["legend"])

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)




def policy_to_uv_grid(
    coords: np.ndarray,
    policy: np.ndarray,
    horizon: int,
    fill_value: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts a next-state policy into quiver U,V grids."""
    coords = np.asarray(coords, dtype=int)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1

    U = np.full((h, w), fill_value, dtype=np.float64)
    V = np.full((h, w), fill_value, dtype=np.float64)
    mask = np.zeros((h, w), dtype=bool)

    for s, (r, c) in enumerate(coords):
        sp = int(policy[int(horizon), int(s)])
        r2, c2 = coords[sp]
        U[int(r), int(c)] = float(c2 - c)
        V[int(r), int(c)] = float(r2 - r)
        mask[int(r), int(c)] = True

    return U, V, mask


def plot_policy_on_heatmap(
    ax,
    coords: np.ndarray,
    values: np.ndarray,
    policy: np.ndarray,
    horizon: int,
    title: str,
    cmap: str = "viridis",
    path: List[int] = None,
    start_idx: int = None,
):
    grid, h, w = values_to_grid(coords, values)
    im = ax.imshow(grid, origin="upper", cmap=cmap, interpolation="nearest")
    set_imshow_cell_limits(ax, grid)

    U, V, mask = policy_to_uv_grid(coords, policy, horizon)
    yy, xx = np.mgrid[0:h, 0:w]
    ax.quiver(
        xx[mask],
        yy[mask],
        U[mask],
        V[mask],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.006,
        headwidth=4.0,
        headlength=5.0,
        headaxislength=4.5,
        pivot="middle",
    )

    if path is not None and len(path) > 0:
        x, y = path_to_xy(coords, path)
        ax.plot(x, y, linewidth=3.0, marker="o", markersize=4)

    if start_idx is not None:
        sx, sy = path_to_xy(coords, [start_idx])
        ax.scatter(sx, sy, s=110, marker="o", edgecolors="black", linewidths=1.4, label="start")

    ax.set_title(title, fontsize=PLOT_FONTS["title"])
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_policy_comparison_panel(
    coords: np.ndarray,
    true_values: np.ndarray,
    learned_values: np.ndarray,
    true_policy: np.ndarray,
    neo_policy: np.ndarray,
    sneo_policy: np.ndarray,
    true_path: List[int],
    neo_path: List[int],
    sneo_path: List[int],
    start_idx: int,
    horizon: int,
    save_path: str,
    title: str,
):
    """
    Saves a three-panel policy visualization:
      1. TRUE policy over true source eigenvector.
      2. NEO policy over learned eigenvector.
      3. SNEO policy over learned eigenvector.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=False)

    im = plot_policy_on_heatmap(
        axes[0], coords, true_values, true_policy, horizon,
        title="TRUE VI policy", cmap="viridis", path=true_path, start_idx=start_idx,
    )
    plot_policy_on_heatmap(
        axes[1], coords, learned_values, neo_policy, horizon,
        title="NEO VI policy", cmap="viridis", path=neo_path, start_idx=start_idx,
    )
    plot_policy_on_heatmap(
        axes[2], coords, learned_values, sneo_policy, horizon,
        title="SNEO VI policy", cmap="viridis", path=sneo_path, start_idx=start_idx,
    )

    fig.suptitle(title, fontsize=PLOT_FONTS["title"] + 2, y=1.04)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.015)
    cbar.ax.tick_params(labelsize=PLOT_FONTS["tick"])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, fontsize=PLOT_FONTS["legend"])

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

# ============================================================
# Main true-endpoint diagnostic
# ============================================================

def run_true_endpoint_diagnostics(
    *,
    seed: int,
    global_step: int,
    train_state: Train_State,
    env,
    args,
    visit_counts: np.ndarray = None,
    out_dir: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    apply_plot_style()

    state_imgs_u8, coords, ij_to_i, _ = get_all_state_pixels(env)
    state_imgs = state_imgs_u8.astype(np.float32)
    n_states = state_imgs.shape[0]

    neigh = build_neighbor_index(env, ij_to_i)
    P = build_transition_matrix_from_neighbors(neigh, n_states)

    compute_embeddings_fn = make_compute_all_state_embeddings(args)
    learned_phi = np.asarray(
        jax.device_get(compute_embeddings_fn(train_state.lap_state, jnp.asarray(state_imgs)))
    ).astype(np.float64)

    # Count-based diagnostic quantities. No RND is used for the TRUE source
    # Laplacian or for the SNEO diagnostic bonus in this version.
    zeta, novelty_n, count_norm = compute_count_novelty_and_zeta(
        visit_counts=visit_counts,
        n_states=n_states,
        delta=SOURCE_ZETA_DELTA,
        k=SOURCE_ZETA_K,
    )

    eval_dims = choose_eval_dims(learned_phi.shape[1], args.algo.n_options)
    max_needed_true_dim = max(eval_dims) + 2 if len(eval_dims) > 0 else args.algo.n_options + 2
    true_vals, true_phi = compute_true_source_laplacian_eigenvectors(
        P=P,
        zeta=zeta,
        n_eigs=max(max_needed_true_dim, args.algo.n_options + 2),
    )

    dim_to_true_dim, dim_to_corr, true_phi_for_eval = build_fixed_dim_map_no_alignment(
        true_phi=true_phi,
        learned_phi=learned_phi,
        eval_dims=eval_dims,
    )
    eval_dims = [d for d in eval_dims if d in dim_to_true_dim]

    option_gamma = float(OPTION_VI_GAMMA)
    option_horizon = int(OPTION_VI_HORIZON)

    raw_rows = []

    for learned_dim in eval_dims:
        true_dim = dim_to_true_dim[learned_dim]
        align_corr = dim_to_corr[learned_dim]

        # Precompute VI policies once per dimension/method.
        true_vi = finite_horizon_option_value_iteration(
            phi=true_phi_for_eval,
            novelty_n=np.zeros_like(novelty_n),
            neighbors=neigh,
            dim=true_dim,
            method="TRUE",
            beta=0.0,
            horizon=option_horizon,
            gamma=option_gamma,
        )
        neo_vi = finite_horizon_option_value_iteration(
            phi=learned_phi,
            novelty_n=novelty_n,
            neighbors=neigh,
            dim=learned_dim,
            method="NEO",
            beta=0.0,
            horizon=option_horizon,
            gamma=option_gamma,
        )
        sneo_vi = finite_horizon_option_value_iteration(
            phi=learned_phi,
            novelty_n=novelty_n,
            neighbors=neigh,
            dim=learned_dim,
            method="SNEO",
            beta=RELIABLE_RND_BETA,
            horizon=option_horizon,
            gamma=option_gamma,
        )

        for start_idx in range(n_states):
            true_rollout = rollout_option_policy(
                start_idx=start_idx,
                dim=true_dim,
                phi=true_phi_for_eval,
                novelty_n=np.zeros_like(novelty_n),
                neighbors=neigh,
                coords=coords,
                method="TRUE",
                beta=0.0,
                horizon=option_horizon,
                gamma=option_gamma,
                precomputed=true_vi,
            )
            true_end = true_rollout["end_idx"]

            for method, vi_pack, beta_used in (
                ("NEO", neo_vi, 0.0),
                ("SNEO", sneo_vi, RELIABLE_RND_BETA),
            ):
                method_rollout = rollout_option_policy(
                    start_idx=start_idx,
                    dim=learned_dim,
                    phi=learned_phi,
                    novelty_n=novelty_n,
                    neighbors=neigh,
                    coords=coords,
                    method=method,
                    beta=beta_used,
                    horizon=option_horizon,
                    gamma=option_gamma,
                    precomputed=vi_pack,
                )

                method_end = method_rollout["end_idx"]
                endpoint_l1 = l1_coord_distance(coords, method_end, true_end)
                endpoint_exact = float(method_end == true_end)

                true_stats = rollout_true_potential_stats(
                    path=true_rollout["path"],
                    true_phi=true_phi_for_eval,
                    true_dim=true_dim,
                )
                method_stats = rollout_true_potential_stats(
                    path=method_rollout["path"],
                    true_phi=true_phi_for_eval,
                    true_dim=true_dim,
                )

                raw_rows.append({
                    "seed": int(seed),
                    "global_step": int(global_step),
                    "method": method,
                    "control": "finite_horizon_value_iteration" if USE_VALUE_ITERATION_ROLLOUTS else "local_greedy",
                    "learned_dim": int(learned_dim),
                    "true_dim": int(true_dim),
                    "true_eigval": float(true_vals[true_dim]),
                    "fixed_order_corr": float(align_corr),
                    "start_idx": int(start_idx),
                    "true_end_idx": int(true_end),
                    "method_end_idx": int(method_end),

                    # Debug endpoint-imitation metric. Not used as headline.
                    "endpoint_l1_to_true": float(endpoint_l1),
                    "endpoint_exact_match": float(endpoint_exact),

                    # Main mechanistic metrics.
                    "true_option_progress": float(true_stats["true_progress"]),
                    "method_true_progress": float(method_stats["true_progress"]),
                    "method_true_pos_progress": float(method_stats["true_pos_progress"]),
                    "method_true_neg_progress": float(method_stats["true_neg_progress"]),
                    "method_true_step_agreement": float(method_stats["true_step_agreement"]),
                    "true_progress_regret": float(true_stats["true_progress"] - method_stats["true_progress"]),

                    "true_duration": float(true_rollout["duration"]),
                    "method_duration": float(method_rollout["duration"]),
                    "true_unique_states": float(true_rollout["unique_states"]),
                    "method_unique_states": float(method_rollout["unique_states"]),
                    "true_max_l1_from_start": float(true_rollout["max_l1_from_start"]),
                    "method_max_l1_from_start": float(method_rollout["max_l1_from_start"]),
                    "true_planned_value": float(true_rollout.get("planned_value", np.nan)),
                    "method_planned_value": float(method_rollout.get("planned_value", np.nan)),
                    "option_horizon": int(option_horizon),
                    "option_gamma": float(option_gamma),
                    "zeta_mean": float(np.mean(zeta)),
                    "zeta_min": float(np.min(zeta)),
                    "zeta_max": float(np.max(zeta)),
                    "count_novelty_mean": float(np.mean(novelty_n)),
                    "count_novelty_min": float(np.min(novelty_n)),
                    "count_novelty_max": float(np.max(novelty_n)),
                })

    rows_df = pd.DataFrame(raw_rows)

    summary_rows = []
    for method in ("NEO", "SNEO"):
        mdf = rows_df[rows_df["method"] == method]
        summary_rows.append({
            "seed": int(seed),
            "global_step": int(global_step),
            "method": method,

            # Main mechanistic metrics.
            "method_true_progress_mean": float(mdf["method_true_progress"].mean()),
            "method_true_progress_sem_over_dims_starts": sem(mdf["method_true_progress"].to_numpy()),
            "method_true_step_agreement_mean": float(mdf["method_true_step_agreement"].mean()),
            "method_true_step_agreement_sem_over_dims_starts": sem(mdf["method_true_step_agreement"].to_numpy()),

            # Useful supporting diagnostics.
            "method_true_pos_progress_mean": float(mdf["method_true_pos_progress"].mean()),
            "method_true_neg_progress_mean": float(mdf["method_true_neg_progress"].mean()),
            "true_progress_regret_mean": float(mdf["true_progress_regret"].mean()),
            "true_option_progress_mean": float(mdf["true_option_progress"].mean()),
            "method_planned_value_mean": float(mdf["method_planned_value"].mean()),

            # Endpoint metrics retained for debugging only.
            "endpoint_l1_to_true_mean": float(mdf["endpoint_l1_to_true"].mean()),
            "endpoint_l1_to_true_sem_over_dims_starts": sem(mdf["endpoint_l1_to_true"].to_numpy()),
            "endpoint_exact_match_mean": float(mdf["endpoint_exact_match"].mean()),

            "method_duration_mean": float(mdf["method_duration"].mean()),
            "method_unique_states_mean": float(mdf["method_unique_states"].mean()),
            "method_max_l1_from_start_mean": float(mdf["method_max_l1_from_start"].mean()),
            "fixed_order_corr_mean": float(mdf["fixed_order_corr"].mean()),
            "zeta_mean": float(mdf["zeta_mean"].mean()),
            "count_novelty_mean": float(mdf["count_novelty_mean"].mean()),
            "option_horizon": int(option_horizon),
            "option_gamma": float(option_gamma),
            "n_dims": int(len(eval_dims)),
            "n_starts": int(n_states),
        })

    summary_df = pd.DataFrame(summary_rows)

    if out_dir is not None:
        vis_dir = os.path.join(out_dir, "mechanism_panels", f"step_{global_step}")
        policy_dir = os.path.join(out_dir, "policy_panels", f"step_{global_step}")
        ensure_dir(vis_dir)
        ensure_dir(policy_dir)

        candidate_starts = []
        first_dim = eval_dims[0] if len(eval_dims) > 0 else 0
        first_true_dim = dim_to_true_dim[first_dim]
        first_true_vi = finite_horizon_option_value_iteration(
            true_phi_for_eval, np.zeros_like(novelty_n), neigh, first_true_dim, "TRUE", 0.0,
            horizon=option_horizon, gamma=option_gamma,
        )

        for s in range(n_states):
            tr = rollout_option_policy(
                start_idx=s,
                dim=first_true_dim,
                phi=true_phi_for_eval,
                novelty_n=np.zeros_like(novelty_n),
                neighbors=neigh,
                coords=coords,
                method="TRUE",
                beta=0.0,
                horizon=option_horizon,
                gamma=option_gamma,
                precomputed=first_true_vi,
            )

            if visit_counts is not None:
                # Prefer states actually reached by the training agent while still
                # favoring starts where the true option moves nontrivially.
                score = float(visit_counts[s]) + 0.01 * float(tr["max_l1_from_start"])
            else:
                score = float(tr["max_l1_from_start"])

            candidate_starts.append((score, s))

        candidate_starts = sorted(candidate_starts, reverse=True)
        vis_starts = [s for _, s in candidate_starts[:N_VIS_STARTS]]
        vis_dims = eval_dims[:N_VIS_DIMS]

        for learned_dim in vis_dims:
            true_dim = dim_to_true_dim[learned_dim]

            true_vi = finite_horizon_option_value_iteration(
                true_phi_for_eval, np.zeros_like(novelty_n), neigh, true_dim, "TRUE", 0.0,
                horizon=option_horizon, gamma=option_gamma,
            )
            neo_vi = finite_horizon_option_value_iteration(
                learned_phi, novelty_n, neigh, learned_dim, "NEO", 0.0,
                horizon=option_horizon, gamma=option_gamma,
            )
            sneo_vi = finite_horizon_option_value_iteration(
                learned_phi, novelty_n, neigh, learned_dim, "SNEO", RELIABLE_RND_BETA,
                horizon=option_horizon, gamma=option_gamma,
            )

            for start_idx in vis_starts:
                true_rollout = rollout_option_policy(
                    start_idx, true_dim, true_phi_for_eval, np.zeros_like(novelty_n), neigh, coords,
                    "TRUE", 0.0, horizon=option_horizon, gamma=option_gamma, precomputed=true_vi,
                )
                neo_rollout = rollout_option_policy(
                    start_idx, learned_dim, learned_phi, novelty_n, neigh, coords,
                    "NEO", 0.0, horizon=option_horizon, gamma=option_gamma, precomputed=neo_vi,
                )
                sneo_rollout = rollout_option_policy(
                    start_idx, learned_dim, learned_phi, novelty_n, neigh, coords,
                    "SNEO", RELIABLE_RND_BETA, horizon=option_horizon, gamma=option_gamma, precomputed=sneo_vi,
                )

                neo_stats = rollout_true_potential_stats(neo_rollout["path"], true_phi_for_eval, true_dim)
                sneo_stats = rollout_true_potential_stats(sneo_rollout["path"], true_phi_for_eval, true_dim)

                panel_title = (
                    fr"VI Step {global_step} | dim {learned_dim} | corr={dim_to_corr[learned_dim]:.2f} | "
                    fr"NEO $\Delta\phi^*$={neo_stats['true_progress']:.2f}, "
                    fr"SNEO $\Delta\phi^*$={sneo_stats['true_progress']:.2f} | "
                    fr"NEO agree={neo_stats['true_step_agreement']:.2f}, "
                    fr"SNEO agree={sneo_stats['true_step_agreement']:.2f}"
                )

                save_path = os.path.join(vis_dir, f"panel_vi_dim{learned_dim}_true{true_dim}_start{start_idx}.png")
                plot_mechanism_panel(
                    coords=coords,
                    true_values=true_phi_for_eval[:, true_dim],
                    learned_values=learned_phi[:, learned_dim],
                    neo_path=neo_rollout["path"],
                    sneo_path=sneo_rollout["path"],
                    true_path=true_rollout["path"],
                    start_idx=start_idx,
                    save_path=save_path,
                    title=panel_title,
                )

                policy_title = (
                    fr"Finite-horizon VI policies | step {global_step} | dim {learned_dim} | "
                    fr"H={option_horizon}, $\gamma_o$={option_gamma:.2f}"
                )
                policy_path = os.path.join(policy_dir, f"policy_vi_dim{learned_dim}_true{true_dim}_start{start_idx}.png")
                plot_policy_comparison_panel(
                    coords=coords,
                    true_values=true_phi_for_eval[:, true_dim],
                    learned_values=learned_phi[:, learned_dim],
                    true_policy=true_vi[1],
                    neo_policy=neo_vi[1],
                    sneo_policy=sneo_vi[1],
                    true_path=true_rollout["path"],
                    neo_path=neo_rollout["path"],
                    sneo_path=sneo_rollout["path"],
                    start_idx=start_idx,
                    horizon=option_horizon,
                    save_path=policy_path,
                    title=policy_title,
                )

        plot_heatmap_with_optional_path(
            coords=coords,
            values=zeta,
            save_path=os.path.join(vis_dir, "source_zeta_weights.png"),
            title=fr"Count source-Laplacian $\zeta(s)$ at step {global_step}",
            cmap="magma",
        )

        plot_heatmap_with_optional_path(
            coords=coords,
            values=novelty_n,
            save_path=os.path.join(vis_dir, "count_novelty_normalized.png"),
            title=fr"Count novelty $n_{{count}}(s)$ at step {global_step}",
            cmap="magma",
        )

    return rows_df, summary_df


# ============================================================
# Aggregation and final plots
# ============================================================

def aggregate_true_endpoint_across_seeds(summary_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["global_step", "method"]
    metric_cols = [
        # Main metrics.
        "method_true_progress_mean",
        "method_true_step_agreement_mean",

        # Supporting diagnostics.
        "method_true_pos_progress_mean",
        "method_true_neg_progress_mean",
        "true_progress_regret_mean",
        "true_option_progress_mean",
        "method_planned_value_mean",

        # Endpoint debug metrics.
        "endpoint_l1_to_true_mean",
        "endpoint_exact_match_mean",

        "method_duration_mean",
        "method_unique_states_mean",
        "method_max_l1_from_start_mean",
        "fixed_order_corr_mean",
        "zeta_mean",
        "count_novelty_mean",
    ]

    rows = []
    for keys, g in summary_df.groupby(group_cols):
        global_step, method = keys
        row = {"global_step": int(global_step), "method": method, "n_seeds": int(g["seed"].nunique())}
        for col in metric_cols:
            vals = g[col].to_numpy(dtype=np.float64)
            row[f"{col}_avg"] = float(np.mean(vals))
            row[f"{col}_sem"] = sem(vals)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["global_step", "method"]).reset_index(drop=True)


def plot_final_metric_bar(
    agg_df: pd.DataFrame,
    final_step: int,
    metric: str,
    save_path: str,
    title: str,
    ylabel: str,
):
    df = agg_df[agg_df["global_step"] == final_step].copy()
    methods = ["NEO", "SNEO"]

    vals, errs = [], []
    for method in methods:
        sub = df[df["method"] == method]
        vals.append(float(sub[f"{metric}_avg"].iloc[0]))
        errs.append(float(sub[f"{metric}_sem"].iloc[0]))

    x = np.arange(len(methods))
    plt.figure(figsize=(6.5, 5.2))
    plt.bar(x, vals, yerr=errs, capsize=5, width=0.55)
    plt.xticks(x, methods, fontsize=PLOT_FONTS["tick"])
    plt.ylabel(ylabel, fontsize=PLOT_FONTS["axis_label"])
    plt.title(title, fontsize=PLOT_FONTS["title"], pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_curve(
    agg_df: pd.DataFrame,
    metric: str,
    save_path: str,
    title: str,
    ylabel: str,
):
    methods = ["NEO", "SNEO"]
    plt.figure(figsize=(7.5, 5.2))

    for method in methods:
        sub = agg_df[agg_df["method"] == method].sort_values("global_step")
        x = sub["global_step"].to_numpy()
        y = sub[f"{metric}_avg"].to_numpy()
        e = sub[f"{metric}_sem"].to_numpy()
        plt.plot(x, y, marker="o", label=method)
        plt.fill_between(x, y - e, y + e, alpha=0.18)

    plt.xlabel("Environment steps", fontsize=PLOT_FONTS["axis_label"])
    plt.ylabel(ylabel, fontsize=PLOT_FONTS["axis_label"])
    plt.title(title, fontsize=PLOT_FONTS["title"], pad=10)
    plt.legend(frameon=False, fontsize=PLOT_FONTS["legend"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_improvement_curve(
    summary_df: pd.DataFrame,
    metric: str,
    save_path: str,
    title: str,
    ylabel: str,
    improvement_label: str,
):
    rows = []

    for (seed, step), g in summary_df.groupby(["seed", "global_step"]):
        g_neo = g[g["method"] == "NEO"]
        g_sneo = g[g["method"] == "SNEO"]
        if len(g_neo) == 0 or len(g_sneo) == 0:
            continue

        diff = float(g_sneo[metric].iloc[0] - g_neo[metric].iloc[0])
        rows.append({"seed": int(seed), "global_step": int(step), "diff": diff})

    diff_df = pd.DataFrame(rows)
    xs, ys, es = [], [], []
    for step, g in diff_df.groupby("global_step"):
        vals = g["diff"].to_numpy(dtype=np.float64)
        xs.append(int(step))
        ys.append(float(np.mean(vals)))
        es.append(sem(vals))

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    es = np.asarray(es)
    order = np.argsort(xs)
    xs, ys, es = xs[order], ys[order], es[order]

    plt.figure(figsize=(7.5, 5.2))
    plt.axhline(0.0, linewidth=1.2)
    plt.plot(xs, ys, marker="o", label=improvement_label)
    plt.fill_between(xs, ys - es, ys + es, alpha=0.18)
    plt.xlabel("Environment steps", fontsize=PLOT_FONTS["axis_label"])
    plt.ylabel(ylabel, fontsize=PLOT_FONTS["axis_label"])
    plt.title(title, fontsize=PLOT_FONTS["title"], pad=10)
    plt.legend(frameon=False, fontsize=PLOT_FONTS["legend"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_all_plots(summary_df: pd.DataFrame, out_root: str):
    apply_plot_style()
    plots_dir = os.path.join(out_root, "plots")
    ensure_dir(plots_dir)

    agg_df = aggregate_true_endpoint_across_seeds(summary_df)
    agg_path = os.path.join(out_root, "aggregate_summary.csv")
    agg_df.to_csv(agg_path, index=False)

    final_step = int(agg_df["global_step"].max())

    # Main metric 1: true eigen-potential progress.
    plot_final_metric_bar(
        agg_df,
        final_step,
        metric="method_true_progress_mean",
        save_path=os.path.join(plots_dir, "final_true_progress_bar.png"),
        title=r"True source-eigen progress",
        ylabel=r"$\phi^*(s_T)-\phi^*(s_0)$",
    )
    plot_metric_curve(
        agg_df,
        metric="method_true_progress_mean",
        save_path=os.path.join(plots_dir, "curve_true_progress.png"),
        title=r"True source-eigen progress over training",
        ylabel=r"$\phi^*(s_T)-\phi^*(s_0)$",
    )
    plot_metric_improvement_curve(
        summary_df,
        metric="method_true_progress_mean",
        save_path=os.path.join(plots_dir, "improvement_true_progress_sneo_minus_neo.png"),
        title="Positive means SNEO makes more true eigen-progress",
        ylabel="SNEO - NEO",
        improvement_label="SNEO progress - NEO progress",
    )

    # Main metric 2: local true-gradient step agreement.
    plot_final_metric_bar(
        agg_df,
        final_step,
        metric="method_true_step_agreement_mean",
        save_path=os.path.join(plots_dir, "final_true_step_agreement_bar.png"),
        title="True-gradient step agreement",
        ylabel="Fraction positive true-gradient steps",
    )
    plot_metric_curve(
        agg_df,
        metric="method_true_step_agreement_mean",
        save_path=os.path.join(plots_dir, "curve_true_step_agreement.png"),
        title="True-gradient step agreement over training",
        ylabel="Fraction positive true-gradient steps",
    )
    plot_metric_improvement_curve(
        summary_df,
        metric="method_true_step_agreement_mean",
        save_path=os.path.join(plots_dir, "improvement_true_step_agreement_sneo_minus_neo.png"),
        title="Positive means SNEO has better local true-gradient agreement",
        ylabel="SNEO - NEO",
        improvement_label="SNEO agreement - NEO agreement",
    )

    # Endpoint L1 remains as a debug plot only.
    plot_final_metric_bar(
        agg_df,
        final_step,
        metric="endpoint_l1_to_true_mean",
        save_path=os.path.join(plots_dir, "debug_final_endpoint_l1_to_true_bar.png"),
        title="L1 to true greedy endpoint",
        ylabel="L1 distance",
    )
    plot_metric_curve(
        agg_df,
        metric="endpoint_l1_to_true_mean",
        save_path=os.path.join(plots_dir, "debug_curve_endpoint_l1_to_true.png"),
        title="L1 to true greedy endpoint",
        ylabel="L1 distance",
    )

    return agg_df


# ============================================================
# Evaluation policy
# ============================================================

def evaluate_policy(train_state: Train_State, env, global_step: int, args):
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

    return terminated, eps_return, eps_step


# ============================================================
# Main single-seed training loop
# ============================================================

def q_learning_one_seed(
    *,
    args,
    seed: int,
    train_state: Train_State,
    rb_state,
    key,
    n_actions,
    env,
    rb,
    out_seed_dir: str,
):
    buffer_device = jax.devices(args.buffer.device)[0]
    total_timesteps = args.algo.tot_steps

    obs, _ = env.reset(seed=args.exp.seed)
    eval_points = set(make_eval_points(total_timesteps))

    # Counts are used by the mechanistic diagnostic to build the exact
    # count-source Laplacian, define count novelty for SNEO VI planning,
    # and choose visualization start nodes. They do not affect training.
    _, _, ij_to_i_for_counts, _ = get_all_state_pixels(env)
    visit_counts = np.zeros((len(ij_to_i_for_counts),), dtype=np.float64)
    safe_count_current_state(env, ij_to_i_for_counts, visit_counts)

    rows_all = []
    summaries_all = []
    test_results = []

    eps_step, eps_return = 1, 0.0
    log_batch = []
    start_time = None

    seq_len = max(args.algo.n_step, args.laplacian.n_step) + 1
    update_q_network = make_update_q_network(args)
    update_lap_network = make_update_grid_neo_laplacian_network(args)

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
        safe_count_current_state(env, ij_to_i_for_counts, visit_counts)

        next_obs_gray = rgb_to_gray_uint8(next_obs)
        next_obs_b = np.expand_dims(next_obs_gray, axis=0)

        rnd_bonus = float(
            np.asarray(jax.device_get(rnd_intrinsic_reward(train_state.rnd_state, jnp.asarray(next_obs_b.astype(np.float32)))))[0]
        )

        with jax.default_device(buffer_device):
            timestep = TimeStep(
                observation=jnp.asarray(obs_b[:, None, ...]),
                next_observation=jnp.asarray(next_obs_b[:, None, ...]),
                action=jnp.expand_dims(jnp.asarray(action, dtype=jnp.int32), (0, 1)),
                reward=jnp.expand_dims(jnp.asarray(reward, dtype=jnp.float32), (0, 1)),
                done=jnp.expand_dims(jnp.asarray(terminated or eps_step >= args.algo.max_eps_len, dtype=jnp.bool_), (0, 1)),
                rnd_bonus=jnp.expand_dims(jnp.asarray(rnd_bonus, dtype=jnp.float32), (0, 1)),
            )
            rb_state = rb.add(rb_state, timestep)

        eps_return += reward

        if terminated or eps_step >= args.algo.max_eps_len:
            log_batch.append(f"Seed {seed} | Step {global_step} | Episode length {eps_step} | Return {eps_return}")
            next_obs, _ = env.reset()
            safe_count_current_state(env, ij_to_i_for_counts, visit_counts)
            eps_step, eps_return = 0, 0.0

        eps_step += 1
        obs = next_obs

        if global_step % 5000 == 0 and len(log_batch) > 0:
            print("\n".join(log_batch))
            log_batch = []

        if global_step == args.algo.starts:
            assert rb.can_sample(rb_state)
            print(f"[seed={seed}] Learning starts")
            start_time = time.time()

        if global_step >= args.algo.starts:
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
                    priorities = compute_is_weights(batch.probabilities, rb_state.running_index, rb_state.sum_tree_state.capacity, seq_len, beta)
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
                    q_batch, priorities, train_state.enc_state, train_state.q_state, args.algo.ddqn
                )

                train_state = train_state.replace(enc_state=enc_state, q_state=q_state, grad_steps=train_state.grad_steps + 1)

                if args.buffer.prioritized:
                    with jax.default_device(buffer_device):
                        rb_state = rb.set_priorities(rb_state, batch.indices, new_priorities)

                if args.laplacian.upd_freq is not None and global_step % args.laplacian.upd_freq == 0:
                    lap_obs = q_data.observation[:, : args.laplacian.n_step + 1, ...]
                    lap_dones = q_data.done[:, : args.laplacian.n_step + 1]
                    lap_actions = q_data.action[:, : args.laplacian.n_step + 1]

                    _, rnd_bonus_seq, _, _, inv_used_seq = compute_laplace_neo_targets_from_obs_seq(train_state, lap_obs.astype(jnp.float32))

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
                        delta=SOURCE_ZETA_DELTA,
                        k=SOURCE_ZETA_K,
                        target_mean=0.3,
                        tau_mean=1e-3,
                    )
                    train_state = train_state.replace(neo_state=neo_state_new)

                    lap_batch = LapBatch(
                        obs=lap_obs.astype(jnp.float32),
                        dones=lap_dones,
                        actions=lap_actions,
                        novelty=rnd_bonus_seq,
                        counts=zeta_seq,
                    )

                    key, lap_key = jax.random.split(key)
                    train_state, _ = update_lap_network(lap_key, train_state, lap_batch, n_actions)

                flat_storage = FlatStorage(rnd_frame=q_data.next_observation[:, 0, ...].astype(jnp.float32))
                if global_step % 8 == 0:
                    rnd_state2, _ = rnd_update_once_per_epoch(train_state.rnd_state, flat_storage, update_proportion=0.01, burnin=False)
                else:
                    rnd_state2 = train_state.rnd_state
                train_state = train_state.replace(rnd_state=rnd_state2)

            if (args.algo.tau == 1 and global_step % args.algo.tgt_upd_freq == 0) or (args.algo.tau < 1):
                enc_state, q_state = update_target_network(args.algo.tau, train_state.enc_state, train_state.q_state)
                train_state = train_state.replace(enc_state=enc_state, q_state=q_state)

            if (global_step >= total_timesteps * 0.5) and global_step % 5000 == 0:
                reached_goal, test_ret, test_len = evaluate_policy(train_state, env, global_step, args)
                test_results.append(float(reached_goal))
                success_rate = float(np.mean(test_results))
                print(f"[seed={seed}] Goal success rate: {success_rate:.3f}")

        train_state = train_state.replace(
            timesteps=global_step,
            n_updates=train_state.n_updates + int(global_step >= args.algo.starts),
        )

        if global_step in eval_points:
            print(f"[seed={seed}] Running true-source endpoint diagnostics at step {global_step}")
            rows_df, summary_df = run_true_endpoint_diagnostics(
                seed=seed,
                global_step=global_step,
                train_state=train_state,
                env=env,
                args=args,
                visit_counts=visit_counts,
                out_dir=out_seed_dir,
            )

            rows_all.append(rows_df)
            summaries_all.append(summary_df)

            rows_so_far = pd.concat(rows_all, ignore_index=True)
            summary_so_far = pd.concat(summaries_all, ignore_index=True)
            rows_so_far.to_csv(os.path.join(out_seed_dir, "diag_rows.csv"), index=False)
            summary_so_far.to_csv(os.path.join(out_seed_dir, "diag_summary.csv"), index=False)

            try:
                make_all_plots(summary_so_far, out_seed_dir)
            except Exception:
                print(f"[seed={seed}] Warning: per-seed plotting failed.")
                traceback.print_exc()

    if start_time is not None:
        print(f"[seed={seed}] Training time: {time.time() - start_time:.2f}s")

    if len(rows_all) == 0:
        rows_all_df = pd.DataFrame()
        summaries_all_df = pd.DataFrame()
    else:
        rows_all_df = pd.concat(rows_all, ignore_index=True)
        summaries_all_df = pd.concat(summaries_all, ignore_index=True)

    return train_state, rb_state, key, rows_all_df, summaries_all_df


# ============================================================
# Multi-seed runner
# ============================================================

def run_one_seed(seed: int, out_root: str):
    args = copy_args_with_seed(base_args, seed)
    set_seed_everywhere(seed)
    key = jax.random.PRNGKey(seed)

    grid = GridAgent(args)
    env = grid.env

    screen_size = args.env.screen_size
    if screen_size is None:
        screen_size = (grid.height, grid.width)
    else:
        screen_size = (args.env.screen_size, args.env.screen_size)

    n_actions = env.action_space.n
    seq_len = max(args.algo.n_step, args.laplacian.n_step) + 1

    out_seed_dir = os.path.join(out_root, f"seed_{seed}")
    ensure_dir(out_seed_dir)

    with jax.default_device(jax.devices(args.buffer.device)[0]):
        rb, rb_state = setup_rb(args, screen_size, seq_len)

    try:
        gpu_device = jax.devices("gpu")[0]
    except Exception:
        gpu_device = jax.devices()[0]

    with jax.default_device(gpu_device):
        try:
            key, train_state = setup_train_state(key, env, args, n_actions)
            train_state, rb_state, key, rows_df, summary_df = q_learning_one_seed(
                args=args,
                seed=seed,
                train_state=train_state,
                rb_state=rb_state,
                key=key,
                n_actions=n_actions,
                env=env,
                rb=rb,
                out_seed_dir=out_seed_dir,
            )
            rows_df.to_csv(os.path.join(out_seed_dir, "diag_rows.csv"), index=False)
            summary_df.to_csv(os.path.join(out_seed_dir, "diag_summary.csv"), index=False)
        except Exception:
            traceback.print_exc()
            rows_df = pd.DataFrame()
            summary_df = pd.DataFrame()
        finally:
            env.close()

    return rows_df, summary_df


# ============================================================
# CLI
# ============================================================

def parse_local_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--diag_seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--diag_out", type=str, default=None)
    parser.add_argument("--primary_k", type=int, default=PRIMARY_K)
    parser.add_argument("--reliable_beta", type=float, default=RELIABLE_RND_BETA)
    parser.add_argument("--n_vis_dims", type=int, default=N_VIS_DIMS)
    parser.add_argument("--n_vis_starts", type=int, default=N_VIS_STARTS)
    parser.add_argument("--use_abs_true_option", action="store_true")
    parser.add_argument("--vi_horizon", type=int, default=OPTION_VI_HORIZON)
    parser.add_argument("--vi_gamma", type=float, default=OPTION_VI_GAMMA)
    local_args, _ = parser.parse_known_args()
    return local_args


if __name__ == "__main__":
    print("Available JAX devices:", jax.devices())
    local_args = parse_local_args()

    PRIMARY_K = int(local_args.primary_k)
    RELIABLE_RND_BETA = float(local_args.reliable_beta)
    N_VIS_DIMS = int(local_args.n_vis_dims)
    N_VIS_STARTS = int(local_args.n_vis_starts)
    USE_ABS_TRUE_OPTION = bool(local_args.use_abs_true_option)
    OPTION_VI_HORIZON = int(local_args.vi_horizon)
    OPTION_VI_GAMMA = float(local_args.vi_gamma)

    seeds = [int(x) for x in local_args.diag_seeds.split(",")]

    run_group = (
        f"{base_args.env.maze_type}"
        f"_steps{base_args.algo.tot_steps}"
        f"_seeds{len(seeds)}"
        f"_beta{RELIABLE_RND_BETA}"
        f"_trueabs{int(USE_ABS_TRUE_OPTION)}"
        f"_viH{OPTION_VI_HORIZON}"
        f"_vigamma{OPTION_VI_GAMMA}"
        f"_{int(time.time())}"
    )

    out_root = local_args.diag_out
    if out_root is None:
        out_root = os.path.join("artifacts", "SNEO_count_source_vi_policy", run_group)

    ensure_dir(out_root)

    all_rows = []
    all_summaries = []

    for seed in seeds:
        print("=" * 80)
        print(f"Running seed {seed}")
        print("=" * 80)

        rows_df, summary_df = run_one_seed(seed, out_root)

        if len(rows_df) > 0:
            all_rows.append(rows_df)
        if len(summary_df) > 0:
            all_summaries.append(summary_df)

        jax.clear_caches()

    if len(all_rows) == 0 or len(all_summaries) == 0:
        raise RuntimeError("No diagnostic rows were collected. Check training/eval_points.")

    aggregate_rows = pd.concat(all_rows, ignore_index=True)
    aggregate_summary = pd.concat(all_summaries, ignore_index=True)

    aggregate_rows.to_csv(os.path.join(out_root, "aggregate_diag_rows.csv"), index=False)
    aggregate_summary.to_csv(os.path.join(out_root, "aggregate_seed_summary.csv"), index=False)

    agg_df = make_all_plots(aggregate_summary, out_root)

    print("Done.")
    print(f"Saved outputs to: {out_root}")
    print(agg_df.tail(20))
