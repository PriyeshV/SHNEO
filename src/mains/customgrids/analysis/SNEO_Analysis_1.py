# reliable_vs_vanilla_escape_success.py
#
# reliable_vs_vanilla_escape_success.py
#
# Purpose:
#   Run multiple seeds and evaluate whether reliable eigen-option rollouts
#   improve over vanilla eigen-option rollouts in:
#
#       1. Escape:
#           whether the rollout reaches distance >= k from its start state.
#
#       2. Relative novelty gain:
#           whether the rollout moves toward states that are more novel than
#           its own start state, rather than requiring it to reach a global
#           top-15% novelty region.
#
# Main metrics:
#   escape_ge_k:
#       1{D_max >= k}
#
#   novelty_gain_final:
#       n_RND(s_final) - n_RND(s_start)
#
#   novelty_gain_max:
#       max_t n_RND(s_t) - n_RND(s_start)
#
#   rel_success_final_ge_k:
#       1{D_max >= k} * 1{n_RND(s_final) > n_RND(s_start)}
#
# Reliable score:
#       phi[s', d] - phi[s, d] + reliable_beta * novelty_n[s']
#
# Vanilla score:
#       phi[s', d] - phi[s, d]
#


import os
import time
import math
import random
import traceback
import argparse
from copy import deepcopy
from functools import partial
from typing import Any, Union, NamedTuple, Dict, List, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from etils import epath

from src.config.customgrids.laplacian import *
base_args = convert_args_to_dict(parse_args())

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


# ============================================================
# Experiment-level diagnostic constants
# ============================================================

DIAG_FRACTIONS = tuple(np.linspace(0.1, 1.0, 10))

# Plot "at least k" thresholds, not exact radius categories.
ESCAPE_KS = (2, 3, 4, 5)

# Primary threshold for paper-style headline plot.
PRIMARY_K = 3

# Greedy diagnostic option rollout horizon.
TRAJ_HORIZON = 25

# Stop after repeated non-positive local progress.
TRAJ_STALL_PATIENCE = 4
TRAJ_STALL_TOL = 1e-6

# Reliable edge score coefficient:
# reliable score = phi(s') - phi(s) + beta * normalized_RND(s')
RELIABLE_RND_BETA = 10.0

# If None, use args.algo.n_options.
N_EVAL_DIMS = None

# Whether to skip dim 0. Usually dim 0 is constant-like / trivial.
SKIP_DIM0 = True

# ============================================================
# Dataclasses from old code
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
    pts = sorted({max(1, int(round(frac * total_steps))) for frac in DIAG_FRACTIONS})
    return pts


def set_seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)


from dataclasses import replace, is_dataclass
import copy

def copy_args_with_seed(args, seed: int):
    """
    Copy the full config and replace args.exp.seed even when args.exp
    is a frozen dataclass.
    """
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
    enc_state = enc_state.replace(
        target_params=optax.incremental_update(enc_state.params, enc_state.target_params, tau)
    )
    q_state = q_state.replace(
        target_params=optax.incremental_update(q_state.params, q_state.target_params, tau)
    )
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
                target = n_step_return + boot_mask * gamma_n * jnp.clip(
                    q_next_target,
                    jnp.min(q_pred),
                    jnp.max(q_pred),
                )
            else:
                target = n_step_return + boot_mask * gamma_n * q_next_target

            td_error = q_pred - jax.lax.stop_gradient(target)
            upd_priorities = jnp.abs(td_error) + 1e-8
            loss = td_error ** 2
            wt_loss = jnp.mean(priorities * loss) if args.buffer.prioritized else jnp.mean(loss)

            return wt_loss, (
                jnp.mean(loss),
                q_pred.mean(),
                q_pred.min(),
                q_pred.max(),
                upd_priorities,
            )

        (wt_td_loss, aux), grads = jax.value_and_grad(
            td_loss_fn,
            has_aux=True,
            argnums=(0, 1),
        )(enc_state.params, q_state.params)

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
    inv_cap = 1e4
    inv_used = jnp.minimum(inv_count, inv_cap)

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
                phis_flat, updates = lap_state.apply_fn(
                    lap_vars,
                    obs_flat,
                    train=True,
                    mutable=mutable,
                )
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

        (eigen_loss, aux), grads = jax.value_and_grad(
            laplacian_loss_fn,
            has_aux=True,
        )(lap_state.params)

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

    return (
        np.stack(imgs, axis=0),
        np.stack(coords, axis=0),
        ij_to_i,
        i_to_ij,
    )


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
# Graph helpers
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
# Greedy option rollout diagnostics
# ============================================================

def option_score_vanilla(phi: np.ndarray, s: int, sp: int, dim: int) -> float:
    return float(phi[sp, dim] - phi[s, dim])


def option_score_reliable(
    phi: np.ndarray,
    novelty_n: np.ndarray,
    s: int,
    sp: int,
    dim: int,
    beta: float,
) -> float:
    return float(phi[sp, dim] - phi[s, dim] + beta * novelty_n[sp])


def greedy_climb_rollout(
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
    assert method in ("NEO", "SNEO")

    path = [start_idx]
    unique = {start_idx}

    stall_ctr = 0
    curr = start_idx

    novelty_start = float(novelty_n[start_idx])
    novelty_sum = 0.0
    novelty_max = novelty_start

    phi_start = float(phi[start_idx, dim])
    best_phi_gain = 0.0

    for _ in range(horizon):
        cand = neighbors[curr]
        if len(cand) == 0:
            break

        if method == "SNEO":
            scores = np.asarray(
                [option_score_reliable(phi, novelty_n, curr, sp, dim, beta) for sp in cand],
                dtype=np.float32,
            )
        else:
            scores = np.asarray(
                [option_score_vanilla(phi, curr, sp, dim) for sp in cand],
                dtype=np.float32,
            )

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

        novelty_sum += float(novelty_n[curr])
        novelty_max = max(novelty_max, float(novelty_n[curr]))
        best_phi_gain = max(best_phi_gain, float(phi[curr, dim] - phi_start))

        if stall_ctr >= stall_patience:
            break

    novelty_final = float(novelty_n[path[-1]])
    novelty_gain_final = novelty_final - novelty_start
    novelty_gain_max = novelty_max - novelty_start

    start_xy = coords[start_idx]
    end_xy = coords[path[-1]]
    end_l1_dist = int(np.abs(end_xy - start_xy).sum())

    max_l1_dist = 0
    for idx in path:
        xy = coords[idx]
        max_l1_dist = max(max_l1_dist, int(np.abs(xy - start_xy).sum()))

    return {
        "path": path,
        "duration": len(path) - 1,
        "unique_states": len(unique),

        "novelty_start": novelty_start,
        "novelty_final": novelty_final,
        "novelty_max": novelty_max,
        "novelty_sum": novelty_sum,
        "novelty_gain_final": novelty_gain_final,
        "novelty_gain_max": novelty_gain_max,

        "end_l1_dist": end_l1_dist,
        "max_l1_dist": max_l1_dist,
        "best_phi_gain": best_phi_gain,
    }

# ============================================================
# Clean diagnostics: escape>=k and success>=k only
# ============================================================

def choose_eval_dims(phi_dim: int, n_options: int) -> List[int]:
    n_eval = n_options if N_EVAL_DIMS is None else N_EVAL_DIMS

    if SKIP_DIM0:
        dims = list(range(1, min(n_eval + 1, phi_dim)))
    else:
        dims = list(range(min(n_eval, phi_dim)))

    return dims


def run_reliable_vs_vanilla_diagnostics(
    *,
    seed: int,
    global_step: int,
    train_state: Train_State,
    env,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        rows_df:
            one row per seed x step x dim x method x start_state.

        summary_df:
            aggregated over starts and dims for each seed x step x method x k.

    Main metrics:
        escape_ge_k:
            Whether the rollout reaches max Manhattan distance >= k.

        novelty_gain_final:
            n_RND(s_final) - n_RND(s_start).

        novelty_gain_max:
            max_t n_RND(s_t) - n_RND(s_start).

        reliable_success_ge_k:
            Optional binary diagnostic:
            escape_ge_k and novelty_gain_final > 0.
            This is weaker than top-15% novelty and better suited for short horizons.
    """
    state_imgs_u8, coords, ij_to_i, i_to_ij = get_all_state_pixels(env)
    state_imgs = state_imgs_u8.astype(np.float32)

    compute_embeddings_fn = make_compute_all_state_embeddings(args)

    phi = np.asarray(
        jax.device_get(
            compute_embeddings_fn(train_state.lap_state, jnp.asarray(state_imgs))
        )
    )
    novelty = np.asarray(
        jax.device_get(
            compute_all_state_novelty(train_state.rnd_state, jnp.asarray(state_imgs))
        )
    )

    # Normalize within the current checkpoint. This makes relative improvement
    # interpretable across training time and seeds.
    novelty_n = novelty / (np.max(novelty) + 1e-8)

    neigh = build_neighbor_index(env, ij_to_i)
    dims = choose_eval_dims(phi.shape[1], args.algo.n_options)

    raw_rows = []

    for dim in dims:
        for start_idx in range(len(coords)):
            for method in ("NEO", "SNEO"):
                rollout = greedy_climb_rollout(
                    start_idx=start_idx,
                    dim=dim,
                    phi=phi,
                    novelty_n=novelty_n,
                    neighbors=neigh,
                    coords=coords,
                    method=method,
                    beta=RELIABLE_RND_BETA,
                )

                row = {
                    "seed": int(seed),
                    "global_step": int(global_step),
                    "dim": int(dim),
                    "method": method,
                    "start_idx": int(start_idx),

                    "duration": float(rollout["duration"]),
                    "unique_states": float(rollout["unique_states"]),

                    "novelty_start": float(rollout["novelty_start"]),
                    "novelty_final": float(rollout["novelty_final"]),
                    "novelty_max": float(rollout["novelty_max"]),
                    "novelty_sum": float(rollout["novelty_sum"]),
                    "novelty_gain_final": float(rollout["novelty_gain_final"]),
                    "novelty_gain_max": float(rollout["novelty_gain_max"]),

                    "end_l1_dist": float(rollout["end_l1_dist"]),
                    "max_l1_dist": float(rollout["max_l1_dist"]),
                    "best_phi_gain": float(rollout["best_phi_gain"]),
                }

                for k in ESCAPE_KS:
                    escaped = float(rollout["max_l1_dist"] >= k)

                    # Relative-novelty success:
                    # the option escapes and ends in a state that is more novel than its start.
                    # This avoids the pathological failure mode of absolute top-15% thresholds.
                    rel_success_final = float(
                        (rollout["max_l1_dist"] >= k)
                        and (rollout["novelty_gain_final"] > 0.0)
                    )

                    # A softer auxiliary metric:
                    # the option escapes and reaches any state more novel than its start.
                    rel_success_max = float(
                        (rollout["max_l1_dist"] >= k)
                        and (rollout["novelty_gain_max"] > 0.0)
                    )

                    row[f"escape_ge_{k}"] = escaped
                    row[f"rel_success_final_ge_{k}"] = rel_success_final
                    row[f"rel_success_max_ge_{k}"] = rel_success_max

                raw_rows.append(row)

    rows_df = pd.DataFrame(raw_rows)

    summary_rows = []
    for method in ("NEO", "SNEO"):
        mdf = rows_df[rows_df["method"] == method]

        for k in ESCAPE_KS:
            summary_rows.append({
                "seed": int(seed),
                "global_step": int(global_step),
                "method": method,
                "k": int(k),

                "escape_mean": float(mdf[f"escape_ge_{k}"].mean()),
                "rel_success_final_mean": float(mdf[f"rel_success_final_ge_{k}"].mean()),
                "rel_success_max_mean": float(mdf[f"rel_success_max_ge_{k}"].mean()),

                "novelty_gain_final_mean": float(mdf["novelty_gain_final"].mean()),
                "novelty_gain_max_mean": float(mdf["novelty_gain_max"].mean()),
                "novelty_start_mean": float(mdf["novelty_start"].mean()),
                "novelty_final_mean": float(mdf["novelty_final"].mean()),
                "novelty_max_mean": float(mdf["novelty_max"].mean()),
                "novelty_sum_mean": float(mdf["novelty_sum"].mean()),

                "duration_mean": float(mdf["duration"].mean()),
                "unique_states_mean": float(mdf["unique_states"].mean()),
                "end_l1_dist_mean": float(mdf["end_l1_dist"].mean()),
                "max_l1_dist_mean": float(mdf["max_l1_dist"].mean()),

                "n_dims": int(len(dims)),
                "n_starts": int(len(coords)),
            })

    summary_df = pd.DataFrame(summary_rows)
    return rows_df, summary_df


# ============================================================
# Plotting utilities
# ============================================================

def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def aggregate_across_seeds(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    summary_df has one row per seed x step x method x k.
    This returns mean and SEM across seeds.
    """
    group_cols = ["global_step", "method", "k"]

    metric_cols = [
        "escape_mean",
        "rel_success_final_mean",
        "rel_success_max_mean",

        "novelty_gain_final_mean",
        "novelty_gain_max_mean",
        "novelty_start_mean",
        "novelty_final_mean",
        "novelty_max_mean",
        "novelty_sum_mean",

        "duration_mean",
        "unique_states_mean",
        "end_l1_dist_mean",
        "max_l1_dist_mean",
    ]

    rows = []
    for keys, g in summary_df.groupby(group_cols):
        global_step, method, k = keys
        row = {
            "global_step": int(global_step),
            "method": method,
            "k": int(k),
            "n_seeds": int(g["seed"].nunique()),
        }

        for col in metric_cols:
            vals = g[col].to_numpy(dtype=np.float64)
            row[f"{col}_avg"] = float(np.mean(vals))
            row[f"{col}_sem"] = sem(vals)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["global_step", "k", "method"]).reset_index(drop=True)


# ============================================================
# Plotting style
# ============================================================

PLOT_FONTS = {
    "title": 20,
    "axis_label": 18,
    "tick": 15,
    "legend": 15,
}

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


def plot_final_bar(
    agg_df: pd.DataFrame,
    metric: str,
    final_step: int,
    save_path: str,
    title: str,
    ylabel: str,
    ylim=None,
):
    """
    metric examples:
        escape_mean
        rel_success_final_mean
        novelty_gain_final_mean
        novelty_gain_max_mean
    """
    df = agg_df[agg_df["global_step"] == final_step].copy()

    ks = sorted(df["k"].unique())
    methods = ["NEO", "SNEO"]

    x = np.arange(len(ks), dtype=np.float32)
    width = 0.35

    plt.figure(figsize=(7.0, 4.5))

    for i, method in enumerate(methods):
        vals = []
        errs = []
        for k in ks:
            sub = df[(df["method"] == method) & (df["k"] == k)]
            vals.append(float(sub[f"{metric}_avg"].iloc[0]))
            errs.append(float(sub[f"{metric}_sem"].iloc[0]))

        offset = (i - 0.5) * width
        plt.bar(
            x + offset,
            vals,
            width=width,
            yerr=errs,
            capsize=4,
            label=method,
        )

    plt.xticks(x, [fr"$\geq {k}$" for k in ks])
    plt.ylabel(ylabel)
    plt.xlabel("Escape threshold k")
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_training_curve(
    agg_df: pd.DataFrame,
    metric: str,
    k: int,
    save_path: str,
    title: str,
    ylabel: str,
    ylim=None,
):
    df = agg_df[agg_df["k"] == k].copy()
    methods = ["NEO", "SNEO"]

    plt.figure(figsize=(7.0, 4.5))

    for method in methods:
        sub = df[df["method"] == method].sort_values("global_step")
        x = sub["global_step"].to_numpy()
        y = sub[f"{metric}_avg"].to_numpy()
        e = sub[f"{metric}_sem"].to_numpy()

        plt.plot(x, y, marker="o", label=method)
        plt.fill_between(x, y - e, y + e, alpha=0.18)

    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_improvement_curve(
    summary_df: pd.DataFrame,
    metric: str,
    k: int,
    save_path: str,
    title: str,
    ylabel: str,
):
    """
    Plots reliable - vanilla over training, with SEM across seeds.
    """
    rows = []

    for (seed, step), g in summary_df[summary_df["k"] == k].groupby(["seed", "global_step"]):
        gv = g[g["method"] == "NEO"]
        gr = g[g["method"] == "SNEO"]

        if len(gv) == 0 or len(gr) == 0:
            continue

        diff = float(gr[metric].iloc[0] - gv[metric].iloc[0])
        rows.append({
            "seed": int(seed),
            "global_step": int(step),
            "diff": diff,
        })

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

    xs = xs[order]
    ys = ys[order]
    es = es[order]

    plt.figure(figsize=(7.0, 4.5))
    plt.axhline(0.0, linewidth=1.0)
    plt.plot(xs, ys, marker="o", label="SNEO - NEO")
    plt.fill_between(xs, ys - es, ys + es, alpha=0.18)

    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def make_all_plots(summary_df: pd.DataFrame, out_root: str):
    apply_plot_style()
    plots_dir = os.path.join(out_root, "plots")
    ensure_dir(plots_dir)

    agg_df = aggregate_across_seeds(summary_df)
    agg_path = os.path.join(out_root, "aggregate_summary.csv")
    agg_df.to_csv(agg_path, index=False)

    final_step = int(agg_df["global_step"].max())

    # ------------------------------------------------------------
    # Final checkpoint bars
    # ------------------------------------------------------------
    plot_final_bar(
        agg_df,
        metric="escape_mean",
        final_step=final_step,
        save_path=os.path.join(plots_dir, "final_escape_ge_k_bar.png"),
        title=fr"Escape rate: $D_\max \geq k$",
        ylabel="Escape rate",
        ylim=(0.0, 1.05),
    )

    plot_final_bar(
        agg_df,
        metric="novelty_gain_final_mean",
        final_step=final_step,
        save_path=os.path.join(plots_dir, "final_novelty_gain_final_ge_k_bar.png"),
        title=fr"Final relative novelty gain",
        ylabel=fr"$n(s_\mathrm{{final}})-n(s_0)$",
        ylim=None,
    )

    plot_final_bar(
        agg_df,
        metric="novelty_gain_max_mean",
        final_step=final_step,
        save_path=os.path.join(plots_dir, "final_novelty_gain_max_ge_k_bar.png"),
        title=fr"Maximum relative novelty gain",
        ylabel=fr"$\max_t n(s_t)-n(s_0)$",
        ylim=None,
    )

    plot_final_bar(
        agg_df,
        metric="rel_success_final_mean",
        final_step=final_step,
        save_path=os.path.join(plots_dir, "final_rel_success_final_ge_k_bar.png"),
        # title=fr"Relative-novelty success: $D_\max \geq k$ and $n(s_\mathrm{{final}})>n(s_0)$",
        title=fr"Relative-novelty success",
        ylabel="Success rate",
        ylim=(0.0, 1.05),
    )

    # ------------------------------------------------------------
    # Training curves at primary k
    # ------------------------------------------------------------
    plot_training_curve(
        agg_df,
        metric="escape_mean",
        k=PRIMARY_K,
        save_path=os.path.join(plots_dir, f"curve_escape_ge_{PRIMARY_K}.png"),
        title=fr"Escape rate during training: $D_\max \geq {PRIMARY_K}$",
        ylabel="Escape rate",
        ylim=(0.0, 1.05),
    )

    plot_training_curve(
        agg_df,
        metric="novelty_gain_final_mean",
        k=PRIMARY_K,
        save_path=os.path.join(plots_dir, f"curve_novelty_gain_final_ge_{PRIMARY_K}.png"),
        title=fr"Final relative novelty gain during training: $D_\max \geq {PRIMARY_K}$",
        ylabel=fr"$n(s_\mathrm{{final}})-n(s_0)$",
        ylim=None,
    )

    plot_training_curve(
        agg_df,
        metric="novelty_gain_max_mean",
        k=PRIMARY_K,
        save_path=os.path.join(plots_dir, f"curve_novelty_gain_max_ge_{PRIMARY_K}.png"),
        title=fr"Maximum relative novelty gain during training: $D_\max \geq {PRIMARY_K}$",
        ylabel=fr"$\max_t n(s_t)-n(s_0)$",
        ylim=None,
    )

    plot_training_curve(
        agg_df,
        metric="rel_success_final_mean",
        k=PRIMARY_K,
        save_path=os.path.join(plots_dir, f"curve_rel_success_final_ge_{PRIMARY_K}.png"),
        title=fr"Relative-novelty success during training: $D_\max \geq {PRIMARY_K}$",
        ylabel="Success rate",
        ylim=(0.0, 1.05),
    )

    # ------------------------------------------------------------
    # Reliable - vanilla improvement curves
    # ------------------------------------------------------------
    plot_improvement_curve(
        summary_df,
        metric="escape_mean",
        k=PRIMARY_K,
        save_path=os.path.join(plots_dir, f"improvement_escape_ge_{PRIMARY_K}.png"),
        title=fr"Reliable improvement in escape: $D_\max \geq {PRIMARY_K}$",
        ylabel="SNEO - NEO",
    )

    plot_improvement_curve(
        summary_df,
        metric="novelty_gain_final_mean",
        k=PRIMARY_K,
        save_path=os.path.join(plots_dir, f"improvement_novelty_gain_final_ge_{PRIMARY_K}.png"),
        title=fr"Reliable improvement in final novelty gain: $D_\max \geq {PRIMARY_K}$",
        ylabel="SNEO - NEO",
    )

    plot_improvement_curve(
        summary_df,
        metric="novelty_gain_max_mean",
        k=PRIMARY_K,
        save_path=os.path.join(plots_dir, f"improvement_novelty_gain_max_ge_{PRIMARY_K}.png"),
        title=fr"Reliable improvement in max novelty gain: $D_\max \geq {PRIMARY_K}$",
        ylabel="SNEO - NEO",
    )

    plot_improvement_curve(
        summary_df,
        metric="rel_success_final_mean",
        k=PRIMARY_K,
        save_path=os.path.join(plots_dir, f"improvement_rel_success_final_ge_{PRIMARY_K}.png"),
        title=fr"Reliable improvement in relative-novelty success: $D_\max \geq {PRIMARY_K}$",
        ylabel="SNEO - NEO",
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

        q_values = get_q_values(
            jnp.asarray(obs_b),
            train_state.enc_state,
            train_state.q_state,
            use_target=False,
        )
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
            action = sample_action(
                action_key,
                s_tau,
                jnp.asarray(obs_b),
                train_state.enc_state,
                train_state.q_state,
                args.algo.sample_action,
            )
            action = int(jax.device_get(action)[0])

        next_obs, reward, terminated, _, _ = env.step(action)
        next_obs_gray = rgb_to_gray_uint8(next_obs)
        next_obs_b = np.expand_dims(next_obs_gray, axis=0)

        rnd_bonus = float(
            np.asarray(
                jax.device_get(
                    rnd_intrinsic_reward(
                        train_state.rnd_state,
                        jnp.asarray(next_obs_b.astype(np.float32)),
                    )
                )
            )[0]
        )

        with jax.default_device(buffer_device):
            timestep = TimeStep(
                observation=jnp.asarray(obs_b[:, None, ...]),
                next_observation=jnp.asarray(next_obs_b[:, None, ...]),
                action=jnp.expand_dims(jnp.asarray(action, dtype=jnp.int32), (0, 1)),
                reward=jnp.expand_dims(jnp.asarray(reward, dtype=jnp.float32), (0, 1)),
                done=jnp.expand_dims(
                    jnp.asarray(terminated or eps_step >= args.algo.max_eps_len, dtype=jnp.bool_),
                    (0, 1),
                ),
                rnd_bonus=jnp.expand_dims(jnp.asarray(rnd_bonus, dtype=jnp.float32), (0, 1)),
            )
            rb_state = rb.add(rb_state, timestep)

        eps_return += reward

        if terminated or eps_step >= args.algo.max_eps_len:
            log_batch.append(
                f"Seed {seed} | Step {global_step} | Episode length {eps_step} | Return {eps_return}"
            )
            next_obs, _ = env.reset()
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
            # -------------------------
            # RND stats update
            # -------------------------
            train_state = train_state.replace(
                rnd_state=rnd_update_obs_rms(
                    train_state.rnd_state,
                    jnp.asarray(next_obs_b.astype(np.float32)),
                )
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

            # -------------------------
            # Learning update
            # -------------------------
            if global_step % args.algo.train_freq == 0:
                beta = linear_schedule(
                    args.buffer.beta,
                    0.1,
                    total_timesteps,
                    global_step - args.algo.starts,
                )

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

                # -------------------------
                # Laplacian update
                # -------------------------
                if args.laplacian.upd_freq is not None and global_step % args.laplacian.upd_freq == 0:
                    lap_obs = q_data.observation[:, : args.laplacian.n_step + 1, ...]
                    lap_dones = q_data.done[:, : args.laplacian.n_step + 1]
                    lap_actions = q_data.action[:, : args.laplacian.n_step + 1]

                    raw_bonus_seq, rnd_bonus_seq, rnd_bonus_n_seq, inv_count_seq, inv_used_seq = (
                        compute_laplace_neo_targets_from_obs_seq(
                            train_state,
                            lap_obs.astype(jnp.float32),
                        )
                    )

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
                    train_state, (eig_loss, pos_loss, ortho_loss, dual_loss) = update_lap_network(
                        lap_key,
                        train_state,
                        lap_batch,
                        n_actions,
                    )

                # -------------------------
                # RND predictor update
                # -------------------------
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

                train_state = train_state.replace(rnd_state=rnd_state2)

            # -------------------------
            # Target network update
            # -------------------------
            if (args.algo.tau == 1 and global_step % args.algo.tgt_upd_freq == 0) or (args.algo.tau < 1):
                enc_state, q_state = update_target_network(
                    args.algo.tau,
                    train_state.enc_state,
                    train_state.q_state,
                )
                train_state = train_state.replace(enc_state=enc_state, q_state=q_state)

            # -------------------------
            # Optional environment policy evaluation
            # -------------------------
            if (global_step >= total_timesteps * 0.5) and global_step % 5000 == 0:
                reached_goal, test_ret, test_len = evaluate_policy(train_state, env, global_step, args)
                test_results.append(float(reached_goal))
                success_rate = float(np.mean(test_results))
                print(f"[seed={seed}] Goal success rate: {success_rate:.3f}")

        train_state = train_state.replace(
            timesteps=global_step,
            n_updates=train_state.n_updates + int(global_step >= args.algo.starts),
        )

        # -------------------------
        # Clean diagnostics
        # -------------------------
        if global_step in eval_points:
            print(f"[seed={seed}] Running escape/success diagnostics at step {global_step}")

            rows_df, summary_df = run_reliable_vs_vanilla_diagnostics(
                seed=seed,
                global_step=global_step,
                train_state=train_state,
                env=env,
                args=args,
            )

            rows_all.append(rows_df)
            summaries_all.append(summary_df)

            rows_so_far = pd.concat(rows_all, ignore_index=True)
            summary_so_far = pd.concat(summaries_all, ignore_index=True)

            rows_so_far.to_csv(os.path.join(out_seed_dir, "diag_rows.csv"), index=False)
            summary_so_far.to_csv(os.path.join(out_seed_dir, "diag_summary.csv"), index=False)

            # Per-seed plots are useful for debugging, but the paper figures should use aggregate plots.
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
        # Use GPU for model state / training.
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


def parse_local_args():
    parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument("--diag_seeds", type=str, default="0")
    parser.add_argument("--diag_seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--diag_out", type=str, default=None)
    parser.add_argument("--primary_k", type=int, default=PRIMARY_K)
    parser.add_argument("--reliable_beta", type=float, default=RELIABLE_RND_BETA)
    parser.add_argument("--escape_ks", type=str, default="2,3,4,5")
    local_args, _ = parser.parse_known_args()
    return local_args

if __name__ == "__main__":
    print("Available JAX devices:", jax.devices())

    local_args = parse_local_args()

    # Update globals from local command-line args.
    ESCAPE_KS = tuple(int(x) for x in local_args.escape_ks.split(","))
    PRIMARY_K = int(local_args.primary_k)
    RELIABLE_RND_BETA = float(local_args.reliable_beta)

    seeds = [int(x) for x in local_args.diag_seeds.split(",")]

    run_group = (
        f"{base_args.env.maze_type}"
        f"_steps{base_args.algo.tot_steps}"
        f"_seeds{len(seeds)}"
        f"_k{PRIMARY_K}"
        f"_beta{RELIABLE_RND_BETA}"
        f"_{int(time.time())}"
    )

    out_root = local_args.diag_out
    if out_root is None:
        out_root = os.path.join("artifacts", "SNEO_Vs_NEO", run_group)

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

        # Clear between seeds to reduce memory fragmentation.
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