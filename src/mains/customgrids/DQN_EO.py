#DQN_EO.py
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, Union

import numpy as np


# Make direct execution (``python src/mains/.../DQN_DCEO.py``) work too.
_REPO_PARENT = str(Path(__file__).resolve().parents[3])
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)


def _has_cli_flag(argv: list[str], flag: str) -> bool:
    return any(item == flag or item.startswith(flag + "=") for item in argv)


def _parse_args():
    """Parse DCEO flags first, then delegate existing flags to the repo config."""
    local = argparse.ArgumentParser(add_help=False)
    local.add_argument("--dceo_mu", type=float, default=0.5)
    local.add_argument("--dceo_option_gamma", type=float, default=0.95)
    local.add_argument("--dceo_option_n_step", type=int, default=5)
    local.add_argument("--dceo_option_temperature", type=float, default=0.25)
    local.add_argument("--dceo_option_lr", type=float, default=2.5e-4)
    local.add_argument("--dceo_log_interval", type=int, default=1000)
    local.add_argument("--dceo_eval_interval", type=int, default=5000)
    local.add_argument("--dceo_eval_episodes", type=int, default=10)
    dceo, remaining = local.parse_known_args()

    # Defaults used for the paper's GridRoom-16 mechanistic experiment.  Every
    # value remains overridable through the existing custom-grid CLI.
    defaults = {
        "--e_maze_type": "GridRoom-16",
        "--a_n_options": "5",
        "--a_opt_dur": "5",
        "--a_tot_steps": "50000",
        "--a_starts": "5000",
        "--a_n_step": "5",
        "--lap_n_step": "5",
        "--lap_br_val": "100",
        "--x_file": "DQN_DCEO",
        "--wb_model": "EO",
    }
    original_argv = sys.argv
    try:
        delegated = list(remaining)
        for flag, value in defaults.items():
            if not _has_cli_flag(delegated, flag):
                delegated.extend([flag, value])
        sys.argv = [original_argv[0], *delegated]
        from src.config.customgrids.laplacian import convert_args_to_dict, parse_args

        base = convert_args_to_dict(parse_args())
    finally:
        sys.argv = original_argv

    if not (0.0 <= dceo.dceo_mu <= 1.0):
        raise ValueError("--dceo_mu must lie in [0, 1]")
    if dceo.dceo_option_n_step < 1:
        raise ValueError("--dceo_option_n_step must be positive")
    if dceo.dceo_option_temperature < 0.0:
        raise ValueError("--dceo_option_temperature must be non-negative")
    if dceo.dceo_log_interval < 1 or dceo.dceo_eval_interval < 1 or dceo.dceo_eval_episodes < 1:
        raise ValueError("DCEO logging/evaluation intervals and episode count must be positive")
    if base.algo.n_options < 1 or base.algo.n_options + 1 > base.laplacian.dims:
        raise ValueError("regular Laplacian options require a_n_options + 1 <= lap_dims")
    if base.laplacian.n_step is None:
        raise ValueError(
            "Do not pass '--dqn true' to DQN_DCEO.py: that legacy config flag disables "
            "the Laplacian module required by DCEO. The primitive learner here is already DQN."
        )
    return base, dceo


ARGS, DCEO = _parse_args()

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import chex
import flax
import jax
import jax.numpy as jnp
import optax
import wandb
from flax import linen as nn
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

import src.fbx_buffers.prioritised_trajectory_buffer as prioritised_trajectory_buffer
import src.fbx_buffers.trajectory_buffer as trajectory_buffer
from src.config.customgrids.laplacian import close_logger, create_wb_logger
from src.losses.laplacian import multi_step_ALLO, update_ALLO_parameters, update_eigenvalues
from src.mains.algorithms.Laplacian import compute_batch_eigenvalues_seq
from src.networks.networks import Conv_Encoder
from src.utils.grid_agent import GridAgent
from src.utils.jax_utils import linear_schedule


class TargetTrainState(TrainState):
    target_params: FrozenDict
    batch_stats: Union[FrozenDict, None] = None
    target_batch_stats: Union[FrozenDict, None] = None


@struct.dataclass
class LapState(TrainState):
    betas: Any
    errors: Any
    quadratic_errors: Any
    barrier_coeffs: Any
    eigen_values: Any
    batch_stats: Union[FrozenDict, None]


@struct.dataclass
class DCEOTrainState:
    dqn_enc: TargetTrainState
    dqn_q: TargetTrainState
    option_enc: TargetTrainState
    option_q: TargetTrainState
    lap: LapState
    grad_steps: int = 0


@chex.dataclass(frozen=True)
class TimeStep:
    observation: chex.Array
    next_observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


class LapBatch(NamedTuple):
    states: jnp.ndarray       # [B, N+1, H, W, C]
    actions: jnp.ndarray      # [B, N]
    loss_mask: jnp.ndarray    # [B, N], false at the terminal transition
    transition_mask: jnp.ndarray  # [B, N], includes the terminal successor


class QHead(nn.Module):
    n_actions: int
    hidden_dims: tuple[int, ...]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        del train
        for width in self.hidden_dims:
            x = nn.Dense(width)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        return nn.Dense(self.n_actions)(x)


class OptionQHeads(nn.Module):
    n_options: int
    n_actions: int
    hidden_dims: tuple[int, ...]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        values = [
            QHead(
                n_actions=self.n_actions,
                hidden_dims=self.hidden_dims,
                layer_norm=self.layer_norm,
                name=f"option_{option_id}",
            )(x, train=train)
            for option_id in range(self.n_options)
        ]
        return jnp.stack(values, axis=1)  # [B, O, A]


def rgb_to_gray_uint8(obs_rgb: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs_rgb, dtype=np.float32)
    if obs.ndim == 2:
        return np.clip(obs, 0, 255).astype(np.uint8)[..., None]
    if obs.shape[-1] == 1:
        return np.clip(obs, 0, 255).astype(np.uint8)
    gray = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
    return np.clip(gray, 0, 255).astype(np.uint8)[..., None]


def set_env_episode_limit(env, max_episode_steps: int) -> bool:
    """Override any Gymnasium TimeLimit wrapper with the configured DQN cap."""
    current = env
    found_time_limit = False
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "_max_episode_steps"):
            current._max_episode_steps = int(max_episode_steps)
            found_time_limit = True
        current = getattr(current, "env", None)
    return found_time_limit


def total_option_count(args) -> int:
    multiplier = 2 if args.algo.bidir_opt else 1
    return int(args.algo.n_options * multiplier)


def _variables(state: TargetTrainState, target: bool = False):
    params = state.target_params if target else state.params
    batch_stats = state.target_batch_stats if target else state.batch_stats
    variables = {"params": params}
    if batch_stats is not None:
        variables["batch_stats"] = batch_stats
    return variables


def _make_target_state(network, key, sample, tx, *, train=False):
    variables = network.init(key, sample, train=train)
    batch_stats = variables.get("batch_stats")
    return TargetTrainState.create(
        apply_fn=network.apply,
        tx=tx,
        params=variables["params"],
        batch_stats=batch_stats,
        target_params=variables["params"],
        target_batch_stats=batch_stats,
    )


def setup_replay(args, observation_shape, sequence_length):
    kwargs = dict(
        max_size=args.buffer.size,
        period=1,
        sample_sequence_length=sequence_length,
        add_batch_size=1,
        min_length_time_axis=args.algo.starts,
        sample_batch_size=max(args.buffer.q_batch_size, args.buffer.lap_batch_size),
    )
    if args.buffer.prioritized:
        kwargs["priority_exponent"] = 0.6
        make_buffer = prioritised_trajectory_buffer.make_prioritised_trajectory_buffer
    else:
        make_buffer = trajectory_buffer.make_trajectory_buffer

    replay = make_buffer(**kwargs)
    replay = replay.replace(
        init=jax.jit(replay.init),
        can_sample=jax.jit(replay.can_sample),
        add=jax.jit(replay.add, donate_argnums=0),
        sample=jax.jit(replay.sample),
    )
    if args.buffer.prioritized:
        replay = replay.replace(set_priorities=jax.jit(replay.set_priorities, donate_argnums=0))

    dummy = TimeStep(
        observation=jnp.zeros(observation_shape, dtype=jnp.uint8),
        next_observation=jnp.zeros(observation_shape, dtype=jnp.uint8),
        action=jnp.int32(0),
        reward=jnp.float32(0.0),
        done=jnp.bool_(False),
    )
    return replay, replay.init(dummy)


def setup_train_state(key, sample_obs, n_actions, args, dceo):
    keys = jax.random.split(key, 5)
    n_options = total_option_count(args)

    dqn_enc_net = Conv_Encoder(
        name="DQN:Enc",
        min_scale=0.5,
        batch_norm=False,
        layer_norm=args.net.layer_norm,
        layer_dims=list(args.net.c_dims),
        activations=[nn.relu] * len(args.net.c_dims),
    )
    dqn_q_net = QHead(n_actions, tuple(args.net.q_dims), args.net.layer_norm, name="DQN:Q")

    option_enc_net = Conv_Encoder(
        name="DCEO:OptionEnc",
        min_scale=0.5,
        batch_norm=False,
        layer_norm=args.net.layer_norm,
        layer_dims=list(args.net.c_dims),
        activations=[nn.relu] * len(args.net.c_dims),
    )
    option_q_net = OptionQHeads(
        n_options=n_options,
        n_actions=n_actions,
        hidden_dims=tuple(args.net.q_dims),
        layer_norm=args.net.layer_norm,
        name="DCEO:OptionQ",
    )
    lap_net = Conv_Encoder(
        name="DCEO:Laplacian",
        min_scale=0.5,
        batch_norm=args.laplacian.batch_norm,
        layer_norm=args.laplacian.layer_norm,
        layer_dims=list(args.net.c_dims) + [args.laplacian.dims],
        activations=[nn.relu] * len(args.net.c_dims) + [None],
    )

    main_tx = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adam(args.net.lr.ffn, eps=1.5e-4),
    )
    option_tx = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adam(dceo.dceo_option_lr, eps=1.5e-4),
    )
    lap_tx = optax.chain(
        optax.clip_by_global_norm(args.laplacian.max_grad_norm),
        optax.adamw(args.laplacian.lr, eps=1.5e-4, weight_decay=1e-4),
    )

    sample_obs = jnp.asarray(sample_obs)
    dummy_main = jnp.ones((1, args.net.c_dims[-1]), dtype=jnp.float32)
    dqn_enc = _make_target_state(dqn_enc_net, keys[0], sample_obs, main_tx)
    dqn_q = _make_target_state(dqn_q_net, keys[1], dummy_main, main_tx)
    option_enc = _make_target_state(option_enc_net, keys[2], sample_obs, option_tx)
    option_q = _make_target_state(option_q_net, keys[3], dummy_main, option_tx)

    lap_variables = lap_net.init(keys[4], sample_obs, train=False)
    lap_batch_stats = lap_variables.get("batch_stats")
    lap = LapState.create(
        apply_fn=lap_net.apply,
        tx=lap_tx,
        params=lap_variables["params"],
        batch_stats=lap_batch_stats,
        barrier_coeffs=jnp.tril(jnp.ones((1, 1), dtype=jnp.float32)) * args.laplacian.barrier.value,
        betas=jnp.zeros((args.laplacian.dims, args.laplacian.dims), dtype=jnp.float32),
        eigen_values=jnp.ones((args.laplacian.dims,), dtype=jnp.float32) * 1e-3,
        errors=jnp.zeros((args.laplacian.dims, args.laplacian.dims), dtype=jnp.float32),
        quadratic_errors=jnp.zeros((args.laplacian.dims, args.laplacian.dims), dtype=jnp.float32),
    )
    return DCEOTrainState(dqn_enc, dqn_q, option_enc, option_q, lap)


@partial(jax.jit, static_argnums=(3,))
def dqn_values(obs, enc_state, q_state, target=False):
    embedding = enc_state.apply_fn(_variables(enc_state, target), obs, train=False)
    return q_state.apply_fn(_variables(q_state, target), embedding, train=False)


@partial(jax.jit, static_argnums=(3,))
def option_values(obs, enc_state, q_state, target=False):
    embedding = enc_state.apply_fn(_variables(enc_state, target), obs, train=False)
    return q_state.apply_fn(_variables(q_state, target), embedding, train=False)


@jax.jit
def greedy_dqn_action(obs, enc_state, q_state):
    return jnp.argmax(dqn_values(obs, enc_state, q_state), axis=-1)


@partial(jax.jit, static_argnums=(5,))
def option_action(key, obs, option_id, enc_state, q_state, temperature):
    q = option_values(obs, enc_state, q_state)[0, option_id]
    return jax.lax.cond(
        temperature > 0.0,
        lambda _: jax.random.categorical(key, q / jnp.maximum(temperature, 1e-8)),
        lambda _: jnp.argmax(q),
        operand=None,
    )


@partial(jax.jit, static_argnums=(0,))
def update_targets(tau, state: DCEOTrainState):
    def one(s):
        target_batch_stats = s.target_batch_stats
        if s.batch_stats is not None:
            target_batch_stats = optax.incremental_update(s.batch_stats, s.target_batch_stats, tau)
        return s.replace(
            target_params=optax.incremental_update(s.params, s.target_params, tau),
            target_batch_stats=target_batch_stats,
        )

    return state.replace(
        dqn_enc=one(state.dqn_enc),
        dqn_q=one(state.dqn_q),
        option_enc=one(state.option_enc),
        option_q=one(state.option_q),
    )


@jax.jit
def replay_key(key):
    return jax.random.split(key)


@jax.jit
def importance_weights(probabilities, running_index, capacity, sequence_length, beta):
    n_valid = jnp.clip(running_index - (sequence_length - 1), a_min=1)
    n_valid = jnp.minimum(n_valid, capacity)
    weights = (1.0 / (n_valid * probabilities + 1e-8)) ** beta
    return weights / (jnp.max(weights) + 1e-8)


def make_dqn_update(args):
    n_step = args.algo.n_step
    gamma = args.env.gamma

    @partial(jax.jit, static_argnums=(4,))
    def update(data, weights, enc_state, q_state, use_ddqn):
        next_obs = data.next_observation[:, n_step - 1]
        q_next_target_all = dqn_values(next_obs, enc_state, q_state, target=True)
        if use_ddqn:
            next_ids = jnp.argmax(dqn_values(next_obs, enc_state, q_state), axis=-1)
            q_next = q_next_target_all[jnp.arange(next_ids.shape[0]), next_ids]
        else:
            q_next = jnp.max(q_next_target_all, axis=-1)

        rewards = data.reward[:, :n_step]
        dones = data.done[:, :n_step]
        alive = jnp.cumprod(
            jnp.concatenate([jnp.ones_like(dones[:, :1], dtype=jnp.float32), 1.0 - dones[:, :-1]], axis=1),
            axis=1,
        )
        returns = jnp.sum(rewards * alive * gamma ** jnp.arange(n_step)[None], axis=1)
        bootstrap = 1.0 - jnp.any(dones, axis=1).astype(jnp.float32)

        def loss_fn(enc_params, q_params):
            embedding = enc_state.apply_fn({"params": enc_params}, data.observation[:, 0], train=True)
            q_all = q_state.apply_fn({"params": q_params}, embedding, train=True)
            q_taken = q_all[jnp.arange(q_all.shape[0]), data.action[:, 0]]
            target = returns + bootstrap * (gamma ** n_step) * q_next
            td_error = q_taken - jax.lax.stop_gradient(target)
            per_sample = jnp.square(td_error)
            loss = jnp.mean(per_sample if weights is None else weights * per_sample)
            return loss, (jnp.mean(per_sample), jnp.mean(q_taken), jnp.abs(td_error) + 1e-8)

        (_, (td_loss, q_mean, priorities)), grads = jax.value_and_grad(
            loss_fn, has_aux=True, argnums=(0, 1)
        )(enc_state.params, q_state.params)
        enc_state = enc_state.apply_gradients(grads=grads[0])
        q_state = q_state.apply_gradients(grads=grads[1])
        return enc_state, q_state, td_loss, q_mean, priorities

    return update


def _lap_embeddings(lap_state, observations, train=False, mutable=False):
    variables = {"params": lap_state.params}
    if lap_state.batch_stats is not None:
        variables["batch_stats"] = lap_state.batch_stats
    mutable_cols = ["batch_stats"] if mutable and lap_state.batch_stats is not None else False
    return lap_state.apply_fn(variables, observations, train=train, mutable=mutable_cols)


def make_option_update(args, dceo):
    n_step = dceo.dceo_option_n_step
    gamma = dceo.dceo_option_gamma
    n_base_options = args.algo.n_options
    n_options = total_option_count(args)
    bidirectional = args.algo.bidir_opt

    @partial(jax.jit, static_argnums=(5,))
    def update(data, weights, state, enc_state, q_state, use_ddqn):
        obs = data.observation[:, :n_step]
        next_obs = data.next_observation[:, :n_step]
        batch, steps = obs.shape[:2]
        flat_obs = obs.reshape((batch * steps,) + obs.shape[2:]).astype(jnp.float32)
        flat_next = next_obs.reshape((batch * steps,) + next_obs.shape[2:]).astype(jnp.float32)

        # The ordinary graph Laplacian has a constant first eigenfunction.
        # Match EO.py by skipping it and using signed potential differences.
        phi_all = _lap_embeddings(state.lap, flat_obs).reshape(batch, steps, -1)
        next_phi_all = _lap_embeddings(state.lap, flat_next).reshape(batch, steps, -1)
        phi = phi_all[..., 1:1 + n_base_options]
        next_phi = next_phi_all[..., 1:1 + n_base_options]
        forward_reward = next_phi - phi
        eigen_reward = (
            jnp.concatenate([forward_reward, -forward_reward], axis=-1)
            if bidirectional
            else forward_reward
        )

        dones = data.done[:, :n_step]
        alive = jnp.cumprod(
            jnp.concatenate([jnp.ones_like(dones[:, :1], dtype=jnp.float32), 1.0 - dones[:, :-1]], axis=1),
            axis=1,
        )
        # Preserve the repository DCEO/EO convention: terminal intrinsic
        # rewards are zero, while earlier rewards remain valid until done.
        option_rewards = eigen_reward * (1.0 - dones.astype(jnp.float32))[..., None]
        returns = jnp.sum(
            option_rewards * alive[..., None] * gamma ** jnp.arange(n_step)[None, :, None], axis=1
        )
        bootstrap = 1.0 - jnp.any(dones, axis=1).astype(jnp.float32)

        last_next = next_obs[:, n_step - 1]
        q_target_all = option_values(last_next, enc_state, q_state, target=True)
        if use_ddqn:
            next_ids = jnp.argmax(option_values(last_next, enc_state, q_state), axis=-1)
            q_boot = jnp.take_along_axis(q_target_all, next_ids[..., None], axis=-1).squeeze(-1)
        else:
            q_boot = jnp.max(q_target_all, axis=-1)

        def loss_fn(enc_params, q_params):
            embedding = enc_state.apply_fn({"params": enc_params}, data.observation[:, 0], train=True)
            q_all = q_state.apply_fn({"params": q_params}, embedding, train=True)
            actions = jnp.broadcast_to(data.action[:, 0, None, None], (batch, n_options, 1))
            q_taken = jnp.take_along_axis(q_all, actions, axis=-1).squeeze(-1)
            target = returns + bootstrap[:, None] * (gamma ** n_step) * q_boot
            td_error = q_taken - jax.lax.stop_gradient(target)
            per_sample = jnp.mean(jnp.square(td_error), axis=1)
            loss = jnp.mean(per_sample if weights is None else weights * per_sample)
            return loss, (jnp.mean(jnp.square(td_error)), jnp.mean(q_taken), jnp.float32(0.0))

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))(
            enc_state.params, q_state.params
        )
        enc_state = enc_state.apply_gradients(grads=grads[0])
        q_state = q_state.apply_gradients(grads=grads[1])
        return enc_state, q_state, metrics

    return update


def make_laplacian_update(args):
    use_bn = args.laplacian.batch_norm
    n_steps = args.laplacian.n_step

    @jax.jit
    def update(key, state: DCEOTrainState, batch: LapBatch):
        lap = state.lap
        bsz, seq_len = batch.states.shape[:2]
        flat = batch.states.reshape((bsz * seq_len,) + batch.states.shape[2:]).astype(jnp.float32)

        def loss_fn(params):
            variables = {"params": params}
            if use_bn:
                variables["batch_stats"] = lap.batch_stats
                flat_phi, updates = lap.apply_fn(variables, flat, train=True, mutable=["batch_stats"])
            else:
                flat_phi = lap.apply_fn(variables, flat, train=True)
                updates = None
            phi = flat_phi.reshape(bsz, seq_len, args.laplacian.dims)
            loss, (pos, ortho, dual), (_, errors) = multi_step_ALLO(
                key=key,
                phis_all=phi[:, 0],
                future_phis_all=phi[:, 1:],
                actions=batch.actions,
                sample_mask=jnp.ones((bsz,), dtype=jnp.bool_),
                # multi_step_ALLO includes the first false entry, so a false
                # at transition t correctly retains the terminal successor.
                masks=batch.loss_mask,
                barrier_coeffs=lap.barrier_coeffs,
                betas=lap.betas,
                weights=None,
                novelty_values=None,
                zeta_values=None,
                use_novelty=False,
                use_zeta=False,
                normalize_zeta=True,
                gamma=args.env.gamma,
                state_action=False,
                uni_policy=False,
                n_steps=n_steps,
                use_dual=args.laplacian.dual.on,
                use_weights=False,
                average_loss=args.laplacian.average_loss,
                rms_norm=getattr(args.laplacian, "rms_norm", False),
                barrier_dims=1,
            )
            eigenvalues = compute_batch_eigenvalues_seq(phi, batch.transition_mask.astype(jnp.float32))
            return loss, (updates, pos, ortho, dual, errors, eigenvalues)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(lap.params)
        updates, pos, ortho, dual, errors, eigenvalues = aux
        lap = lap.apply_gradients(grads=grads)
        if use_bn:
            lap = lap.replace(batch_stats=updates["batch_stats"])
        if args.laplacian.barrier.on or args.laplacian.dual.on:
            lap = update_ALLO_parameters(
                lap,
                errors,
                update_rates=args.laplacian.rate,
                barrier_args=args.laplacian.barrier,
                dual_args=args.laplacian.dual,
                average_loss=args.laplacian.average_loss,
            )
        lap = lap.replace(eigen_values=update_eigenvalues(lap.eigen_values, eigenvalues, alpha=0.1))
        return state.replace(lap=lap), (loss, pos, ortho, dual)

    return update


def evaluate(state, env, args, episodes):
    returns, lengths, successes = [], [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=args.exp.seed + 10000 + episode)
        total = 0.0
        terminated = truncated = False
        length = 0
        while not (terminated or truncated) and length < args.algo.max_eps_len:
            obs_b = jnp.asarray(rgb_to_gray_uint8(obs)[None])
            action = int(jax.device_get(greedy_dqn_action(obs_b, state.dqn_enc, state.dqn_q))[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            length += 1
        returns.append(total)
        lengths.append(length)
        successes.append(float(terminated and total > 0.0))
    return {
        "test/episodic_return": float(np.mean(returns)),
        "test/episodic_length": float(np.mean(lengths)),
        "test/success_rate": float(np.mean(successes)),
    }


def train(args=ARGS, dceo=DCEO):
    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    key = jax.random.PRNGKey(args.exp.seed)

    grid = GridAgent(args)
    env = grid.env
    eval_grid = GridAgent(args)
    eval_env = eval_grid.env
    train_limit_overridden = set_env_episode_limit(env, args.algo.max_eps_len)
    eval_limit_overridden = set_env_episode_limit(eval_env, args.algo.max_eps_len)
    if train_limit_overridden or eval_limit_overridden:
        print(f"Environment TimeLimit set to {args.algo.max_eps_len} steps")
    obs, _ = env.reset(seed=args.exp.seed)
    obs_gray = rgb_to_gray_uint8(obs)
    n_actions = env.action_space.n
    n_options = total_option_count(args)

    sequence_length = max(args.algo.n_step, args.laplacian.n_step, dceo.dceo_option_n_step)
    replay_sequence_length = sequence_length
    buffer_device = jax.devices(args.buffer.device)[0]
    with jax.default_device(buffer_device):
        replay, replay_state = setup_replay(args, obs_gray.shape, replay_sequence_length)

    key, setup_key = jax.random.split(key)
    state = setup_train_state(setup_key, obs_gray[None], n_actions, args, dceo)
    dqn_update = make_dqn_update(args)
    option_update = make_option_update(args, dceo)
    lap_update = make_laplacian_update(args)

    active_option = -1
    option_steps_left = 0
    episode_return = 0.0
    episode_length = 0
    completed_returns = []
    action_source_counts = {"dqn": 0, "random": 0, "option": 0}
    start_time = time.time()
    last_metrics = {}

    try:
        for step in range(args.algo.tot_steps + 1):
            obs_gray = rgb_to_gray_uint8(obs)
            obs_b = jnp.asarray(obs_gray[None])
            key, action_key, explore_key, mu_key, choice_key = jax.random.split(key, 5)

            epsilon = linear_schedule(
                args.algo.eps.start,
                args.algo.eps.end,
                int(args.algo.eps.frac * args.algo.tot_steps),
                step,
            )
            if active_option >= 0 and option_steps_left > 0:
                action_source = "option"
                action = int(jax.device_get(option_action(
                    action_key, obs_b, active_option, state.option_enc, state.option_q,
                    dceo.dceo_option_temperature,
                )))
                option_steps_left -= 1
                if option_steps_left == 0:
                    active_option = -1
            elif float(jax.random.uniform(explore_key)) < epsilon:
                if float(jax.random.uniform(mu_key)) < dceo.dceo_mu:
                    action_source = "option"
                    active_option = int(jax.random.randint(choice_key, (), 0, n_options))
                    action = int(jax.device_get(option_action(
                        action_key, obs_b, active_option, state.option_enc, state.option_q,
                        dceo.dceo_option_temperature,
                    )))
                    option_steps_left = max(args.algo.opt_dur - 1, 0)
                    if option_steps_left == 0:
                        active_option = -1
                else:
                    action_source = "random"
                    action = int(jax.random.randint(choice_key, (), 0, n_actions))
            else:
                action_source = "dqn"
                action = int(jax.device_get(greedy_dqn_action(obs_b, state.dqn_enc, state.dqn_q))[0])
            action_source_counts[action_source] += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_length += 1
            forced_truncation = episode_length >= args.algo.max_eps_len
            done = bool(terminated or truncated or forced_truncation)
            next_gray = rgb_to_gray_uint8(next_obs)

            with jax.default_device(buffer_device):
                transition = TimeStep(
                    observation=jnp.asarray(obs_gray[None, None]),
                    next_observation=jnp.asarray(next_gray[None, None]),
                    action=jnp.asarray([[action]], dtype=jnp.int32),
                    reward=jnp.asarray([[reward]], dtype=jnp.float32),
                    done=jnp.asarray([[done]], dtype=jnp.bool_),
                )
                replay_state = replay.add(replay_state, transition)

            episode_return += float(reward)
            obs = next_obs
            if done:
                if args.wb.on and step >= args.algo.starts:
                    wandb.log(
                        {
                            "values/episodic_return": episode_return,
                            "values/episodic_length": episode_length,
                            "values/epsilon": float(epsilon),
                        },
                        step=step,
                    )
                completed_returns.append(episode_return)
                obs, _ = env.reset()
                active_option = -1
                option_steps_left = 0
                episode_return = 0.0
                episode_length = 0

            if step == args.algo.starts:
                if not bool(replay.can_sample(replay_state)):
                    raise RuntimeError("Replay buffer cannot sample at learning start")
                print(f"Learning starts at step {step}")

            if step >= args.algo.starts and step % args.algo.train_freq == 0:
                beta = linear_schedule(args.buffer.beta, 1.0, args.algo.tot_steps, step)
                with jax.default_device(buffer_device):
                    key, sample_key = replay_key(key)
                    sampled = replay.sample(replay_state, sample_key)
                data = sampled.experience
                if args.buffer.prioritized:
                    weights = importance_weights(
                        sampled.probabilities,
                        replay_state.running_index,
                        replay_state.sum_tree_state.capacity,
                        replay_sequence_length,
                        beta,
                    )
                else:
                    weights = None

                dqn_enc, dqn_q, td_loss, q_mean, priorities = dqn_update(
                    data, weights, state.dqn_enc, state.dqn_q, args.algo.ddqn
                )
                option_enc, option_q, option_metrics = option_update(
                    data, weights, state, state.option_enc, state.option_q, args.algo.ddqn
                )
                state = state.replace(
                    dqn_enc=dqn_enc,
                    dqn_q=dqn_q,
                    option_enc=option_enc,
                    option_q=option_q,
                    grad_steps=state.grad_steps + 2,
                )

                if args.buffer.prioritized:
                    with jax.default_device(buffer_device):
                        replay_state = replay.set_priorities(replay_state, sampled.indices, priorities)

                if args.laplacian.upd_freq is not None and step % args.laplacian.upd_freq == 0:
                    n_lap = args.laplacian.n_step
                    lap_states = jnp.concatenate(
                        [data.observation[:, :1], data.next_observation[:, :n_lap]], axis=1
                    )
                    lap_dones = data.done[:, :n_lap]
                    lap_transition_mask = jnp.cumprod(
                        jnp.concatenate(
                            [jnp.ones_like(lap_dones[:, :1], dtype=jnp.float32),
                             1.0 - lap_dones[:, :-1]], axis=1
                        ), axis=1,
                    ).astype(jnp.bool_)
                    lap_loss_mask = jnp.cumprod(
                        1.0 - lap_dones.astype(jnp.float32), axis=1
                    ).astype(jnp.bool_)
                    lap_batch = LapBatch(
                        states=lap_states,
                        actions=data.action[:, :n_lap],
                        loss_mask=lap_loss_mask,
                        transition_mask=lap_transition_mask,
                    )
                    key, lap_key = jax.random.split(key)
                    state, lap_metrics = lap_update(lap_key, state, lap_batch)
                else:
                    lap_metrics = (jnp.nan, jnp.nan, jnp.nan, jnp.nan)

                # Keep the W&B schema identical to DQN_SNEO.  These two
                # metrics are exactly zero because DCEO has no RND/novelty.
                rnd_loss = jnp.float32(0.0)

                if (args.algo.tau == 1 and step % args.algo.tgt_upd_freq == 0) or args.algo.tau < 1:
                    state = update_targets(args.algo.tau, state)

                last_metrics = {
                    "losses/dqn_td_loss": float(jax.device_get(td_loss)),
                    "losses/option_td_loss": float(jax.device_get(option_metrics[0])),
                    "losses/laplacian_loss": float(jax.device_get(lap_metrics[0])),
                    "losses/rnd_loss": float(jax.device_get(rnd_loss)),
                    "values/q_values": float(jax.device_get(q_mean)),
                    "values/option_q_values": float(jax.device_get(option_metrics[1])),
                    "values/replay_novelty": float(jax.device_get(option_metrics[2])),
                }

            if step > 0 and step % dceo.dceo_eval_interval == 0:
                eval_metrics = evaluate(state, eval_env, args, dceo.dceo_eval_episodes)
                print(
                    f"Evaluation at step {step} | Return: "
                    f"{eval_metrics['test/episodic_return']:.3f} | Success Rate: "
                    f"{eval_metrics['test/success_rate']:.3f}"
                )
                if args.wb.on:
                    wandb.log(eval_metrics, step=step)
                    wandb.run.summary["Success Rate"] = eval_metrics["test/success_rate"]

            if step > 0 and step % dceo.dceo_log_interval == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                action_count = max(sum(action_source_counts.values()), 1)
                record = {
                    "values/epsilon": float(epsilon),
                    "values/episodes": len(completed_returns),
                    "values/recent_return_mean": (
                        float(np.mean(completed_returns[-20:])) if completed_returns else 0.0
                    ),
                    "values/SPS": int(step / elapsed),
                    "behavior/dqn_action_fraction": action_source_counts["dqn"] / action_count,
                    "behavior/random_action_fraction": action_source_counts["random"] / action_count,
                    "behavior/option_action_fraction": action_source_counts["option"] / action_count,
                    **last_metrics,
                }
                print(
                    f"Global Step: {step} | Episodes: {len(completed_returns)} | "
                    f"Recent Return: {record['values/recent_return_mean']:.3f} | "
                    f"Option Actions: {record['behavior/option_action_fraction']:.3f} | "
                    f"SPS: {record['values/SPS']}"
                )
                if args.wb.on:
                    wandb.log(record, step=step)
                action_source_counts = {"dqn": 0, "random": 0, "option": 0}

        return state
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    print("Available JAX devices:", jax.devices())
    create_wb_logger(ARGS)
    logger_closed = False
    if ARGS.wb.on:
        wandb.config.update({f"dceo/{key}": value for key, value in vars(DCEO).items()})
    try:
        train()
        print("Finished training.")
    except Exception:
        if ARGS.wb.on:
            wandb.finish(exit_code=1)
            logger_closed = True
        raise
    finally:
        if ARGS.wb.on and not logger_closed:
            close_logger()
