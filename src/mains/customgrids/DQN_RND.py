from __future__ import annotations

import argparse
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np


_REPO_PARENT = str(Path(__file__).resolve().parents[3])
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)


def _has_cli_flag(argv: list[str], flag: str) -> bool:
    return any(item == flag or item.startswith(flag + "=") for item in argv)


def _parse_args():
    """Parse RND-specific flags, then delegate repository-wide flags."""
    local = argparse.ArgumentParser(add_help=False)
    local.add_argument("--rnd_beta", type=float, default=1.0)
    local.add_argument("--rnd_int_gamma", type=float, default=0.99)
    local.add_argument("--rnd_lr", type=float, default=1e-4)
    local.add_argument("--rnd_update_proportion", type=float, default=0.25)
    local.add_argument("--rnd_burnin_steps", type=int, default=0)
    local.add_argument("--rnd_log_interval", type=int, default=1000)
    local.add_argument("--rnd_eval_interval", type=int, default=5000)
    local.add_argument("--rnd_eval_episodes", type=int, default=10)
    rnd, remaining = local.parse_known_args()

    defaults = {
        "--dqn": "true",
        "--e_maze_type": "GridRoom-16",
        "--a_tot_steps": "50000",
        "--a_starts": "5000",
        "--a_n_step": "5",
        "--x_file": "DQN_RND",
        "--wb_model": "RND",
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

    if rnd.rnd_beta < 0.0:
        raise ValueError("--rnd_beta must be non-negative")
    if not (0.0 <= rnd.rnd_int_gamma <= 1.0):
        raise ValueError("--rnd_int_gamma must lie in [0, 1]")
    if rnd.rnd_lr <= 0.0:
        raise ValueError("--rnd_lr must be positive")
    if not (0.0 < rnd.rnd_update_proportion <= 1.0):
        raise ValueError("--rnd_update_proportion must lie in (0, 1]")
    if rnd.rnd_burnin_steps < 0:
        raise ValueError("--rnd_burnin_steps must be non-negative")
    if rnd.rnd_log_interval < 1 or rnd.rnd_eval_interval < 1 or rnd.rnd_eval_episodes < 1:
        raise ValueError("RND logging/evaluation intervals and episode count must be positive")
    return base, rnd


ARGS, RND = _parse_args()

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import chex
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
from src.mains.algorithms.RND import (
    RNDState,
    rnd_intrinsic_reward,
    rnd_predictor_loss_and_update,
    rnd_setup,
    rnd_update_obs_rms,
    rnd_update_rew_rms,
)
from src.networks.networks import Conv_Encoder
from src.utils.grid_agent import GridAgent
from src.utils.jax_utils import linear_schedule


class TargetTrainState(TrainState):
    target_params: FrozenDict
    batch_stats: Union[FrozenDict, None] = None
    target_batch_stats: Union[FrozenDict, None] = None


@struct.dataclass
class DQNRNDTrainState:
    enc: TargetTrainState
    q_ext: TargetTrainState
    q_int: TargetTrainState
    rnd: RNDState
    grad_steps: int = 0


@chex.dataclass(frozen=True)
class TimeStep:
    observation: chex.Array
    next_observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


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


def rgb_to_gray_uint8(obs_rgb: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs_rgb, dtype=np.float32)
    if obs.ndim == 2:
        return np.clip(obs, 0, 255).astype(np.uint8)[..., None]
    if obs.shape[-1] == 1:
        return np.clip(obs, 0, 255).astype(np.uint8)
    gray = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
    return np.clip(gray, 0, 255).astype(np.uint8)[..., None]


RND_MIN_SPATIAL_SIZE = 42


def prepare_rnd_frames(frames: jnp.ndarray) -> jnp.ndarray:
    """Resize small grid images for the shared Atari-style RND CNN."""
    frames = frames.astype(jnp.float32)
    height, width = frames.shape[-3:-1]
    target_height = max(int(height), RND_MIN_SPATIAL_SIZE)
    target_width = max(int(width), RND_MIN_SPATIAL_SIZE)
    if target_height == height and target_width == width:
        return frames
    target_shape = (*frames.shape[:-3], target_height, target_width, frames.shape[-1])
    return jax.image.resize(frames, target_shape, method="nearest")


def set_env_episode_limit(env, max_episode_steps: int) -> bool:
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


def _variables(state: TargetTrainState, target: bool = False):
    params = state.target_params if target else state.params
    batch_stats = state.target_batch_stats if target else state.batch_stats
    variables = {"params": params}
    if batch_stats is not None:
        variables["batch_stats"] = batch_stats
    return variables


def _make_target_state(network, key, sample, tx):
    variables = network.init(key, sample, train=False)
    batch_stats = variables.get("batch_stats")
    return TargetTrainState.create(
        apply_fn=network.apply,
        tx=tx,
        params=variables["params"],
        batch_stats=batch_stats,
        target_params=variables["params"],
        target_batch_stats=batch_stats,
    )


def setup_replay(args, observation_shape):
    kwargs = dict(
        max_size=args.buffer.size,
        period=1,
        sample_sequence_length=args.algo.n_step,
        add_batch_size=1,
        min_length_time_axis=args.algo.starts,
        sample_batch_size=args.buffer.q_batch_size,
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


def setup_train_state(key, sample_obs, n_actions, args, rnd_args):
    enc_key, ext_key, int_key, rnd_key = jax.random.split(key, 4)
    enc_net = Conv_Encoder(
        name="DQN_RND:Enc",
        min_scale=0.5,
        batch_norm=False,
        layer_norm=args.net.layer_norm,
        layer_dims=list(args.net.c_dims),
        activations=[nn.relu] * len(args.net.c_dims),
    )
    q_ext_net = QHead(n_actions, tuple(args.net.q_dims), args.net.layer_norm, name="DQN_RND:Q_ext")
    q_int_net = QHead(n_actions, tuple(args.net.q_dims), args.net.layer_norm, name="DQN_RND:Q_int")

    q_tx = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adam(args.net.lr.ffn, eps=1.5e-4),
    )
    sample_obs = jnp.asarray(sample_obs)
    dummy_embedding = jnp.ones((1, args.net.c_dims[-1]), dtype=jnp.float32)
    enc = _make_target_state(enc_net, enc_key, sample_obs, q_tx)
    q_ext = _make_target_state(q_ext_net, ext_key, dummy_embedding, q_tx)
    q_int = _make_target_state(q_int_net, int_key, dummy_embedding, q_tx)

    rnd_sample = prepare_rnd_frames(sample_obs)
    rnd = rnd_setup(
        key=rnd_key,
        lr=rnd_args.rnd_lr,
        max_grad_norm=args.net.max_grad_norm,
        n_envs=1,
        int_gamma=rnd_args.rnd_int_gamma,
        update_proportion=rnd_args.rnd_update_proportion,
        obs_shape=tuple(rnd_sample.shape[1:]),
    )
    return DQNRNDTrainState(enc=enc, q_ext=q_ext, q_int=q_int, rnd=rnd)


@partial(jax.jit, static_argnums=(3,))
def q_values(obs, enc_state, q_state, target=False):
    embedding = enc_state.apply_fn(_variables(enc_state, target), obs, train=False)
    return q_state.apply_fn(_variables(q_state, target), embedding, train=False)


@jax.jit
def greedy_extrinsic_action(obs, state):
    return jnp.argmax(q_values(obs, state.enc, state.q_ext), axis=-1)


@jax.jit
def greedy_behavior_action(obs, state, intrinsic_scale):
    q_ext = q_values(obs, state.enc, state.q_ext)
    q_int = q_values(obs, state.enc, state.q_int)
    return jnp.argmax(q_ext + intrinsic_scale * q_int, axis=-1)


@partial(jax.jit, static_argnums=(0,))
def update_targets(tau, state: DQNRNDTrainState):
    def one(network_state):
        target_batch_stats = network_state.target_batch_stats
        if network_state.batch_stats is not None:
            target_batch_stats = optax.incremental_update(
                network_state.batch_stats, network_state.target_batch_stats, tau
            )
        return network_state.replace(
            target_params=optax.incremental_update(
                network_state.params, network_state.target_params, tau
            ),
            target_batch_stats=target_batch_stats,
        )

    return state.replace(enc=one(state.enc), q_ext=one(state.q_ext), q_int=one(state.q_int))


@jax.jit
def replay_key(key):
    return jax.random.split(key)


@jax.jit
def importance_weights(probabilities, running_index, capacity, sequence_length, beta):
    n_valid = jnp.clip(running_index - (sequence_length - 1), a_min=1)
    n_valid = jnp.minimum(n_valid, capacity)
    weights = (1.0 / (n_valid * probabilities + 1e-8)) ** beta
    return weights / (jnp.max(weights) + 1e-8)


def make_q_update(args, rnd_args):
    n_step = args.algo.n_step
    ext_gamma = args.env.gamma
    int_gamma = rnd_args.rnd_int_gamma

    @partial(jax.jit, static_argnums=(3,))
    def update(data, weights, state: DQNRNDTrainState, use_ddqn):
        next_obs = data.next_observation[:, n_step - 1]
        q_ext_target_all = q_values(next_obs, state.enc, state.q_ext, target=True)
        q_int_target_all = q_values(next_obs, state.enc, state.q_int, target=True)
        if use_ddqn:
            ext_ids = jnp.argmax(q_values(next_obs, state.enc, state.q_ext), axis=-1)
            int_ids = jnp.argmax(q_values(next_obs, state.enc, state.q_int), axis=-1)
            q_ext_next = q_ext_target_all[jnp.arange(ext_ids.shape[0]), ext_ids]
            q_int_next = q_int_target_all[jnp.arange(int_ids.shape[0]), int_ids]
        else:
            q_ext_next = jnp.max(q_ext_target_all, axis=-1)
            q_int_next = jnp.max(q_int_target_all, axis=-1)

        rewards_ext = data.reward[:, :n_step]
        dones = data.done[:, :n_step]
        alive = jnp.cumprod(
            jnp.concatenate(
                [jnp.ones_like(dones[:, :1], dtype=jnp.float32), 1.0 - dones[:, :-1]],
                axis=1,
            ),
            axis=1,
        )
        flat_next = data.next_observation[:, :n_step].reshape(
            (-1,) + data.next_observation.shape[2:]
        )
        raw_intrinsic = rnd_intrinsic_reward(
            state.rnd, prepare_rnd_frames(flat_next)
        ).reshape(rewards_ext.shape)
        rewards_int = raw_intrinsic / jnp.sqrt(state.rnd.rew_rms.var + 1e-8)

        ext_return = jnp.sum(
            rewards_ext * alive * ext_gamma ** jnp.arange(n_step)[None], axis=1
        )
        # RND uses a continuing intrinsic objective: environment episode
        # boundaries terminate extrinsic returns, but not the novelty stream.
        int_return = jnp.sum(
            rewards_int * int_gamma ** jnp.arange(n_step)[None], axis=1
        )
        bootstrap = 1.0 - jnp.any(dones, axis=1).astype(jnp.float32)
        ext_target = ext_return + bootstrap * (ext_gamma ** n_step) * q_ext_next
        int_target = int_return + (int_gamma ** n_step) * q_int_next

        def loss_fn(enc_params, ext_params, int_params):
            embedding = state.enc.apply_fn(
                {"params": enc_params}, data.observation[:, 0], train=True
            )
            q_ext_all = state.q_ext.apply_fn({"params": ext_params}, embedding, train=True)
            q_int_all = state.q_int.apply_fn({"params": int_params}, embedding, train=True)
            batch_ids = jnp.arange(q_ext_all.shape[0])
            actions = data.action[:, 0]
            q_ext_taken = q_ext_all[batch_ids, actions]
            q_int_taken = q_int_all[batch_ids, actions]
            ext_error = q_ext_taken - jax.lax.stop_gradient(ext_target)
            int_error = q_int_taken - jax.lax.stop_gradient(int_target)
            ext_per_sample = jnp.square(ext_error)
            int_per_sample = jnp.square(int_error)
            if weights is None:
                loss = jnp.mean(ext_per_sample + int_per_sample)
            else:
                loss = jnp.mean(weights * (ext_per_sample + int_per_sample))
            metrics = (
                jnp.mean(ext_per_sample),
                jnp.mean(int_per_sample),
                jnp.mean(q_ext_taken),
                jnp.mean(q_int_taken),
                jnp.mean(rewards_int),
                jnp.abs(ext_error) + 1e-8,
            )
            return loss, metrics

        (_, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True, argnums=(0, 1, 2)
        )(state.enc.params, state.q_ext.params, state.q_int.params)
        state = state.replace(
            enc=state.enc.apply_gradients(grads=grads[0]),
            q_ext=state.q_ext.apply_gradients(grads=grads[1]),
            q_int=state.q_int.apply_gradients(grads=grads[2]),
            grad_steps=state.grad_steps + 1,
        )
        return state, metrics

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
            action = int(jax.device_get(greedy_extrinsic_action(obs_b, state))[0])
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


def train(args=ARGS, rnd_args=RND):
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
    buffer_device = jax.devices(args.buffer.device)[0]
    with jax.default_device(buffer_device):
        replay, replay_state = setup_replay(args, obs_gray.shape)

    key, setup_key = jax.random.split(key)
    state = setup_train_state(setup_key, obs_gray[None], n_actions, args, rnd_args)
    q_update = make_q_update(args, rnd_args)

    episode_return = 0.0
    episode_length = 0
    completed_returns = []
    action_source_counts = {"dqn": 0, "random": 0}
    start_time = time.time()
    last_metrics = {}

    try:
        for step in range(args.algo.tot_steps + 1):
            obs_gray = rgb_to_gray_uint8(obs)
            obs_b = jnp.asarray(obs_gray[None])
            key, action_key, explore_key = jax.random.split(key, 3)
            epsilon = linear_schedule(
                args.algo.eps.start,
                args.algo.eps.end,
                int(args.algo.eps.frac * args.algo.tot_steps),
                step,
            )
            if float(jax.random.uniform(explore_key)) < epsilon:
                action_source = "random"
                action = int(jax.random.randint(action_key, (), 0, n_actions))
            else:
                action_source = "dqn"
                action = int(
                    jax.device_get(
                        greedy_behavior_action(obs_b, state, rnd_args.rnd_beta)
                    )[0]
                )
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

            # RND running statistics observe the same next states that define
            # the intrinsic reward. Predictor fitting remains replay-based.
            next_frame = prepare_rnd_frames(jnp.asarray(next_gray[None], dtype=jnp.float32))
            raw_bonus = rnd_intrinsic_reward(state.rnd, next_frame)
            rnd_state = rnd_update_obs_rms(state.rnd, next_frame)
            rnd_state, normalized_bonus = rnd_update_rew_rms(
                rnd_state, raw_bonus[None, :]
            )
            state = state.replace(rnd=rnd_state)

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
                        args.algo.n_step,
                        beta,
                    )
                else:
                    weights = None

                state, q_metrics = q_update(data, weights, state, args.algo.ddqn)
                ext_loss, int_loss, q_ext_mean, q_int_mean, replay_novelty, priorities = q_metrics

                if args.buffer.prioritized:
                    with jax.default_device(buffer_device):
                        replay_state = replay.set_priorities(
                            replay_state, sampled.indices, priorities
                        )

                if step < rnd_args.rnd_burnin_steps:
                    rnd_loss = jnp.float32(0.0)
                else:
                    replay_frames = data.next_observation[:, :args.algo.n_step].reshape(
                        (-1,) + data.next_observation.shape[2:]
                    )
                    rnd_state, rnd_loss = rnd_predictor_loss_and_update(
                        state.rnd,
                        prepare_rnd_frames(replay_frames),
                        rnd_args.rnd_update_proportion,
                    )
                    state = state.replace(rnd=rnd_state)

                if (args.algo.tau == 1 and step % args.algo.tgt_upd_freq == 0) or args.algo.tau < 1:
                    state = update_targets(args.algo.tau, state)

                last_metrics = {
                    # Common names shared with DQN/SNEO.
                    "losses/dqn_td_loss": float(jax.device_get(ext_loss)),
                    "losses/rnd_loss": float(jax.device_get(rnd_loss)),
                    "values/q_values": float(jax.device_get(q_ext_mean)),
                    "values/replay_novelty": float(jax.device_get(replay_novelty)),
                    # RND-only diagnostics.
                    "losses/intrinsic_td_loss": float(jax.device_get(int_loss)),
                    "values/intrinsic_q_values": float(jax.device_get(q_int_mean)),
                    "values/online_novelty": float(jax.device_get(normalized_bonus[0, 0])),
                }

            if step > 0 and step % rnd_args.rnd_eval_interval == 0:
                eval_metrics = evaluate(state, eval_env, args, rnd_args.rnd_eval_episodes)
                print(
                    f"Evaluation at step {step} | Return: "
                    f"{eval_metrics['test/episodic_return']:.3f} | Success Rate: "
                    f"{eval_metrics['test/success_rate']:.3f}"
                )
                if args.wb.on:
                    wandb.log(eval_metrics, step=step)
                    wandb.run.summary["Success Rate"] = eval_metrics["test/success_rate"]

            if step > 0 and step % rnd_args.rnd_log_interval == 0:
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
                    **last_metrics,
                }
                print(
                    f"Global Step: {step} | Episodes: {len(completed_returns)} | "
                    f"Recent Return: {record['values/recent_return_mean']:.3f} | "
                    f"SPS: {record['values/SPS']}"
                )
                if args.wb.on:
                    wandb.log(record, step=step)
                action_source_counts = {"dqn": 0, "random": 0}

        return state
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    print("Available JAX devices:", jax.devices())
    create_wb_logger(ARGS)
    logger_closed = False
    if ARGS.wb.on:
        wandb.config.update({f"rnd/{key}": value for key, value in vars(RND).items()})
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
