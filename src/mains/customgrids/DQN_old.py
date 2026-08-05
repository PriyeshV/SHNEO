import math
import random
import traceback
from copy import deepcopy
from typing import Any, Union

import numpy as np
from etils import epath
from functools import partial

from src.config.customgrids.laplacian import *
args = convert_args_to_dict(parse_args())
# args = DotDict(args)
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.gpu.perc)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Helps with fragmentation
# os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"


import jax
import flax
import chex
import optax
import gymnasium as gym
from flax import linen as nn
import orbax.checkpoint as ocp
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from minigrid.wrappers import RGBImgPartialObsWrapper, FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper, FlatObsWrapper
print('Available Jax Devices: ', jax.devices())

from src.utils.jax_utils import *
from src.utils.grid_agent import GridAgent
import src.fbx_buffers.trajectory_buffer as trajectory_buffer
import src.fbx_buffers.prioritised_trajectory_buffer as prioritised_trajectory_buffer
from src.networks.networks import FFNetwork, Conv_Encoder


# ------------------------------------------------------------------------------
# JAX Dataclasses
# ------------------------------------------------------------------------------
class DQN_Enc_State(TrainState):
    target_params: flax.core.FrozenDict
    batch_stats: Union[None, flax.core.FrozenDict] = None
    target_batch_stats: Union[None, flax.core.FrozenDict] = None


class DQN_FFN_State(TrainState):
    target_params: flax.core.FrozenDict
    batch_stats: Union[None, flax.core.FrozenDict] = None
    target_batch_stats: Union[None, flax.core.FrozenDict] = None


@chex.dataclass(frozen=True)
class TimeStep:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


# ------------------------------------------------------------------------------
# Setup Replay Buffer with JIT-compiled functions and device handle caching
# ------------------------------------------------------------------------------

def setup_rb(args, screen_size):
    buffer_args = {'max_size': args.buffer.size, 'period': 1,  # sequence overlap
                   'sample_sequence_length': args.algo.n_step+1,
                   'add_batch_size': 1, 'min_length_time_axis': args.algo.starts,
                   'sample_batch_size': args.buffer.q_batch_size}
    if args.buffer.prioritized:
        buffer_args['priority_exponent'] = 0.6
        fbx_buffer = prioritised_trajectory_buffer.make_prioritised_trajectory_buffer
    else:
        fbx_buffer = trajectory_buffer.make_trajectory_buffer
    rb = fbx_buffer(**buffer_args)

    # JIT compile key replay buffer functions
    rb = rb.replace(init=jax.jit(rb.init), can_sample=jax.jit(rb.can_sample),
                    add=jax.jit(rb.add, donate_argnums=0), sample=jax.jit(rb.sample))
    if args.buffer.prioritized:
        rb = rb.replace(set_priorities=jax.jit(rb.set_priorities, donate_argnums=0))

    dummy_timestep = TimeStep(
        observation=jnp.ones((screen_size[0], screen_size[1], 1), dtype=jnp.uint8),
        action=jnp.int32(0), reward=jnp.float32(0.0), done=jnp.bool_(True))
    rb_state = rb.init(dummy_timestep)
    return rb, rb_state


def rgb_to_gray_uint8(obs_rgb: np.ndarray) -> np.ndarray:
    """
    obs_rgb: (H, W, 3) uint8 or float
    returns: (H, W, 1) uint8
    """
    if obs_rgb.dtype != np.uint8:
        # if already float in [0,1] or [0,255], handle it
        obs = obs_rgb.astype(np.float32)
    else:
        obs = obs_rgb.astype(np.float32)

    # ITU-R BT.601 luma transform (common)
    gray = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray[..., None]  # keep channel dim

# ------------------------------------------------------------------------------
# Setup Networks, Optimizers and Training States
# ------------------------------------------------------------------------------

def setup(key, env, args, n_actions):
    key, q_key = jax.random.split(key, 2)
    # Setup encoder network
    enc_network = Conv_Encoder(name='DQN:Enc',
                               min_scale=1.,
                               batch_norm=args.laplacian.batch_norm,
                               layer_norm=args.laplacian.layer_norm,
                               layer_dims=list(args.net.c_dims),
                               activations=[nn.relu] * len(args.net.c_dims))
    # Setup Q-network
    q_network = FFNetwork(name='DQN:Q',
                          stop_grad=False,
                          ip_batch_norm=False,
                          batch_norm=args.net.batch_norm,
                          layer_norm=args.net.layer_norm,
                          layer_dims=list(args.net.q_dims) + [n_actions],
                          activations=[nn.relu] * len(args.net.q_dims) + [None])

    # Visualize network architecture
    obs, _ = env.reset(seed=args.exp.seed)
    obs = rgb_to_gray_uint8(obs)
    obs, dummy_embedding = np.expand_dims(obs, 0), np.ones((1, args.net.c_dims[-1]), dtype=jnp.float32)
    # print(enc_network.tabulate(key, obs))
    # print(q_network.tabulate(key, dummy_embedding))

    # Initialize networks and training states.
    enc_variables = enc_network.init(q_key, obs, train=False)
    enc_lr_schedule = learning_rate_fn(init_lr=args.laplacian.lr, alpha=args.algo.lr_alpha,
                                       warmup_epochs=int(args.algo.tot_steps / 4), num_epochs=args.algo.tot_steps,
                                       steps_per_epoch=args.laplacian.upd_freq if args.laplacian.upd_freq is not None else 1)
    enc_optimizer = optax.chain(optax.clip_by_global_norm(args.laplacian.max_grad_norm),
                                optax.adam(learning_rate=enc_lr_schedule, eps=1.5e-4))

    enc_batch_stats = enc_variables['batch_stats'] if args.laplacian.batch_norm else None
    enc_state = DQN_Enc_State.create(apply_fn=enc_network.apply, tx=enc_optimizer,
                                     params=enc_variables['params'], batch_stats=enc_batch_stats,
                                     target_params=enc_variables['params'], target_batch_stats=enc_batch_stats)

    q_variables = q_network.init(q_key, dummy_embedding, train=False)
    q_lr_schedule = learning_rate_fn(init_lr=args.net.lr.ffn, alpha=args.algo.lr_alpha,
                                     warmup_epochs=int(args.algo.tot_steps / 4), num_epochs=args.algo.tot_steps,
                                     steps_per_epoch=args.algo.train_freq)
    q_optimizer = optax.chain(optax.clip_by_global_norm(args.net.max_grad_norm),
                              optax.adam(learning_rate=q_lr_schedule, eps=1.5e-4))

    q_batch_stats = q_variables['batch_stats'] if args.net.batch_norm else None
    q_state = DQN_FFN_State.create(apply_fn=q_network.apply, tx=q_optimizer,
                                   params=q_variables['params'], batch_stats=q_batch_stats,
                                   target_params=q_variables['params'], target_batch_stats=q_batch_stats)
    return key, (enc_state, enc_lr_schedule), (q_state, q_lr_schedule)


# ------------------------------------------------------------------------------
# Functions for Q-value computation and action selection
# ------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(3, 4, 5))
def get_q_values(obs, enc_state, q_state, use_enc_bn, use_q_bn, use_target: bool = False):
    enc_p = enc_state.target_params if use_target else enc_state.params
    q_p   = q_state.target_params   if use_target else q_state.params

    # (handle batch_stats similarly if you use BN)
    enc_vars = {'params': enc_p, 'batch_stats': enc_state.target_batch_stats} if (use_enc_bn and use_target) else \
               {'params': enc_p, 'batch_stats': enc_state.batch_stats} if use_enc_bn else {'params': enc_p}

    embeddings = enc_state.apply_fn(enc_vars, obs, train=False)

    q_vars = {'params': q_p, 'batch_stats': q_state.target_batch_stats} if (use_q_bn and use_target) else \
             {'params': q_p, 'batch_stats': q_state.batch_stats} if use_q_bn else {'params': q_p}

    q_values = q_state.apply_fn(q_vars, embeddings, train=False)
    return q_values


@partial(jax.jit, static_argnums=(4, 5, 6))
def get_action_probs(s_tau, obs, enc_state, q_state, use_enc_bn, use_q_bn, use_temp=False):
    q_values = get_q_values(obs, enc_state, q_state, use_enc_bn, use_q_bn)
    q_probs = jax.lax.cond(use_temp,  # If `use_temp` is True, temperature scaling is applied.
                           lambda _: jax.nn.softmax(q_values / (s_tau + 1e-8), axis=-1),
                           lambda _: jax.nn.softmax(q_values, axis=-1), operand=None)
    return q_probs


@partial(jax.jit, static_argnums=(5, 6, 7))
def sample_action(key, s_tau, obs, enc_state, q_state, use_enc_bn, use_q_bn, sample=False):
    q_probs = get_action_probs(s_tau, obs, enc_state, q_state, use_enc_bn, use_q_bn, use_temp=sample)
    action = jax.lax.cond(sample,  # If `sample` is True, samples from the distribution using categorical sampling
                          lambda _: jax.random.categorical(key, logits=jnp.log(q_probs)),
                          lambda _: jnp.argmax(q_probs, axis=-1), operand=None)
    return action


@partial(jax.jit, static_argnums=(3, 4))
def get_greedy_action(obs, enc_state, q_state, use_enc_bn, use_q_bn):
    q_values = get_q_values(obs, enc_state, q_state, use_enc_bn, use_q_bn)
    return jnp.argmax(q_values, axis=-1)


# ------------------------------------------------------------------------------
# Update Target Networks Incrementally
# ------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=0)
def update_target_network(tau, enc_state, q_state):
    enc_state = enc_state.replace(
        target_params=optax.incremental_update(enc_state.params, enc_state.target_params, tau),
        target_batch_stats=optax.incremental_update(enc_state.batch_stats, enc_state.target_batch_stats, tau))
    q_state = q_state.replace(
        target_params=optax.incremental_update(q_state.params, q_state.target_params, tau),
        target_batch_stats=optax.incremental_update(q_state.batch_stats, q_state.target_batch_stats, tau))
    return enc_state, q_state


# ------------------------------------------------------------------------------
# Update Function: Computes gradients and applies updates.
# ------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(4, 5, 6))
def update(data, priorities, enc_state, q_state, use_enc_bn, use_q_bn, use_ddqn):
    # Use the last observation in the sequence (s_{t+n}) for target Q-value estimation.
    q_next_target = get_q_values(data.observation[:, -1, ::], enc_state, q_state, use_enc_bn, use_q_bn, use_target=True)
    if use_ddqn:
        q_next = get_q_values(data.observation[:, -1, ::], enc_state, q_state, use_enc_bn, use_q_bn)
        next_action_id = jnp.argmax(jax.lax.stop_gradient(q_next), axis=-1)
        q_next_target = q_next_target[jnp.arange(next_action_id.shape[0]), next_action_id]
    else:
        q_next_target = jnp.max(q_next_target, axis=-1)

    # rewards and dones for first n steps (NOT n+1)
    n = args.algo.n_step
    r = data.reward[:, :n]
    d = data.done[:, :n]

    gamma_n = args.env.gamma ** n
    discounts = jnp.power(args.env.gamma, jnp.arange(n))[None, :]

    # d: [B,n] done flags aligned with rewards r[:,0..n-1]
    not_done = 1.0 - d.astype(jnp.float32)

    # prefix mask: [1, (1-d0), (1-d0)(1-d1), ...]  length n
    prefix = jnp.concatenate([jnp.ones((d.shape[0], 1), jnp.float32), not_done[:, :-1]], axis=1)
    mask = jnp.cumprod(prefix, axis=1)  # r0 always counted

    n_step_return = jnp.sum(r * discounts * mask, axis=1)
    boot_mask = 1.0 - jnp.any(d, axis=1).astype(jnp.float32)

    def td_loss_fn(enc_params, q_params):
        # Compute embeddings with optional batch norm.
        if use_enc_bn:
            embeddings, enc_bn_update = enc_state.apply_fn({'params': enc_params, 'batch_stats': enc_state.batch_stats},
                                                          data.observation[:, 0, :], train=True, mutable=['batch_stats'])
        else:
            embeddings, enc_bn_update = enc_state.apply_fn({'params': enc_params}, data.observation[:, 0, :]), None

        # Compute Q-values from embeddings with optional batch norm.
        if use_q_bn:
            (q_values, q_bn_update) = q_state.apply_fn({'params': q_params, 'batch_stats': q_state.batch_stats},
                                                      embeddings, train=True, mutable=['batch_stats'])
        else:
            q_values, q_bn_update = q_state.apply_fn({'params': q_params}, embeddings), None

        # Select the predicted Q-value for the taken action.
        q_pred = q_values[jnp.arange(q_values.shape[0]), data.action[:, 0]]

        # Calculate the TD target using the aggregated n-step reward.
        if args.algo.clip_q_pred:
            target = n_step_return + boot_mask * gamma_n * jnp.clip(q_next_target, jnp.min(q_pred), jnp.max(q_pred))
        else:
            target = n_step_return + boot_mask * gamma_n * q_next_target

        td_error = (q_pred - jax.lax.stop_gradient(target))
        upd_priorities = jnp.abs(td_error) + 1e-8

        loss = td_error ** 2
        wt_loss = jnp.mean(priorities * loss) if args.buffer.prioritized else jnp.mean(loss)
        return wt_loss, (jnp.mean(loss), enc_bn_update, q_bn_update, q_pred.mean(), q_pred.min(), q_pred.max(), upd_priorities)

    (wt_td_loss, (td_loss, enc_bn_updates, q_bn_updates, q_mean, q_min, q_max, new_priorities)), grads = (
        jax.value_and_grad(td_loss_fn, has_aux=True, argnums=(0, 1))(enc_state.params, q_state.params)
    )
    grads_enc_w, grads_q_w = grads
    enc_state = enc_state.apply_gradients(grads=grads_enc_w)
    q_state = q_state.apply_gradients(grads=grads_q_w)
    if use_enc_bn:
        enc_state = enc_state.replace(batch_stats=enc_bn_updates['batch_stats'])
    if use_q_bn:
        q_state = q_state.replace(batch_stats=q_bn_updates['batch_stats'])

    return enc_state, (q_state, td_loss, q_mean, new_priorities)


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


# ------------------------------------------------------------------------------
# Main Q-Learning Loop with improved device management, random handling, and logging.
# ------------------------------------------------------------------------------
def q_learning(enc_state, q_state, rb_state, key, n_actions):
    buffer_device = jax.devices(args.buffer.device)[0]

    # Reset environment and initialize episode variables.
    obs, _ = Env.reset(seed=args.exp.seed)

    test_results = []
    eps_step, eps_return = 1, 0
    total_timesteps = args.algo.tot_steps
    log_batch = []  # Initialize a list to accumulate log messages.

    for global_step in range(total_timesteps + 1):
        obs = rgb_to_gray_uint8(obs)
        obs = np.expand_dims(obs, axis=0)
        key, epsilon_key, action_key = jax.random.split(key, 3)
        epsilon = linear_schedule(args.algo.eps.start, args.algo.eps.end,
                                  int(args.algo.eps.frac * (total_timesteps + args.algo.starts)), global_step)
        s_tau = linear_schedule(args.algo.s_tau.start, args.algo.s_tau.end,
                                int(args.algo.s_tau.frac * (total_timesteps + args.algo.starts)), global_step)

        if jax.random.uniform(epsilon_key) < epsilon:
            action = int(jax.random.randint(action_key, (), 0, n_actions))
        else:
            action = sample_action(action_key, s_tau, obs, enc_state, q_state,
                                   args.laplacian.batch_norm, args.net.batch_norm, args.algo.sample_action)

            action = int(jax.device_get(action)[0])

        next_obs, reward, terminated, _, _ = Env.step(action)

        # Add the transition to the replay buffer on the cached device.
        with jax.default_device(buffer_device):
            timestep = TimeStep(observation=obs[:, None, ...],
                                action=jnp.expand_dims(action, [0, 1]),
                                reward=jnp.expand_dims(jnp.asarray(reward, dtype=jnp.float32), [0, 1]),
                                done=jnp.expand_dims(jnp.asarray(terminated or eps_step >= args.algo.max_eps_len), [0, 1]))
            rb_state = RB.add(rb_state, timestep)

        eps_return += reward
        if terminated or eps_step >= args.algo.max_eps_len:
            if args.wb.on and global_step >= args.algo.starts:
                wandb.log({'values/episodic_return': eps_return,
                           'values/episodic_length': eps_step,
                           'values/epsilon': epsilon}, step=global_step)
            log_batch.append(f"Global Step: {global_step} | Episodic Steps: {eps_step} | Return: {eps_return}")
            next_obs, _ = Env.reset()
            eps_step, eps_return = 0, 0
        eps_step += 1
        obs = next_obs

        if global_step % 5_000 == 0:
            print("\n".join(log_batch))
            log_batch = []

        if global_step == args.algo.starts:
            assert RB.can_sample(rb_state)
            print('Learning Starts')
            start_time = time.time()

        if global_step >= args.algo.starts:
            if global_step % args.algo.train_freq == 0:
                beta = linear_schedule(args.buffer.beta, 0.1, total_timesteps, global_step - args.algo.starts)

                # sample
                with jax.default_device(buffer_device):
                    key, sample_key = get_buffer_key(key)
                    batch = RB.sample(rb_state, sample_key)

                if args.buffer.prioritized:
                    priorities = compute_is_weights(batch.probabilities,
                                                    rb_state.running_index, rb_state.sum_tree_state.capacity,
                                                    args.algo.n_step + 1, beta)
                else:
                    priorities = None

                enc_state, (q_state, td_loss, q_mean, new_priorities_unnorm) = update(
                    batch.experience, priorities, enc_state, q_state,
                    args.laplacian.batch_norm, args.net.batch_norm, args.algo.ddqn
                )

                if args.buffer.prioritized:
                    with jax.default_device(buffer_device):
                        rb_state = RB.set_priorities(rb_state, batch.indices, new_priorities_unnorm)

                if global_step % 1_000 == 0 and args.wb.on:
                    wandb.log({'losses/td_loss': jax.device_get(td_loss),
                               'values/q_values': jax.device_get(q_mean),
                               'values/SPS': int((global_step - args.algo.starts) / (time.time() - start_time))
                           }, step=global_step)

            if (args.algo.tau == 1 and global_step % args.algo.tgt_upd_freq == 0) or (args.algo.tau < 1):
                enc_state, q_state = update_target_network(args.algo.tau, enc_state, q_state)

            if (global_step >= total_timesteps * 0.5) and global_step % 5_000 == 0:
                key, reached_goal = evaluate_policy(key, enc_state, q_state, global_step)
                test_results.append(reached_goal)

                n_success = np.asarray(test_results, np.int32).sum()
                success_rate = n_success / len(test_results)
                print('Success Rate: ', str(n_success) + '/' + str(len(test_results)), ' = ', success_rate)
                if args.wb.on:
                    wandb.run.summary['Success Rate: '] = success_rate

    end_time = time.time()
    print('The training session started at ', start_time, 'and ended at ', end_time)
    print('Took ', end_time - start_time)
    return enc_state, q_state, key


def evaluate_policy(key, enc_state, q_state, global_step):
    env = deepcopy(Env)
    obs, _ = env.reset(seed=args.exp.seed)

    eps_return, terminated = 0, False
    key, sample_key = jax.random.split(key, 2)
    for eps_step in range(args.algo.max_eps_len+1):
        obs = rgb_to_gray_uint8(obs)
        obs = np.expand_dims(obs, axis=0)
        action = sample_action(sample_key, 0, obs,
                               enc_state, q_state,
                               use_enc_bn=args.laplacian.batch_norm, use_q_bn=args.net.batch_norm,
                               sample=args.algo.sample_action)[0]
        obs, reward, terminated, _, _ = env.step(action)
        eps_return += reward
        if terminated:
            break
        if args.algo.sample_action:
            key, sample_key = jax.random.split(key, 2)

    if args.wb.on:
        wandb.log({'test/episodic_return': eps_return,
                   'test/episodic_length': eps_step}, step=global_step)
    return key, terminated


# ------------------------------------------------------------------------------
# Main Entry Point: Environment and Logger Initialization
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    Env = GridAgent(args)
    screen_size = args.env.screen_size
    if screen_size is None:
        screen_size = (Env.height, Env.width)
    else:
        screen_size = (args.env.screen_size, args.env.screen_size)

    Env = Env.env
    n_actions = Env.action_space.n
    create_wb_logger(args)

    # Set seeds for reproducibility.
    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    key = jax.random.PRNGKey(args.exp.seed)

    with jax.default_device(jax.devices(args.buffer.device)[0]):
        RB, RB_state = setup_rb(args, screen_size)

    with jax.default_device(jax.devices('gpu')[0]):
        try:
            key, (Enc_state, Enc_Lr_Schedule), (Q_state, Q_Lr_Schedule) = setup(key, Env, args, n_actions)
            Enc_state, Q_state, Key = q_learning(Enc_state, Q_state, RB_state, key, n_actions)
        except Exception as e:
            traceback.print_exc()  # This prints the full traceback with line numbers.
            wandb.finish(exit_code=1)
        finally:
            Env.close()
            close_logger() if args.wb.on else None
            jax.clear_backends()