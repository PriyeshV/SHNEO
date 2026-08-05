import math
import random
import traceback
from copy import deepcopy
from typing import Any, Union

import envpool
import numpy as np
from etils import epath
from functools import partial

from src.config.atari.ppo_laplacian import *
args = convert_args_to_dict(parse_args())

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Helps with fragmentation


import jax
import flax
import chex
import optax
import gymnasium as gym
from flax import struct
from flax import linen as nn
import orbax.checkpoint as ocp
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from src.utils.jax_utils import *
from src.networks.ppo_networks import AtariConvEnc, Actor, Critic
from src.utils.purejax_atari_wrappers import JaxLogEnvPoolWrapper


@struct.dataclass
class Agent_State():
    enc: TrainState
    actor: TrainState
    critic: TrainState
    timesteps: jnp.int32 = jnp.int32(0)
    n_updates: jnp.int32 = jnp.int32(0)
    grad_steps: jnp.int32 = jnp.int32(0)
    next_eval_step: jnp.int32 = jnp.int32(0)   # NEW


@struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


def linear_schedule(count):
    # anneal learning rate linearly after one training iteration which contains
    # (args.num_minibatches * args.update_epochs) gradient updates
    frac = 1.0 - (count // (args.algo.n_mini_batches * args.algo.upd_epochs)) / args.algo.num_iterations
    return args.net.lr * frac


def make_env():
    env_kwargs = {'episodic_life': True, 'reward_clip': True, 'repeat_action_probability': 0.,
                  'frame_skip': 4, 'noop_max': 30, 'full_action_space': False}
    env = envpool.make(args.env.id+'-v5', env_type="gym", num_envs=args.env.n_total_envs, seed=args.exp.seed, **env_kwargs)
    env.num_envs = args.env.n_total_envs
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    env.name = args.env.id+'-v5'
    return env


def make_eval_env(n_eval_envs: int):
    env_kwargs = dict(
        episodic_life=True,
        reward_clip=True,
        repeat_action_probability=0.0,
        frame_skip=4,
        noop_max=30,
        full_action_space=False,
    )
    env = envpool.make(args.env.id+'-v5', env_type="gym", num_envs=n_eval_envs, seed=args.exp.seed + 999, **env_kwargs)
    env.num_envs = n_eval_envs
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    env.name = args.env.id+'-v5'
    return JaxLogEnvPoolWrapper(env)


def setup(key, args, obs, n_actions):
    e_key, a_key, c_key = jax.random.split(key, 3)

    enc_network = AtariConvEnc(use_batch_norm=False, use_layer_norm=False,
                               mlp_dims=args.net.c_dims, mlp_activations=[nn.relu])
    actor = Actor(action_dim=n_actions)
    critic = Critic()

    obs = np.transpose(obs[0], (1, 2, 0))
    obs = jnp.expand_dims(obs, 0)
    dummy_embedding = jnp.ones((1, args.net.c_dims[-1]))

    enc_variables = enc_network.init(e_key, obs, train=False)
    actor_variables = actor.init(a_key, dummy_embedding, train=False)
    critic_variables = critic.init(c_key, dummy_embedding, train=False)

    enc_state = TrainState.create(apply_fn=jax.jit(enc_network.apply), params=enc_variables['params'],
                                  tx=optax.chain(optax.clip_by_global_norm(args.net.max_grad_norm),
                                                 optax.inject_hyperparams(optax.adam)(eps=1e-5,
                                                 learning_rate=linear_schedule if args.algo.anneal_lr else args.net.lr)))
    actor_state = TrainState.create(apply_fn=jax.jit(actor.apply), params=actor_variables['params'],
                                    tx=optax.chain(optax.clip_by_global_norm(args.net.max_grad_norm),
                                                   optax.inject_hyperparams(optax.adam)(eps=1e-5,
                                                   learning_rate=linear_schedule if args.algo.anneal_lr else args.net.lr)))
    critic_state = TrainState.create(apply_fn=jax.jit(critic.apply), params=critic_variables['params'],
                                     tx=optax.chain(optax.clip_by_global_norm(args.net.max_grad_norm),
                                                    optax.inject_hyperparams(optax.adam)(eps=1e-5,
                                                    learning_rate=linear_schedule if args.algo.anneal_lr else args.net.lr)))
    eval_interval = args.algo.tot_steps // 25
    return Agent_State(enc=enc_state, actor=actor_state, critic=critic_state, next_eval_step=jnp.int32(eval_interval))


# ---------- small helper ----------
@jax.jit
def chw_to_nhwc(obs_chw: jnp.ndarray) -> jnp.ndarray:
    # obs_chw: (N, C, H, W) -> (N, H, W, C)
    return jnp.transpose(obs_chw, (0, 2, 3, 1))


# NOTE: your get_action_and_value must pass NHWC into encoder
@jax.jit
def get_action_and_value(agent_state, obs_chw, key):
    obs = chw_to_nhwc(obs_chw)
    hidden = agent_state.enc.apply_fn({'params': agent_state.enc.params}, obs, train=False)
    logits = agent_state.actor.apply_fn({'params': agent_state.actor.params}, hidden, train=False)
    value = agent_state.critic.apply_fn({'params': agent_state.critic.params}, hidden, train=False).squeeze(-1)

    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=logits.shape)
    action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
    logp = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
    return action, logp, value, key


def compute_gae_once(carry, inp, gamma: float, gae_lambda: float):
    adv_next = carry
    next_done, next_value, cur_value, reward = inp
    next_nonterminal = 1.0 - next_done.astype(jnp.float32)

    delta = reward + gamma * next_value * next_nonterminal - cur_value
    adv = delta + gamma * gae_lambda * next_nonterminal * adv_next
    return adv, adv


@partial(jax.jit, static_argnums=(4, 5))
def compute_gae(agent_state, next_obs_chw, next_done, storage, gamma: float, gae_lambda: float):
    # bootstrap V(next_obs)
    hidden = agent_state.enc.apply_fn({'params': agent_state.enc.params}, chw_to_nhwc(next_obs_chw), train=False)
    next_value = agent_state.critic.apply_fn({'params': agent_state.critic.params}, hidden, train=False).squeeze(-1)

    dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)      # [T+1,N]
    values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)  # [T+1,N]

    adv0 = jnp.zeros_like(next_value)
    scan_inp = (dones[1:], values[1:], values[:-1], storage.rewards)
    _, adv = jax.lax.scan(
        partial(compute_gae_once, gamma=gamma, gae_lambda=gae_lambda),
        adv0,
        scan_inp,
        reverse=True,
    )
    returns = adv + storage.values
    return storage.replace(advantages=adv, returns=returns)


# -----------------------------
# env_take_steps: one env step per env (vectorized)
# -----------------------------
@partial(jax.jit, static_argnums=(3, 4,))
def env_take_steps(carry, _unused, agent_state, env, n_total_envs: int):
    """
    carry: (last_obs_chw, env_state, key)
      last_obs_chw: (N,C,H,W) uint8
      env_state: LogEnvState
      key: PRNGKey
    returns:
      new_carry, (storage_t, infos_t)
    """
    last_obs, env_state, key = carry
    key, step_key = jax.random.split(key)

    action, logp, value, key = get_action_and_value(agent_state, last_obs, step_key)
    next_obs, next_env_state, reward, next_done, infos = env.step(env_state, action)

    storage_t = Storage(
        obs=last_obs,                     # keep CHW in storage
        actions=action,
        logprobs=logp,
        dones=next_done,                  # done after step (consistent with CleanRL style)
        values=value,
        rewards=reward,
        advantages=jnp.zeros((n_total_envs,), dtype=jnp.float32),
        returns=jnp.zeros((n_total_envs,), dtype=jnp.float32),
    )
    return (next_obs, next_env_state, key), (storage_t, infos)


@partial(jax.jit, static_argnums=(2,))
def ppo_minibatch_update(agent_state, minibatch, args):
    B = minibatch.actions.shape[0]

    def loss_fn(enc_params, actor_params, critic_params):
        hidden = agent_state.enc.apply_fn({'params': enc_params}, chw_to_nhwc(minibatch.obs), train=True)
        logits = agent_state.actor.apply_fn({'params': actor_params}, hidden, train=True)
        new_value = agent_state.critic.apply_fn({'params': critic_params}, hidden, train=True).squeeze(-1)

        new_logp = jax.nn.log_softmax(logits)[jnp.arange(B), minibatch.actions]
        logratio = new_logp - minibatch.logprobs
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1.0) - logratio).mean()

        adv = minibatch.advantages
        if args.algo.norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pg1 = -adv * ratio
        pg2 = -adv * jnp.clip(ratio, 1.0 - args.algo.clip_coef, 1.0 + args.algo.clip_coef)
        pg_loss = jnp.maximum(pg1, pg2).mean()

        # value loss (clipped option as in CleanRL)
        v_loss_unclipped = 0.5 * jnp.square(new_value - minibatch.returns)
        if args.algo.clip_vloss:
            v_clipped = minibatch.values + jnp.clip(
                new_value - minibatch.values, -args.algo.clip_coef, args.algo.clip_coef
            )
            v_loss_clipped = 0.5 * jnp.square(v_clipped - minibatch.returns)
            v_loss = jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = v_loss_unclipped.mean()

        # entropy (stable)
        l = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        l = l.clip(min=jnp.finfo(l.dtype).min)
        entropy = -(l * jax.nn.softmax(l)).sum(-1).mean()

        loss = pg_loss - args.algo.ent_coef * entropy + args.algo.vf_coef * v_loss
        return loss, (pg_loss, v_loss, entropy, jax.lax.stop_gradient(approx_kl))

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1, 2))(
        agent_state.enc.params, agent_state.actor.params, agent_state.critic.params
    )
    g_enc, g_actor, g_critic = grads

    agent_state = agent_state.replace(
        enc=agent_state.enc.apply_gradients(grads=g_enc),
        actor=agent_state.actor.apply_gradients(grads=g_actor),
        critic=agent_state.critic.apply_gradients(grads=g_critic),
        grad_steps=agent_state.grad_steps + 1,
    )
    return agent_state, (loss, *aux)


def _get_lr_from_opt_state(opt_state):
    # Works with: optax.chain(clip, inject_hyperparams(adam)(...))
    # Typically opt_state = (clip_state, injected_adam_state)
    try:
        return opt_state[1].hyperparams["learning_rate"]
    except Exception:
        return None


def _current_lr(agent_state, args):
    lr = _get_lr_from_opt_state(agent_state.enc.opt_state)
    if lr is None:
        # fallback: use grad_steps as the "count" (same convention as CleanRL)
        lr = linear_schedule(agent_state.grad_steps) if args.algo.anneal_lr else args.net.lr
    return lr


from functools import partial
import time
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def update_step(runner_state, _unused, args, train_env, eval_env, start_time: float, eval_steps: int):
    """
    runner_state:
      (agent_state, (train_obs, train_env_state), (eval_obs, eval_env_state), key)

    Notes:
      - Train logging happens at step=update_steps
      - Eval logging happens ONLY when eval ran, at step=env_step
      - metrics always includes eval keys (dummy 0) to satisfy lax.cond PyTree constraints
    """
    agent_state, train_state, eval_state, key = runner_state
    train_obs, train_env_state = train_state

    # ------------------------------------------------------------
    # (1) Rollout: collect T steps of on-policy data
    # ------------------------------------------------------------
    key, rollout_key = jax.random.split(key)
    step_fn = partial(
        env_take_steps,
        agent_state=agent_state,
        env=train_env,
        n_total_envs=args.env.n_total_envs,
    )
    (train_obs, train_env_state, key), (storage, infos) = jax.lax.scan(
        step_fn,
        (train_obs, train_env_state, rollout_key),
        None,
        length=args.env.n_steps,
    )

    # ------------------------------------------------------------
    # (2) GAE + returns
    # ------------------------------------------------------------
    next_done = storage.dones[-1]
    storage = compute_gae(agent_state, train_obs, next_done, storage, args.env.gamma, args.algo.gae_lambda)

    # ------------------------------------------------------------
    # (3) Flatten T,N -> B and run PPO update epochs/minibatches
    # ------------------------------------------------------------
    def flatten_TN(x):
        return x.reshape((-1,) + x.shape[2:])  # [T*N, ...]

    flat = jax.tree_util.tree_map(flatten_TN, storage)
    B = flat.actions.shape[0]

    def make_minibatches(rng, flat_storage):
        perm = jax.random.permutation(rng, B)

        def take(x):
            x = x[perm]
            return x.reshape((args.algo.n_mini_batches, -1) + x.shape[1:])

        return jax.tree_util.tree_map(take, flat_storage)

    def one_epoch(carry, _):
        state, rng = carry
        rng, subkey = jax.random.split(rng)
        mbs = make_minibatches(subkey, flat)

        def one_mb(s, mb):
            s, stats = ppo_minibatch_update(s, mb, args)
            return s, stats

        state, stats = jax.lax.scan(one_mb, state, mbs)
        return (state, rng), stats  # stats: [M, 5]

    key, upd_key = jax.random.split(key)
    (agent_state, _), stats = jax.lax.scan(one_epoch, (agent_state, upd_key), None, length=args.algo.upd_epochs)

    # stats is [E, M, 5] where 5=(loss, pg, v, ent, kl)
    loss, pg_loss, v_loss, ent, approx_kl = [s.mean() for s in stats]

    # ------------------------------------------------------------
    # (4) Counters + LR
    # ------------------------------------------------------------
    agent_state = agent_state.replace(
        timesteps=agent_state.timesteps + args.env.n_steps * args.env.n_total_envs,
        n_updates=agent_state.n_updates + 1,
    )
    lr = _current_lr(agent_state, args)

    # ------------------------------------------------------------
    # (5) Training metrics (always)
    # ------------------------------------------------------------
    metrics = {
        "env_step": agent_state.timesteps,
        "update_steps": agent_state.n_updates,
        "grad_steps": agent_state.grad_steps,
        "learning_rate": lr,
        "ppo/loss": loss,
        "ppo/policy_loss": pg_loss,
        "ppo/value_loss": v_loss,
        "ppo/entropy": ent,
        "ppo/approx_kl": approx_kl,
    }
    metrics.update({k: v.mean() for k, v in infos.items()})

    # Always include eval keys (dummy) to keep PyTree structure fixed
    metrics = {
        **metrics,
        "eval/episodes": jnp.array(0, dtype=jnp.int32),  # FIXED
        "eval/avg_return": jnp.array(0.0, dtype=jnp.float32),
        "eval/avg_ep_len": jnp.array(0.0, dtype=jnp.float32),
        "eval/steps": jnp.array(0, dtype=jnp.int32),
    }

    # ------------------------------------------------------------
    # (6) Optional JIT-eval (runs only when scheduled)
    # ------------------------------------------------------------
    do_eval = agent_state.timesteps >= agent_state.next_eval_step

    def _do_eval_fn(carry):
        st, ev_st, met = carry
        new_ev_st, ev_met = run_eval_until_episodes(st, ev_st, eval_env, eval_steps)
        met = {
            **met,
            "eval/episodes": ev_met["eval/episodes"],
            "eval/avg_return": ev_met["eval/avg_return"],
            "eval/avg_ep_len": ev_met["eval/avg_ep_len"],
            "eval/steps": ev_met["eval/steps"],
        }
        eval_interval = jnp.int32(args.algo.tot_steps // 25)  # 1% cadence; change to //10 for 10%
        st = st.replace(next_eval_step=st.next_eval_step + eval_interval)
        return st, new_ev_st, met

    def _no_eval_fn(carry):
        return carry

    agent_state, eval_state, metrics = jax.lax.cond(
        do_eval, _do_eval_fn, _no_eval_fn, (agent_state, eval_state, metrics)
    )

    # ------------------------------------------------------------
    # (7) Logging callbacks
    #   - Train logs at update_steps
    #   - Eval logs only when eval ran, at env_step (prevents zeros in eval plots)
    # ------------------------------------------------------------
    def _train_log_cb(m):
        elapsed = time.time() - start_time
        sps = float(m["env_step"]) / max(elapsed, 1e-8)

        out = {
            # x-axis fields
            "env_step": int(m["env_step"]),
            "update_steps": int(m["update_steps"]),
            "grad_steps": int(m["grad_steps"]),
            "learning_rate": float(m["learning_rate"]),
            "SPS": float(sps),

            # PPO losses
            "ppo/loss": float(m["ppo/loss"]),
            "ppo/policy_loss": float(m["ppo/policy_loss"]),
            "ppo/value_loss": float(m["ppo/value_loss"]),
            "ppo/entropy": float(m["ppo/entropy"]),
            "ppo/approx_kl": float(m["ppo/approx_kl"]),

            # env stats you care about (example — add your keys)
            "returned_episode_returns": float(m.get("returned_episode_returns", 0.0)),
            "returned_episode_lengths": float(m.get("returned_episode_lengths", 0.0)),
            "normalized_returned_episode_returns": float(m.get("normalized_returned_episode_returns", 0.0)),
        }

        wandb.log(out, step=int(out["env_step"]))

    def _eval_log_cb(m):
        wandb.log(
            {
                "eval/avg_return": float(m["eval/avg_return"]),
                "eval/avg_ep_len": float(m["eval/avg_ep_len"]),
                "eval/episodes": float(m["eval/episodes"]),
                "eval/steps": float(m["eval/steps"]),
                "eval/env_step": int(m["env_step"]),
            },
            step=int(m["env_step"]),
        )

    def _train_print_cb(m):
        elapsed = time.time() - start_time
        sps = float(m["env_step"]) / max(elapsed, 1e-8)
        print(
            f"upd={int(m['update_steps'])} env_step={int(m['env_step'])} "
            f"grad_steps={int(m['grad_steps'])} lr={float(m['learning_rate']):.3e} "
            f"SPS={int(sps)}"
        )

    def _eval_print_cb(m):
        print(
            f"[EVAL] env_step={int(m['env_step'])} "
            f"avg_return={float(m['eval/avg_return'])} episodes={float(m['eval/episodes'])}"
        )

    do_train_log = (agent_state.n_updates % 10 == 0) | (agent_state.n_updates == (args.algo.num_iterations - 1))
    _ = jax.lax.cond(
        do_train_log,
        lambda m: jax.debug.callback(_train_log_cb if args.wb.on else _train_print_cb, m),
        lambda m: None,
        metrics,
    )

    _ = jax.lax.cond(
        do_eval,
        lambda m: jax.debug.callback(_eval_log_cb if args.wb.on else _eval_print_cb, m),
        lambda m: None,
        metrics,
    )

    # ------------------------------------------------------------
    # (8) Return updated runner_state
    # ------------------------------------------------------------
    new_runner_state = (agent_state, (train_obs, train_env_state), eval_state, key)
    return new_runner_state, metrics


@jax.jit
def greedy_action(agent_state, obs_chw):
    obs = chw_to_nhwc(obs_chw)
    hidden = agent_state.enc.apply_fn({'params': agent_state.enc.params}, obs, train=False)
    logits = agent_state.actor.apply_fn({'params': agent_state.actor.params}, hidden, train=False)
    return jnp.argmax(logits, axis=-1)


def _eval_body(carry, eval_env):
    # carry: (obs, env_state, ep_count, ret_sum, steps)
    obs, env_state, ep_count, ret_sum, steps = carry
    a = greedy_action(agent_state=carry_agent_state, obs_chw=obs)  # closed over via partial below
    next_obs, next_env_state, r, d, info = eval_env.step(env_state, a)

    done_mask = info["returned_episode"].astype(jnp.float32)              # [N]
    done_ret  = info["returned_episode_returns"]                          # [N]

    ep_count = ep_count + done_mask.sum()
    ret_sum  = ret_sum + (done_mask * done_ret).sum()
    steps    = steps + 1

    return (next_obs, next_env_state, ep_count, ret_sum, steps)


def eval_cond(carry, n_episodes, max_steps):
    _, _, ep_count, _, steps = carry
    return (ep_count < n_episodes) & (steps < max_steps)


@partial(jax.jit, static_argnums=(2, 3, 4))
def run_eval_until_episodes(agent_state, eval_state, eval_env, n_episodes: int, max_steps: int = 5_000):
    obs, env_state = eval_state

    ep_count0 = jnp.array(0, dtype=jnp.int32)
    ret_sum0  = jnp.array(0.0, dtype=jnp.float32)
    len_sum0  = jnp.array(0.0, dtype=jnp.float32)   # NEW
    steps0    = jnp.array(0, dtype=jnp.int32)

    def cond(carry):
        _, _, ep_count, _, _, steps = carry
        return (ep_count < n_episodes) & (steps < max_steps)

    def body(carry):
        obs, env_state, ep_count, ret_sum, len_sum, steps = carry

        a = greedy_action(agent_state, obs)
        next_obs, next_env_state, r, d, info = eval_env.step(env_state, a)

        done_mask = info["returned_episode"].astype(jnp.float32)  # [N]
        done_ret  = info["returned_episode_returns"].astype(jnp.float32)  # [N]

        # This key should exist with your wrapper; if not, see note below.
        done_len  = info["returned_episode_lengths"].astype(jnp.float32)  # [N]

        ep_count = ep_count + done_mask.sum().astype(jnp.int32)
        ret_sum  = ret_sum  + (done_mask * done_ret).sum()
        len_sum  = len_sum  + (done_mask * done_len).sum()
        steps    = steps + 1

        return (next_obs, next_env_state, ep_count, ret_sum, len_sum, steps)

    carry0 = (obs, env_state, ep_count0, ret_sum0, len_sum0, steps0)
    obsF, env_stateF, ep_countF, ret_sumF, len_sumF, stepsF = jax.lax.while_loop(cond, body, carry0)

    ep_count_f = jnp.maximum(ep_countF.astype(jnp.float32), 1.0)
    avg_ret = ret_sumF / ep_count_f
    avg_len = len_sumF / ep_count_f

    new_eval_state = (obsF, env_stateF)
    eval_metrics = {
        "eval/episodes": ep_countF,
        "eval/avg_return": avg_ret,
        "eval/avg_ep_len": avg_len,   # NEW
        "eval/steps": stepsF,
    }
    return new_eval_state, eval_metrics


def ppo_learning(key, train_env, eval_env, agent_state):
    train_obs, train_env_state = train_env.reset()
    eval_obs, eval_env_state = eval_env.reset()

    runner_state = (agent_state, (train_obs, train_env_state), (eval_obs, eval_env_state), key)

    start_time = time.time()
    step_fn = partial(update_step, args=args, train_env=train_env, eval_env=eval_env,
                      start_time=start_time, eval_steps=args.env.n_steps * 10)  # example

    runner_state, metrics = jax.lax.scan(step_fn, runner_state, None, length=args.algo.num_iterations)
    return runner_state, metrics


if __name__ == '__main__':
    Env = make_env()
    n_actions = Env.single_action_space.n
    if args.wb.on:
        create_wb_logger(args)
        wandb.save('src/networks/networks.py', policy='now')

    # Set seeds for reproducibility.
    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    Key, setup_key, algo_key = jax.random.split(jax.random.PRNGKey(args.exp.seed), 3)

    with jax.default_device(jax.devices('gpu')[0]):
        try:
            TrainEnv = JaxLogEnvPoolWrapper(make_env())
            EvalEnv = make_eval_env(n_eval_envs=1)

            init_obs, _ = TrainEnv.reset()
            agent_state = setup(setup_key, args, init_obs, n_actions)
            ppo_learning(algo_key, TrainEnv, EvalEnv, agent_state)

        except Exception as e:
            traceback.print_exc()  # This prints the full traceback with line numbers.
            wandb.finish(exit_code=1)
        finally:
            Env.close()
            close_logger() if args.wb.on else None
            jax.clear_backends()
