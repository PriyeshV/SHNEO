import math
import time
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
from src.mains.algorithms.RND import *
from src.networks.ppo_networks import AtariConvEnc, Actor, Critic
from src.utils.purejax_atari_wrappers import JaxLogEnvPoolWrapper


@struct.dataclass
class Agent_State():
    enc: TrainState
    actor: TrainState
    critic_ext: TrainState
    critic_int: TrainState
    rnd: RND_State

    timesteps: jnp.int32 = jnp.int32(0)
    n_updates: jnp.int32 = jnp.int32(0)
    grad_steps: jnp.int32 = jnp.int32(0)
    next_eval_step: jnp.int32 = jnp.int32(0)


@struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    ext_values: jnp.array
    int_values: jnp.array
    ext_adv: jnp.array
    int_adv: jnp.array
    ext_returns: jnp.array
    int_returns: jnp.array
    ext_rewards: jnp.array
    int_rewards: jnp.array
    rnd_frame: jnp.array     # Storage.rnd_frame will be [N,84,84,1]


def linear_schedule(count):
    # anneal learning rate linearly after one training iteration which contains
    # (args.num_minibatches * args.update_epochs) gradient updates
    frac = 1.0 - (count // (args.algo.n_mini_batches * args.algo.upd_epochs)) / args.algo.num_iterations
    return args.net.lr * frac


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
    e_key, a_key, c_key, c_key2, rnd_key = jax.random.split(key, 5)

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

    critic_int = Critic()
    critic_int_vars = critic_int.init(c_key2, dummy_embedding, train=False)  # add new key split

    critic_int_state = TrainState.create(apply_fn=jax.jit(critic_int.apply), params=critic_int_vars["params"],
                                         tx=optax.chain(optax.clip_by_global_norm(args.net.max_grad_norm),
                                                        optax.inject_hyperparams(optax.adam)(eps=1e-5,
                                                        learning_rate=linear_schedule if args.algo.anneal_lr else args.net.lr)))

    rnd_state = rnd_setup(key=rnd_key, lr=args.net.lr,
                          max_grad_norm=args.net.max_grad_norm,
                          n_envs=args.env.n_total_envs, int_gamma=args.algo.int_gamma)

    return Agent_State(enc=enc_state, actor=actor_state, critic_ext=critic_state, critic_int=critic_int_state,
                       rnd=rnd_state, next_eval_step=jnp.int32(eval_interval))


# ---------- small helper ----------
@jax.jit
def chw_to_nhwc(obs_chw: jnp.ndarray) -> jnp.ndarray:
    # obs_chw: (N, C, H, W) -> (N, H, W, C)
    return jnp.transpose(obs_chw, (0, 2, 3, 1))


@jax.jit
def get_action_and_value(agent_state, obs_chw, key):
    obs = chw_to_nhwc(obs_chw)
    hid = agent_state.enc.apply_fn({"params": agent_state.enc.params}, obs, train=False)

    logits = agent_state.actor.apply_fn({"params": agent_state.actor.params}, hid, train=False)
    v_ext = agent_state.critic_ext.apply_fn({"params": agent_state.critic_ext.params}, hid, train=False).squeeze(-1)
    v_int = agent_state.critic_int.apply_fn({"params": agent_state.critic_int.params}, hid, train=False).squeeze(-1)

    key, sub = jax.random.split(key)
    u = jax.random.uniform(sub, shape=logits.shape)
    action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
    logp = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
    return action, logp, v_ext, v_int, key


def compute_gae_once(carry, inp, gamma: float, gae_lambda: float):
    adv_next = carry
    next_done, next_value, cur_value, reward = inp
    next_nonterminal = 1.0 - next_done.astype(jnp.float32)

    delta = reward + gamma * next_value * next_nonterminal - cur_value
    adv = delta + gamma * gae_lambda * next_nonterminal * adv_next
    return adv, adv


@partial(jax.jit, static_argnums=(4,5,6,7))
def compute_gae_ext_int(agent_state, next_obs_chw, next_done, storage,
                        gamma: float, int_gamma: float, gae_lambda: float):
    obs = chw_to_nhwc(next_obs_chw)
    hid = agent_state.enc.apply_fn({"params": agent_state.enc.params}, obs, train=False)
    next_v_ext = agent_state.critic_ext.apply_fn({"params": agent_state.critic_ext.params}, hid, train=False).squeeze(-1)
    next_v_int = agent_state.critic_int.apply_fn({"params": agent_state.critic_int.params}, hid, train=False).squeeze(-1)

    # append bootstrap
    dones = jnp.concatenate([storage.dones, next_done[None,:]], axis=0)  # [T+1,N]
    vext  = jnp.concatenate([storage.ext_values, next_v_ext[None,:]], axis=0)
    vint  = jnp.concatenate([storage.int_values, next_v_int[None,:]], axis=0)

    def scan_fn(carry, inp):
        adv_ext_next, adv_int_next = carry
        done_t1, vext_t1, vext_t, r_ext, vint_t1, vint_t, r_int = inp

        nt_ext = 1.0 - done_t1.astype(jnp.float32)
        nt_int = 1.0  # by design

        delta_ext = r_ext + gamma * vext_t1 * nt_ext - vext_t
        adv_ext = delta_ext + gamma * gae_lambda * nt_ext * adv_ext_next

        delta_int = r_int + int_gamma * vint_t1 * nt_int - vint_t
        adv_int = delta_int + int_gamma * gae_lambda * nt_int * adv_int_next

        return (adv_ext, adv_int), (adv_ext, adv_int)

    inp = (dones[1:], vext[1:], vext[:-1], storage.ext_rewards,
           vint[1:], vint[:-1], storage.int_rewards)

    ( _, _), (adv_ext, adv_int) = jax.lax.scan(
        scan_fn,
        (jnp.zeros_like(next_v_ext), jnp.zeros_like(next_v_int)),
        inp,
        reverse=True
    )

    ext_returns = adv_ext + storage.ext_values
    int_returns = adv_int + storage.int_values

    return storage.replace(ext_adv=adv_ext, int_adv=adv_int,
                           ext_returns=ext_returns, int_returns=int_returns)


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

    action, logp, v_ext, v_int, key = get_action_and_value(agent_state, last_obs, step_key)
    next_obs, next_env_state, reward, next_done, infos = env.step(env_state, action)

    # in env_take_steps
    frame_hwc1 = next_obs[:, 3, :, :].astype(jnp.float32)[..., None]  # [N,84,84,1]
    int_r = rnd_intrinsic_reward(agent_state.rnd, frame_hwc1)

    storage_t = Storage(
        obs=last_obs,
        actions=action,
        logprobs=logp,
        dones=next_done,

        ext_values=v_ext,
        int_values=v_int,

        ext_rewards=reward,
        int_rewards=int_r,

        rnd_frame=frame_hwc1,  # save for rms update + predictor training
        ext_adv=jnp.zeros((n_total_envs,), jnp.float32),
        int_adv=jnp.zeros((n_total_envs,), jnp.float32),
        ext_returns=jnp.zeros((n_total_envs,), jnp.float32),
        int_returns=jnp.zeros((n_total_envs,), jnp.float32),
    )
    return (next_obs, next_env_state, key), (storage_t, infos)


@partial(jax.jit, static_argnums=(2,))
def ppo_minibatch_update(agent_state: Agent_State, minibatch, args):
    B = minibatch.actions.shape[0]

    def loss_fn(enc_params, actor_params, ext_params, int_params):
        hid = agent_state.enc.apply_fn({"params": enc_params}, chw_to_nhwc(minibatch.obs), train=True)
        logits = agent_state.actor.apply_fn({"params": actor_params}, hid, train=True)

        new_v_ext = agent_state.critic_ext.apply_fn({"params": ext_params}, hid, train=True).squeeze(-1)
        new_v_int = agent_state.critic_int.apply_fn({"params": int_params}, hid, train=True).squeeze(-1)

        new_logp = jax.nn.log_softmax(logits)[jnp.arange(B), minibatch.actions]
        logratio = new_logp - minibatch.logprobs
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1.0) - logratio).mean()

        adv = args.algo.ext_coef * minibatch.ext_adv + args.algo.int_coef * minibatch.int_adv
        if args.algo.norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pg1 = -adv * ratio
        pg2 = -adv * jnp.clip(ratio, 1.0 - args.algo.clip_coef, 1.0 + args.algo.clip_coef)
        pg_loss = jnp.maximum(pg1, pg2).mean()

        ext_target = minibatch.ext_returns
        ext_oldv = minibatch.ext_values
        ext_unclipped = 0.5 * jnp.square(new_v_ext - ext_target)
        if args.algo.clip_vloss:
            ext_v_clipped = ext_oldv + jnp.clip(new_v_ext - ext_oldv, -args.algo.clip_coef, args.algo.clip_coef)
            ext_clipped = 0.5 * jnp.square(ext_v_clipped - ext_target)
            v_loss_ext = jnp.maximum(ext_unclipped, ext_clipped).mean()
        else:
            v_loss_ext = ext_unclipped.mean()

        int_target = minibatch.int_returns
        v_loss_int = 0.5 * jnp.square(new_v_int - int_target).mean()

        logp_all = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        entropy = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()

        v_loss = v_loss_ext + v_loss_int
        loss = pg_loss - args.algo.ent_coef * entropy + args.algo.vf_coef * v_loss
        return loss, (pg_loss, v_loss_ext, v_loss_int, entropy, jax.lax.stop_gradient(approx_kl))

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0,1,2,3))(
        agent_state.enc.params,
        agent_state.actor.params,
        agent_state.critic_ext.params,
        agent_state.critic_int.params,
    )

    g_enc, g_actor, g_ext, g_int = grads
    agent_state = agent_state.replace(
        enc=agent_state.enc.apply_gradients(grads=g_enc),
        actor=agent_state.actor.apply_gradients(grads=g_actor),
        critic_ext=agent_state.critic_ext.apply_gradients(grads=g_ext),
        critic_int=agent_state.critic_int.apply_gradients(grads=g_int),
        grad_steps=agent_state.grad_steps + 1,
    )

    pg_loss, v_loss_ext, v_loss_int, entropy, approx_kl = aux
    return agent_state, (loss, pg_loss, v_loss_ext, v_loss_int, entropy, approx_kl)


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
    # (2) RND: obs RMS update + intrinsic reward normalization + GAE (ext/int)
    # ------------------------------------------------------------
    next_done = storage.dones[-1]
    burnin = agent_state.timesteps < jnp.int32(args.algo.obs_burnin)

    frames = storage.rnd_frame.reshape((-1, 84, 84, 1))  # [T*N,84,84,1]
    rnd = rnd_update_obs_rms(agent_state.rnd, frames)

    # burnin => kill intrinsic reward
    storage = jax.lax.cond(
        burnin,
        lambda s: s.replace(int_rewards=jnp.zeros_like(s.int_rewards)),
        lambda s: s,
        storage,
    )

    # after burnin => normalize intrinsic reward using forward filter + reward rms
    rnd, int_rewards_norm = jax.lax.cond(
        burnin,
        lambda r_s: (r_s, storage.int_rewards),
        lambda r_s: rnd_update_rew_rms(r_s, storage.int_rewards),
        rnd,
    )
    storage = storage.replace(int_rewards=int_rewards_norm)
    agent_state = agent_state.replace(rnd=rnd)

    # compute BOTH GAEs + returns
    storage = compute_gae_ext_int(
        agent_state,
        next_obs_chw=train_obs,
        next_done=next_done,
        storage=storage,
        gamma=args.env.gamma,
        int_gamma=args.algo.int_gamma,
        gae_lambda=args.algo.gae_lambda,
    )
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

        # ---- RND predictor update ONCE per epoch (use *state*, not outer agent_state) ----
        rnd, fwd_loss = rnd_update_once_per_epoch(state.rnd, flat, args.algo.upd_prop, burnin)
        state = state.replace(rnd=rnd)

        def one_mb(s, mb):
            s, stats = ppo_minibatch_update(s, mb, args)  # stats: (loss, pg, vext, vint, ent, kl)
            return s, stats

        state, stats = jax.lax.scan(one_mb, state, mbs)  # stats: [M, 6]
        return (state, rng), (stats, fwd_loss)  # fwd_loss: scalar

    key, upd_key = jax.random.split(key)
    (agent_state, _), (stats, fwd_losses) = jax.lax.scan(
        one_epoch,
        (agent_state, upd_key),
        None,
        length=args.algo.upd_epochs
    )
    # stats: [E, M, 6], fwd_losses: [E]
    loss_arr, pg_arr, vext_arr, vint_arr, ent_arr, kl_arr = stats  # each is [E, M]
    loss = loss_arr.mean()
    pg_loss = pg_arr.mean()
    vext_loss = vext_arr.mean()
    vint_loss = vint_arr.mean()
    ent = ent_arr.mean()
    approx_kl = kl_arr.mean()
    fwd_loss = fwd_losses.mean()

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
        "ppo/value_loss": vext_loss + vint_loss,
        "ppo/value_loss_extr": vext_loss,
        "ppo/value_loss_int": vint_loss,
        "ppo/entropy": ent,
        "ppo/approx_kl": approx_kl,
    }
    metrics.update({k: v.mean() for k, v in infos.items()})
    metrics.update({"Max_Return": infos["returned_episode_returns"].max()})

    int_r = storage.int_rewards
    metrics_int = {
        "rnd/int_reward_mean": int_r.mean(),
        "rnd/int_reward_max": int_r.max(),
        "rnd/int_reward_sum_per_env_mean": int_r.sum(axis=0).mean(),
    }
    metrics.update(metrics_int)
    metrics["rnd/fwd_loss"] = fwd_loss
    metrics["rnd/burnin"] = burnin.astype(jnp.float32)

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

            "Max_Return": float(m.get("Max_Return", 0.0)),
            "rnd/fwd_loss": float(m.get("rnd/fwd_loss", 0.0)),
            "rnd/burnin": float(m.get("rnd/burnin", 0.0)),

            "rnd/int_reward_mean": float(m.get("rnd/int_reward_mean", 0.0)),
            "rnd/int_reward_max": float(m.get("rnd/int_reward_max", 0.0)),
            "rnd/int_reward_sum_per_env_mean": float(m.get("rnd/int_reward_sum_per_env_mean", 0.0)),
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
