from functools import partial
from typing import Callable

import chex
import jax
import optax
import jax.numpy as jnp
from flax import struct

from src.mains.algorithms.RND import (
    RNDState,
    rnd_intrinsic_reward,
    rnd_update_obs_rms,
    rnd_update_rew_rms,
    rnd_update_once_per_epoch,
)
from src.mains.algorithms.PQN import (
    value_rescale,
    inv_value_rescale,
    compute_td_lambda_targets,
    compute_transformed_td_lambda_targets,
    learn_phase_dual,
)


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    actions: chex.Array
    ext_rewards: chex.Array
    int_rewards: chex.Array
    dones: chex.Array
    next_obs: chex.Array
    q_ext_vals: chex.Array
    q_int_vals: chex.Array
    rnd_frame: chex.Array


class FlatStorage(struct.PyTreeNode):
    rnd_frame: chex.Array


@partial(jax.jit, static_argnums=range(7, 19))
def env_take_steps_common(
    carry,
    _unused,
    enc_state,
    q_ext_state,
    q_int_state,
    rnd_state,
    n_updates,
    env,
    eps_scheduler,
    n_tr_envs,
    n_te_envs,
    use_enc_bn,
    use_q_bn,
    int_coef,
    use_value_rescale,
    reward_clip,
    eps_greedy_exploration_fn: Callable,
    get_dual_q_values_fn: Callable,
    rnd_frame_fn: Callable,
):
    last_obs, env_state, b_key = carry
    E = n_tr_envs + n_te_envs

    b_key, a_key = jax.random.split(b_key, 2)
    expl_keys = jax.random.split(a_key, E)

    q_ext_vals, q_int_vals = get_dual_q_values_fn(
        last_obs,
        enc_state,
        q_ext_state,
        q_int_state,
        use_enc_bn,
        use_q_bn,
        use_q_bn,
    )

    if use_value_rescale:
        q_sel_raw = inv_value_rescale(q_ext_vals) + int_coef * inv_value_rescale(q_int_vals)
        q_sel = value_rescale(q_sel_raw)
    else:
        q_sel = q_ext_vals + int_coef * q_int_vals

    eps_val = eps_scheduler(n_updates)
    eps = jnp.concatenate([
        jnp.full((n_tr_envs,), eps_val, dtype=jnp.float32),
        jnp.zeros((n_te_envs,), dtype=jnp.float32),
    ], axis=0)

    new_actions = jax.vmap(eps_greedy_exploration_fn)(expl_keys, q_sel, eps)
    new_obs, new_env_state, ext_reward, new_done, info = env.step(env_state, new_actions)

    if reward_clip:
        ext_reward = jnp.clip(ext_reward, -10.0, 10.0)

    rnd_frame = rnd_frame_fn(new_obs)
    int_reward = rnd_intrinsic_reward(rnd_state, rnd_frame)

    transition = Transition(
        obs=last_obs,
        actions=new_actions,
        ext_rewards=ext_reward,
        int_rewards=int_reward,
        dones=new_done,
        next_obs=new_obs,
        q_ext_vals=q_ext_vals,
        q_int_vals=q_int_vals,
        rnd_frame=rnd_frame,
    )
    return (new_obs, new_env_state, b_key), (transition, info)


@partial(jax.jit, static_argnums=range(2, 10))
def update_step_common(
    runner_state,
    _unused,
    args,
    env,
    eps_scheduler,
    start_time,
    eps_greedy_exploration_fn: Callable,
    get_dual_q_values_fn: Callable,
    rnd_frame_fn: Callable,
    make_wandb_log_callback_fn: Callable,
):
    train_state, expl_state, key = runner_state
    key, b_key = jax.random.split(key)

    step_fn_env = partial(
        env_take_steps_common,
        enc_state=train_state.enc_state,
        q_ext_state=train_state.q_ext_state,
        q_int_state=train_state.q_int_state,
        rnd_state=train_state.rnd_state,
        n_updates=train_state.n_updates,
        env=env,
        eps_scheduler=eps_scheduler,
        n_tr_envs=args.env.n_envs,
        n_te_envs=args.env.n_test_envs,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        int_coef=args.algo.int_coef,
        use_value_rescale=not args.algo.rew_scale,
        reward_clip=getattr(args.env, "reward_clip", False),
        eps_greedy_exploration_fn=eps_greedy_exploration_fn,
        get_dual_q_values_fn=get_dual_q_values_fn,
        rnd_frame_fn=rnd_frame_fn,
    )

    (*expl_state, _), (transitions_all, infos) = jax.lax.scan(
        step_fn_env,
        (*expl_state, b_key),
        None,
        args.env.n_steps,
    )
    expl_state = tuple(expl_state)

    if args.env.n_test_envs > 0:
        transitions = jax.tree_util.tree_map(lambda x: x[:, :args.env.n_envs], transitions_all)
    else:
        transitions = transitions_all

    train_state = train_state.replace(
        timesteps=train_state.timesteps + args.env.n_steps * args.env.n_envs
    )

    burnin = train_state.timesteps < jnp.int32(args.rnd.obs_burnin)

    flat_rnd_frames = transitions.rnd_frame.reshape((-1, *transitions.rnd_frame.shape[2:]))
    rnd_state = rnd_update_obs_rms(train_state.rnd_state, flat_rnd_frames)

    int_rewards = jax.lax.cond(
        burnin,
        lambda _: jnp.zeros_like(transitions.int_rewards),
        lambda _: transitions.int_rewards,
        operand=None,
    )

    rnd_state, int_rewards_norm = jax.lax.cond(
        burnin,
        lambda r: (r, int_rewards),
        lambda r: rnd_update_rew_rms(r, int_rewards),
        rnd_state,
    )

    transitions = transitions.replace(int_rewards=int_rewards_norm)
    train_state = train_state.replace(rnd_state=rnd_state)

    next_q_ext_vals, next_q_int_vals = get_dual_q_values_fn(
        transitions.next_obs[-1],
        train_state.enc_state,
        train_state.q_ext_state,
        train_state.q_int_state,
        args.net.batch_norm,
        args.net.batch_norm,
        args.net.batch_norm,
    )

    next_q_ext = jnp.max(next_q_ext_vals, axis=-1)
    next_q_int = jnp.max(next_q_int_vals, axis=-1)

    if args.algo.rew_scale:
        compute_ext_targets = partial(
            compute_td_lambda_targets,
            gamma=args.env.gamma,
            td_lambda=args.algo.td_lambda,
        )
        compute_int_targets = partial(
            compute_td_lambda_targets,
            gamma=args.rnd.int_gamma,
            td_lambda=args.algo.td_lambda,
        )
    else:
        compute_ext_targets = partial(
            compute_transformed_td_lambda_targets,
            gamma=args.env.gamma,
            td_lambda=args.algo.td_lambda,
        )
        compute_int_targets = partial(
            compute_transformed_td_lambda_targets,
            gamma=args.rnd.int_gamma,
            td_lambda=args.algo.td_lambda,
        )

    ext_targets = compute_ext_targets(
        next_q_ext,
        transitions.q_ext_vals,
        transitions.ext_rewards,
        transitions.dones,
    )

    int_dones = jnp.zeros_like(transitions.dones)
    int_targets = compute_int_targets(
        next_q_int,
        transitions.q_int_vals,
        transitions.int_rewards,
        int_dones,
    )

    _learn_phase = partial(
        learn_phase_dual,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        int_coef=args.algo.int_coef,
    )

    def _learn_epoch(carry, _):
        train_state, key = carry

        def preprocess_transition(x, key):
            x = x.reshape(-1, *x.shape[2:])
            x = jax.random.permutation(key, x)
            x = x.reshape(args.algo.n_mini_batches, -1, *x.shape[1:])
            return x

        key, _key = jax.random.split(key)
        mini_batches = jax.tree_util.tree_map(lambda x: preprocess_transition(x, _key), transitions)
        ext_tgts = preprocess_transition(ext_targets, _key)
        int_tgts = preprocess_transition(int_targets, _key)

        flat_storage = FlatStorage(rnd_frame=flat_rnd_frames)
        rnd_state, rnd_fwd_loss = rnd_update_once_per_epoch(
            train_state.rnd_state,
            flat_storage,
            args.rnd.upd_prop,
            burnin,
        )
        train_state = train_state.replace(rnd_state=rnd_state)

        (train_state, rng), (loss, qstats) = jax.lax.scan(
            _learn_phase,
            (train_state, _key),
            (mini_batches, ext_tgts, int_tgts),
        )
        return (train_state, rng), (loss, qstats, rnd_fwd_loss)

    key, _key = jax.random.split(key)
    (train_state, _), (losses, q_stats, rnd_fwd_losses) = jax.lax.scan(
        _learn_epoch,
        (train_state, _key),
        None,
        args.algo.n_epochs,
    )

    grad_steps_inc = args.algo.n_epochs * args.algo.n_mini_batches
    train_state = train_state.replace(
        n_updates=train_state.n_updates + 1,
        grad_steps=train_state.grad_steps + grad_steps_inc,
    )

    ext_loss = q_stats[0].mean()
    int_loss = q_stats[1].mean()
    q_ext_mean_t = q_stats[2].mean()
    q_ext_mean_raw = q_stats[3].mean()
    q_int_mean_t = q_stats[4].mean()
    q_int_mean_raw = q_stats[5].mean()

    metrics = {
        "env_step": train_state.timesteps,
        "update_steps": train_state.n_updates,
        "env_frame": train_state.timesteps * args.env.n_frames,
        "grad_steps": train_state.grad_steps,
        "loss/total_td": losses.mean(),
        "loss/ext_td": ext_loss,
        "loss/int_td": int_loss,
        "Q_mean_transformed": q_ext_mean_t,
        "Q_mean_raw": q_ext_mean_raw,
        "Q_int_mean_transformed": q_int_mean_t,
        "Q_int_mean_raw": q_int_mean_raw,
        "RND/int_reward_mean": transitions.int_rewards.mean(),
        "RND/int_reward_max": transitions.int_rewards.max(),
        "RND/int_reward_sum_per_env_mean": transitions.int_rewards.sum(axis=0).mean(),
        "RND/fwd_loss": rnd_fwd_losses.mean(),
        "RND/burnin": burnin.astype(jnp.float32),
    }

    if args.env.n_test_envs > 0:
        train_infos = jax.tree_util.tree_map(lambda x: x[:, :-args.env.n_test_envs], infos)
        test_infos = jax.tree_util.tree_map(lambda x: x[:, -args.env.n_test_envs:], infos)

        metrics.update({k: v.mean() for k, v in train_infos.items()})
        metrics.update({f"test/{k}": v.mean() for k, v in test_infos.items()})

        metrics["Max_Return"] = train_infos["returned_episode_returns"].max()
        metrics["Max_Normalized_Return"] = train_infos["returned_episode_normalized_returns"].max()

        metrics["test/Max_Return"] = test_infos["returned_episode_returns"].max()
        metrics["test/Max_Normalized_Return"] = test_infos["returned_episode_normalized_returns"].max()
    else:
        metrics.update({k: v.mean() for k, v in infos.items()})
        metrics["Max_Return"] = infos["returned_episode_returns"].max()
        metrics["Max_Normalized_Return"] = infos["returned_episode_normalized_returns"].max()

    pass_and_log = make_wandb_log_callback_fn(args, start_time)
    if args.wb.on:
        do_log = (
            (train_state.n_updates % 10 == 0)
            | (train_state.n_updates == (args.algo.n_upds - 1))
        )
        _ = jax.lax.cond(do_log, lambda m: pass_and_log(m), lambda m: None, metrics)

    runner_state = (train_state, tuple(expl_state), key)
    return runner_state, metrics