from functools import partial
from typing import Callable

import chex
import jax
import optax
import jax.numpy as jnp
from flax import struct

from src.mains.algorithms.PQN import (
    compute_td_lambda_targets,
    compute_transformed_td_lambda_targets,
    learn_phase,
)


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    actions: chex.Array
    rewards: chex.Array
    dones: chex.Array
    next_obs: chex.Array
    q_vals: chex.Array


@partial(jax.jit, static_argnums=range(5, 14))
def env_take_steps_common(
    carry,
    _unused,
    enc_state,
    q_state,
    n_updates,
    env,
    eps_scheduler,
    n_tr_envs,
    n_te_envs,
    use_enc_bn,
    use_q_bn,
    reward_clip,
    eps_greedy_exploration_fn: Callable,
    get_q_values_fn: Callable,
):
    last_obs, env_state, b_key = carry
    E = n_tr_envs + n_te_envs

    b_key, a_key = jax.random.split(b_key, 2)
    expl_keys = jax.random.split(a_key, E)

    q_vals = get_q_values_fn(last_obs, enc_state, q_state, use_enc_bn, use_q_bn)

    eps_val = eps_scheduler(n_updates)
    eps = jnp.concatenate([
        jnp.full((n_tr_envs,), eps_val, dtype=jnp.float32),
        jnp.zeros((n_te_envs,), dtype=jnp.float32),
    ], axis=0)

    new_actions = jax.vmap(eps_greedy_exploration_fn)(expl_keys, q_vals, eps)
    new_obs, new_env_state, reward, new_done, info = env.step(env_state, new_actions)

    if reward_clip:
        reward = jnp.clip(reward, -10.0, 10.0)

    transition = Transition(
        obs=last_obs,
        actions=new_actions,
        rewards=reward,
        dones=new_done,
        next_obs=new_obs,
        q_vals=q_vals,
    )
    return (new_obs, new_env_state, b_key), (transition, info)


@partial(jax.jit, static_argnums=range(2, 9))
def update_step_common(
    runner_state,
    _unused,
    args,
    env,
    eps_scheduler,
    start_time,
    eps_greedy_exploration_fn: Callable,
    get_q_values_fn: Callable,
    make_wandb_log_callback_fn: Callable,
):
    train_state, expl_state, key = runner_state
    key, b_key = jax.random.split(key)

    step_fn_env = partial(
        env_take_steps_common,
        enc_state=train_state.enc_state,
        q_state=train_state.q_state,
        n_updates=train_state.n_updates,
        env=env,
        eps_scheduler=eps_scheduler,
        n_tr_envs=args.env.n_envs,
        n_te_envs=args.env.n_test_envs,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        reward_clip=getattr(args.env, "reward_clip", False),
        eps_greedy_exploration_fn=eps_greedy_exploration_fn,
        get_q_values_fn=get_q_values_fn,
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

    new_q = jnp.max(
        get_q_values_fn(
            transitions.next_obs[-1],
            train_state.enc_state,
            train_state.q_state,
            args.net.batch_norm,
            args.net.batch_norm,
        ),
        axis=-1,
    )

    if args.algo.rew_scale:
        compute_td_targets = partial(
            compute_td_lambda_targets,
            gamma=args.env.gamma,
            td_lambda=args.algo.td_lambda,
        )
    else:
        compute_td_targets = partial(
            compute_transformed_td_lambda_targets,
            gamma=args.env.gamma,
            td_lambda=args.algo.td_lambda,
        )

    lambda_targets = compute_td_targets(
        new_q,
        transitions.q_vals,
        transitions.rewards,
        transitions.dones,
    )

    _learn_phase = partial(
        learn_phase,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
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
        targets = preprocess_transition(lambda_targets, _key)

        (train_state, rng), (loss, qstats) = jax.lax.scan(
            _learn_phase,
            (train_state, _key),
            (mini_batches, targets),
        )
        return (train_state, rng), (loss, qstats)

    key, _key = jax.random.split(key)
    (train_state, _), (losses, q_stats) = jax.lax.scan(
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

    q_mean_t = q_stats[0].mean()
    q_mean_raw = q_stats[1].mean()

    metrics = {
        "env_step": train_state.timesteps,
        "update_steps": train_state.n_updates,
        "env_frame": train_state.timesteps * args.env.n_frames,
        "grad_steps": train_state.grad_steps,
        "TD_loss": losses.mean(),
        "Q_mean_transformed": q_mean_t,
        "Q_mean_raw": q_mean_raw,
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