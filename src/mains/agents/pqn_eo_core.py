import time
from functools import partial
from typing import Any, List, NamedTuple, Union

import chex
import jax
import optax
import numpy as np
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from src.networks.networks import Conv_Encoder
from src.utils.jax_utils import learning_rate_fn
from src.mains.algorithms.PQN import (
    compute_td_lambda_targets,
    compute_transformed_td_lambda_targets,
    learn_phase,
)
from src.mains.algorithms.EO import (
    compute_options_lambda_targets_shared,
    update_option_networks_shared,
)
from src.mains.algorithms.Laplacian import update_laplacian_network


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    actions: chex.Array
    rewards: chex.Array
    dones: chex.Array
    next_obs: chex.Array

    q_vals: chex.Array
    option_idx: chex.Array
    option_q_vals: chex.Array
    action_probs: chex.Array


@struct.dataclass
class Lap_State(TrainState):
    betas: Any
    errors: Any
    quadratic_errors: Any
    barrier_coeffs: Any
    eigen_values: Any
    batch_stats: Union[FrozenDict, None]


class LapBatch(NamedTuple):
    obs: jnp.ndarray
    dones: jnp.ndarray
    actions: jnp.ndarray
    action_probs: jnp.ndarray


@jax.jit
def eps_greedy_exploration(rng, q_vals, eps):
    rng_a, rng_e = jax.random.split(rng)
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    chosen_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape) < eps,
        jax.random.randint(rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]),
        greedy_actions,
    )
    return chosen_actions


@partial(jax.jit, static_argnums=(3, 4))
def get_q_values_shared(obs, enc_state, q_state, use_enc_bn, use_q_bn):
    obs = jnp.transpose(obs, (0, 2, 3, 1))

    enc_vars = {"params": enc_state.params}
    if use_enc_bn:
        enc_vars["batch_stats"] = enc_state.batch_stats
    embeddings = enc_state.apply_fn(enc_vars, obs, train=False)

    q_vars = {"params": q_state.params}
    if use_q_bn:
        q_vars["batch_stats"] = q_state.batch_stats
    q_values = q_state.apply_fn(q_vars, embeddings, train=False)
    return q_values


@partial(jax.jit, static_argnums=(3, 4))
def compute_option_qs_shared(obs, opt_enc_state, opt_q_states, use_enc_bn, use_q_bn):
    obs = jnp.transpose(obs, (0, 2, 3, 1))

    enc_vars = {"params": opt_enc_state.params}
    if use_enc_bn:
        enc_vars["batch_stats"] = opt_enc_state.batch_stats
    emb = opt_enc_state.apply_fn(enc_vars, obs, train=False)

    q_params = jax.tree_util.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in opt_q_states])
    q_bs = (
        jax.tree_util.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in opt_q_states])
        if use_q_bn else None
    )
    q_apply = opt_q_states[0].apply_fn

    def q_one(p_q, bs_q):
        vars_q = {"params": p_q}
        if use_q_bn:
            vars_q["batch_stats"] = bs_q
        return q_apply(vars_q, emb, train=False)

    q_opts = jax.vmap(q_one, in_axes=(0, 0 if use_q_bn else None))(q_params, q_bs)
    return q_opts  # [O,E,A]


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def eigenvalues_to_option_durations_skip_first(
    eigs: jnp.ndarray,
    n_base_options: int,
    bidir_options: bool,
    min_dur: int,
    max_dur: int,
    eps: float = 1e-6,
):
    eigs = jnp.reshape(eigs, (-1,))
    eigs = jnp.nan_to_num(eigs, nan=0.0, posinf=0.0, neginf=0.0)
    eigs = jnp.clip(eigs, 0.0, jnp.inf)

    usable = eigs[1:1 + n_base_options]
    lam = jnp.maximum(usable, eps)
    dur = jnp.ceil(1.0 / lam).astype(jnp.int32)
    dur = jnp.clip(dur, min_dur, max_dur)

    if bidir_options:
        dur = jnp.concatenate([dur, dur], axis=0)
    return dur


def build_lap_state_common(key, args, obs):
    obs0 = np.transpose(obs[0], (1, 2, 0))[None]

    lap_network = Conv_Encoder(
        name="Lap:Enc",
        layer_dims=list(args.net.c_dims) + [args.laplacian.dims],
        activations=[nn.relu] * len(args.net.c_dims) + [None],
        min_scale=1,
        width_mul=1,
        impala_cnn=getattr(args.net, "impala", False),
        batch_norm=args.laplacian.batch_norm,
        layer_norm=args.laplacian.layer_norm,
    )

    lap_variables = lap_network.init(key, obs0, train=False)
    lap_lr_schedule = learning_rate_fn(
        init_lr=args.laplacian.lr,
        alpha=1.0,
        num_epochs=args.algo.n_upds,
        warmup_epochs=max(1, args.algo.n_upds // 4),
        steps_per_epoch=max(1, args.algo.n_epochs),
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
    return lap_state


def make_wandb_log_callback(args, start_time):
    def _log_to_wandb(m):
        import wandb
        elapsed = time.time() - start_time
        total_env = m["update_steps"] * args.env.n_steps * args.env.n_envs
        m["SPS"] = total_env / max(elapsed, 1e-6)
        wandb.log(m, step=int(m["update_steps"]))

    @jax.jit
    def _pass_and_log(m):
        return jax.debug.callback(_log_to_wandb, m)

    return _pass_and_log


@partial(jax.jit, static_argnums=range(8, 17))
def env_take_steps_common(
    carry,
    _unused,

    # main primitive controller
    enc_state,
    q_state,

    # option controllers
    opt_enc_state,
    opt_q_states,

    # schedules / durations
    n_updates,
    opt_dur_table,

    # env/config
    env,
    eps_scheduler,
    mu_prob,
    n_tr_envs,
    n_te_envs,
    use_enc_bn,
    use_q_bn,
    save_act_probs,
    n_options,
):
    (
        obs_t,              # [E,C,H,W]
        env_state,

        active_option,      # [E], -1 if no active option
        opt_steps_left,     # [E], remaining option steps

        key,
    ) = carry

    E = n_tr_envs + n_te_envs
    A = env.single_action_space.n
    O = n_options

    key, k_eps, k_mu, k_option, k_primitive = jax.random.split(key, 5)

    # ------------------------------------------------------------
    # 1. Main primitive Q-values
    # ------------------------------------------------------------
    q_main = get_q_values_shared(
        obs_t,
        enc_state,
        q_state,
        use_enc_bn,
        use_q_bn,
    )

    greedy_primitive = jnp.argmax(q_main, axis=-1)

    # ------------------------------------------------------------
    # 2. Option Q-values
    # ------------------------------------------------------------
    q_opts = compute_option_qs_shared(
        obs_t,
        opt_enc_state,
        opt_q_states,
        use_enc_bn,
        use_q_bn,
    )  # [O,E,A]

    # ------------------------------------------------------------
    # 3. Decide whether a new behavior source is needed
    # ------------------------------------------------------------
    need_new_option = (active_option < 0) | (opt_steps_left <= 0)

    eps_val = eps_scheduler(n_updates)

    eps = jnp.concatenate(
        [
            jnp.full((n_tr_envs,), eps_val, dtype=jnp.float32),
            jnp.zeros((n_te_envs,), dtype=jnp.float32),
        ],
        axis=0,
    )

    u_eps = jax.random.uniform(k_eps, shape=(E,))
    u_mu = jax.random.uniform(k_mu, shape=(E,))

    do_explore = need_new_option & (u_eps < eps)
    explore_option = do_explore & (u_mu < mu_prob)
    explore_primitive = do_explore & (u_mu >= mu_prob)

    sampled_option = jax.random.randint(
        k_option,
        shape=(E,),
        minval=0,
        maxval=O,
    )

    sampled_primitive = jax.random.randint(
        k_primitive,
        shape=(E,),
        minval=0,
        maxval=A,
    )

    # ------------------------------------------------------------
    # 4. Choose/update active option
    # ------------------------------------------------------------
    new_active_option = jnp.where(
        explore_option,
        sampled_option,
        jnp.where(
            need_new_option,
            -1,
            active_option,
        ),
    )

    is_option_active = new_active_option >= 0

    active_opt = jnp.clip(new_active_option, 0, O - 1)
    q_active_option = q_opts[active_opt, jnp.arange(E), :]
    option_action = jnp.argmax(q_active_option, axis=-1)

    primitive_action = jnp.where(
        explore_option,
        option_action,
        jnp.where(
            explore_primitive,
            sampled_primitive,
            jnp.where(
                is_option_active,
                option_action,
                greedy_primitive,
            ),
        ),
    )
    primitive_action = jnp.clip(primitive_action, 0, A - 1)

    # ------------------------------------------------------------
    # 5. Update option remaining duration
    # ------------------------------------------------------------
    sampled_option_dur = opt_dur_table[sampled_option]

    opt_steps_left_after = jnp.where(
        explore_option,
        sampled_option_dur - 1,
        jnp.where(
            is_option_active,
            opt_steps_left - 1,
            0,
        ),
    )
    opt_steps_left_after = jnp.maximum(opt_steps_left_after, 0)

    # ------------------------------------------------------------
    # 6. Step environment
    # ------------------------------------------------------------
    obs_tp1, env_state_tp1, reward, done, info = env.step(
        env_state,
        primitive_action,
    )

    # ------------------------------------------------------------
    # 7. Reset active option on option termination or env termination
    # ------------------------------------------------------------
    option_done = is_option_active & (opt_steps_left_after <= 0)

    next_active_option = jnp.where(
        option_done | done,
        -1,
        new_active_option,
    )

    next_opt_steps_left = jnp.where(
        option_done | done,
        0,
        opt_steps_left_after,
    )

    # ------------------------------------------------------------
    # 8. Optional primitive behavior probability
    # ------------------------------------------------------------
    action_probs = None
    if save_act_probs:
        # Main primitive behavior is epsilon-greedy over primitive actions.
        main_greedy_onehot = jax.nn.one_hot(
            greedy_primitive,
            A,
            dtype=jnp.float32,
        )
        main_base = eps[:, None] / jnp.float32(A)
        main_action_probs = main_base + (1.0 - eps)[:, None] * main_greedy_onehot

        # Option policy executes greedily.
        opt_action_probs = jax.nn.one_hot(
            option_action,
            A,
            dtype=jnp.float32,
        )

        # This is only an approximate primitive behavior probability because
        # EO mixes primitive exploration, option exploration, and option continuation.
        action_probs = jnp.where(
            is_option_active[:, None] | explore_option[:, None],
            opt_action_probs,
            main_action_probs,
        )

    # ------------------------------------------------------------
    # 9. Primitive transition
    # ------------------------------------------------------------
    transition = Transition(
        obs=obs_t,
        actions=primitive_action,
        rewards=reward,
        dones=done,
        next_obs=obs_tp1,

        q_vals=q_main,
        option_idx=next_active_option,
        option_q_vals=jnp.transpose(q_opts, (1, 0, 2)),

        action_probs=action_probs,
    )

    new_carry = (
        obs_tp1,
        env_state_tp1,

        next_active_option,
        next_opt_steps_left,

        key,
    )

    return new_carry, (transition, info)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def update_step_common(runner_state, _unused, args, env, eps_scheduler, start_time):
    train_state, expl_state, key, opt_dur_table = runner_state

    n_base_options = args.options.n_options
    n_total_options = 2 * n_base_options if args.options.bidir_options else n_base_options
    n_prim_actions = env.single_action_space.n

    key, b_key = jax.random.split(key)

    step_fn_env = partial(
        env_take_steps_common,
        enc_state=train_state.enc_state,
        q_state=train_state.q_state,
        opt_enc_state=train_state.opt_enc_state,
        opt_q_states=train_state.opt_q_states,
        n_updates=train_state.n_updates,
        opt_dur_table=opt_dur_table,  # fixed name
        env=env,
        eps_scheduler=eps_scheduler,
        mu_prob=args.options.mu_prob,
        n_tr_envs=args.env.n_envs,
        n_te_envs=args.env.n_test_envs,
        use_enc_bn=args.net.batch_norm,  # fixed name
        use_q_bn=args.net.batch_norm,
        save_act_probs=False,
        n_options=n_total_options,
    )

    (*expl_state, _), (transitions_all, infos) = jax.lax.scan(
        step_fn_env,
        (*expl_state, b_key),
        None,
        args.env.n_steps,
    )
    expl_state = tuple(expl_state)

    if args.env.n_test_envs > 0:
        transitions = jax.tree_util.tree_map(lambda x: x[:, :-args.env.n_test_envs], transitions_all)
    else:
        transitions = transitions_all

    train_state = train_state.replace(
        timesteps=train_state.timesteps + args.env.n_steps * args.env.n_envs
    )

    next_q_vals = get_q_values_shared(
        transitions.next_obs[-1],
        train_state.enc_state,
        train_state.q_state,
        args.net.batch_norm,
        args.net.batch_norm,
    )
    next_q = jnp.max(next_q_vals, axis=-1)

    if args.algo.rew_scale:
        compute_main_targets = partial(
            compute_td_lambda_targets,
            gamma=args.env.gamma,
            td_lambda=args.algo.td_lambda,
        )
    else:
        compute_main_targets = partial(
            compute_transformed_td_lambda_targets,
            gamma=args.env.gamma,
            td_lambda=args.algo.td_lambda,
        )

    lambda_targets = compute_main_targets(
        next_q,
        transitions.q_vals,
        transitions.rewards,
        transitions.dones,
    )

    _main_learn_phase = partial(
        learn_phase,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
    )

    opt_lambda_targets = compute_options_lambda_targets_shared(
        train_state=train_state,
        transitions=transitions,
        gamma=args.env.gamma,
        td_lambda=args.algo.td_lambda,
        use_opt_bn=args.net.batch_norm,
        use_lap_bn=args.laplacian.batch_norm,
        n_base_options=n_base_options,
        bidir_options=args.options.bidir_options,
        use_value_rescale=not args.algo.rew_scale,
    )

    _opt_learn_phase = partial(
        update_option_networks_shared,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
    )

    _update_lap_network = partial(
        update_laplacian_network,
        args=args,
        n_actions=n_prim_actions,
    )

    def _learn_epoch(carry, _):
        train_state, key = carry
        key, _key, lap_key = jax.random.split(key, 3)

        def preprocess_transition(x, key, n_shards):
            x = x.reshape(-1, *x.shape[2:])
            x = jax.random.permutation(key, x)
            x = x.reshape(n_shards, -1, *x.shape[1:])
            return x

        lap_batches = LapBatch(
            obs=transitions.obs.transpose(1, 0, 2, 3, 4),
            dones=transitions.dones.T,
            actions=transitions.actions.T,
            action_probs=None,
        )

        train_state, (eig_loss, pos_loss, ortho_loss, dual_loss) = _update_lap_network(
            lap_key, train_state, lap_batches
        )

        main_batches = jax.tree_util.tree_map(
            lambda x: preprocess_transition(x, _key, args.algo.n_mini_batches),
            transitions,
        )
        main_targets_mb = preprocess_transition(
            lambda_targets, _key, args.algo.n_mini_batches
        )

        (train_state, _), (main_losses, main_qstats) = jax.lax.scan(
            _main_learn_phase,
            (train_state, _key),
            (main_batches, main_targets_mb),
        )

        opt_batches = jax.tree_util.tree_map(
            lambda x: preprocess_transition(x, _key, args.options.n_mini_batches),
            transitions,
        )

        opt_targets = jax.vmap(
            lambda targ_o: preprocess_transition(targ_o, _key, args.options.n_mini_batches),
            in_axes=0,
        )(opt_lambda_targets)
        opt_targets = jnp.transpose(opt_targets, (1, 0, 2))

        def _opt_mb_step(ts, shards):
            mb_data, mb_opt_tgt = shards
            ts, (opt_loss, opt_q_t, opt_q_raw) = _opt_learn_phase(ts, mb_data, mb_opt_tgt)
            return ts, (opt_loss, opt_q_t, opt_q_raw)

        train_state, (opt_losses, opt_q_t, opt_q_raw) = jax.lax.scan(
            _opt_mb_step,
            train_state,
            (opt_batches, opt_targets),
        )

        return (train_state, key), (
            main_losses,
            main_qstats,
            opt_losses,
            opt_q_t,
            opt_q_raw,
            eig_loss,
            pos_loss,
            ortho_loss,
            dual_loss,
        )

    key, _key = jax.random.split(key)
    (train_state, _), epoch_stats = jax.lax.scan(
        _learn_epoch,
        (train_state, _key),
        None,
        args.algo.n_epochs,
    )

    grad_steps_inc = args.algo.n_epochs * (args.algo.n_mini_batches + args.options.n_mini_batches)
    train_state = train_state.replace(
        n_updates=train_state.n_updates + 1,
        grad_steps=train_state.grad_steps + grad_steps_inc,
    )

    main_losses = epoch_stats[0]
    main_qstats = epoch_stats[1]
    opt_losses = epoch_stats[2]
    opt_q_t = epoch_stats[3]
    opt_q_raw = epoch_stats[4]
    eig_loss = epoch_stats[5]
    pos_loss = epoch_stats[6]
    ortho_loss = epoch_stats[7]
    dual_loss = epoch_stats[8]

    q_mean_t = main_qstats[0].mean()
    q_mean_raw = main_qstats[1].mean()

    opt_td_loss = opt_losses.mean(axis=(0, 1))
    opt_q_mean_t = opt_q_t.mean(axis=(0, 1))
    opt_q_mean_raw = opt_q_raw.mean(axis=(0, 1))

    if args.options.eigdur_options:
        opt_dur_table = eigenvalues_to_option_durations_skip_first(
            train_state.lap_state.eigen_values,
            n_base_options=n_base_options,
            bidir_options=args.options.bidir_options,
            min_dur=args.options.opt_dur_min,
            max_dur=args.options.opt_dur_max,
            eps=1e-6,
        )
    else:
        opt_dur_table = jnp.full(
            (n_total_options,),
            jnp.int32(args.options.opt_dur),
            dtype=jnp.int32,
        )

    metrics = {
        "env_step": train_state.timesteps,
        "update_steps": train_state.n_updates,
        "env_frame": train_state.timesteps * args.env.n_frames,
        "grad_steps": train_state.grad_steps,

        "TD_loss": main_losses.mean(),
        "Q_mean_transformed": q_mean_t,
        "Q_mean_raw": q_mean_raw,

        "Eigen/Total_loss": eig_loss.mean(),
        "Eigen/Pos_loss": pos_loss.mean(),
        "Eigen/Ortho_loss": ortho_loss.mean(),
        "Eigen/Dual_loss": dual_loss.mean(),
        "Eigen/Barrier": train_state.lap_state.barrier_coeffs[0, 0],
        "Eigen/Max_EigenValue": jnp.max(train_state.lap_state.eigen_values),

        "Options/dur_min": opt_dur_table.min(),
        "Options/dur_max": opt_dur_table.max(),
        "Options/dur_mean": opt_dur_table.astype(jnp.float32).mean(),
    }

    for i in range(n_total_options):
        metrics[f"Options/TD_loss_{i}"] = opt_td_loss[i]
        metrics[f"Options/Q_mean_t_{i}"] = opt_q_mean_t[i]
        metrics[f"Options/Q_mean_raw_{i}"] = opt_q_mean_raw[i]

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

    pass_and_log = make_wandb_log_callback(args, start_time)
    if args.wb.on:
        do_log = (
            (train_state.n_updates % 10 == 0)
            | (train_state.n_updates == (args.algo.n_upds - 1))
        )
        _ = jax.lax.cond(do_log, lambda m: pass_and_log(m), lambda m: None, metrics)

    runner_state = (train_state, tuple(expl_state), key, opt_dur_table)
    return runner_state, metrics