# mains/agents/pqn_neo_core.py
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
from src.utils.jax_utils import learning_rate_fn, qema_update
from src.mains.algorithms.RND import (
    rnd_intrinsic_reward,
    rnd_update_obs_rms,
    rnd_update_rew_rms,
    rnd_update_once_per_epoch,
)
from src.mains.algorithms.PQN import (
    value_rescale,
    inv_value_rescale,
    learn_phase_dual,
    compute_td_lambda_targets,
    compute_transformed_td_lambda_targets,
)
from src.mains.algorithms.EO import (
    update_option_networks_shared_dual,
)
from src.mains.algorithms.Laplacian import update_neo_laplacian_network
from src.mains.algorithms.NEO import (
    NEOScaleState,
    neo_scale_from_state_calibrated,
    compute_neo_options_lambda_targets_shared_dual,
)


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    actions: chex.Array
    rewards: chex.Array
    int_rewards: chex.Array
    dones: chex.Array
    next_obs: chex.Array

    q_vals: chex.Array
    q_int_vals: chex.Array

    option_idx: chex.Array
    option_q_ext_vals: chex.Array
    option_q_int_vals: chex.Array

    action_probs: chex.Array

    rnd_frame: chex.Array
    rnd_bonus: chex.Array
    rnd_bonus_n: chex.Array
    inv_count: chex.Array


class FlatStorage(struct.PyTreeNode):
    rnd_frame: chex.Array


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
    counts: jnp.ndarray
    novelty: jnp.ndarray


@jax.jit
def eps_greedy_exploration(rng, q_vals, eps):
    rng_a, rng_e = jax.random.split(rng)
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    chosen_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape) < eps,
        jax.random.randint(
            rng_a,
            shape=greedy_actions.shape,
            minval=0,
            maxval=q_vals.shape[-1],
        ),
        greedy_actions,
    )
    return chosen_actions


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


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def compute_option_dual_qs(
    obs,
    opt_enc_state,
    opt_q_ext_states,
    opt_q_int_states,
    use_enc_bn,
    use_q_bn,
    opt_int_coef,
    use_value_rescale,
):
    """
    obs: [E,C,H,W]
    returns:
      q_ext: [O,E,A]
      q_int: [O,E,A]
      q_sel: [O,E,A]
    """
    obs = jnp.transpose(obs, (0, 2, 3, 1))

    enc_vars = {"params": opt_enc_state.params}
    if use_enc_bn:
        enc_vars["batch_stats"] = opt_enc_state.batch_stats
    emb = opt_enc_state.apply_fn(enc_vars, obs, train=False)

    qext_params = jax.tree_util.tree_map(
        lambda *ps: jnp.stack(ps), *[s.params for s in opt_q_ext_states]
    )
    qint_params = jax.tree_util.tree_map(
        lambda *ps: jnp.stack(ps), *[s.params for s in opt_q_int_states]
    )

    qext_bs = (
        jax.tree_util.tree_map(
            lambda *bs: jnp.stack(bs), *[s.batch_stats for s in opt_q_ext_states]
        )
        if use_q_bn else None
    )
    qint_bs = (
        jax.tree_util.tree_map(
            lambda *bs: jnp.stack(bs), *[s.batch_stats for s in opt_q_int_states]
        )
        if use_q_bn else None
    )

    qext_apply = opt_q_ext_states[0].apply_fn
    qint_apply = opt_q_int_states[0].apply_fn

    def qext_one(p_q, bs_q):
        vars_q = {"params": p_q}
        if use_q_bn:
            vars_q["batch_stats"] = bs_q
        return qext_apply(vars_q, emb, train=False)

    def qint_one(p_q, bs_q):
        vars_q = {"params": p_q}
        if use_q_bn:
            vars_q["batch_stats"] = bs_q
        return qint_apply(vars_q, emb, train=False)

    q_ext = jax.vmap(qext_one, in_axes=(0, 0 if use_q_bn else None))(qext_params, qext_bs)
    q_int = jax.vmap(qint_one, in_axes=(0, 0 if use_q_bn else None))(qint_params, qint_bs)

    if use_value_rescale:
        q_sel_raw = inv_value_rescale(q_ext) + opt_int_coef * inv_value_rescale(q_int)
        q_sel = value_rescale(q_sel_raw)
    else:
        q_sel = q_ext + opt_int_coef * q_int

    return q_ext, q_int, q_sel


@partial(jax.jit, static_argnums=range(11, 25))
def env_take_steps_common(
    carry,
    _unused,

    # main primitive controller
    enc_state,
    q_ext_state,
    q_int_state,

    # option controllers
    opt_enc_state,
    opt_q_ext_states,
    opt_q_int_states,

    # RND
    rnd_state,

    # schedules / durations
    n_updates,
    option_dur_table,

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
    int_coef,
    opt_int_coef,
    use_value_rescale,
    get_dual_q_values_fn,
    rnd_frame_fn,
):
    (
        obs_t,              # [E,C,H,W]
        env_state,
        active_opt,         # [E], -1 means no active option
        opt_steps_left,     # [E], remaining primitive steps for active option
        key,
    ) = carry

    key, k_eps, k_mu, k_option, k_primitive = jax.random.split(key, 5)

    E = n_tr_envs + n_te_envs
    A = env.single_action_space.n
    O = n_options

    # ------------------------------------------------------------
    # 1. Main primitive Q-values
    # ------------------------------------------------------------
    q_ext, q_int = get_dual_q_values_fn(
        obs_t,
        enc_state,
        q_ext_state,
        q_int_state,
        use_enc_bn,
        use_q_bn,
        use_q_bn,
    )

    if use_value_rescale:
        q_sel_raw = inv_value_rescale(q_ext) + int_coef * inv_value_rescale(q_int)
        q_sel = value_rescale(q_sel_raw)
    else:
        q_sel_raw = q_ext + int_coef * q_int
        q_sel = q_sel_raw

    greedy_primitive = jnp.argmax(q_sel, axis=-1)

    # ------------------------------------------------------------
    # 2. Option Q-values
    # ------------------------------------------------------------
    q_opts_ext, q_opts_int, q_opts_sel = compute_option_dual_qs(
        obs_t,
        opt_enc_state,
        opt_q_ext_states,
        opt_q_int_states,
        use_enc_bn,
        use_q_bn,
        opt_int_coef,
        use_value_rescale,
    )

    # ------------------------------------------------------------
    # 3. Decide whether a new primitive/option choice is needed
    # ------------------------------------------------------------
    need_new_choice = (active_opt < 0) | (opt_steps_left <= 0)

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

    do_explore = need_new_choice & (u_eps < eps)
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

    continuing_option = ~need_new_choice

    # ------------------------------------------------------------
    # 4. Select option index used for option-action computation
    # ------------------------------------------------------------
    safe_active_opt = jnp.clip(active_opt, 0, O - 1)

    executed_opt = jnp.where(
        explore_option,
        sampled_option,
        safe_active_opt,
    )

    q_executed_option = q_opts_sel[executed_opt, jnp.arange(E), :]
    greedy_option_action = jnp.argmax(q_executed_option, axis=-1)

    # ------------------------------------------------------------
    # 5. Select primitive action
    # ------------------------------------------------------------
    primitive_action = jnp.where(
        explore_option,
        greedy_option_action,
        jnp.where(
            explore_primitive,
            sampled_primitive,
            jnp.where(
                continuing_option,
                greedy_option_action,
                greedy_primitive,
            ),
        ),
    )

    primitive_action = jnp.clip(primitive_action, 0, A - 1)

    # ------------------------------------------------------------
    # 6. Update active option state before env step
    # ------------------------------------------------------------
    selected_option_dur = option_dur_table[sampled_option]

    next_active_opt_pre = jnp.where(
        explore_option,
        sampled_option,
        jnp.where(
            continuing_option,
            active_opt,
            -1,
        ),
    )

    next_opt_steps_left_pre = jnp.where(
        explore_option,
        selected_option_dur - 1,
        jnp.where(
            continuing_option,
            opt_steps_left - 1,
            0,
        ),
    )
    next_opt_steps_left_pre = jnp.maximum(next_opt_steps_left_pre, 0)

    # For the transition, store the option that generated this primitive action.
    # If the action came from the main primitive policy, this is -1.
    transition_option_idx = jnp.where(
        explore_option | continuing_option,
        next_active_opt_pre,
        -1,
    )

    # ------------------------------------------------------------
    # 7. Optional primitive action probability
    # ------------------------------------------------------------
    action_probs = None

    if save_act_probs:
        main_base = eps[:, None] * (1.0 - mu_prob) / jnp.float32(A)

        main_greedy_onehot = jax.nn.one_hot(
            greedy_primitive,
            A,
            dtype=jnp.float32,
        )

        main_action_probs = (
            main_base
            + (1.0 - eps)[:, None] * main_greedy_onehot
        )

        opt_greedy_onehot = jax.nn.one_hot(
            greedy_option_action,
            A,
            dtype=jnp.float32,
        )

        # Options are executed greedily here, so the primitive action under
        # an active option has probability 1 for the greedy option action.
        opt_action_probs = opt_greedy_onehot

        behav_probs_all = jnp.where(
            (explore_option | continuing_option)[:, None],
            opt_action_probs,
            main_action_probs,
        )

        action_probs = jnp.take_along_axis(
            behav_probs_all,
            primitive_action[:, None],
            axis=-1,
        ).squeeze(-1)

        action_probs = jnp.clip(action_probs, 1e-8, 1.0)

    # ------------------------------------------------------------
    # 8. Step environment
    # ------------------------------------------------------------
    obs_tp1, env_state_tp1, ext_reward, done, info = env.step(
        env_state,
        primitive_action,
    )

    # ------------------------------------------------------------
    # 9. RND bonus
    # ------------------------------------------------------------
    rnd_frame = rnd_frame_fn(obs_tp1)
    rnd_bonus = rnd_intrinsic_reward(rnd_state, rnd_frame)

    # ------------------------------------------------------------
    # 10. Reset option state on env termination
    # ------------------------------------------------------------
    next_active_opt = jnp.where(done, -1, next_active_opt_pre)
    next_opt_steps_left = jnp.where(done, 0, next_opt_steps_left_pre)

    # ------------------------------------------------------------
    # 11. Primitive transition
    # ------------------------------------------------------------
    transition = Transition(
        obs=obs_t,
        actions=primitive_action,
        rewards=ext_reward,
        int_rewards=jnp.zeros_like(rnd_bonus),
        dones=done,
        next_obs=obs_tp1,

        q_vals=q_ext,
        q_int_vals=q_int,

        option_idx=transition_option_idx,
        option_q_ext_vals=jnp.transpose(q_opts_ext, (1, 0, 2)),
        option_q_int_vals=jnp.transpose(q_opts_int, (1, 0, 2)),

        action_probs=(
            action_probs
            if action_probs is not None
            else jnp.ones((E,), dtype=jnp.float32)
        ),

        rnd_frame=rnd_frame,
        rnd_bonus=rnd_bonus,
        rnd_bonus_n=jnp.zeros_like(rnd_bonus),
        inv_count=jnp.zeros_like(rnd_bonus),
    )

    new_carry = (
        obs_tp1,
        env_state_tp1,
        next_active_opt,
        next_opt_steps_left,
        key,
    )
    return new_carry, (transition, info)


@partial(jax.jit, static_argnums=range(2, 9))
def update_step_common(
    runner_state,
    _unused,
    args,
    env,
    eps_scheduler,
    start_time,
    get_dual_q_values_fn,
    rnd_frame_fn,
    make_wandb_log_callback_fn,
):
    train_state, expl_state, key, opt_dur_table = runner_state

    n_base_options = args.options.n_options
    n_total_options = args.options.n_options
    n_prim_actions = env.single_action_space.n

    key, b_key = jax.random.split(key)

    step_fn_env = partial(
        env_take_steps_common,
        enc_state=train_state.enc_state,
        q_ext_state=train_state.q_ext_state,
        q_int_state=train_state.q_int_state,
        opt_enc_state=train_state.opt_enc_state,
        opt_q_ext_states=train_state.opt_q_ext_states,
        opt_q_int_states=train_state.opt_q_int_states,
        rnd_state=train_state.rnd_state,
        n_updates=train_state.n_updates,
        option_dur_table=opt_dur_table,
        env=env,
        eps_scheduler=eps_scheduler,
        mu_prob=args.options.mu_prob,
        n_tr_envs=args.env.n_envs,
        n_te_envs=args.env.n_test_envs,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        save_act_probs=False,
        n_options=n_total_options,
        int_coef=args.algo.int_coef,
        opt_int_coef=args.neo.rnd_coef,
        use_value_rescale=not args.algo.rew_scale,
        get_dual_q_values_fn=get_dual_q_values_fn,
        rnd_frame_fn=rnd_frame_fn,
    )

    (*expl_state, _), (transitions, infos) = jax.lax.scan(
        step_fn_env,
        (*expl_state, b_key),
        None,
        args.env.n_steps,
    )
    expl_state = tuple(expl_state)

    if args.env.n_test_envs > 0:
        transitions = jax.tree_util.tree_map(lambda x: x[:, :-args.env.n_test_envs], transitions)

    train_state = train_state.replace(
        timesteps=train_state.timesteps + args.env.n_steps * args.env.n_envs
    )

    burnin = train_state.timesteps < jnp.int32(args.rnd.obs_burnin)

    flat_rnd_frames = transitions.rnd_frame.reshape((-1, *transitions.rnd_frame.shape[2:]))

    rnd_state = rnd_update_obs_rms(train_state.rnd_state, flat_rnd_frames)

    raw_bonus = jax.lax.cond(
        burnin,
        lambda _: jnp.zeros_like(transitions.rnd_bonus),
        lambda _: transitions.rnd_bonus,
        operand=None,
    )

    rnd_state, rnd_bonus = jax.lax.cond(
        burnin,
        lambda r: (r, raw_bonus),
        lambda r: rnd_update_rew_rms(r, raw_bonus),
        rnd_state,
    )

    bonus_max_ema = qema_update(
        prev=rnd_state.rnd_bonus_max_ema,
        x=rnd_bonus,
        tau=getattr(args.neo, "tau_bonus_max", 1e-3),
        q=getattr(args.neo, "bonus_q", 0.99),
        monotone=True,
    )
    rnd_state = rnd_state.replace(rnd_bonus_max_ema=bonus_max_ema)

    rnd_bonus_n = rnd_bonus / (bonus_max_ema + 1e-8)
    inv_count = 1.0 / (rnd_bonus_n + 1e-6)

    inv_cap = getattr(args.neo, "inv_cap", 1e4)
    inv_used = jnp.minimum(inv_count, inv_cap)

    neo_max = qema_update(
        prev=train_state.neo_state.max_inv_ema,
        x=inv_used,
        tau=getattr(args.neo, "tau_max", 1e-3),
        q=getattr(args.neo, "q", 0.99),
        monotone=True,
    )
    neo_state = train_state.neo_state.replace(max_inv_ema=neo_max)

    zeta_rollout, neo_state = neo_scale_from_state_calibrated(
        inv_count=inv_used,
        neo_state=neo_state,
        delta=args.neo.delta,
        k=args.neo.k,
        target_mean=args.neo.target_mean,
        tau_mean=getattr(args.neo, "tau_mean", 1e-3),
    )

    transitions = transitions.replace(
        int_rewards=rnd_bonus,
        rnd_bonus=rnd_bonus,
        rnd_bonus_n=rnd_bonus_n,
        inv_count=inv_count,
    )

    train_state = train_state.replace(
        rnd_state=rnd_state,
        neo_state=neo_state,
    )

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
        transitions.q_vals,
        transitions.rewards,
        transitions.dones,
    )

    int_dones = jnp.zeros_like(transitions.dones)
    int_targets = compute_int_targets(
        next_q_int,
        transitions.q_int_vals,
        transitions.int_rewards,
        int_dones,
    )

    _main_learn_phase = partial(
        learn_phase_dual,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        int_coef=args.algo.int_coef,
    )

    opt_ext_lambda_targets, opt_int_lambda_targets, eig_rewards, stall_mean = \
        compute_neo_options_lambda_targets_shared_dual(
            train_state=train_state,
            transitions=transitions,
            gamma=args.options.gamma,
            td_lambda=args.options.td_lambda,
            use_opt_bn=args.net.batch_norm,
            n_options=n_total_options,
            use_value_rescale=not args.algo.rew_scale,
            stall_coef=args.options.stall_coef,
            rnd_coef=args.neo.rnd_coef,
        )

    _opt_learn_phase = partial(
        update_option_networks_shared_dual,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        int_coef=args.neo.rnd_coef,
    )

    _update_lap_network = partial(
        update_neo_laplacian_network,
        args=args,
        n_actions=n_prim_actions,
    )

    def _learn_epoch(carry, _):
        train_state, key = carry
        key, shard_key, lap_key = jax.random.split(key, 3)

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
            counts=zeta_rollout.T,
            novelty=transitions.rnd_bonus.T,
        )

        train_state, (eig_loss, pos_loss, ortho_loss, dual_loss) = _update_lap_network(
            lap_key,
            train_state,
            lap_batches,
        )

        flat_storage = FlatStorage(rnd_frame=flat_rnd_frames)
        rnd_state, rnd_fwd_loss = rnd_update_once_per_epoch(
            train_state.rnd_state,
            flat_storage,
            args.rnd.upd_prop,
            burnin,
        )
        train_state = train_state.replace(rnd_state=rnd_state)

        rnd_fwd_l2 = transitions.rnd_bonus.mean()

        main_batches = jax.tree_util.tree_map(
            lambda x: preprocess_transition(x, shard_key, args.algo.n_mini_batches),
            transitions,
        )
        ext_targets_mb = preprocess_transition(ext_targets, shard_key, args.algo.n_mini_batches)
        int_targets_mb = preprocess_transition(int_targets, shard_key, args.algo.n_mini_batches)

        (train_state, _), (main_losses, main_qstats) = jax.lax.scan(
            _main_learn_phase,
            (train_state, shard_key),
            (main_batches, ext_targets_mb, int_targets_mb),
        )

        opt_batches = jax.tree_util.tree_map(
            lambda x: preprocess_transition(x, shard_key, args.options.n_mini_batches),
            transitions,
        )

        opt_ext_targets = jax.vmap(
            lambda targ_o: preprocess_transition(targ_o, shard_key, args.options.n_mini_batches),
            in_axes=0,
        )(opt_ext_lambda_targets)
        opt_ext_targets = jnp.transpose(opt_ext_targets, (1, 0, 2))

        opt_int_targets = jax.vmap(
            lambda targ_o: preprocess_transition(targ_o, shard_key, args.options.n_mini_batches),
            in_axes=0,
        )(opt_int_lambda_targets)
        opt_int_targets = jnp.transpose(opt_int_targets, (1, 0, 2))

        def _opt_mb_step(ts, shards):
            mb_data, mb_opt_ext_tgt, mb_opt_int_tgt = shards
            ts, (opt_loss, opt_q_t, opt_q_raw) = _opt_learn_phase(
                ts, mb_data, mb_opt_ext_tgt, mb_opt_int_tgt
            )
            return ts, (opt_loss, opt_q_t, opt_q_raw)

        train_state, (opt_losses, opt_q_t, opt_q_raw) = jax.lax.scan(
            _opt_mb_step,
            train_state,
            (opt_batches, opt_ext_targets, opt_int_targets),
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
            rnd_fwd_loss,
            rnd_fwd_l2,
        )

    key, epoch_key = jax.random.split(key)
    (train_state, _), epoch_stats = jax.lax.scan(
        _learn_epoch,
        (train_state, epoch_key),
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
    rnd_fwd_losses = epoch_stats[9]
    rnd_fwd_l2s = epoch_stats[10]

    ext_loss = main_qstats[0].mean()
    int_loss = main_qstats[1].mean()
    q_ext_mean_t = main_qstats[2].mean()
    q_ext_mean_raw = main_qstats[3].mean()
    q_int_mean_t = main_qstats[4].mean()
    q_int_mean_raw = main_qstats[5].mean()
    q_sel_mean_t = main_qstats[6].mean()
    q_sel_mean_raw = main_qstats[7].mean()

    opt_td_loss = opt_losses.mean(axis=(0, 1))
    opt_q_mean_t = opt_q_t.mean(axis=(0, 1))
    opt_q_mean_raw = opt_q_raw.mean(axis=(0, 1))

    rnd_fwd_loss = rnd_fwd_losses.mean()
    rnd_fwd_l2 = rnd_fwd_l2s.mean()

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

    env_frame = train_state.timesteps * getattr(args.env, "n_frames", 1)

    metrics = {
        "env_step": train_state.timesteps,
        "update_steps": train_state.n_updates,
        "env_frame": env_frame,
        "grad_steps": train_state.grad_steps,

        "loss/total_td": main_losses.mean(),
        "loss/ext_td": ext_loss,
        "loss/int_td": int_loss,

        "Q_mean_transformed": q_ext_mean_t,
        "Q_mean_raw": q_ext_mean_raw,
        "Q_int_mean_transformed": q_int_mean_t,
        "Q_int_mean_raw": q_int_mean_raw,
        "Q_sel_mean_transformed": q_sel_mean_t,
        "Q_sel_mean_raw": q_sel_mean_raw,

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

    nov = transitions.rnd_bonus
    nov_n = transitions.rnd_bonus_n
    inv = transitions.inv_count
    bonus_max = nov.max()

    metrics.update({
        "Explore/Min_Novelty": nov.min(),
        "Explore/Mean_Novelty": nov.mean(),
        "Explore/Max_Novelty": nov.max(),

        "Explore/Min_Novelty_Norm": nov_n.min(),
        "Explore/Mean_Novelty_Norm": nov_n.mean(),
        "Explore/Max_Novelty_Norm": nov_n.max(),

        "Counts/MinInv": inv.min(),
        "Counts/MeanInv": inv.mean(),
        "Counts/MaxInv": inv.max(),

        "RND/int_reward_mean": transitions.int_rewards.mean(),
        "RND/int_reward_max": transitions.int_rewards.max(),
        "RND/int_reward_sum_per_env_mean": transitions.int_rewards.sum(axis=0).mean(),
        "RND/fwd_loss": rnd_fwd_loss,
        "RND/fwd_l2": rnd_fwd_l2,
        "RND/obs_std": jnp.sqrt(train_state.rnd_state.obs_rms.var).mean(),
        "RND/rew_std": jnp.sqrt(train_state.rnd_state.rew_rms.var),
        "RND/burnin": burnin.astype(jnp.float32),

        "Explore/bonus_p50": jnp.percentile(nov, 50.0),
        "Explore/bonus_p90": jnp.percentile(nov, 90.0),
        "Explore/bonus_p99": jnp.percentile(nov, 99.0),

        "NEO/max_inv_ema": train_state.neo_state.max_inv_ema,
        "NEO/zeta_min": zeta_rollout.min(),
        "NEO/zeta_mean": zeta_rollout.mean(),
        "NEO/zeta_max": zeta_rollout.max(),

        "EigenRew/max": eig_rewards.max(),
        "EigenRew/mean": eig_rewards.mean(),
        "EigenRew/min": eig_rewards.min(),

        "RND/bonus_max_rollout": bonus_max,
        "RND/bonus_max_ema": bonus_max_ema,
    })

    if args.env.n_test_envs > 0:
        test_infos = jax.tree_util.tree_map(lambda x: x[:, -args.env.n_test_envs:], infos)
        infos = jax.tree_util.tree_map(lambda x: x[:, :-args.env.n_test_envs], infos)

        metrics.update({k: v.mean() for k, v in infos.items()})
        metrics.update({f"test/{k}": v.mean() for k, v in test_infos.items()})

        metrics["Max_Return"] = infos["returned_episode_returns"].max()
        metrics["Max_Normalized_Return"] = infos["returned_episode_normalized_returns"].max()

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

    runner_state = (train_state, tuple(expl_state), key, opt_dur_table)
    return runner_state, metrics