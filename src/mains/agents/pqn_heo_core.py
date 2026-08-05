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
from src.mains.algorithms.HEO import (
    MetaRingBuffer,
    unwrap_meta_ring,
    init_meta_buffer,
    get_meta_q_seq,
    compute_meta_td_lambda_targets_no_retrace,
    compute_intra_option_meta_targets_1step,
    smdp_learn_phase,
    primitive_learn_phase,
    update_intra_option_meta_shared,
)
from src.mains.algorithms.Laplacian import update_laplacian_network


@chex.dataclass(frozen=True)
class Prim_Transition:
    obs: chex.Array
    actions: chex.Array
    next_obs: chex.Array
    dones: chex.Array
    rewards: chex.Array
    meta_action: chex.Array
    meta_q_vals: chex.Array
    option_q_vals: chex.Array
    action_probs: chex.Array


@chex.dataclass(frozen=True)
class Meta_Transition:
    obs: chex.Array
    actions: chex.Array
    rewards: chex.Array
    dones: chex.Array
    next_obs: chex.Array
    discounts: chex.Array
    behav_probs: chex.Array
    valid: chex.Array


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


@partial(jax.jit, static_argnums=range(8, 20))
def env_take_steps_common(
    carry,
    _unused,

    # main/meta controller
    enc_state,
    meta_q_state,

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
    n_base_options,
    bidir_options,
    n_options,
    gamma,
):
    (
        seg_start_obs,          # [E,C,H,W]
        seg_start_meta,         # [E]
        seg_start_behav_prob,   # [E]

        obs_t,                  # [E,C,H,W]
        env_state,

        active_meta,            # [E], -1 if no active meta-action
        opt_steps_left,         # [E], remaining option steps

        seg_ext_return,         # [E]
        seg_ext_discount,       # [E]
        seg_len,                # [E]

        meta_buf,
        key,
    ) = carry

    E = n_tr_envs + n_te_envs
    A = env.single_action_space.n
    O = n_options
    A_meta = A + O

    key, k_eps, k_mu, k_meta_opt, k_meta_prim = jax.random.split(key, 5)

    # ------------------------------------------------------------
    # 1. Main/meta Q-values
    # ------------------------------------------------------------
    q_meta = get_q_values_shared(
        obs_t,
        enc_state,
        meta_q_state,
        use_enc_bn,
        use_q_bn,
    )

    greedy_meta = jnp.argmax(q_meta, axis=-1)

    # ------------------------------------------------------------
    # 2. Decide whether a new meta-action is needed
    # ------------------------------------------------------------
    need_new_meta = (active_meta < 0) | (opt_steps_left <= 0)

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

    do_explore = need_new_meta & (u_eps < eps)
    explore_option = do_explore & (u_mu < mu_prob)
    explore_primitive = do_explore & (u_mu >= mu_prob)

    sampled_primitive = jax.random.randint(
        k_meta_prim,
        shape=(E,),
        minval=0,
        maxval=A,
    )

    # ------------------------------------------------------------
    # 3. Sample exploratory option meta-action
    # ------------------------------------------------------------
    if bidir_options:
        n_lap = n_base_options

        sampled_dim = jax.random.randint(
            k_meta_opt,
            shape=(E,),
            minval=0,
            maxval=n_lap,
        )

        pos_meta = A + sampled_dim
        neg_meta = A + sampled_dim + n_lap

        pos_q = q_meta[jnp.arange(E), pos_meta]
        neg_q = q_meta[jnp.arange(E), neg_meta]

        sampled_option_meta = jnp.where(pos_q >= neg_q, pos_meta, neg_meta)
    else:
        sampled_option = jax.random.randint(
            k_meta_opt,
            shape=(E,),
            minval=0,
            maxval=O,
        )
        sampled_option_meta = A + sampled_option

    selected_meta = jnp.where(
        explore_option,
        sampled_option_meta,
        jnp.where(
            explore_primitive,
            sampled_primitive,
            greedy_meta,
        ),
    )

    new_active_meta = jnp.where(
        need_new_meta,
        selected_meta,
        active_meta,
    )

    # ------------------------------------------------------------
    # 4. Behavior probability of selected meta-action
    # ------------------------------------------------------------
    p_explore_prim = eps * (1.0 - mu_prob)
    p_explore_opt = eps * mu_prob

    base_prim = p_explore_prim / jnp.float32(A)
    base_opt = p_explore_opt / jnp.float32(O)

    base_meta_probs = jnp.concatenate(
        [
            jnp.full((E, A), 0.0, dtype=jnp.float32) + base_prim[:, None],
            jnp.full((E, O), 0.0, dtype=jnp.float32) + base_opt[:, None],
        ],
        axis=-1,
    )

    greedy_onehot = jax.nn.one_hot(
        greedy_meta,
        A_meta,
        dtype=jnp.float32,
    )

    meta_action_probs = base_meta_probs + (1.0 - eps)[:, None] * greedy_onehot

    selected_meta_prob = jnp.take_along_axis(
        meta_action_probs,
        selected_meta[:, None],
        axis=-1,
    ).squeeze(-1)
    selected_meta_prob = jnp.clip(selected_meta_prob, 1e-8, 1.0)

    # ------------------------------------------------------------
    # 5. Initialize new SMDP segment
    # ------------------------------------------------------------
    seg_start_obs = jnp.where(
        need_new_meta[:, None, None, None],
        obs_t,
        seg_start_obs,
    )

    seg_start_meta = jnp.where(
        need_new_meta,
        selected_meta,
        seg_start_meta,
    )

    seg_start_behav_prob = jnp.where(
        need_new_meta,
        selected_meta_prob,
        seg_start_behav_prob,
    )

    seg_ext_return = jnp.where(need_new_meta, 0.0, seg_ext_return)
    seg_ext_discount = jnp.where(need_new_meta, 1.0, seg_ext_discount)
    seg_len = jnp.where(need_new_meta, 0, seg_len)

    # ------------------------------------------------------------
    # 6. Compute primitive action to execute
    # ------------------------------------------------------------
    is_primitive_meta = new_active_meta < A
    is_option_meta = new_active_meta >= A

    active_opt = jnp.clip(new_active_meta - A, 0, O - 1)

    q_opts = compute_option_qs_shared(
        obs_t,
        opt_enc_state,
        opt_q_states,
        use_enc_bn,
        use_q_bn,
    )

    q_active_option = q_opts[active_opt, jnp.arange(E), :]
    opt_action = jnp.argmax(q_active_option, axis=-1)

    primitive_action = jnp.where(
        is_primitive_meta,
        new_active_meta,
        opt_action,
    )
    primitive_action = jnp.clip(primitive_action, 0, A - 1)

    # Optional primitive behavior probability.
    primitive_behav_prob = None
    if save_act_probs:
        primitive_from_meta_prob = jnp.take_along_axis(
            meta_action_probs,
            primitive_action[:, None],
            axis=-1,
        ).squeeze(-1)

        # Option policies execute greedily in HEO.
        opt_action_prob = jnp.ones((E,), dtype=jnp.float32)

        primitive_behav_prob = jnp.where(
            is_primitive_meta,
            primitive_from_meta_prob,
            seg_start_behav_prob * opt_action_prob,
        )
        primitive_behav_prob = jnp.clip(primitive_behav_prob, 1e-8, 1.0)

    # ------------------------------------------------------------
    # 7. Step environment
    # ------------------------------------------------------------
    obs_tp1, env_state_tp1, reward, done, info = env.step(
        env_state,
        primitive_action,
    )

    # ------------------------------------------------------------
    # 8. Update segment return
    # ------------------------------------------------------------
    seg_ext_return = seg_ext_return + seg_ext_discount * reward
    seg_ext_discount = seg_ext_discount * jnp.float32(gamma)
    seg_len = seg_len + 1

    # ------------------------------------------------------------
    # 9. Update option remaining duration
    # ------------------------------------------------------------
    selected_opt = jnp.clip(selected_meta - A, 0, O - 1)
    selected_opt_dur = opt_dur_table[selected_opt]

    opt_steps_left_after = jnp.where(
        need_new_meta & is_option_meta,
        selected_opt_dur - 1,
        jnp.where(
            is_option_meta,
            opt_steps_left - 1,
            0,
        ),
    )
    opt_steps_left_after = jnp.maximum(opt_steps_left_after, 0)

    # ------------------------------------------------------------
    # 10. Determine meta-transition termination
    # ------------------------------------------------------------
    meta_done = is_primitive_meta | (opt_steps_left_after <= 0) | done

    # ------------------------------------------------------------
    # 11. Write completed meta-transition
    # ------------------------------------------------------------
    ptr = meta_buf.ptr
    env_idx = jnp.arange(E)
    write_mask = meta_done.astype(jnp.int32)

    ext_rewards_b = meta_buf.ext_rewards
    int_rewards_b = meta_buf.int_rewards
    ext_discounts_b = meta_buf.ext_discounts
    int_discounts_b = meta_buf.int_discounts
    dones_b = meta_buf.dones
    start_obs_b = meta_buf.start_obs
    term_obs_b = meta_buf.term_obs
    actions_b = meta_buf.actions
    durations_b = meta_buf.durations
    behav_probs_b = meta_buf.behav_probs

    ext_rewards_b = ext_rewards_b.at[ptr, env_idx].set(
        jnp.where(write_mask, seg_ext_return, ext_rewards_b[ptr, env_idx])
    )

    int_rewards_b = int_rewards_b.at[ptr, env_idx].set(
        jnp.where(write_mask, jnp.zeros_like(seg_ext_return), int_rewards_b[ptr, env_idx])
    )

    ext_discounts_b = ext_discounts_b.at[ptr, env_idx].set(
        jnp.where(write_mask, seg_ext_discount, ext_discounts_b[ptr, env_idx])
    )

    int_discounts_b = int_discounts_b.at[ptr, env_idx].set(
        jnp.where(write_mask, jnp.ones_like(seg_ext_discount), int_discounts_b[ptr, env_idx])
    )

    dones_b = dones_b.at[ptr, env_idx].set(
        jnp.where(write_mask, done.astype(jnp.int32), dones_b[ptr, env_idx])
    )

    start_obs_b = start_obs_b.at[ptr, env_idx].set(
        jnp.where(
            write_mask[:, None, None, None],
            seg_start_obs,
            start_obs_b[ptr, env_idx],
        )
    )

    term_obs_b = term_obs_b.at[ptr, env_idx].set(
        jnp.where(
            write_mask[:, None, None, None],
            obs_tp1,
            term_obs_b[ptr, env_idx],
        )
    )

    actions_b = actions_b.at[ptr, env_idx].set(
        jnp.where(write_mask, seg_start_meta, actions_b[ptr, env_idx])
    )

    durations_b = durations_b.at[ptr, env_idx].set(
        jnp.where(write_mask, seg_len, durations_b[ptr, env_idx])
    )

    behav_probs_b = behav_probs_b.at[ptr, env_idx].set(
        jnp.where(write_mask, seg_start_behav_prob, behav_probs_b[ptr, env_idx])
    )

    T_buf = meta_buf.ext_rewards.shape[0]
    new_ptr = (ptr + write_mask) % T_buf
    new_len = jnp.minimum(meta_buf.length + write_mask, T_buf)
    overflow = (meta_buf.length == T_buf) & (write_mask == 1)
    new_head = jnp.where(overflow, (meta_buf.head + 1) % T_buf, meta_buf.head)

    meta_buf = MetaRingBuffer(
        ptr=new_ptr,
        head=new_head,
        length=new_len,
        start_obs=start_obs_b,
        term_obs=term_obs_b,
        actions=actions_b,
        ext_rewards=ext_rewards_b,
        int_rewards=int_rewards_b,
        ext_discounts=ext_discounts_b,
        int_discounts=int_discounts_b,
        dones=dones_b,
        durations=durations_b,
        behav_probs=behav_probs_b,
    )

    # ------------------------------------------------------------
    # 12. Reset active meta-state after termination
    # ------------------------------------------------------------
    next_active_meta = jnp.where(meta_done | done, -1, new_active_meta)
    next_opt_steps_left = jnp.where(meta_done | done, 0, opt_steps_left_after)

    seg_ext_return = jnp.where(meta_done, 0.0, seg_ext_return)
    seg_ext_discount = jnp.where(meta_done, 1.0, seg_ext_discount)
    seg_len = jnp.where(meta_done, 0, seg_len)

    # ------------------------------------------------------------
    # 13. Primitive transition
    # ------------------------------------------------------------
    prim_trans = Prim_Transition(
        obs=obs_t,
        actions=primitive_action,
        next_obs=obs_tp1,
        dones=done,
        rewards=reward,

        meta_action=new_active_meta,
        meta_q_vals=q_meta,
        option_q_vals=jnp.transpose(q_opts, (1, 0, 2)),

        action_probs=(
            primitive_behav_prob
            if primitive_behav_prob is not None
            else jnp.ones((E,), dtype=jnp.float32)
        ),
    )

    new_carry = (
        seg_start_obs,
        seg_start_meta,
        seg_start_behav_prob,

        obs_tp1,
        env_state_tp1,

        next_active_meta,
        next_opt_steps_left,

        seg_ext_return,
        seg_ext_discount,
        seg_len,

        meta_buf,
        key,
    )

    return new_carry, (prim_trans, info)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def update_step_common(runner_state, _unused, args, env, eps_scheduler, start_time):
    train_state, expl_state, key, opt_dur_table = runner_state

    n_base_options = args.options.n_options
    n_total_options = 2 * n_base_options if args.options.bidir_options else n_base_options
    n_prim_actions = env.single_action_space.n
    n_meta_actions = n_prim_actions + n_total_options
    n_train_envs = args.env.n_envs

    key, b_key = jax.random.split(key)

    step_fn_env = partial(
        env_take_steps_common,
        enc_state=train_state.enc_state,
        meta_q_state=train_state.q_state,
        opt_enc_state=train_state.opt_enc_state,
        opt_q_states=train_state.opt_q_states,
        n_updates=train_state.n_updates,
        opt_dur_table=opt_dur_table,
        env=env,
        eps_scheduler=eps_scheduler,
        mu_prob=args.options.mu_prob,
        n_tr_envs=args.env.n_envs,
        n_te_envs=args.env.n_test_envs,

        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,

        save_act_probs=False,
        n_base_options=n_base_options,
        bidir_options=args.options.bidir_options,

        n_options=n_total_options,
        gamma=args.env.gamma,
    )

    (*expl_state, _), (prim_transitions_all, infos) = jax.lax.scan(
        step_fn_env,
        (*expl_state, b_key),
        None,
        args.env.n_steps,
    )
    expl_state = tuple(expl_state)

    meta_buf_all = expl_state[10]

    if args.env.n_test_envs > 0:
        prim_transitions = jax.tree_util.tree_map(lambda x: x[:, :n_train_envs], prim_transitions_all)
        meta_buf = MetaRingBuffer(
            ptr=meta_buf_all.ptr[:n_train_envs],
            head=meta_buf_all.head[:n_train_envs],
            length=meta_buf_all.length[:n_train_envs],
            start_obs=meta_buf_all.start_obs[:, :n_train_envs],
            term_obs=meta_buf_all.term_obs[:, :n_train_envs],
            actions=meta_buf_all.actions[:, :n_train_envs],
            ext_rewards=meta_buf_all.ext_rewards[:, :n_train_envs],
            int_rewards=meta_buf_all.int_rewards[:, :n_train_envs],
            ext_discounts=meta_buf_all.ext_discounts[:, :n_train_envs],
            int_discounts=meta_buf_all.int_discounts[:, :n_train_envs],
            dones=meta_buf_all.dones[:, :n_train_envs],
            durations=meta_buf_all.durations[:, :n_train_envs],
            behav_probs=meta_buf_all.behav_probs[:, :n_train_envs],
        )
    else:
        prim_transitions = prim_transitions_all
        meta_buf = meta_buf_all

    train_state = train_state.replace(
        timesteps=train_state.timesteps + args.env.n_steps * args.env.n_envs
    )

    next_q_all = get_q_values_shared(
        prim_transitions.next_obs[-1],
        train_state.enc_state,
        train_state.q_state,
        args.net.batch_norm,
        args.net.batch_norm,
    )

    next_prim_q = jnp.max(next_q_all[:, :n_prim_actions], axis=-1)

    if args.algo.rew_scale:
        prim_lambda_targets = compute_td_lambda_targets(
            next_q=next_prim_q,
            q_vals=prim_transitions.meta_q_vals[:, :, :n_prim_actions],
            rewards=prim_transitions.rewards,
            dones=prim_transitions.dones.astype(jnp.float32),
            gamma=args.env.gamma,
            td_lambda=args.algo.td_lambda,
        )
    else:
        prim_lambda_targets = compute_transformed_td_lambda_targets(
            next_q_t=next_prim_q,
            q_vals_t=prim_transitions.meta_q_vals[:, :, :n_prim_actions],
            rewards=prim_transitions.rewards,
            dones=prim_transitions.dones.astype(jnp.float32),
            gamma=args.env.gamma,
            td_lambda=args.algo.td_lambda,
        )

    opt_lambda_targets = compute_options_lambda_targets_shared(
        train_state=train_state,
        transitions=prim_transitions,
        gamma=args.options.gamma,
        td_lambda=args.options.td_lambda,
        use_opt_bn=args.net.batch_norm,
        use_lap_bn=args.laplacian.batch_norm,
        n_base_options=n_base_options,
        bidir_options=args.options.bidir_options,
        use_value_rescale=not args.algo.rew_scale,
    )

    meta_seq = unwrap_meta_ring(meta_buf, args.options.m_steps)

    meta_q_seq = get_meta_q_seq(
        meta_seq["start_obs"],
        train_state.enc_state,
        train_state.q_state,
        args.net.batch_norm,
        args.net.batch_norm,
    )

    seg_start_obs_live = expl_state[0][:n_train_envs]
    obs_t_live = expl_state[3][:n_train_envs]
    active_meta_live = expl_state[5][:n_train_envs]
    opt_steps_left_live = expl_state[6][:n_train_envs]

    need_new_meta_live = (active_meta_live < 0) | (opt_steps_left_live <= 0)

    meta_boot_obs = jnp.where(
        need_new_meta_live[:, None, None, None],
        obs_t_live,
        seg_start_obs_live,
    )

    meta_boot_q = get_q_values_shared(
        meta_boot_obs,
        train_state.enc_state,
        train_state.q_state,
        args.net.batch_norm,
        args.net.batch_norm,
    )
    # next_meta_q = jnp.max(meta_boot_q, axis=-1)

    meta_lambda_targets = compute_meta_td_lambda_targets_no_retrace(
        next_q_all=meta_boot_q,
        rewards=meta_seq["ext_rewards"],
        dones=meta_seq["dones"].astype(jnp.float32),
        discounts=meta_seq["ext_discounts"],
        q_vals=meta_q_seq,
        td_lambda=args.options.meta_td_lambda,
        use_value_rescale=not args.algo.rew_scale,
    )

    ioq_lambda_targets = compute_intra_option_meta_targets_1step(
        transitions=prim_transitions,
        next_meta_q_all=next_q_all,
        opt_dur_table=opt_dur_table,
        gamma=args.env.gamma,
        n_prim_actions=n_prim_actions,
        use_value_rescale=not args.algo.rew_scale,
    )

    _smdp_learn_phase = partial(
        smdp_learn_phase,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        n_meta_actions=n_meta_actions,
        n_prim_actions=n_prim_actions,
    )

    _prim_learn_phase = partial(
        primitive_learn_phase,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        n_primitives=n_prim_actions,
    )

    _ioq_learn_phase = partial(
        update_intra_option_meta_shared,
        n_prim_actions=n_prim_actions,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        deterministic_match=True,
    )

    _opt_policy_learn_phase = partial(
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
        key, shard_key, lap_key = jax.random.split(key, 3)

        def preprocess_pytree(x, key, n_shards):
            x = x.reshape(-1, *x.shape[2:])
            x = jax.random.permutation(key, x)
            x = x.reshape(n_shards, -1, *x.shape[1:])
            return x

        def preprocess_array(x, key, n_shards):
            x = x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)
            x = jax.random.permutation(key, x)
            x = x.reshape(n_shards, -1, *x.shape[1:])
            return x

        lap_batch = LapBatch(
            obs=prim_transitions.obs.transpose(1, 0, 2, 3, 4),
            dones=prim_transitions.dones.T,
            actions=prim_transitions.actions.T,
            action_probs=None,
        )

        train_state, (eig_loss, pos_loss, ortho_loss, dual_loss) = _update_lap_network(
            lap_key, train_state, lap_batch
        )

        meta_transitions = Meta_Transition(
            obs=meta_seq["start_obs"],
            actions=meta_seq["actions"],
            rewards=meta_seq["ext_rewards"],
            dones=meta_seq["dones"],
            next_obs=meta_seq["term_obs"],
            discounts=meta_seq["ext_discounts"],
            behav_probs=meta_seq["behav_probs"],
            valid=meta_seq["valid"],
        )

        meta_batches = jax.tree_util.tree_map(
            lambda x: preprocess_pytree(x, shard_key, args.algo.n_mini_batches),
            meta_transitions,
        )
        meta_targets_mb = preprocess_array(
            meta_lambda_targets, shard_key, args.algo.n_mini_batches
        )

        (train_state, _), (
            meta_losses, meta_qstats,
            meta_option_taken_count_mb, meta_primitive_taken_count_mb,
            meta_option_greedy_count_mb, meta_primitive_greedy_count_mb,
            meta_valid_count_mb,
        ) = jax.lax.scan(
            _smdp_learn_phase,
            (train_state, shard_key),
            (meta_batches, meta_targets_mb),
        )

        prim_batches = jax.tree_util.tree_map(
            lambda x: preprocess_pytree(x, shard_key, args.algo.n_mini_batches),
            prim_transitions,
        )
        prim_targets_mb = preprocess_array(
            prim_lambda_targets, shard_key, args.algo.n_mini_batches
        )

        (train_state, _), (prim_losses, prim_qstats) = jax.lax.scan(
            _prim_learn_phase,
            (train_state, shard_key),
            (prim_batches, prim_targets_mb),
        )

        opt_batches = jax.tree_util.tree_map(
            lambda x: preprocess_pytree(x, shard_key, args.options.n_mini_batches),
            prim_transitions,
        )

        opt_targets = jax.vmap(
            lambda targ_o: preprocess_array(targ_o, shard_key, args.options.n_mini_batches),
            in_axes=0,
        )(opt_lambda_targets)
        opt_targets = jnp.transpose(opt_targets, (1, 0, 2))

        def _opt_mb_step(ts, shard):
            mb_data, mb_opt_tgt = shard
            ts, (opt_loss, opt_q_t, opt_q_raw) = _opt_policy_learn_phase(ts, mb_data, mb_opt_tgt)
            return ts, (opt_loss, opt_q_t, opt_q_raw)

        train_state, (optpol_losses, optpol_q_t, optpol_q_raw) = jax.lax.scan(
            _opt_mb_step,
            train_state,
            (opt_batches, opt_targets),
        )

        # ioq_batches = jax.tree_util.tree_map(
        #     lambda x: preprocess_pytree(x, shard_key, args.options.n_mini_batches),
        #     prim_transitions,
        # )
        # 
        # ioq_targets = jax.vmap(
        #     lambda targ_o: preprocess_array(targ_o, shard_key, args.options.n_mini_batches),
        #     in_axes=0,
        # )(ioq_lambda_targets)
        # ioq_targets = jnp.transpose(ioq_targets, (1, 0, 2))
        #
        # def _ioq_mb_step(ts, shard):
        #     mb_data, mb_opt_tgt = shard
        #     ts, (opt_loss, opt_q_t, opt_q_raw, elig_any, elig_count) = _ioq_learn_phase(ts, mb_data, mb_opt_tgt)
        #     return ts, (opt_loss, opt_q_t, opt_q_raw, elig_any, elig_count)
        # #
        # # train_state, (ioq_losses, ioq_q_t, ioq_q_raw, ioq_elig_any, ioq_elig_count) = jax.lax.scan(
        # #     _ioq_mb_step,
        # #     train_state,
        # #     (ioq_batches, ioq_targets),
        # # )
        ioq_losses = ioq_q_t = ioq_q_raw = ioq_elig_any = ioq_elig_count = jnp.zeros_like(optpol_losses)

        return (train_state, key), (
            meta_losses, meta_qstats,
            meta_option_taken_count_mb, meta_primitive_taken_count_mb,
            meta_option_greedy_count_mb, meta_primitive_greedy_count_mb,
            meta_valid_count_mb,
            prim_losses, prim_qstats,
            optpol_losses, optpol_q_t, optpol_q_raw,
            ioq_losses, ioq_q_t, ioq_q_raw, ioq_elig_any, ioq_elig_count,
            eig_loss, pos_loss, ortho_loss, dual_loss,
        )

    key, epoch_key = jax.random.split(key)
    (train_state, _), epoch_stats = jax.lax.scan(
        _learn_epoch,
        (train_state, epoch_key),
        None,
        args.algo.n_epochs,
    )

    grad_steps_inc = args.algo.n_epochs * (
        args.algo.n_mini_batches
        # + args.options.n_mini_batches
        # + args.options.n_mini_batches
    )
    train_state = train_state.replace(
        n_updates=train_state.n_updates + 1,
        grad_steps=train_state.grad_steps + grad_steps_inc,
    )

    meta_losses = epoch_stats[0]
    meta_qstats = epoch_stats[1]
    meta_option_taken_count_mb = epoch_stats[2]
    meta_primitive_taken_count_mb = epoch_stats[3]
    meta_option_greedy_count_mb = epoch_stats[4]
    meta_primitive_greedy_count_mb = epoch_stats[5]
    meta_valid_count_mb = epoch_stats[6]

    prim_losses = epoch_stats[7]
    prim_qstats = epoch_stats[8]

    optpol_losses = epoch_stats[9]
    optpol_q_t = epoch_stats[10]
    optpol_q_raw = epoch_stats[11]

    ioq_losses = epoch_stats[12]
    ioq_q_t = epoch_stats[13]
    ioq_q_raw = epoch_stats[14]
    ioq_elig_any = epoch_stats[15]
    ioq_elig_count = epoch_stats[16]

    eig_loss = epoch_stats[17]
    pos_loss = epoch_stats[18]
    ortho_loss = epoch_stats[19]
    dual_loss = epoch_stats[20]

    meta_q_mean_t = meta_qstats[0].mean()
    meta_q_mean_raw = meta_qstats[1].mean()

    meta_option_taken_count = meta_option_taken_count_mb.sum()
    meta_primitive_taken_count = meta_primitive_taken_count_mb.sum()
    meta_option_greedy_count = meta_option_greedy_count_mb.sum()
    meta_primitive_greedy_count = meta_primitive_greedy_count_mb.sum()
    meta_valid_count = meta_valid_count_mb.sum()

    prim_q_mean_t = prim_qstats[0].mean()
    prim_q_mean_raw = prim_qstats[1].mean()

    optpol_td_loss = optpol_losses.mean(axis=(0, 1))
    optpol_q_mean_t = optpol_q_t.mean(axis=(0, 1))
    optpol_q_mean_raw = optpol_q_raw.mean(axis=(0, 1))

    ioq_td_loss = ioq_losses.mean(axis=(0, 1))
    ioq_q_mean_t = ioq_q_t.mean(axis=(0, 1))
    ioq_q_mean_raw = ioq_q_raw.mean(axis=(0, 1))
    ioq_elig_any = ioq_elig_any.mean()
    ioq_elig_count = ioq_elig_count.mean(axis=(0, 1))

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

        "Loss/SMDP": meta_losses.mean(),
        "Loss/Primitive": prim_losses.mean(),
        "Loss/OptionPolicy": optpol_losses.mean(),
        "Loss/IntraOption": ioq_losses.mean(),

        "Q/SMDP_mean_t": meta_q_mean_t,
        "Q/SMDP_mean_raw": meta_q_mean_raw,
        "Q/Primitive_mean_t": prim_q_mean_t,
        "Q/Primitive_mean_raw": prim_q_mean_raw,

        "Eigen/Total_loss": eig_loss.mean(),
        "Eigen/Pos_loss": pos_loss.mean(),
        "Eigen/Ortho_loss": ortho_loss.mean(),
        "Eigen/Dual_loss": dual_loss.mean(),
        "Eigen/Barrier": train_state.lap_state.barrier_coeffs[0, 0],
        "Eigen/Max_EigenValue": jnp.max(train_state.lap_state.eigen_values),

        "IntraOption/elig_any": ioq_elig_any,

        "Options/dur_min": opt_dur_table.min(),
        "Options/dur_max": opt_dur_table.max(),
        "Options/dur_mean": opt_dur_table.astype(jnp.float32).mean(),

        "Meta/option_taken_frac": meta_option_taken_count / (meta_valid_count + 1e-8),
        "Meta/primitive_taken_frac": meta_primitive_taken_count / (meta_valid_count + 1e-8),
        "Meta/option_greedy_frac": meta_option_greedy_count / (meta_valid_count + 1e-8),
        "Meta/primitive_greedy_frac": meta_primitive_greedy_count / (meta_valid_count + 1e-8),
    }

    for i in range(n_total_options):
        metrics[f"Options/TD_loss_{i}"] = optpol_td_loss[i]
        metrics[f"Options/Q_mean_t_{i}"] = optpol_q_mean_t[i]
        metrics[f"Options/Q_mean_raw_{i}"] = optpol_q_mean_raw[i]

        metrics[f"IntraOption/TD_loss_{i}"] = ioq_td_loss[i]
        metrics[f"IntraOption/Q_mean_t_{i}"] = ioq_q_mean_t[i]
        metrics[f"IntraOption/Q_mean_raw_{i}"] = ioq_q_mean_raw[i]
        metrics[f"IntraOption/elig_count_{i}"] = ioq_elig_count[i]

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