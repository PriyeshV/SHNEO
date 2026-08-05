import time
from functools import partial
from typing import Any, Union, List, NamedTuple, Callable

import chex
import flax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from src.utils.jax_utils import qema_update
from src.mains.algorithms.RND import (
    RNDState,
    rnd_intrinsic_reward,
    rnd_update_obs_rms,
    rnd_update_rew_rms,
    rnd_update_once_per_epoch,
)
from src.mains.algorithms.PQN import (
    value_rescale, inv_value_rescale, compute_td_lambda_targets, compute_transformed_td_lambda_targets)
from src.mains.algorithms.EO import (
    eigenvalues_to_option_durations_skip_first,
    update_option_networks_shared_dual,
)
from src.mains.algorithms.HEO import (
    MetaRingBuffer,
    unwrap_meta_ring,
    get_meta_dual_q_seq,
    compute_meta_td_lambda_targets_dual_no_retrace,
    smdp_learn_phase_dual,
    primitive_learn_phase_dual,
)
from src.mains.algorithms.Laplacian import update_neo_laplacian_network
from src.mains.algorithms.NEO import (
    NEOScaleState,
    neo_scale_from_state_calibrated,
    compute_neo_options_lambda_targets_shared_dual,
)


# ---------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------
@chex.dataclass(frozen=True)
class Prim_Transition:
    obs: chex.Array
    actions: chex.Array
    next_obs: chex.Array
    dones: chex.Array

    rewards: chex.Array
    int_rewards: chex.Array

    meta_action: chex.Array
    meta_q_ext_vals: chex.Array
    meta_q_int_vals: chex.Array

    option_q_ext_vals: chex.Array   # [E, O, A]
    option_q_int_vals: chex.Array   # [E, O, A]

    action_probs: chex.Array

    rnd_frame: chex.Array
    rnd_bonus: chex.Array
    rnd_bonus_n: chex.Array
    inv_count: chex.Array


@chex.dataclass(frozen=True)
class Meta_Transition:
    obs: chex.Array
    actions: chex.Array
    ext_rewards: chex.Array
    int_rewards: chex.Array
    dones: chex.Array
    next_obs: chex.Array
    ext_discounts: chex.Array
    int_discounts: chex.Array
    behav_probs: chex.Array
    valid: chex.Array


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


# ---------------------------------------------------------------------
# Shared option helpers
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnums=(2, 3))
def select_option_action(
    q_opt_sel,
    key,
    option_policy_greedy: bool = True,
    option_policy_temp: float = 1.0,
):
    if option_policy_greedy:
        act = jnp.argmax(q_opt_sel, axis=-1)
        dist = jax.nn.one_hot(act, q_opt_sel.shape[-1], dtype=jnp.float32)
        prob = jnp.ones((q_opt_sel.shape[0],), dtype=jnp.float32)
    else:
        temp = jnp.maximum(jnp.float32(option_policy_temp), 1e-6)
        logits = q_opt_sel / temp
        dist = jax.nn.softmax(logits, axis=-1)
        act = jax.random.categorical(key, logits, axis=-1)
        prob = jnp.take_along_axis(dist, act[:, None], axis=-1).squeeze(-1)
        prob = jnp.clip(prob, 1e-8, 1.0)
    return act, prob, dist


@partial(jax.jit, static_argnums=(3,))
def select_option_action_mixed(
    q_opt_sel,
    key,
    stochastic_mask,
    option_policy_temp: float = 1.0,
):
    temp = jnp.maximum(jnp.float32(option_policy_temp), 1e-6)
    logits = q_opt_sel / temp
    dist = jax.nn.softmax(logits, axis=-1)

    greedy_act = jnp.argmax(q_opt_sel, axis=-1)
    samp_act = jax.random.categorical(key, logits, axis=-1)
    act = jnp.where(stochastic_mask, samp_act, greedy_act)

    prob = jnp.take_along_axis(dist, act[:, None], axis=-1).squeeze(-1)
    prob = jnp.where(
        stochastic_mask,
        jnp.clip(prob, 1e-8, 1.0),
        jnp.ones_like(prob, dtype=jnp.float32),
    )
    return act, prob, dist


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
    obs = jnp.transpose(obs, (0, 2, 3, 1))

    enc_vars = {"params": opt_enc_state.params}
    if use_enc_bn:
        enc_vars["batch_stats"] = opt_enc_state.batch_stats
    emb = opt_enc_state.apply_fn(enc_vars, obs, train=False)

    qext_params = jax.tree_util.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in opt_q_ext_states])
    qint_params = jax.tree_util.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in opt_q_int_states])

    qext_bs = (
        jax.tree_util.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in opt_q_ext_states])
        if use_q_bn else None
    )
    qint_bs = (
        jax.tree_util.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in opt_q_int_states])
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


# ---------------------------------------------------------------------
# Shared rollout core
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnums=range(11, 29))
def env_take_steps_common(
    carry,
    _unused,

    # main/meta controller
    enc_state,
    q_ext_state,
    q_int_state,

    # option controllers
    opt_enc_state,
    opt_q_ext_states,
    opt_q_int_states,

    # optional RND
    rnd_state,

    # schedules / durations
    n_updates,
    opt_dur_table,

    # env/config
    env,
    eps_scheduler,
    mu_scheduler,
    gamma_ext,
    gamma_int,
    n_tr_envs,
    n_te_envs,
    use_enc_bn,
    use_q_bn,
    save_act_probs,
    n_options,
    int_coef,
    opt_int_coef,
    use_value_rescale,
    option_policy_greedy,
    option_policy_temp,
    get_dual_q_values_fn,
    rnd_frame_fn,
):
    (
        seg_start_obs,          # [E,C,H,W]
        seg_start_meta,         # [E]
        seg_start_behav_prob,   # [E]

        obs_t,                  # [E,C,H,W]
        env_state,

        active_meta,            # [E], -1 if no active meta-action
        opt_steps_left,         # [E], remaining option steps

        option_stoch,           # [E] bool

        seg_ext_return,         # [E]
        seg_int_return,         # [E]
        seg_ext_discount,       # [E]
        seg_int_discount,       # [E]
        seg_len,                # [E]

        meta_buf,
        key,
    ) = carry

    E = n_tr_envs + n_te_envs
    A = env.single_action_space.n
    O = n_options
    A_meta = A + O

    key, k_eps, k_mu, k_meta_opt, k_meta_prim, k_opt_act = jax.random.split(key, 6)

    # ------------------------------------------------------------
    # 1. Main/meta Q-values
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

    greedy_meta = jnp.argmax(q_sel, axis=-1)

    # ------------------------------------------------------------
    # 2. Decide whether a new meta-action is needed
    # ------------------------------------------------------------
    need_new_meta = (active_meta < 0) | (opt_steps_left <= 0)

    eps_val = eps_scheduler(n_updates)
    mu_val = mu_scheduler(n_updates)

    eps = jnp.concatenate(
        [
            jnp.full((n_tr_envs,), eps_val, dtype=jnp.float32),
            jnp.zeros((n_te_envs,), dtype=jnp.float32),
        ],
        axis=0,
    )

    u_eps = jax.random.uniform(k_eps, (E,))
    u_mu = jax.random.uniform(k_mu, (E,))

    do_explore = need_new_meta & (u_eps < eps)
    explore_option = do_explore & (u_mu < mu_val)
    explore_primitive = do_explore & (u_mu >= mu_val)

    sampled_option = jax.random.randint(k_meta_opt, (E,), 0, O)
    sampled_option_meta = A + sampled_option
    sampled_primitive = jax.random.randint(k_meta_prim, (E,), 0, A)

    selected_meta = jnp.where(
        explore_option,
        sampled_option_meta,
        jnp.where(
            explore_primitive,
            sampled_primitive,
            greedy_meta,
        ),
    )

    new_active_meta = jnp.where(need_new_meta, selected_meta, active_meta)

    # ------------------------------------------------------------
    # 3. Behavior probability of selected meta-action
    # ------------------------------------------------------------
    p_explore_prim = eps * (1.0 - mu_val)
    p_explore_opt = eps * mu_val

    base_prim = p_explore_prim / jnp.float32(A)
    base_opt = p_explore_opt / jnp.float32(O)

    base_meta_probs = jnp.concatenate(
        [
            jnp.full((E, A), 0.0, dtype=jnp.float32) + base_prim[:, None],
            jnp.full((E, O), 0.0, dtype=jnp.float32) + base_opt[:, None],
        ],
        axis=-1,
    )

    # For off-policy diagnostics / retrace, this is closer to the actual
    # mixed epsilon-greedy behavior than the old "soft * chosen_onehot" form.
    greedy_onehot = jax.nn.one_hot(greedy_meta, A_meta, dtype=jnp.float32)
    target_meta_probs = (1.0 - eps)[:, None] * greedy_onehot
    meta_action_probs = base_meta_probs + target_meta_probs

    selected_meta_prob = jnp.take_along_axis(
        meta_action_probs,
        selected_meta[:, None],
        axis=-1,
    ).squeeze(-1)
    selected_meta_prob = jnp.clip(selected_meta_prob, 1e-8, 1.0)

    # ------------------------------------------------------------
    # 4. Initialize new segment when a new meta-action is chosen
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
    seg_int_return = jnp.where(need_new_meta, 0.0, seg_int_return)
    seg_ext_discount = jnp.where(need_new_meta, 1.0, seg_ext_discount)
    seg_int_discount = jnp.where(need_new_meta, 1.0, seg_int_discount)
    seg_len = jnp.where(need_new_meta, 0, seg_len)

    # ------------------------------------------------------------
    # 5. Compute primitive action to execute
    # ------------------------------------------------------------
    is_primitive_meta = new_active_meta < A
    is_option_meta = new_active_meta >= A

    active_opt = jnp.clip(new_active_meta - A, 0, O - 1)

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

    q_active_option = q_opts_sel[active_opt, jnp.arange(E), :]

    new_option_stoch = explore_option
    option_stoch = jnp.where(need_new_meta, new_option_stoch, option_stoch)

    if option_policy_greedy:
        opt_action = jnp.argmax(q_active_option, axis=-1)
        opt_action_prob = jnp.ones((E,), dtype=jnp.float32)
    else:
        temp = jnp.maximum(jnp.float32(option_policy_temp), 1e-6)
        opt_logits = q_active_option / temp
        opt_pi = jax.nn.softmax(opt_logits, axis=-1)

        sampled_opt_action = jax.random.categorical(k_opt_act, opt_logits, axis=-1)
        greedy_opt_action = jnp.argmax(q_active_option, axis=-1)

        opt_action = jnp.where(
            is_option_meta & option_stoch,
            sampled_opt_action,
            greedy_opt_action,
        )

        sampled_prob = jnp.take_along_axis(
            opt_pi,
            sampled_opt_action[:, None],
            axis=-1,
        ).squeeze(-1)

        opt_action_prob = jnp.where(
            is_option_meta & option_stoch,
            jnp.clip(sampled_prob, 1e-8, 1.0),
            1.0,
        )

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

        primitive_behav_prob = jnp.where(
            is_primitive_meta,
            primitive_from_meta_prob,
            seg_start_behav_prob * opt_action_prob,
        )

    # ------------------------------------------------------------
    # 6. Step environment
    # ------------------------------------------------------------
    obs_tp1, env_state_tp1, ext_reward, done, info = env.step(
        env_state,
        primitive_action,
    )

    rnd_frame = rnd_frame_fn(obs_tp1)
    raw_rnd_bonus = rnd_intrinsic_reward(rnd_state, rnd_frame)

    # ------------------------------------------------------------
    # 7. Update segment accumulators
    # ------------------------------------------------------------
    seg_ext_return = seg_ext_return + seg_ext_discount * ext_reward
    seg_int_return = seg_int_return + seg_int_discount * raw_rnd_bonus

    seg_ext_discount = seg_ext_discount * gamma_ext
    seg_int_discount = seg_int_discount * gamma_int

    seg_len = seg_len + 1

    # ------------------------------------------------------------
    # 8. Update option remaining duration
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
    # 9. Determine meta-transition termination
    # ------------------------------------------------------------
    meta_done = is_primitive_meta | (opt_steps_left_after <= 0) | done

    # ------------------------------------------------------------
    # 10. Write completed meta-transition
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
        jnp.where(write_mask, seg_int_return, int_rewards_b[ptr, env_idx])
    )

    ext_discounts_b = ext_discounts_b.at[ptr, env_idx].set(
        jnp.where(write_mask, seg_ext_discount, ext_discounts_b[ptr, env_idx])
    )
    int_discounts_b = int_discounts_b.at[ptr, env_idx].set(
        jnp.where(write_mask, seg_int_discount, int_discounts_b[ptr, env_idx])
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
    # 11. Reset active meta-state after termination
    # ------------------------------------------------------------
    next_active_meta = jnp.where(meta_done | done, -1, new_active_meta)
    next_opt_steps_left = jnp.where(meta_done | done, 0, opt_steps_left_after)
    next_option_stoch = jnp.where(meta_done | done, False, option_stoch)

    seg_ext_return = jnp.where(meta_done, 0.0, seg_ext_return)
    seg_int_return = jnp.where(meta_done, 0.0, seg_int_return)
    seg_ext_discount = jnp.where(meta_done, 1.0, seg_ext_discount)
    seg_int_discount = jnp.where(meta_done, 1.0, seg_int_discount)
    seg_len = jnp.where(meta_done, 0, seg_len)

    # ------------------------------------------------------------
    # 12. Primitive transition
    # ------------------------------------------------------------
    prim_trans = Prim_Transition(
        obs=obs_t,
        actions=primitive_action,
        next_obs=obs_tp1,
        dones=done,

        rewards=ext_reward,
        int_rewards=jnp.zeros_like(raw_rnd_bonus),

        meta_action=new_active_meta,
        meta_q_ext_vals=q_ext,
        meta_q_int_vals=q_int,

        option_q_ext_vals=jnp.transpose(q_opts_ext, (1, 0, 2)),
        option_q_int_vals=jnp.transpose(q_opts_int, (1, 0, 2)),

        action_probs=(
            primitive_behav_prob
            if primitive_behav_prob is not None
            else jnp.ones((E,), dtype=jnp.float32)
        ),

        rnd_frame=rnd_frame,
        rnd_bonus=raw_rnd_bonus,
        rnd_bonus_n=jnp.zeros_like(raw_rnd_bonus),
        inv_count=jnp.zeros_like(raw_rnd_bonus),
    )

    new_carry = (
        seg_start_obs,
        seg_start_meta,
        seg_start_behav_prob,

        obs_tp1,
        env_state_tp1,

        next_active_meta,
        next_opt_steps_left,

        next_option_stoch,

        seg_ext_return,
        seg_int_return,
        seg_ext_discount,
        seg_int_discount,
        seg_len,

        meta_buf,
        key,
    )

    return new_carry, (prim_trans, info)


# ---------------------------------------------------------------------
# Shared update core
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnums=range(2, 9))
def update_step_common(
    runner_state, _unused, args, env, eps_scheduler, mu_scheduler, start_time,
    get_dual_q_values_fn: Callable,
    rnd_frame_fn: Callable,
):
    train_state, expl_state, key, opt_dur_table = runner_state

    n_total_options = n_base_options = args.options.n_options
    n_prim_actions = env.single_action_space.n
    n_meta_actions = n_prim_actions + n_total_options
    n_train_envs = args.env.n_envs

    key, b_key = jax.random.split(key)

    # ------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------
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
        opt_dur_table=opt_dur_table,
        env=env,
        eps_scheduler=eps_scheduler,
        mu_scheduler=mu_scheduler,
        gamma_ext=args.env.gamma,
        gamma_int=args.rnd.int_gamma,
        n_tr_envs=args.env.n_envs,
        n_te_envs=args.env.n_test_envs,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        save_act_probs=False,
        n_options=n_total_options,
        int_coef=args.algo.int_coef,
        opt_int_coef=args.neo.rnd_coef,
        use_value_rescale=not args.algo.rew_scale,
        option_policy_greedy=args.options.policy_greedy,
        option_policy_temp=args.options.policy_temp,
        get_dual_q_values_fn=get_dual_q_values_fn,
        rnd_frame_fn=rnd_frame_fn,
    )

    carryT, (prim_transitions_all, infos) = jax.lax.scan(
        step_fn_env,
        (*expl_state, b_key),
        None,
        args.env.n_steps,
    )

    (*expl_state, _) = carryT
    expl_state = tuple(expl_state)

    # New unified carry layout:
    # 0 seg_start_obs
    # 1 seg_start_meta
    # 2 seg_start_behav_prob
    # 3 obs_t
    # 4 env_state
    # 5 active_meta
    # 6 opt_steps_left
    # 7 option_stoch
    # 8 seg_ext_return
    # 9 seg_int_return
    # 10 seg_ext_discount
    # 11 seg_int_discount
    # 12 seg_len
    # 13 meta_buf
    meta_buf_all = expl_state[13]

    # ------------------------------------------------------------
    # Slice train envs only
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # RND + NEO
    # ------------------------------------------------------------
    burnin = train_state.timesteps < jnp.int32(args.rnd.obs_burnin)

    flat_rnd_frames = prim_transitions.rnd_frame.reshape((-1, *prim_transitions.rnd_frame.shape[2:]))

    rnd_state = rnd_update_obs_rms(train_state.rnd_state, flat_rnd_frames)

    raw_bonus = jax.lax.cond(
        burnin,
        lambda _: jnp.zeros_like(prim_transitions.rnd_bonus),
        lambda _: prim_transitions.rnd_bonus,
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

    prim_transitions = prim_transitions.replace(
        int_rewards=rnd_bonus,
        rnd_bonus=rnd_bonus,
        rnd_bonus_n=rnd_bonus_n,
        inv_count=inv_count,
    )
    train_state = train_state.replace(rnd_state=rnd_state, neo_state=neo_state)

    # ------------------------------------------------------------
    # Primitive targets
    # ------------------------------------------------------------
    next_q_ext_all, next_q_int_all = get_dual_q_values_fn(
        prim_transitions.next_obs[-1],
        train_state.enc_state,
        train_state.q_ext_state,
        train_state.q_int_state,
        args.net.batch_norm,
        args.net.batch_norm,
        args.net.batch_norm,
    )

    next_ext_prim = next_q_ext_all[:, :n_prim_actions]
    next_int_prim = next_q_int_all[:, :n_prim_actions]

    if not args.algo.rew_scale:
        next_ext_prim_raw = inv_value_rescale(next_ext_prim)
        next_int_prim_raw = inv_value_rescale(next_int_prim)

        next_sel_prim_raw = next_ext_prim_raw + args.algo.int_coef * next_int_prim_raw
        next_a_prim = jnp.argmax(next_sel_prim_raw, axis=-1)

        # Gather transformed head values because compute_transformed_td_lambda_targets expects transformed Qs.
        next_prim_q_ext = jnp.take_along_axis(
            next_ext_prim, next_a_prim[:, None], axis=-1
        ).squeeze(-1)

        next_prim_q_int = jnp.take_along_axis(
            next_int_prim, next_a_prim[:, None], axis=-1
        ).squeeze(-1)
    else:
        next_sel_prim = next_ext_prim + args.algo.int_coef * next_int_prim
        next_a_prim = jnp.argmax(next_sel_prim, axis=-1)

        next_prim_q_ext = jnp.take_along_axis(
            next_ext_prim, next_a_prim[:, None], axis=-1
        ).squeeze(-1)

        next_prim_q_int = jnp.take_along_axis(
            next_int_prim, next_a_prim[:, None], axis=-1
        ).squeeze(-1)

    if args.algo.rew_scale:
        prim_ext_targets = compute_td_lambda_targets(
            next_q=next_prim_q_ext,
            q_vals=prim_transitions.meta_q_ext_vals[:, :, :n_prim_actions],
            rewards=prim_transitions.rewards,
            dones=prim_transitions.dones.astype(jnp.float32),
            gamma=args.env.gamma,
            td_lambda=args.algo.td_lambda,
        )

        prim_int_targets = compute_td_lambda_targets(
            next_q=next_prim_q_int,
            q_vals=prim_transitions.meta_q_int_vals[:, :, :n_prim_actions],
            rewards=prim_transitions.int_rewards,
            dones=jnp.zeros_like(prim_transitions.dones, dtype=jnp.float32),
            gamma=args.rnd.int_gamma,
            td_lambda=args.algo.td_lambda,
        )
    else:
        prim_ext_targets = compute_transformed_td_lambda_targets(
            next_q_t=next_prim_q_ext,
            q_vals_t=prim_transitions.meta_q_ext_vals[:, :, :n_prim_actions],
            rewards=prim_transitions.rewards,
            dones=prim_transitions.dones.astype(jnp.float32),
            gamma=args.env.gamma,
            td_lambda=args.algo.td_lambda,
        )

        prim_int_targets = compute_transformed_td_lambda_targets(
            next_q_t=next_prim_q_int,
            q_vals_t=prim_transitions.meta_q_int_vals[:, :, :n_prim_actions],
            rewards=prim_transitions.int_rewards,
            dones=jnp.zeros_like(prim_transitions.dones, dtype=jnp.float32),
            gamma=args.rnd.int_gamma,
            td_lambda=args.algo.td_lambda,
        )

    # ------------------------------------------------------------
    # NEO option targets
    # ------------------------------------------------------------
    opt_ext_lambda_targets, opt_int_lambda_targets, eig_rewards, stall_mean = \
        compute_neo_options_lambda_targets_shared_dual(
            train_state=train_state,
            transitions=prim_transitions,
            gamma=args.options.gamma,
            td_lambda=args.options.td_lambda,
            use_opt_bn=args.net.batch_norm,
            n_options=n_total_options,
            use_value_rescale=not args.algo.rew_scale,
            stall_coef=args.options.stall_coef,
            rnd_coef=args.neo.rnd_coef,
        )

    # ------------------------------------------------------------
    # Meta targets
    # ------------------------------------------------------------
    meta_seq = unwrap_meta_ring(meta_buf, args.options.m_steps)

    meta_q_ext_seq, meta_q_int_seq = get_meta_dual_q_seq(
        meta_seq["start_obs"],
        train_state.enc_state,
        train_state.q_ext_state,
        train_state.q_int_state,
        args.net.batch_norm,
        args.net.batch_norm,
        args.net.batch_norm,
    )

    # ------------------------------------------------------------
    # Meta bootstrap obs from unified carry
    # ------------------------------------------------------------
    seg_start_obs_live = expl_state[0][:n_train_envs]
    obs_live = expl_state[3][:n_train_envs]
    active_meta_live = expl_state[5][:n_train_envs]
    opt_steps_left_live = expl_state[6][:n_train_envs]

    need_new_meta_live = (active_meta_live < 0) | (opt_steps_left_live <= 0)

    meta_boot_obs = jnp.where(
        need_new_meta_live[:, None, None, None],
        obs_live,
        seg_start_obs_live,
    )

    next_meta_q_ext_all, next_meta_q_int_all = get_dual_q_values_fn(
        meta_boot_obs,
        train_state.enc_state,
        train_state.q_ext_state,
        train_state.q_int_state,
        args.net.batch_norm,
        args.net.batch_norm,
        args.net.batch_norm,
    )

    meta_ext_targets, meta_int_targets = compute_meta_td_lambda_targets_dual_no_retrace(
        next_q_ext_all=next_meta_q_ext_all,
        next_q_int_all=next_meta_q_int_all,
        ext_rewards=meta_seq["ext_rewards"],
        int_rewards=meta_seq["int_rewards"],
        dones=meta_seq["dones"].astype(jnp.float32),
        ext_discounts=meta_seq["ext_discounts"],
        int_discounts=meta_seq["int_discounts"],
        q_ext_vals=meta_q_ext_seq,
        q_int_vals=meta_q_int_seq,
        td_lambda=args.options.meta_td_lambda,
        int_coef=args.algo.int_coef,
        use_value_rescale=not args.algo.rew_scale,
    )

    _smdp_learn_phase = partial(
        smdp_learn_phase_dual,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        int_coef=args.algo.int_coef,
        n_meta_actions=n_meta_actions,
        n_prim_actions=n_prim_actions,
    )

    _prim_learn_phase = partial(
        primitive_learn_phase_dual,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        int_coef=args.algo.int_coef,
        n_primitives=n_prim_actions,
    )

    _opt_policy_learn_phase = partial(
        update_option_networks_shared_dual,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        int_coef=args.neo.rnd_coef,
    )

    _update_lap_network = partial(update_neo_laplacian_network, args=args, n_actions=n_prim_actions)

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
            counts=zeta_rollout.T,
            novelty=prim_transitions.rnd_bonus.T,
        )

        train_state, (eig_loss, pos_loss, ortho_loss, dual_loss) = _update_lap_network(
            lap_key, train_state, lap_batch
        )

        flat_storage = FlatStorage(rnd_frame=flat_rnd_frames)
        rnd_state, rnd_fwd_loss = rnd_update_once_per_epoch(
            train_state.rnd_state,
            flat_storage,
            args.rnd.upd_prop,
            burnin,
        )
        train_state = train_state.replace(rnd_state=rnd_state)
        rnd_fwd_l2 = prim_transitions.rnd_bonus.mean()

        meta_transitions = Meta_Transition(
            obs=meta_seq["start_obs"],
            actions=meta_seq["actions"],
            ext_rewards=meta_seq["ext_rewards"],
            int_rewards=meta_seq["int_rewards"],
            dones=meta_seq["dones"],
            next_obs=meta_seq["term_obs"],
            ext_discounts=meta_seq["ext_discounts"],
            int_discounts=meta_seq["int_discounts"],
            behav_probs=meta_seq["behav_probs"],
            valid=meta_seq["valid"],
        )

        meta_batches = jax.tree_util.tree_map(
            lambda x: preprocess_pytree(x, shard_key, args.algo.n_mini_batches),
            meta_transitions,
        )
        meta_ext_targets_mb = preprocess_array(meta_ext_targets, shard_key, args.algo.n_mini_batches)
        meta_int_targets_mb = preprocess_array(meta_int_targets, shard_key, args.algo.n_mini_batches)

        (train_state, _), (
            meta_losses, meta_qstats,
            meta_option_taken_count_mb, meta_primitive_taken_count_mb,
            meta_option_greedy_count_mb, meta_primitive_greedy_count_mb,
            meta_valid_count_mb
        ) = jax.lax.scan(
            _smdp_learn_phase,
            (train_state, shard_key),
            (meta_batches, meta_ext_targets_mb, meta_int_targets_mb),
        )

        prim_batches = jax.tree_util.tree_map(
            lambda x: preprocess_pytree(x, shard_key, args.algo.n_mini_batches),
            prim_transitions,
        )
        prim_ext_targets_mb = preprocess_array(prim_ext_targets, shard_key, args.algo.n_mini_batches)
        prim_int_targets_mb = preprocess_array(prim_int_targets, shard_key, args.algo.n_mini_batches)

        (train_state, _), (prim_losses, prim_qstats) = jax.lax.scan(
            _prim_learn_phase,
            (train_state, shard_key),
            (prim_batches, prim_ext_targets_mb, prim_int_targets_mb),
        )

        opt_batches = jax.tree_util.tree_map(
            lambda x: preprocess_pytree(x, shard_key, args.options.n_mini_batches),
            prim_transitions,
        )

        opt_ext_targets = jax.vmap(
            lambda targ_o: preprocess_array(targ_o, shard_key, args.options.n_mini_batches),
            in_axes=0,
        )(opt_ext_lambda_targets)
        opt_ext_targets = jnp.transpose(opt_ext_targets, (1, 0, 2))

        opt_int_targets = jax.vmap(
            lambda targ_o: preprocess_array(targ_o, shard_key, args.options.n_mini_batches),
            in_axes=0,
        )(opt_int_lambda_targets)
        opt_int_targets = jnp.transpose(opt_int_targets, (1, 0, 2))

        def _opt_mb_step(ts, shard):
            mb_data, mb_opt_ext_tgt, mb_opt_int_tgt = shard
            ts, (opt_loss, opt_q_t, opt_q_raw) = _opt_policy_learn_phase(
                ts, mb_data, mb_opt_ext_tgt, mb_opt_int_tgt
            )
            return ts, (opt_loss, opt_q_t, opt_q_raw)

        train_state, (optpol_losses, optpol_q_t, optpol_q_raw) = jax.lax.scan(
            _opt_mb_step,
            train_state,
            (opt_batches, opt_ext_targets, opt_int_targets),
        )

        return (train_state, key), (
            meta_losses, meta_qstats,
            meta_option_taken_count_mb, meta_primitive_taken_count_mb,
            meta_option_greedy_count_mb, meta_primitive_greedy_count_mb,
            meta_valid_count_mb,
            prim_losses, prim_qstats,
            optpol_losses, optpol_q_t, optpol_q_raw,
            eig_loss, pos_loss, ortho_loss, dual_loss,
            rnd_fwd_loss, rnd_fwd_l2,
        )

    key, epoch_key = jax.random.split(key)
    (train_state, _), epoch_stats = jax.lax.scan(_learn_epoch, (train_state, epoch_key), None, args.algo.n_epochs)
    train_state = train_state.replace(n_updates=train_state.n_updates + 1)

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

    eig_loss = epoch_stats[12]
    pos_loss = epoch_stats[13]
    ortho_loss = epoch_stats[14]
    dual_loss = epoch_stats[15]

    rnd_fwd_losses = epoch_stats[16]
    rnd_fwd_l2s = epoch_stats[17]

    meta_ext_loss = meta_qstats[0].mean()
    meta_int_loss = meta_qstats[1].mean()
    meta_q_ext_mean_t = meta_qstats[2].mean()
    meta_q_ext_mean_raw = meta_qstats[3].mean()
    meta_q_int_mean_t = meta_qstats[4].mean()
    meta_q_int_mean_raw = meta_qstats[5].mean()
    meta_q_sel_mean_t = meta_qstats[6].mean()
    meta_q_sel_mean_raw = meta_qstats[7].mean()

    prim_ext_loss = prim_qstats[0].mean()
    prim_int_loss = prim_qstats[1].mean()
    prim_q_ext_mean_t = prim_qstats[2].mean()
    prim_q_ext_mean_raw = prim_qstats[3].mean()
    prim_q_int_mean_t = prim_qstats[4].mean()
    prim_q_int_mean_raw = prim_qstats[5].mean()
    prim_q_sel_mean_t = prim_qstats[6].mean()
    prim_q_sel_mean_raw = prim_qstats[7].mean()

    meta_option_taken_count = meta_option_taken_count_mb.sum()
    meta_primitive_taken_count = meta_primitive_taken_count_mb.sum()
    meta_option_greedy_count = meta_option_greedy_count_mb.sum()
    meta_primitive_greedy_count = meta_primitive_greedy_count_mb.sum()
    meta_valid_count = meta_valid_count_mb.sum()

    optpol_td_loss = optpol_losses.mean(axis=(0, 1))
    optpol_q_mean_t = optpol_q_t.mean(axis=(0, 1))
    optpol_q_mean_raw = optpol_q_raw.mean(axis=(0, 1))

    rnd_fwd_loss = rnd_fwd_losses.mean()
    rnd_fwd_l2 = rnd_fwd_l2s.mean()

    if args.options.eigdur_options:
        opt_dur_table = eigenvalues_to_option_durations_skip_first(
            train_state.lap_state.eigen_values,
            n_base_options=n_base_options,
            bidir_options=False,
            min_dur=args.options.opt_dur_min,
            max_dur=args.options.opt_dur_max,
            eps=1e-6,
        )
    else:
        opt_dur_table = jnp.full((n_total_options,), jnp.int32(args.options.opt_dur), dtype=jnp.int32)

    env_frame_mult = getattr(args.env, "n_frames", 1)

    metrics = {
        "env_step": train_state.timesteps,
        "update_steps": train_state.n_updates,
        "env_frame": train_state.timesteps * env_frame_mult,
        "grad_steps": train_state.grad_steps,

        "Loss/SMDP_total": meta_losses.mean(),
        "Loss/SMDP_ext": meta_ext_loss,
        "Loss/SMDP_int": meta_int_loss,

        "Loss/Primitive_total": prim_losses.mean(),
        "Loss/Primitive_ext": prim_ext_loss,
        "Loss/Primitive_int": prim_int_loss,

        "Loss/OptionPolicy": optpol_losses.mean(),

        "Q/SMDP_ext_mean_t": meta_q_ext_mean_t,
        "Q/SMDP_ext_mean_raw": meta_q_ext_mean_raw,
        "Q/SMDP_int_mean_t": meta_q_int_mean_t,
        "Q/SMDP_int_mean_raw": meta_q_int_mean_raw,
        "Q/SMDP_sel_mean_t": meta_q_sel_mean_t,
        "Q/SMDP_sel_mean_raw": meta_q_sel_mean_raw,

        "Q/Primitive_ext_mean_t": prim_q_ext_mean_t,
        "Q/Primitive_ext_mean_raw": prim_q_ext_mean_raw,
        "Q/Primitive_int_mean_t": prim_q_int_mean_t,
        "Q/Primitive_int_mean_raw": prim_q_int_mean_raw,
        "Q/Primitive_sel_mean_t": prim_q_sel_mean_t,
        "Q/Primitive_sel_mean_raw": prim_q_sel_mean_raw,

        "Eigen/Total_loss": eig_loss.mean(),
        "Eigen/Pos_loss": pos_loss.mean(),
        "Eigen/Ortho_loss": ortho_loss.mean(),
        "Eigen/Dual_loss": dual_loss.mean(),
        "Eigen/Barrier": train_state.lap_state.barrier_coeffs[0, 0],
        "Eigen/Max_EigenValue": jnp.max(train_state.lap_state.eigen_values),

        "Options/dur_min": opt_dur_table.min(),
        "Options/dur_max": opt_dur_table.max(),
        "Options/dur_mean": opt_dur_table.astype(jnp.float32).mean(),
        "Options/stall_mean": stall_mean,

        # "Meta/rho_mean": meta_rho.mean(),
        # "Meta/rho_max": meta_rho.max(),

        "Meta/option_taken_count": meta_option_taken_count,
        "Meta/primitive_taken_count": meta_primitive_taken_count,
        "Meta/option_greedy_count": meta_option_greedy_count,
        "Meta/primitive_greedy_count": meta_primitive_greedy_count,
        "Meta/valid_count": meta_valid_count,

        "Meta/option_taken_frac": meta_option_taken_count / (meta_valid_count + 1e-8),
        "Meta/primitive_taken_frac": meta_primitive_taken_count / (meta_valid_count + 1e-8),
        "Meta/option_greedy_frac": meta_option_greedy_count / (meta_valid_count + 1e-8),
        "Meta/primitive_greedy_frac": meta_primitive_greedy_count / (meta_valid_count + 1e-8),
    }

    # valid = meta_seq["valid"]
    #
    # metrics.update({
    #     # "Meta/rho_mean": jnp.sum(valid * meta_rho) / (jnp.sum(valid) + 1e-8),
    #     # "Meta/rho_max": jnp.max(jnp.where(valid > 0, meta_rho, 0.0)),
    #     "Meta/behav_prob_mean": jnp.sum(valid * meta_seq["behav_probs"]) / (jnp.sum(valid) + 1e-8),
    #     "Meta/behav_prob_min": jnp.min(jnp.where(valid > 0, meta_seq["behav_probs"], 1.0)),
    #     "Meta/greedy_match_frac": jnp.sum(valid * (meta_rho > 0).astype(jnp.float32)) / (jnp.sum(valid) + 1e-8),
    # })

    for i in range(n_total_options):
        metrics[f"Options/TD_loss_{i}"] = optpol_td_loss[i]
        metrics[f"Options/Q_mean_t_{i}"] = optpol_q_mean_t[i]
        metrics[f"Options/Q_mean_raw_{i}"] = optpol_q_mean_raw[i]

    nov = prim_transitions.rnd_bonus
    nov_n = prim_transitions.rnd_bonus_n
    inv = prim_transitions.inv_count
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

        "RND/int_reward_mean": prim_transitions.int_rewards.mean(),
        "RND/int_reward_max": prim_transitions.int_rewards.max(),
        "RND/int_reward_sum_per_env_mean": prim_transitions.int_rewards.sum(axis=0).mean(),

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
        "RND/bonus_max_ema": train_state.rnd_state.rnd_bonus_max_ema,
    })

    if args.env.n_test_envs > 0:
        train_infos = jax.tree_util.tree_map(lambda x: x[:, :-args.env.n_test_envs], infos)
        test_infos = jax.tree_util.tree_map(lambda x: x[:, -args.env.n_test_envs:], infos)

        # Means
        metrics.update({k: v.mean() for k, v in train_infos.items()})
        metrics.update({f"test/{k}": v.mean() for k, v in test_infos.items()})

        # Max raw returns
        metrics["Max_Return"] = train_infos["returned_episode_returns"].max()
        metrics["test/Max_Return"] = test_infos["returned_episode_returns"].max()

        # Max normalized returns
        metrics["Max_Normalized_Return"] = train_infos["returned_episode_normalized_returns"].max()
        metrics["test/Max_Normalized_Return"] = test_infos["returned_episode_normalized_returns"].max()

    else:
        metrics.update({k: v.mean() for k, v in infos.items()})

        # Max raw returns
        metrics["Max_Return"] = infos["returned_episode_returns"].max()

        # Max normalized returns
        metrics["Max_Normalized_Return"] = infos["returned_episode_normalized_returns"].max()

    def _log_to_wandb(m):
        elapsed = time.time() - start_time
        total_env = m["update_steps"] * args.env.n_steps * args.env.n_envs
        m["SPS"] = total_env / elapsed
        import wandb
        wandb.log(m, step=int(m["update_steps"]))

    @jax.jit
    def _pass_and_log(m):
        return jax.debug.callback(_log_to_wandb, m)

    if args.wb.on:
        do_log = (
            (train_state.n_updates % 10 == 0)
            | (train_state.n_updates == (args.algo.n_upds - 1))
        )
        _ = jax.lax.cond(do_log, lambda m: _pass_and_log(m), lambda m: None, metrics)

    runner_state = (train_state, tuple(expl_state), key, opt_dur_table)
    return runner_state, metrics