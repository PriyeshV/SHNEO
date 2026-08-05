import jax
import jax.numpy as jnp
from flax import struct
from functools import partial

from src.mains.algorithms.PQN import (
    value_rescale,
    inv_value_rescale,
    compute_td_lambda_targets,
    compute_transformed_td_lambda_targets,
)


@struct.dataclass
class MetaRingBuffer:
    ptr: jnp.ndarray          # [E]
    head: jnp.ndarray         # [E]
    length: jnp.ndarray       # [E]

    start_obs: jnp.ndarray    # [T_max,E,C,H,W]
    term_obs: jnp.ndarray     # [T_max,E,C,H,W]
    actions: jnp.ndarray      # [T_max,E]      meta action in [0..A+O-1]
    rewards: jnp.ndarray      # [T_max,E]      discounted segment return
    dones: jnp.ndarray        # [T_max,E]
    discounts: jnp.ndarray    # [T_max,E]      gamma^k for that segment
    durations: jnp.ndarray    # [T_max,E]      k
    behav_probs: jnp.ndarray  # [T_max,E]      behavior prob of chosen meta action at start


# ============================================================
# Ring-buffer unwrap
# ============================================================

@partial(jax.jit, static_argnums=(1,))
def unwrap_meta_ring(meta_buf, T_max: int):
    """
    Right-align valid entries so the newest valid transition is always at index T_max-1.
    This makes fixed-length backward recursion compatible with variable filled lengths.

    Returns tensors of shape [T_max, E, ...] and valid mask [T_max,E].
    """
    E = meta_buf.ptr.shape[0]
    t = jnp.arange(T_max)[:, None]                         # [T,1]
    e = jnp.arange(E)[None, :]                            # [1,E]
    length = meta_buf.length[None, :]                     # [1,E]
    shift = T_max - length                                # [1,E]

    valid = t >= shift                                    # [T,E]
    src_k = jnp.maximum(t - shift, 0)                     # [T,E]
    idx = (meta_buf.head[None, :] + src_k) % T_max       # [T,E]

    start_obs = meta_buf.start_obs[idx, e, ...]
    term_obs = meta_buf.term_obs[idx, e, ...]
    actions = meta_buf.actions[idx, e]
    rewards = meta_buf.rewards[idx, e]
    dones = meta_buf.dones[idx, e]
    discounts = meta_buf.discounts[idx, e]
    durations = meta_buf.durations[idx, e]
    behav_probs = meta_buf.behav_probs[idx, e]

    # sanitize invalid prefix
    rewards = jnp.where(valid, rewards, 0.0)
    dones = jnp.where(valid, dones, 1)
    discounts = jnp.where(valid, discounts, 0.0)
    durations = jnp.where(valid, durations, 1)
    behav_probs = jnp.where(valid, behav_probs, 1.0)
    actions = jnp.where(valid, actions, 0)

    return {
        "start_obs": start_obs,
        "term_obs": term_obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "discounts": discounts,
        "durations": durations,
        "behav_probs": behav_probs,
        "valid": valid.astype(jnp.float32),
    }


def init_meta_buffer(init_obs, n_total_envs, t_max):
    C, H, W = init_obs.shape[1:]
    return MetaRingBuffer(
        ptr=jnp.zeros((n_total_envs,), dtype=jnp.int32),
        head=jnp.zeros((n_total_envs,), dtype=jnp.int32),
        length=jnp.zeros((n_total_envs,), dtype=jnp.int32),
        start_obs=jnp.zeros((t_max, n_total_envs, C, H, W), dtype=init_obs.dtype),
        term_obs=jnp.zeros((t_max, n_total_envs, C, H, W), dtype=init_obs.dtype),
        actions=jnp.zeros((t_max, n_total_envs), dtype=jnp.int32),
        rewards=jnp.zeros((t_max, n_total_envs), dtype=jnp.float32),
        dones=jnp.ones((t_max, n_total_envs), dtype=jnp.int32),
        discounts=jnp.zeros((t_max, n_total_envs), dtype=jnp.float32),
        durations=jnp.zeros((t_max, n_total_envs), dtype=jnp.int32),
        behav_probs=jnp.ones((t_max, n_total_envs), dtype=jnp.float32),
    )


# ============================================================
# Fresh meta-Q on a [T,E,...] sequence of observations
# ============================================================

@partial(jax.jit, static_argnums=(3, 4))
def get_meta_q_seq(start_obs, enc_state, q_state, use_enc_bn: bool, use_q_bn: bool):
    """
    start_obs: [T,E,C,H,W]
    returns: [T,E,A_meta]
    """
    T, E = start_obs.shape[:2]
    flat = start_obs.reshape((T * E,) + start_obs.shape[2:])

    obs = jnp.transpose(flat, (0, 2, 3, 1))  # NHWC

    enc_vars = {"params": enc_state.params}
    if use_enc_bn:
        enc_vars["batch_stats"] = enc_state.batch_stats
    emb = enc_state.apply_fn(enc_vars, obs, train=False)

    q_vars = {"params": q_state.params}
    if use_q_bn:
        q_vars["batch_stats"] = q_state.batch_stats
    q = q_state.apply_fn(q_vars, emb, train=False)

    return q.reshape((T, E, -1))


# ============================================================
# Meta SMDP Retrace / TD-lambda target
# ============================================================

@partial(jax.jit, static_argnums=(7, 8, 9, 10))
def compute_meta_retrace_targets(
    actions,          # [T,E]
    behav_probs,      # [T,E]
    next_q,           # [E]
    rewards,          # [T,E]
    dones,            # [T,E]
    discounts,        # [T,E] = gamma^k
    q_vals,           # [T,E,A_meta] fresh online q-values at start states
    td_lambda,        # scalar
    use_safe_rho,     # bool
    retrace_clip_rho, # float
    retrace_clip_c,   # float
):
    """
    Retrace-style SMDP targets on meta transitions.
    Returns:
      targets [T,E]
      rho     [T,E]
    """
    pi = jax.nn.softmax(q_vals, axis=-1)  # [T,E,A_meta]
    pi_a = jnp.take_along_axis(pi, actions[..., None], axis=-1).squeeze(-1)  # [T,E]

    rho = pi_a / (behav_probs + 1e-8)

    if use_safe_rho:
        rho_bar = jnp.minimum(retrace_clip_rho, rho)
        c = td_lambda * jnp.minimum(retrace_clip_c, rho)
    else:
        rho_bar = rho
        c = td_lambda * rho

    V = jnp.max(q_vals, axis=-1)  # [T,E]

    G_last = rewards[-1] + discounts[-1] * (1.0 - dones[-1]) * next_q
    V_last = V[-1]

    def _step(carry, elems):
        G_next, V_next = carry
        r_t, V_t, done_t, disc_t, rho_t, c_t = elems

        delta_t = rho_t * (r_t + disc_t * (1.0 - done_t) * V_next - V_t)
        G_t = V_t + delta_t + disc_t * (1.0 - done_t) * c_t * (G_next - V_next)
        G_t = jnp.where(done_t, r_t, G_t)
        return (G_t, V_t), G_t

    (_, _), Gs = jax.lax.scan(
        _step,
        (G_last, V_last),
        (rewards[:-1], V[:-1], dones[:-1], discounts[:-1], rho_bar[:-1], c[:-1]),
        reverse=True,
    )

    targets = jnp.concatenate([Gs, G_last[None]], axis=0)
    return targets, rho


# ============================================================
# Primitive TD-lambda targets on primitive rollout
# ============================================================

@partial(jax.jit, static_argnums=(4, 5))
def compute_meta_primitive_targets(
    next_prim_q,      # [E]
    meta_q_vals,      # [T,E,A_meta]
    rewards,          # [T,E]
    dones,            # [T,E]
    gamma,            # scalar
    td_lambda,        # scalar
):
    """
    Primitive branch updates ONLY primitive rows of the meta-Q head.
    Bootstrap uses primitive rows only.
    """
    prim_q_vals = meta_q_vals  # caller should already pass [:,:,:A]
    targets = compute_td_lambda_targets(
        next_q=next_prim_q,
        q_vals=prim_q_vals,
        rewards=rewards,
        dones=dones,
        gamma=gamma,
        td_lambda=td_lambda,
    )
    return targets


# ============================================================
# Update 1: SMDP meta update on executed meta actions
# ============================================================
@partial(jax.jit, static_argnums=range(2, 7))
def smdp_learn_phase(
    carry,
    minibatch_and_target,
    use_enc_bn: bool,
    use_q_bn: bool,
    use_value_rescale: bool,
    n_meta_actions: int,
    n_prim_actions: int,
):
    train_state, _key = carry
    minibatch, target = minibatch_and_target
    obs = jnp.transpose(minibatch.obs, (0, 2, 3, 1))
    valid = minibatch.valid.astype(jnp.float32)   # [B]

    A = n_prim_actions

    def _loss_fn(enc_params, q_params):
        if use_enc_bn:
            emb, enc_mut = train_state.enc_state.apply_fn(
                {"params": enc_params, "batch_stats": train_state.enc_state.batch_stats},
                obs,
                train=True,
                mutable=["batch_stats"],
            )
            new_enc_bs = enc_mut["batch_stats"]
        else:
            emb = train_state.enc_state.apply_fn({"params": enc_params}, obs, train=True)
            new_enc_bs = None

        if use_q_bn:
            qv, q_mut = train_state.q_state.apply_fn(
                {"params": q_params, "batch_stats": train_state.q_state.batch_stats},
                emb,
                train=True,
                mutable=["batch_stats"],
            )
            new_q_bs = q_mut["batch_stats"]
        else:
            qv = train_state.q_state.apply_fn({"params": q_params}, emb, train=True)
            new_q_bs = None

        q_pred = jnp.take_along_axis(qv, minibatch.actions[:, None], axis=-1).squeeze(-1)
        sq = 0.5 * jnp.square(q_pred - target)
        td_loss = jnp.sum(valid * sq) / (jnp.sum(valid) + 1e-8)

        q_mean_t = jnp.sum(valid * q_pred) / (jnp.sum(valid) + 1e-8)
        q_mean_raw = (
            jnp.sum(valid * inv_value_rescale(q_pred)) / (jnp.sum(valid) + 1e-8)
            if use_value_rescale else q_mean_t
        )

        # --------------------------------------------------
        # Total option vs primitive stats from this minibatch
        # --------------------------------------------------
        actions = minibatch.actions                # [B]
        greedy = jnp.argmax(qv, axis=-1)           # [B]

        taken_is_option = (actions >= A).astype(jnp.float32)
        greedy_is_option = (greedy >= A).astype(jnp.float32)

        option_taken_count = jnp.sum(valid * taken_is_option)
        primitive_taken_count = jnp.sum(valid * (1.0 - taken_is_option))

        option_greedy_count = jnp.sum(valid * greedy_is_option)
        primitive_greedy_count = jnp.sum(valid * (1.0 - greedy_is_option))

        aux = (
            (new_enc_bs, new_q_bs),
            (q_mean_t, q_mean_raw),
            option_taken_count,
            primitive_taken_count,
            option_greedy_count,
            primitive_greedy_count,
            jnp.sum(valid),
        )
        return td_loss, aux

    (loss, (
        (new_enc_bs, new_q_bs),
        q_stats,
        option_taken_count,
        primitive_taken_count,
        option_greedy_count,
        primitive_greedy_count,
        valid_count,
    )), grads = jax.value_and_grad(
        _loss_fn, argnums=(0, 1), has_aux=True
    )(train_state.enc_state.params, train_state.q_state.params)

    new_enc_state = train_state.enc_state.apply_gradients(grads=grads[0])
    new_q_state = train_state.q_state.apply_gradients(grads=grads[1])

    if use_enc_bn and new_enc_bs is not None:
        new_enc_state = new_enc_state.replace(batch_stats=new_enc_bs)
    if use_q_bn and new_q_bs is not None:
        new_q_state = new_q_state.replace(batch_stats=new_q_bs)

    train_state = train_state.replace(
        enc_state=new_enc_state,
        q_state=new_q_state,
        grad_steps=train_state.grad_steps + 1,
    )

    return (train_state, _key), (
        loss,
        q_stats,
        option_taken_count,
        primitive_taken_count,
        option_greedy_count,
        primitive_greedy_count,
        valid_count,
    )


# ============================================================
# Update 2: primitive update on primitive executed actions
# ============================================================

@partial(jax.jit, static_argnums=(2, 3, 4))
def primitive_learn_phase(
    carry,
    minibatch_and_target,
    use_enc_bn: bool,
    use_q_bn: bool,
    use_value_rescale: bool,
):
    """
    Same as your primitive learn phase, but only regresses primitive columns.
    minibatch.actions are primitive actions in [0..A-1].
    minibatch.meta_q_vals shape [B, A+O] is not used except for diagnostics if needed.
    """
    train_state, _key = carry
    minibatch, target = minibatch_and_target
    obs = jnp.transpose(minibatch.obs, (0, 2, 3, 1))
    A = minibatch.option_q_vals.shape[-1]  # primitive action count

    def _loss_fn(enc_params, q_params):
        if use_enc_bn:
            emb, enc_mut = train_state.enc_state.apply_fn(
                {"params": enc_params, "batch_stats": train_state.enc_state.batch_stats},
                obs, train=True, mutable=["batch_stats"]
            )
            new_enc_bs = enc_mut["batch_stats"]
        else:
            emb = train_state.enc_state.apply_fn({"params": enc_params}, obs, train=True)
            new_enc_bs = None

        if use_q_bn:
            qv, q_mut = train_state.q_state.apply_fn(
                {"params": q_params, "batch_stats": train_state.q_state.batch_stats},
                emb, train=True, mutable=["batch_stats"]
            )
            new_q_bs = q_mut["batch_stats"]
        else:
            qv = train_state.q_state.apply_fn({"params": q_params}, emb, train=True)
            new_q_bs = None

        prim_q = qv[:, :A]
        q_pred = jnp.take_along_axis(prim_q, minibatch.actions[:, None], axis=-1).squeeze(-1)

        td_loss = 0.5 * jnp.mean((q_pred - target) ** 2)
        q_mean_t = jnp.mean(q_pred)
        q_mean_raw = jnp.mean(inv_value_rescale(q_pred)) if use_value_rescale else q_mean_t

        return td_loss, ((new_enc_bs, new_q_bs), (q_mean_t, q_mean_raw))

    (loss, ((new_enc_bs, new_q_bs), q_stats)), grads = jax.value_and_grad(
        _loss_fn, argnums=(0, 1), has_aux=True
    )(train_state.enc_state.params, train_state.q_state.params)

    new_enc_state = train_state.enc_state.apply_gradients(grads=grads[0])
    new_q_state = train_state.q_state.apply_gradients(grads=grads[1])

    if use_enc_bn and new_enc_bs is not None:
        new_enc_state = new_enc_state.replace(batch_stats=new_enc_bs)
    if use_q_bn and new_q_bs is not None:
        new_q_state = new_q_state.replace(batch_stats=new_q_bs)

    train_state = train_state.replace(
        enc_state=new_enc_state,
        q_state=new_q_state,
        grad_steps=train_state.grad_steps + 1,
    )
    return (train_state, _key), (loss, q_stats)


# ============================================================
# Update 3: intra-option Q-learning update
# ============================================================
@partial(jax.jit, static_argnums=(3, 4, 5))
def compute_intra_option_meta_targets_1step(
    transitions,          # Prim_Transition [T,E,...]
    next_meta_q_all,      # [E, A+O] fresh meta-Q at final bootstrap state
    opt_dur_table,        # [O]
    gamma: float,
    n_prim_actions: int,
    use_value_rescale: bool,
):
    """
    One-step EXTRINSIC intra-option targets for meta-option rows Q_meta(s, A+o).

    Returns:
        ioq_targets: [O, T, E]
    """
    rewards = transitions.rewards                    # [T,E]
    dones = transitions.dones.astype(jnp.float32)   # [T,E]
    meta_q = transitions.meta_q_vals                 # [T,E,A+O]

    A = n_prim_actions
    O = opt_dur_table.shape[0]
    T, E = rewards.shape

    if use_value_rescale:
        meta_q_raw = inv_value_rescale(meta_q)
        next_meta_q_raw = inv_value_rescale(next_meta_q_all)
    else:
        meta_q_raw = meta_q
        next_meta_q_raw = next_meta_q_all

    q_opt_rows = meta_q_raw[:, :, A:A+O]             # [T,E,O]
    v_rows = jnp.max(meta_q_raw, axis=-1)            # [T,E]

    q_opt_next = jnp.concatenate(
        [q_opt_rows[1:], next_meta_q_raw[None, :, A:A+O]],
        axis=0
    )                                                # [T,E,O]

    v_next = jnp.concatenate(
        [v_rows[1:], jnp.max(next_meta_q_raw, axis=-1)[None, :]],
        axis=0
    )                                                # [T,E]

    dur = jnp.maximum(opt_dur_table.astype(jnp.float32), 1.0)   # [O]
    beta_o = 1.0 / dur                                           # [O]

    targets = rewards[:, :, None] + gamma * (1.0 - dones)[:, :, None] * (
        (1.0 - beta_o[None, None, :]) * q_opt_next
        + beta_o[None, None, :] * v_next[:, :, None]
    )                                                            # [T,E,O]

    targets = jnp.transpose(targets, (2, 0, 1))                  # [O,T,E]

    if use_value_rescale:
        targets = value_rescale(targets)

    return targets


@partial(jax.jit, static_argnums=range(3, 8))
def update_intra_option_meta_shared(
    train_state,
    transitions,         # flattened primitive batch
    option_targets,      # [O,B]
    n_prim_actions: int,
    use_enc_bn: bool,
    use_q_bn: bool,
    use_value_rescale: bool,
    deterministic_match: bool = True,
):
    """
    Update META-Q option entries Q_meta(s, A+o) for ALL options whose
    option-policy primitive action matches the executed primitive action.

    - train_state.q_state / enc_state are updated
    - train_state.opt_enc_state / opt_q_states are NOT updated
    - option networks are used only to compute eligibility
    """
    obs = jnp.transpose(transitions.obs, (0, 2, 3, 1))   # [B,H,W,C]
    act = transitions.actions                              # [B]

    A = n_prim_actions
    O = option_targets.shape[0]

    # --------------------------------------------------
    # 1) Compute option-policy outputs for eligibility
    # --------------------------------------------------
    opt_enc_vars = {"params": train_state.opt_enc_state.params}
    if use_enc_bn:
        opt_enc_vars["batch_stats"] = train_state.opt_enc_state.batch_stats
    opt_emb = train_state.opt_enc_state.apply_fn(opt_enc_vars, obs, train=False)   # [B,D]

    opt_q_params = jax.tree_util.tree_map(
        lambda *ps: jnp.stack(ps), *[s.params for s in train_state.opt_q_states]
    )
    opt_q_bs = (
        jax.tree_util.tree_map(
            lambda *bs: jnp.stack(bs), *[s.batch_stats for s in train_state.opt_q_states]
        )
        if use_q_bn else None
    )
    opt_q_apply = train_state.opt_q_states[0].apply_fn

    def one_opt_head(qp, qb):
        vars_q = {"params": qp}
        if use_q_bn:
            vars_q["batch_stats"] = qb
        return opt_q_apply(vars_q, opt_emb, train=False)   # [B,A]

    opt_q_OBA = jax.vmap(
        one_opt_head, in_axes=(0, 0 if use_q_bn else None)
    )(opt_q_params, opt_q_bs)  # [O,B,A]

    if deterministic_match:
        greedy_OB = jnp.argmax(opt_q_OBA, axis=-1)                         # [O,B]
        elig = (greedy_OB == act[None, :]).astype(jnp.float32)             # [O,B]
    else:
        pi_OBA = jax.nn.softmax(opt_q_OBA, axis=-1)
        elig = jnp.take_along_axis(
            pi_OBA, act[None, :, None], axis=-1
        ).squeeze(-1)                                                      # [O,B]

    # NOTE:
    # We no longer exclude the currently executed option.
    # Any option whose primitive policy matches the executed action is updated.

    # --------------------------------------------------
    # 2) Update META network option rows A:(A+O)
    # --------------------------------------------------
    def loss_fn(enc_params, q_params):
        if use_enc_bn:
            emb, enc_mut = train_state.enc_state.apply_fn(
                {"params": enc_params, "batch_stats": train_state.enc_state.batch_stats},
                obs,
                train=True,
                mutable=["batch_stats"],
            )
            new_enc_bs = enc_mut["batch_stats"]
        else:
            emb = train_state.enc_state.apply_fn({"params": enc_params}, obs, train=True)
            new_enc_bs = None

        if use_q_bn:
            qv, q_mut = train_state.q_state.apply_fn(
                {"params": q_params, "batch_stats": train_state.q_state.batch_stats},
                emb,
                train=True,
                mutable=["batch_stats"],
            )
            new_q_bs = q_mut["batch_stats"]
        else:
            qv = train_state.q_state.apply_fn({"params": q_params}, emb, train=True)
            new_q_bs = None

        meta_opt_q = qv[:, A:A + O].T                                      # [O,B]
        td_OB = meta_opt_q - option_targets                                # [O,B]

        per_opt_loss = jnp.sum(elig * (0.5 * td_OB ** 2), axis=1) / (
            jnp.sum(elig, axis=1) + 1e-8
        )
        total_loss = jnp.sum(per_opt_loss)

        q_mean_t = jnp.sum(elig * meta_opt_q, axis=1) / (jnp.sum(elig, axis=1) + 1e-8)
        if use_value_rescale:
            q_mean_raw = jnp.sum(elig * inv_value_rescale(meta_opt_q), axis=1) / (
                jnp.sum(elig, axis=1) + 1e-8
            )
        else:
            q_mean_raw = q_mean_t

        elig_any = jnp.mean((jnp.sum(elig, axis=0) > 0).astype(jnp.float32))
        elig_count = jnp.sum(elig, axis=1)  # [O]

        return total_loss, ((new_enc_bs, new_q_bs), per_opt_loss, q_mean_t, q_mean_raw, elig_any, elig_count)

    (loss, ((new_enc_bs, new_q_bs), per_opt_loss, q_mean_t, q_mean_raw, elig_any, elig_count)), grads = \
        jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            train_state.enc_state.params,
            train_state.q_state.params,
        )

    new_enc_state = train_state.enc_state.apply_gradients(grads=grads[0])
    new_q_state = train_state.q_state.apply_gradients(grads=grads[1])

    if use_enc_bn and new_enc_bs is not None:
        new_enc_state = new_enc_state.replace(batch_stats=new_enc_bs)
    if use_q_bn and new_q_bs is not None:
        new_q_state = new_q_state.replace(batch_stats=new_q_bs)

    train_state = train_state.replace(
        enc_state=new_enc_state,
        q_state=new_q_state,
        grad_steps=train_state.grad_steps + 1,
    )
    return train_state, (per_opt_loss, q_mean_t, q_mean_raw, elig_any, elig_count)


@partial(jax.jit, static_argnums=range(3, 8))
def update_intra_option_meta_shared_exclude_exec(
    train_state,
    transitions,         # flattened primitive batch
    option_targets,      # [O,B]
    n_prim_actions: int,
    use_enc_bn: bool,
    use_q_bn: bool,
    use_value_rescale: bool,
    deterministic_match: bool = True,
):
    """
    Update META-Q option entries Q_meta(s, A+o) for non-executed options whose
    option policy matches the executed primitive action.

    - train_state.q_state / enc_state are updated
    - train_state.opt_enc_state / opt_q_states are NOT updated
    - option networks are used only to compute eligibility
    """
    obs = jnp.transpose(transitions.obs, (0, 2, 3, 1))   # [B,H,W,C]
    act = transitions.actions                              # [B]
    meta_action = transitions.meta_action                  # [B]

    A = n_prim_actions
    O = option_targets.shape[0]
    B = act.shape[0]

    # --------------------------------------------------
    # 1) Compute option-policy outputs for eligibility
    #    NO gradients through option policy here
    # --------------------------------------------------
    opt_enc_vars = {"params": train_state.opt_enc_state.params}
    if use_enc_bn:
        opt_enc_vars["batch_stats"] = train_state.opt_enc_state.batch_stats
    opt_emb = train_state.opt_enc_state.apply_fn(opt_enc_vars, obs, train=False)   # [B,D]

    opt_q_params = jax.tree_util.tree_map(
        lambda *ps: jnp.stack(ps), *[s.params for s in train_state.opt_q_states]
    )
    opt_q_bs = (
        jax.tree_util.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in train_state.opt_q_states])
        if use_q_bn else None
    )
    opt_q_apply = train_state.opt_q_states[0].apply_fn

    def one_opt_head(qp, qb):
        vars_q = {"params": qp}
        if use_q_bn:
            vars_q["batch_stats"] = qb
        return opt_q_apply(vars_q, opt_emb, train=False)   # [B,A]

    opt_q_OBA = jax.vmap(
        one_opt_head, in_axes=(0, 0 if use_q_bn else None)
    )(opt_q_params, opt_q_bs)  # [O,B,A]

    if deterministic_match:
        greedy_OB = jnp.argmax(opt_q_OBA, axis=-1)                         # [O,B]
        elig = (greedy_OB == act[None, :]).astype(jnp.float32)             # [O,B]
    else:
        pi_OBA = jax.nn.softmax(opt_q_OBA, axis=-1)
        elig = jnp.take_along_axis(
            pi_OBA, act[None, :, None], axis=-1
        ).squeeze(-1)                                                      # [O,B]

    # exclude currently executing option
    # exec_opt = meta_action - A                                             # [B]
    # is_exec_opt = (meta_action >= A)                                       # [B]
    # same_exec = (jnp.arange(O)[:, None] == exec_opt[None, :]) & is_exec_opt[None, :]
    # elig = elig * (1.0 - same_exec.astype(jnp.float32))                    # [O,B]

    # --------------------------------------------------
    # 2) Update META network option rows A:(A+O)
    # --------------------------------------------------
    def loss_fn(enc_params, q_params):
        if use_enc_bn:
            emb, enc_mut = train_state.enc_state.apply_fn(
                {"params": enc_params, "batch_stats": train_state.enc_state.batch_stats},
                obs, train=True, mutable=["batch_stats"]
            )
            new_enc_bs = enc_mut["batch_stats"]
        else:
            emb = train_state.enc_state.apply_fn({"params": enc_params}, obs, train=True)
            new_enc_bs = None

        if use_q_bn:
            qv, q_mut = train_state.q_state.apply_fn(
                {"params": q_params, "batch_stats": train_state.q_state.batch_stats},
                emb, train=True, mutable=["batch_stats"]
            )
            new_q_bs = q_mut["batch_stats"]
        else:
            qv = train_state.q_state.apply_fn({"params": q_params}, emb, train=True)
            new_q_bs = None

        meta_opt_q = qv[:, A:A + O].T                                      # [O,B]
        td_OB = meta_opt_q - option_targets                                # [O,B]

        per_opt_loss = jnp.sum(elig * (0.5 * td_OB ** 2), axis=1) / (jnp.sum(elig, axis=1) + 1e-8)
        total_loss = jnp.sum(per_opt_loss)

        q_mean_t = jnp.sum(elig * meta_opt_q, axis=1) / (jnp.sum(elig, axis=1) + 1e-8)
        if use_value_rescale:
            q_mean_raw = jnp.sum(elig * inv_value_rescale(meta_opt_q), axis=1) / (jnp.sum(elig, axis=1) + 1e-8)
        else:
            q_mean_raw = q_mean_t

        elig_any = jnp.mean((jnp.sum(elig, axis=0) > 0).astype(jnp.float32))

        return total_loss, ((new_enc_bs, new_q_bs), per_opt_loss, q_mean_t, q_mean_raw, elig_any)

    (loss, ((new_enc_bs, new_q_bs), per_opt_loss, q_mean_t, q_mean_raw, elig_any)), grads = \
        jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            train_state.enc_state.params,
            train_state.q_state.params,
        )

    new_enc_state = train_state.enc_state.apply_gradients(grads=grads[0])
    new_q_state = train_state.q_state.apply_gradients(grads=grads[1])

    if use_enc_bn and new_enc_bs is not None:
        new_enc_state = new_enc_state.replace(batch_stats=new_enc_bs)
    if use_q_bn and new_q_bs is not None:
        new_q_state = new_q_state.replace(batch_stats=new_q_bs)

    train_state = train_state.replace(
        enc_state=new_enc_state,
        q_state=new_q_state,
        grad_steps=train_state.grad_steps + 1,
    )
    return train_state, (per_opt_loss, q_mean_t, q_mean_raw, elig_any)