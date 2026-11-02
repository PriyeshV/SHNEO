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
    ptr: jnp.ndarray              # [E]
    head: jnp.ndarray             # [E]
    length: jnp.ndarray           # [E]

    start_obs: jnp.ndarray        # [T_max,E,C,H,W]
    term_obs: jnp.ndarray         # [T_max,E,C,H,W]
    actions: jnp.ndarray          # [T_max,E] meta action in [0..A+O-1]

    ext_rewards: jnp.ndarray      # [T_max,E] discounted segment extrinsic return
    int_rewards: jnp.ndarray      # [T_max,E] discounted segment intrinsic return

    ext_discounts: jnp.ndarray    # [T_max,E] gamma_ext^k
    int_discounts: jnp.ndarray    # [T_max,E] gamma_int^k

    dones: jnp.ndarray            # [T_max,E]
    durations: jnp.ndarray        # [T_max,E]
    behav_probs: jnp.ndarray      # [T_max,E]


# ============================================================
# Ring-buffer unwrap
# ============================================================

@partial(jax.jit, static_argnums=(1,))
def unwrap_meta_ring(meta_buf, T_max: int):
    E = meta_buf.ptr.shape[0]
    t = jnp.arange(T_max)[:, None]
    e = jnp.arange(E)[None, :]
    length = meta_buf.length[None, :]
    shift = T_max - length

    valid = t >= shift
    src_k = jnp.maximum(t - shift, 0)
    idx = (meta_buf.head[None, :] + src_k) % T_max

    start_obs = meta_buf.start_obs[idx, e, ...]
    term_obs = meta_buf.term_obs[idx, e, ...]
    actions = meta_buf.actions[idx, e]

    ext_rewards = meta_buf.ext_rewards[idx, e]
    int_rewards = meta_buf.int_rewards[idx, e]

    ext_discounts = meta_buf.ext_discounts[idx, e]
    int_discounts = meta_buf.int_discounts[idx, e]

    dones = meta_buf.dones[idx, e]
    durations = meta_buf.durations[idx, e]
    behav_probs = meta_buf.behav_probs[idx, e]

    ext_rewards = jnp.where(valid, ext_rewards, 0.0)
    int_rewards = jnp.where(valid, int_rewards, 0.0)

    ext_discounts = jnp.where(valid, ext_discounts, 0.0)
    int_discounts = jnp.where(valid, int_discounts, 0.0)

    dones = jnp.where(valid, dones, 1)
    durations = jnp.where(valid, durations, 1)
    behav_probs = jnp.where(valid, behav_probs, 1.0)
    actions = jnp.where(valid, actions, 0)

    return {
        "start_obs": start_obs,
        "term_obs": term_obs,
        "actions": actions,
        "ext_rewards": ext_rewards,
        "int_rewards": int_rewards,
        "ext_discounts": ext_discounts,
        "int_discounts": int_discounts,
        "dones": dones,
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

        ext_rewards=jnp.zeros((t_max, n_total_envs), dtype=jnp.float32),
        int_rewards=jnp.zeros((t_max, n_total_envs), dtype=jnp.float32),

        ext_discounts=jnp.zeros((t_max, n_total_envs), dtype=jnp.float32),
        int_discounts=jnp.zeros((t_max, n_total_envs), dtype=jnp.float32),

        dones=jnp.ones((t_max, n_total_envs), dtype=jnp.int32),
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


@partial(jax.jit, static_argnums=(4, 5, 6))
def get_meta_dual_q_seq(
    start_obs,
    enc_state,
    q_ext_state,
    q_int_state,
    use_enc_bn: bool,
    use_qext_bn: bool,
    use_qint_bn: bool,
):
    """
    start_obs: [T,E,C,H,W]
    returns:
      q_ext_seq: [T,E,A_meta]
      q_int_seq: [T,E,A_meta]
    """
    T, E = start_obs.shape[:2]
    flat = start_obs.reshape((T * E,) + start_obs.shape[2:])
    obs = jnp.transpose(flat, (0, 2, 3, 1))  # NHWC

    enc_vars = {"params": enc_state.params}
    if use_enc_bn:
        enc_vars["batch_stats"] = enc_state.batch_stats
    emb = enc_state.apply_fn(enc_vars, obs, train=False)

    qext_vars = {"params": q_ext_state.params}
    if use_qext_bn:
        qext_vars["batch_stats"] = q_ext_state.batch_stats
    q_ext = q_ext_state.apply_fn(qext_vars, emb, train=False)

    qint_vars = {"params": q_int_state.params}
    if use_qint_bn:
        qint_vars["batch_stats"] = q_int_state.batch_stats
    q_int = q_int_state.apply_fn(qint_vars, emb, train=False)

    return q_ext.reshape((T, E, -1)), q_int.reshape((T, E, -1))


# ============================================================
# Meta SMDP Retrace / TD-lambda target
# ============================================================

@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11))
def compute_meta_retrace_targets(
    actions,
    behav_probs,
    next_q,
    rewards,
    dones,
    discounts,
    q_vals,
    td_lambda,
    use_safe_rho,
    retrace_clip_rho,
    retrace_clip_c,
    use_value_rescale: bool,
):
    pi = jax.nn.softmax(q_vals, axis=-1)
    pi_a = jnp.take_along_axis(pi, actions[..., None], axis=-1).squeeze(-1)

    rho = pi_a / (behav_probs + 1e-8)

    if use_safe_rho:
        rho_bar = jnp.minimum(retrace_clip_rho, rho)
        c = td_lambda * jnp.minimum(retrace_clip_c, rho)
    else:
        rho_bar = rho
        c = td_lambda * rho

    if use_value_rescale:
        q_raw = inv_value_rescale(q_vals)
        next_q_raw = inv_value_rescale(next_q)
    else:
        q_raw = q_vals
        next_q_raw = next_q

    V = jnp.max(q_raw, axis=-1)

    G_last = rewards[-1] + discounts[-1] * (1.0 - dones[-1]) * next_q_raw
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

    targets_raw = jnp.concatenate([Gs, G_last[None]], axis=0)

    targets = value_rescale(targets_raw) if use_value_rescale else targets_raw
    return targets, rho


@partial(jax.jit, static_argnums=range(11, 17))
def compute_meta_retrace_targets_dual(
    actions,
    behav_probs,
    next_q_ext_all,
    next_q_int_all,
    ext_rewards,
    int_rewards,
    dones,
    ext_discounts,
    int_discounts,
    q_ext_vals,
    q_int_vals,
    td_lambda,
    use_safe_rho,
    retrace_clip_rho,
    retrace_clip_c,
    int_coef,
    use_value_rescale,
):
    if use_value_rescale:
        q_ext_raw = inv_value_rescale(q_ext_vals)
        q_int_raw = inv_value_rescale(q_int_vals)
        next_q_ext_raw = inv_value_rescale(next_q_ext_all)
        next_q_int_raw = inv_value_rescale(next_q_int_all)
    else:
        q_ext_raw = q_ext_vals
        q_int_raw = q_int_vals
        next_q_ext_raw = next_q_ext_all
        next_q_int_raw = next_q_int_all

    q_sel_raw = q_ext_raw + int_coef * q_int_raw
    next_q_sel_raw = next_q_ext_raw + int_coef * next_q_int_raw

    # Greedy target policy under joint ext+int selection.
    greedy_actions = jnp.argmax(q_sel_raw, axis=-1)  # [T,E]
    pi_a = (actions == greedy_actions).astype(jnp.float32)

    rho = pi_a / (behav_probs + 1e-8)

    if use_safe_rho:
        c = td_lambda * jnp.minimum(retrace_clip_c, rho)
    else:
        c = td_lambda * rho

    # Current joint-greedy action values.
    a_star = jnp.argmax(q_sel_raw, axis=-1)  # [T,E]

    V_ext = jnp.take_along_axis(
        q_ext_raw, a_star[..., None], axis=-1
    ).squeeze(-1)
    V_int = jnp.take_along_axis(
        q_int_raw, a_star[..., None], axis=-1
    ).squeeze(-1)

    # Actually sampled action values.
    Q_taken_ext = jnp.take_along_axis(
        q_ext_raw, actions[..., None], axis=-1
    ).squeeze(-1)
    Q_taken_int = jnp.take_along_axis(
        q_int_raw, actions[..., None], axis=-1
    ).squeeze(-1)

    # Final bootstrap from joint-greedy action.
    next_a_star = jnp.argmax(next_q_sel_raw, axis=-1)  # [E]

    next_V_ext = jnp.take_along_axis(
        next_q_ext_raw, next_a_star[:, None], axis=-1
    ).squeeze(-1)
    next_V_int = jnp.take_along_axis(
        next_q_int_raw, next_a_star[:, None], axis=-1
    ).squeeze(-1)

    def _one_head_targets(rewards, dones_head, discounts, V, Q_taken, next_V):
        G_last = rewards[-1] + discounts[-1] * (1.0 - dones_head[-1]) * next_V

        def _step(G_next, elems):
            r_t, done_t, disc_t, V_next, Q_next_taken, c_next = elems

            G_t = r_t + disc_t * (1.0 - done_t) * (
                V_next + c_next * (G_next - Q_next_taken)
            )
            G_t = jnp.where(done_t, r_t, G_t)
            return G_t, G_t

        _, Gs = jax.lax.scan(
            _step,
            G_last,
            (
                rewards[:-1],
                dones_head[:-1],
                discounts[:-1],
                V[1:],
                Q_taken[1:],
                c[1:],
            ),
            reverse=True,
        )

        return jnp.concatenate([Gs, G_last[None]], axis=0)

    dones_ext = dones.astype(jnp.float32)
    dones_int = jnp.zeros_like(dones_ext)

    ext_targets_raw = _one_head_targets(
        rewards=ext_rewards,
        dones_head=dones_ext,
        discounts=ext_discounts,
        V=V_ext,
        Q_taken=Q_taken_ext,
        next_V=next_V_ext,
    )

    int_targets_raw = _one_head_targets(
        rewards=int_rewards,
        dones_head=dones_int,
        discounts=int_discounts,
        V=V_int,
        Q_taken=Q_taken_int,
        next_V=next_V_int,
    )

    if use_value_rescale:
        ext_targets = value_rescale(ext_targets_raw)
        int_targets = value_rescale(int_targets_raw)
    else:
        ext_targets = ext_targets_raw
        int_targets = int_targets_raw

    return ext_targets, int_targets, rho


@partial(jax.jit, static_argnums=(5, 6))
def compute_meta_td_lambda_targets_no_retrace(
    next_q_all,          # [E,A_meta]
    rewards,             # [T,E]
    dones,               # [T,E]
    discounts,           # [T,E], already gamma^duration
    q_vals,              # [T,E,A_meta]
    td_lambda: float,
    use_value_rescale: bool,
):
    """
    SMDP TD(lambda) targets for single-head meta-Q.

    No importance sampling.
    No behavior probabilities.
    No Retrace correction.

    Returns:
        targets: [T,E]
    """
    if use_value_rescale:
        q_raw = inv_value_rescale(q_vals)
        next_q_raw = inv_value_rescale(next_q_all)
    else:
        q_raw = q_vals
        next_q_raw = next_q_all

    V = jnp.max(q_raw, axis=-1)          # [T,E]
    next_V_final = jnp.max(next_q_raw, axis=-1)  # [E]

    # V(s_{t+1}) for each meta transition t.
    V_next = jnp.concatenate(
        [V[1:], next_V_final[None, :]],
        axis=0,
    )  # [T,E]

    def _step(G_next, elems):
        r_t, done_t, disc_t, V_next_t = elems

        G_t = r_t + disc_t * (1.0 - done_t) * (
            (1.0 - td_lambda) * V_next_t
            + td_lambda * G_next
        )
        G_t = jnp.where(done_t, r_t, G_t)
        return G_t, G_t

    _, targets_rev = jax.lax.scan(
        _step,
        next_V_final,
        (
            rewards,
            dones.astype(jnp.float32),
            discounts,
            V_next,
        ),
        reverse=True,
    )

    targets_raw = targets_rev
    return value_rescale(targets_raw) if use_value_rescale else targets_raw


@partial(jax.jit, static_argnums=(10, 11, 12))
def compute_meta_td_lambda_targets_dual_no_retrace(
    next_q_ext_all,       # [E,A_meta]
    next_q_int_all,       # [E,A_meta]
    ext_rewards,          # [T,E]
    int_rewards,          # [T,E]
    dones,                # [T,E]
    ext_discounts,        # [T,E], gamma_ext^duration
    int_discounts,        # [T,E], gamma_int^duration
    q_ext_vals,           # [T,E,A_meta]
    q_int_vals,           # [T,E,A_meta]
    td_lambda: float,
    int_coef: float,
    use_value_rescale: bool,
):
    """
    SMDP TD(lambda) targets for dual-head meta-Q.

    Selection is joint-greedy under:
        Q_sel = Q_ext + int_coef * Q_int

    Ext and int heads receive separate TD(lambda) targets using the same
    joint-greedy action selection.

    No importance sampling.
    No behavior probabilities.
    No Retrace correction.

    Returns:
        ext_targets: [T,E]
        int_targets: [T,E]
    """
    if use_value_rescale:
        q_ext_raw = inv_value_rescale(q_ext_vals)
        q_int_raw = inv_value_rescale(q_int_vals)
        next_q_ext_raw = inv_value_rescale(next_q_ext_all)
        next_q_int_raw = inv_value_rescale(next_q_int_all)
    else:
        q_ext_raw = q_ext_vals
        q_int_raw = q_int_vals
        next_q_ext_raw = next_q_ext_all
        next_q_int_raw = next_q_int_all

    q_sel_raw = q_ext_raw + int_coef * q_int_raw
    next_q_sel_raw = next_q_ext_raw + int_coef * next_q_int_raw

    # Joint-greedy actions on sequence states.
    a_star = jnp.argmax(q_sel_raw, axis=-1)  # [T,E]

    V_ext = jnp.take_along_axis(
        q_ext_raw, a_star[..., None], axis=-1
    ).squeeze(-1)  # [T,E]

    V_int = jnp.take_along_axis(
        q_int_raw, a_star[..., None], axis=-1
    ).squeeze(-1)  # [T,E]

    # Joint-greedy bootstrap action on final live state.
    next_a_star = jnp.argmax(next_q_sel_raw, axis=-1)  # [E]

    next_V_ext_final = jnp.take_along_axis(
        next_q_ext_raw, next_a_star[:, None], axis=-1
    ).squeeze(-1)  # [E]

    next_V_int_final = jnp.take_along_axis(
        next_q_int_raw, next_a_star[:, None], axis=-1
    ).squeeze(-1)  # [E]

    # V(s_{t+1}) for each meta transition t.
    V_ext_next = jnp.concatenate(
        [V_ext[1:], next_V_ext_final[None, :]],
        axis=0,
    )  # [T,E]

    V_int_next = jnp.concatenate(
        [V_int[1:], next_V_int_final[None, :]],
        axis=0,
    )  # [T,E]

    def _one_head_targets(rewards, dones_head, discounts, V_next, next_V_final):
        """
        rewards:      [T,E]
        dones_head:   [T,E]
        discounts:    [T,E], SMDP gamma^duration
        V_next:       [T,E], V(s_{t+1}) for each meta transition
        next_V_final: [E], bootstrap V after the last stored transition
        """

        def _step(G_next, elems):
            r_t, done_t, disc_t, V_next_t = elems

            G_t = r_t + disc_t * (1.0 - done_t) * (
                    (1.0 - td_lambda) * V_next_t
                    + td_lambda * G_next
            )
            G_t = jnp.where(done_t, r_t, G_t)
            return G_t, G_t

        _, targets = jax.lax.scan(
            _step,
            next_V_final,
            (
                rewards,
                dones_head.astype(jnp.float32),
                discounts,
                V_next,
            ),
            reverse=True,
        )
        return targets

    dones_ext = dones.astype(jnp.float32)

    # Keep intrinsic continuing across env termination, same as your primitive int target.
    dones_int = jnp.zeros_like(dones_ext)

    ext_targets_raw = _one_head_targets(
        rewards=ext_rewards,
        dones_head=dones_ext,
        discounts=ext_discounts,
        V_next=V_ext_next,
        next_V_final=next_V_ext_final,
    )

    int_targets_raw = _one_head_targets(
        rewards=int_rewards,
        dones_head=dones_int,
        discounts=int_discounts,
        V_next=V_int_next,
        next_V_final=next_V_int_final,
    )

    if use_value_rescale:
        ext_targets = value_rescale(ext_targets_raw)
        int_targets = value_rescale(int_targets_raw)
    else:
        ext_targets = ext_targets_raw
        int_targets = int_targets_raw

    return ext_targets, int_targets


@partial(jax.jit, static_argnums=(11, 12, 13))
def compute_meta_td_lambda_targets_dual_no_retrace2(
    next_q_ext_all,       # [E,A_meta]
    next_q_int_all,       # [E,A_meta]
    ext_rewards,          # [T,E]
    int_rewards,          # [T,E]
    dones,                # [T,E]
    ext_discounts,        # [T,E] = gamma_ext^duration
    int_discounts,        # [T,E] = gamma_int^duration
    durations,            # [T,E]
    q_ext_vals,           # [T,E,A_meta]
    q_int_vals,           # [T,E,A_meta]
    valid,                # [T,E]
    td_lambda: float,
    int_coef: float,
    use_value_rescale: bool,
):
    """
    SMDP TD(lambda) targets for dual-head meta-Q.

    Uses lambda^duration because one meta transition may span k primitive
    transitions. Targets are computed in raw value space, then transformed
    if use_value_rescale=True.
    """
    if use_value_rescale:
        q_ext_raw = inv_value_rescale(q_ext_vals)
        q_int_raw = inv_value_rescale(q_int_vals)
        next_q_ext_raw = inv_value_rescale(next_q_ext_all)
        next_q_int_raw = inv_value_rescale(next_q_int_all)
    else:
        q_ext_raw = q_ext_vals
        q_int_raw = q_int_vals
        next_q_ext_raw = next_q_ext_all
        next_q_int_raw = next_q_int_all

    q_sel_raw = q_ext_raw + int_coef * q_int_raw
    next_q_sel_raw = next_q_ext_raw + int_coef * next_q_int_raw

    a_star = jnp.argmax(q_sel_raw, axis=-1)  # [T,E]

    V_ext = jnp.take_along_axis(
        q_ext_raw, a_star[..., None], axis=-1
    ).squeeze(-1)

    V_int = jnp.take_along_axis(
        q_int_raw, a_star[..., None], axis=-1
    ).squeeze(-1)

    next_a_star = jnp.argmax(next_q_sel_raw, axis=-1)  # [E]

    next_V_ext_final = jnp.take_along_axis(
        next_q_ext_raw, next_a_star[:, None], axis=-1
    ).squeeze(-1)

    next_V_int_final = jnp.take_along_axis(
        next_q_int_raw, next_a_star[:, None], axis=-1
    ).squeeze(-1)

    V_ext_next = jnp.concatenate([V_ext[1:], next_V_ext_final[None, :]], axis=0)
    V_int_next = jnp.concatenate([V_int[1:], next_V_int_final[None, :]], axis=0)

    durations_f = jnp.maximum(durations.astype(jnp.float32), 1.0)
    lambda_decay = td_lambda ** durations_f
    lambda_decay = jnp.where(valid > 0, lambda_decay, 0.0)

    def _one_head_targets(rewards, dones_head, discounts, V_next, next_V_final):
        def _step(G_next, elems):
            r_t, done_t, disc_t, lam_t, V_next_t, valid_t = elems

            G_t = r_t + disc_t * (1.0 - done_t) * (
                (1.0 - lam_t) * V_next_t + lam_t * G_next
            )

            G_t = jnp.where(done_t > 0, r_t, G_t)

            # Invalid padded entries should not leak bootstraps/returns.
            G_t = jnp.where(valid_t > 0, G_t, 0.0)
            return G_t, G_t

        _, targets = jax.lax.scan(
            _step,
            next_V_final,
            (
                rewards,
                dones_head.astype(jnp.float32),
                discounts,
                lambda_decay,
                V_next,
                valid,
            ),
            reverse=True,
        )
        return targets

    dones_ext = dones.astype(jnp.float32)
    dones_int = jnp.zeros_like(dones_ext)

    ext_targets_raw = _one_head_targets(
        rewards=ext_rewards,
        dones_head=dones_ext,
        discounts=ext_discounts,
        V_next=V_ext_next,
        next_V_final=next_V_ext_final,
    )

    int_targets_raw = _one_head_targets(
        rewards=int_rewards,
        dones_head=dones_int,
        discounts=int_discounts,
        V_next=V_int_next,
        next_V_final=next_V_int_final,
    )

    if use_value_rescale:
        ext_targets = value_rescale(ext_targets_raw)
        int_targets = value_rescale(int_targets_raw)
    else:
        ext_targets = ext_targets_raw
        int_targets = int_targets_raw

    return ext_targets, int_targets


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
    valid = minibatch.valid.astype(jnp.float32)

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

        actions = minibatch.actions
        greedy = jnp.argmax(qv, axis=-1)

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


@partial(jax.jit, static_argnums=range(2, 8))
def smdp_learn_phase_dual(
    carry,
    minibatch_and_targets,
    use_enc_bn: bool,
    use_q_bn: bool,
    use_value_rescale: bool,
    int_coef: float,
    n_meta_actions: int,
    n_prim_actions: int,
):
    """
    Dual-head SMDP meta update on executed meta actions.

    Expects:
      minibatch_and_targets = (minibatch, ext_target, int_target)

    train_state must contain:
      enc_state, q_ext_state, q_int_state
    """
    train_state, _key = carry
    minibatch, ext_target, int_target = minibatch_and_targets
    obs = jnp.transpose(minibatch.obs, (0, 2, 3, 1))
    valid = minibatch.valid.astype(jnp.float32)

    A = n_prim_actions

    def _loss_fn(enc_params, qext_params, qint_params):
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
            q_ext_vals, qext_mut = train_state.q_ext_state.apply_fn(
                {"params": qext_params, "batch_stats": train_state.q_ext_state.batch_stats},
                emb,
                train=True,
                mutable=["batch_stats"],
            )
            q_int_vals, qint_mut = train_state.q_int_state.apply_fn(
                {"params": qint_params, "batch_stats": train_state.q_int_state.batch_stats},
                emb,
                train=True,
                mutable=["batch_stats"],
            )
            new_qext_bs = qext_mut["batch_stats"]
            new_qint_bs = qint_mut["batch_stats"]
        else:
            q_ext_vals = train_state.q_ext_state.apply_fn({"params": qext_params}, emb, train=True)
            q_int_vals = train_state.q_int_state.apply_fn({"params": qint_params}, emb, train=True)
            new_qext_bs = None
            new_qint_bs = None

        aidx = minibatch.actions[:, None]
        q_ext_pred = jnp.take_along_axis(q_ext_vals, aidx, axis=-1).squeeze(-1)
        q_int_pred = jnp.take_along_axis(q_int_vals, aidx, axis=-1).squeeze(-1)

        ext_sq = 0.5 * jnp.square(q_ext_pred - ext_target)
        int_sq = 0.5 * jnp.square(q_int_pred - int_target)

        ext_loss = jnp.sum(valid * ext_sq) / (jnp.sum(valid) + 1e-8)
        int_loss = jnp.sum(valid * int_sq) / (jnp.sum(valid) + 1e-8)
        total_loss = ext_loss + int_coef * int_loss

        q_ext_mean_t = jnp.sum(valid * q_ext_pred) / (jnp.sum(valid) + 1e-8)
        q_int_mean_t = jnp.sum(valid * q_int_pred) / (jnp.sum(valid) + 1e-8)

        if use_value_rescale:
            q_ext_mean_raw = jnp.sum(valid * inv_value_rescale(q_ext_pred)) / (jnp.sum(valid) + 1e-8)
            q_int_mean_raw = jnp.sum(valid * inv_value_rescale(q_int_pred)) / (jnp.sum(valid) + 1e-8)

            q_sel_pred = value_rescale(inv_value_rescale(q_ext_pred) + int_coef * inv_value_rescale(q_int_pred))
            q_sel_raw = inv_value_rescale(q_ext_pred) + int_coef * inv_value_rescale(q_int_pred)
            q_sel_mean_t = jnp.sum(valid * q_sel_pred) / (jnp.sum(valid) + 1e-8)
            q_sel_mean_raw = jnp.sum(valid * q_sel_raw) / (jnp.sum(valid) + 1e-8)

            q_ext_raw_all = inv_value_rescale(q_ext_vals)
            q_int_raw_all = inv_value_rescale(q_int_vals)
            q_sel_all = value_rescale(q_ext_raw_all + int_coef * q_int_raw_all)
        else:
            q_ext_mean_raw = q_ext_mean_t
            q_int_mean_raw = q_int_mean_t

            q_sel_raw = q_ext_pred + int_coef * q_int_pred
            q_sel_mean_raw = jnp.sum(valid * q_sel_raw) / (jnp.sum(valid) + 1e-8)
            q_sel_mean_t = q_sel_mean_raw

            q_sel_all = q_ext_vals + int_coef * q_int_vals

        actions = minibatch.actions
        greedy = jnp.argmax(q_sel_all, axis=-1)

        taken_is_option = (actions >= A).astype(jnp.float32)
        greedy_is_option = (greedy >= A).astype(jnp.float32)

        option_taken_count = jnp.sum(valid * taken_is_option)
        primitive_taken_count = jnp.sum(valid * (1.0 - taken_is_option))

        option_greedy_count = jnp.sum(valid * greedy_is_option)
        primitive_greedy_count = jnp.sum(valid * (1.0 - greedy_is_option))

        aux = (
            (new_enc_bs, new_qext_bs, new_qint_bs),
            (
                ext_loss, int_loss,
                q_ext_mean_t, q_ext_mean_raw,
                q_int_mean_t, q_int_mean_raw,
                q_sel_mean_t, q_sel_mean_raw,
            ),
            option_taken_count,
            primitive_taken_count,
            option_greedy_count,
            primitive_greedy_count,
            jnp.sum(valid),
        )
        return total_loss, aux

    (loss, (
        (new_enc_bs, new_qext_bs, new_qint_bs),
        q_stats,
        option_taken_count,
        primitive_taken_count,
        option_greedy_count,
        primitive_greedy_count,
        valid_count,
    )), grads = jax.value_and_grad(
        _loss_fn, argnums=(0, 1, 2), has_aux=True
    )(
        train_state.enc_state.params,
        train_state.q_ext_state.params,
        train_state.q_int_state.params,
    )

    new_enc_state = train_state.enc_state.apply_gradients(grads=grads[0])
    new_qext_state = train_state.q_ext_state.apply_gradients(grads=grads[1])
    new_qint_state = train_state.q_int_state.apply_gradients(grads=grads[2])

    if use_enc_bn and new_enc_bs is not None:
        new_enc_state = new_enc_state.replace(batch_stats=new_enc_bs)
    if use_q_bn and new_qext_bs is not None:
        new_qext_state = new_qext_state.replace(batch_stats=new_qext_bs)
    if use_q_bn and new_qint_bs is not None:
        new_qint_state = new_qint_state.replace(batch_stats=new_qint_bs)

    train_state = train_state.replace(
        enc_state=new_enc_state,
        q_ext_state=new_qext_state,
        q_int_state=new_qint_state,
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


@partial(jax.jit, static_argnums=range(2, 8))
def smdp_learn_phase_dual2(
    carry,
    minibatch_and_targets,
    use_enc_bn: bool,
    use_q_bn: bool,
    use_value_rescale: bool,
    int_coef: float,
    n_meta_actions: int,
    n_prim_actions: int,
):
    """
    Dual-head SMDP meta update.

    Important:
      - Targets may be computed using the full meta TD(lambda) sequence.
      - Loss is applied only to option meta-actions, i.e. actions >= n_prim_actions
      - Primitive rows are updated separately by primitive_learn_phase_dual.

    Expects:
      minibatch_and_targets = (minibatch, ext_target, int_target)

    train_state must contain:
      enc_state, q_ext_state, q_int_state
    """
    train_state, _key = carry
    minibatch, ext_target, int_target = minibatch_and_targets

    obs = jnp.transpose(minibatch.obs, (0, 2, 3, 1))

    A = n_prim_actions

    # Padding-valid mask from the meta ring buffer.
    valid_all = minibatch.valid.astype(jnp.float32)

    # SMDP loss mask: update only option rows.
    is_option_action = (minibatch.actions >= A).astype(jnp.float32)
    valid_loss = valid_all * is_option_action

    def _safe_mean(x, mask):
        return jnp.sum(mask * x) / (jnp.sum(mask) + 1e-8)

    def _loss_fn(enc_params, qext_params, qint_params):
        if use_enc_bn:
            emb, enc_mut = train_state.enc_state.apply_fn(
                {
                    "params": enc_params,
                    "batch_stats": train_state.enc_state.batch_stats,
                },
                obs,
                train=True,
                mutable=["batch_stats"],
            )
            new_enc_bs = enc_mut["batch_stats"]
        else:
            emb = train_state.enc_state.apply_fn(
                {"params": enc_params},
                obs,
                train=True,
            )
            new_enc_bs = None

        if use_q_bn:
            q_ext_vals, qext_mut = train_state.q_ext_state.apply_fn(
                {
                    "params": qext_params,
                    "batch_stats": train_state.q_ext_state.batch_stats,
                },
                emb,
                train=True,
                mutable=["batch_stats"],
            )
            q_int_vals, qint_mut = train_state.q_int_state.apply_fn(
                {
                    "params": qint_params,
                    "batch_stats": train_state.q_int_state.batch_stats,
                },
                emb,
                train=True,
                mutable=["batch_stats"],
            )
            new_qext_bs = qext_mut["batch_stats"]
            new_qint_bs = qint_mut["batch_stats"]
        else:
            q_ext_vals = train_state.q_ext_state.apply_fn(
                {"params": qext_params},
                emb,
                train=True,
            )
            q_int_vals = train_state.q_int_state.apply_fn(
                {"params": qint_params},
                emb,
                train=True,
            )
            new_qext_bs = None
            new_qint_bs = None

        aidx = minibatch.actions[:, None]

        q_ext_pred = jnp.take_along_axis(
            q_ext_vals,
            aidx,
            axis=-1,
        ).squeeze(-1)

        q_int_pred = jnp.take_along_axis(
            q_int_vals,
            aidx,
            axis=-1,
        ).squeeze(-1)

        ext_sq = 0.5 * jnp.square(q_ext_pred - ext_target)
        int_sq = 0.5 * jnp.square(q_int_pred - int_target)

        # Critical change:
        # SMDP loss updates only option rows.
        ext_loss = _safe_mean(ext_sq, valid_loss)
        int_loss = _safe_mean(int_sq, valid_loss)

        # Since your algo.int_coef = 0, this becomes ext_loss only.
        total_loss = ext_loss + int_coef * int_loss

        q_ext_mean_t = _safe_mean(q_ext_pred, valid_loss)
        q_int_mean_t = _safe_mean(q_int_pred, valid_loss)

        if use_value_rescale:
            q_ext_pred_raw = inv_value_rescale(q_ext_pred)
            q_int_pred_raw = inv_value_rescale(q_int_pred)

            q_ext_mean_raw = _safe_mean(q_ext_pred_raw, valid_loss)
            q_int_mean_raw = _safe_mean(q_int_pred_raw, valid_loss)

            q_sel_pred_raw = q_ext_pred_raw + int_coef * q_int_pred_raw
            q_sel_pred_t = value_rescale(q_sel_pred_raw)

            q_sel_mean_raw = _safe_mean(q_sel_pred_raw, valid_loss)
            q_sel_mean_t = _safe_mean(q_sel_pred_t, valid_loss)

            q_ext_raw_all = inv_value_rescale(q_ext_vals)
            q_int_raw_all = inv_value_rescale(q_int_vals)
            q_sel_all = value_rescale(q_ext_raw_all + int_coef * q_int_raw_all)
        else:
            q_ext_mean_raw = q_ext_mean_t
            q_int_mean_raw = q_int_mean_t

            q_sel_pred_raw = q_ext_pred + int_coef * q_int_pred
            q_sel_mean_raw = _safe_mean(q_sel_pred_raw, valid_loss)
            q_sel_mean_t = q_sel_mean_raw

            q_sel_all = q_ext_vals + int_coef * q_int_vals

        actions = minibatch.actions
        greedy = jnp.argmax(q_sel_all, axis=-1)

        taken_is_option = (actions >= A).astype(jnp.float32)
        greedy_is_option = (greedy >= A).astype(jnp.float32)

        # Logging uses valid_all, not valid_loss, so you can still see
        # what the meta sequence contains.
        option_taken_count = jnp.sum(valid_all * taken_is_option)
        primitive_taken_count = jnp.sum(valid_all * (1.0 - taken_is_option))

        option_greedy_count = jnp.sum(valid_all * greedy_is_option)
        primitive_greedy_count = jnp.sum(valid_all * (1.0 - greedy_is_option))

        option_loss_valid_count = jnp.sum(valid_loss)
        valid_count = jnp.sum(valid_all)

        aux = (
            (new_enc_bs, new_qext_bs, new_qint_bs),
            (
                ext_loss,
                int_loss,
                q_ext_mean_t,
                q_ext_mean_raw,
                q_int_mean_t,
                q_int_mean_raw,
                q_sel_mean_t,
                q_sel_mean_raw,
            ),
            option_taken_count,
            primitive_taken_count,
            option_greedy_count,
            primitive_greedy_count,
            valid_count,
            option_loss_valid_count,
        )
        return total_loss, aux

    (
        loss,
        (
            (new_enc_bs, new_qext_bs, new_qint_bs),
            q_stats,
            option_taken_count,
            primitive_taken_count,
            option_greedy_count,
            primitive_greedy_count,
            valid_count,
            option_loss_valid_count,
        ),
    ), grads = jax.value_and_grad(
        _loss_fn,
        argnums=(0, 1, 2),
        has_aux=True,
    )(
        train_state.enc_state.params,
        train_state.q_ext_state.params,
        train_state.q_int_state.params,
    )

    new_enc_state = train_state.enc_state.apply_gradients(grads=grads[0])
    new_qext_state = train_state.q_ext_state.apply_gradients(grads=grads[1])
    new_qint_state = train_state.q_int_state.apply_gradients(grads=grads[2])

    if use_enc_bn and new_enc_bs is not None:
        new_enc_state = new_enc_state.replace(batch_stats=new_enc_bs)
    if use_q_bn and new_qext_bs is not None:
        new_qext_state = new_qext_state.replace(batch_stats=new_qext_bs)
    if use_q_bn and new_qint_bs is not None:
        new_qint_state = new_qint_state.replace(batch_stats=new_qint_bs)

    train_state = train_state.replace(
        enc_state=new_enc_state,
        q_ext_state=new_qext_state,
        q_int_state=new_qint_state,
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
        option_loss_valid_count,
    )


# ============================================================
# Update 2: primitive update on primitive executed actions
# ============================================================

@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def primitive_learn_phase(
    carry,
    minibatch_and_target,
    use_enc_bn: bool,
    use_q_bn: bool,
    use_value_rescale: bool,
    n_primitives: int,
):
    """
    Regresses only primitive columns of q_state.
    minibatch.actions are primitive actions in [0..A-1].
    """
    train_state, _key = carry
    minibatch, target = minibatch_and_target
    obs = jnp.transpose(minibatch.obs, (0, 2, 3, 1))
    A = n_primitives

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


@partial(jax.jit, static_argnums=range(2, 7))
def primitive_learn_phase_dual(
    carry,
    minibatch_and_targets,
    use_enc_bn: bool,
    use_q_bn: bool,
    use_value_rescale: bool,
    int_coef: float,
    n_primitives: int,
):
    """
    Dual primitive-row update on the primitive subset of the meta heads.

    Expects:
      minibatch_and_targets = (minibatch, ext_target, int_target)

    train_state must contain:
      enc_state, q_ext_state, q_int_state
    """
    train_state, _key = carry
    minibatch, ext_target, int_target = minibatch_and_targets
    obs = jnp.transpose(minibatch.obs, (0, 2, 3, 1))
    A = n_primitives

    def _loss_fn(enc_params, qext_params, qint_params):
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
            q_ext_vals, qext_mut = train_state.q_ext_state.apply_fn(
                {"params": qext_params, "batch_stats": train_state.q_ext_state.batch_stats},
                emb, train=True, mutable=["batch_stats"]
            )
            q_int_vals, qint_mut = train_state.q_int_state.apply_fn(
                {"params": qint_params, "batch_stats": train_state.q_int_state.batch_stats},
                emb, train=True, mutable=["batch_stats"]
            )
            new_qext_bs = qext_mut["batch_stats"]
            new_qint_bs = qint_mut["batch_stats"]
        else:
            q_ext_vals = train_state.q_ext_state.apply_fn({"params": qext_params}, emb, train=True)
            q_int_vals = train_state.q_int_state.apply_fn({"params": qint_params}, emb, train=True)
            new_qext_bs = None
            new_qint_bs = None

        prim_q_ext = q_ext_vals[:, :A]
        prim_q_int = q_int_vals[:, :A]

        aidx = minibatch.actions[:, None]
        q_ext_pred = jnp.take_along_axis(prim_q_ext, aidx, axis=-1).squeeze(-1)
        q_int_pred = jnp.take_along_axis(prim_q_int, aidx, axis=-1).squeeze(-1)

        ext_loss = 0.5 * jnp.mean(jnp.square(q_ext_pred - ext_target))
        int_loss = 0.5 * jnp.mean(jnp.square(q_int_pred - int_target))
        total_loss = ext_loss + int_coef * int_loss

        q_ext_mean_t = jnp.mean(q_ext_pred)
        q_int_mean_t = jnp.mean(q_int_pred)

        if use_value_rescale:
            q_ext_mean_raw = jnp.mean(inv_value_rescale(q_ext_pred))
            q_int_mean_raw = jnp.mean(inv_value_rescale(q_int_pred))
            q_sel_mean_raw = jnp.mean(inv_value_rescale(q_ext_pred) + int_coef * inv_value_rescale(q_int_pred))
            q_sel_mean_t = jnp.mean(value_rescale(inv_value_rescale(q_ext_pred) + int_coef * inv_value_rescale(q_int_pred)))
        else:
            q_ext_mean_raw = q_ext_mean_t
            q_int_mean_raw = q_int_mean_t
            q_sel_mean_raw = jnp.mean(q_ext_pred + int_coef * q_int_pred)
            q_sel_mean_t = q_sel_mean_raw

        aux = (
            (new_enc_bs, new_qext_bs, new_qint_bs),
            (
                ext_loss, int_loss,
                q_ext_mean_t, q_ext_mean_raw,
                q_int_mean_t, q_int_mean_raw,
                q_sel_mean_t, q_sel_mean_raw,
            )
        )
        return total_loss, aux

    (loss, ((new_enc_bs, new_qext_bs, new_qint_bs), q_stats)), grads = jax.value_and_grad(
        _loss_fn, argnums=(0, 1, 2), has_aux=True
    )(
        train_state.enc_state.params,
        train_state.q_ext_state.params,
        train_state.q_int_state.params,
    )

    new_enc_state = train_state.enc_state.apply_gradients(grads=grads[0])
    new_qext_state = train_state.q_ext_state.apply_gradients(grads=grads[1])
    new_qint_state = train_state.q_int_state.apply_gradients(grads=grads[2])

    if use_enc_bn and new_enc_bs is not None:
        new_enc_state = new_enc_state.replace(batch_stats=new_enc_bs)
    if use_q_bn and new_qext_bs is not None:
        new_qext_state = new_qext_state.replace(batch_stats=new_qext_bs)
    if use_q_bn and new_qint_bs is not None:
        new_qint_state = new_qint_state.replace(batch_stats=new_qint_bs)

    train_state = train_state.replace(
        enc_state=new_enc_state,
        q_ext_state=new_qext_state,
        q_int_state=new_qint_state,
        grad_steps=train_state.grad_steps + 1,
    )
    return (train_state, _key), (loss, q_stats)


# ============================================================
# Update 3: intra-option Q-learning update
# ============================================================

@partial(jax.jit, static_argnums=(3, 4, 5))
def compute_intra_option_meta_targets_1step(
    transitions,          # Prim_Transition [T,E,...]
    next_meta_q_all,      # [E, A+O]
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
    rewards = transitions.rewards
    dones = transitions.dones.astype(jnp.float32)
    meta_q = transitions.meta_q_vals

    A = n_prim_actions
    O = opt_dur_table.shape[0]

    if use_value_rescale:
        meta_q_raw = inv_value_rescale(meta_q)
        next_meta_q_raw = inv_value_rescale(next_meta_q_all)
    else:
        meta_q_raw = meta_q
        next_meta_q_raw = next_meta_q_all

    q_opt_rows = meta_q_raw[:, :, A:A + O]                 # [T,E,O]
    v_rows = jnp.max(meta_q_raw, axis=-1)                  # [T,E]

    q_opt_next = jnp.concatenate(
        [q_opt_rows[1:], next_meta_q_raw[None, :, A:A + O]],
        axis=0
    )
    v_next = jnp.concatenate(
        [v_rows[1:], jnp.max(next_meta_q_raw, axis=-1)[None, :]],
        axis=0
    )

    dur = jnp.maximum(opt_dur_table.astype(jnp.float32), 1.0)
    beta_o = 1.0 / dur

    targets = rewards[:, :, None] + gamma * (1.0 - dones)[:, :, None] * (
        (1.0 - beta_o[None, None, :]) * q_opt_next
        + beta_o[None, None, :] * v_next[:, :, None]
    )

    targets = jnp.transpose(targets, (2, 0, 1))
    if use_value_rescale:
        targets = value_rescale(targets)
    return targets


@partial(jax.jit, static_argnums=(5, 6, 7))
def compute_intra_option_meta_targets_1step_dual(
    rewards_ext,          # [T,E]
    rewards_int,          # [T,E]
    dones_ext,            # [T,E]
    q_ext_vals,           # [T,E,A+O]
    q_int_vals,           # [T,E,A+O]
    next_q_ext_all,       # [E,A+O]
    next_q_int_all,       # [E,A+O]
    opt_dur_table,        # [O]
    gamma_ext: float,
    gamma_int: float,
    n_prim_actions: int,
    use_value_rescale: bool,
):
    """
    Dual one-step targets for meta-option rows Q_meta_ext(s, A+o) and Q_meta_int(s, A+o).

    Returns:
      ext_targets: [O,T,E]
      int_targets: [O,T,E]
    """
    A = n_prim_actions
    O = opt_dur_table.shape[0]

    dones_ext = dones_ext.astype(jnp.float32)
    dones_int = jnp.zeros_like(dones_ext)

    if use_value_rescale:
        q_ext_raw = inv_value_rescale(q_ext_vals)
        q_int_raw = inv_value_rescale(q_int_vals)
        next_q_ext_raw = inv_value_rescale(next_q_ext_all)
        next_q_int_raw = inv_value_rescale(next_q_int_all)
    else:
        q_ext_raw = q_ext_vals
        q_int_raw = q_int_vals
        next_q_ext_raw = next_q_ext_all
        next_q_int_raw = next_q_int_all

    dur = jnp.maximum(opt_dur_table.astype(jnp.float32), 1.0)
    beta_o = 1.0 / dur

    def _one_head_targets(rewards, dones, gamma, q_raw, next_q_raw):
        q_opt_rows = q_raw[:, :, A:A + O]                        # [T,E,O]
        v_rows = jnp.max(q_raw, axis=-1)                         # [T,E]

        q_opt_next = jnp.concatenate(
            [q_opt_rows[1:], next_q_raw[None, :, A:A + O]],
            axis=0
        )
        v_next = jnp.concatenate(
            [v_rows[1:], jnp.max(next_q_raw, axis=-1)[None, :]],
            axis=0
        )

        targets = rewards[:, :, None] + gamma * (1.0 - dones)[:, :, None] * (
            (1.0 - beta_o[None, None, :]) * q_opt_next
            + beta_o[None, None, :] * v_next[:, :, None]
        )
        targets = jnp.transpose(targets, (2, 0, 1))             # [O,T,E]
        return value_rescale(targets) if use_value_rescale else targets

    ext_targets = _one_head_targets(rewards_ext, dones_ext, gamma_ext, q_ext_raw, next_q_ext_raw)
    int_targets = _one_head_targets(rewards_int, dones_int, gamma_int, q_int_raw, next_q_int_raw)
    return ext_targets, int_targets


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
    """
    obs = jnp.transpose(transitions.obs, (0, 2, 3, 1))
    act = transitions.actions

    A = n_prim_actions
    O = option_targets.shape[0]

    opt_enc_vars = {"params": train_state.opt_enc_state.params}
    if use_enc_bn:
        opt_enc_vars["batch_stats"] = train_state.opt_enc_state.batch_stats
    opt_emb = train_state.opt_enc_state.apply_fn(opt_enc_vars, obs, train=False)

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
        return opt_q_apply(vars_q, opt_emb, train=False)

    opt_q_OBA = jax.vmap(
        one_opt_head, in_axes=(0, 0 if use_q_bn else None)
    )(opt_q_params, opt_q_bs)

    if deterministic_match:
        greedy_OB = jnp.argmax(opt_q_OBA, axis=-1)
        elig = (greedy_OB == act[None, :]).astype(jnp.float32)
    else:
        pi_OBA = jax.nn.softmax(opt_q_OBA, axis=-1)
        elig = jnp.take_along_axis(
            pi_OBA, act[None, :, None], axis=-1
        ).squeeze(-1)

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

        meta_opt_q = qv[:, A:A + O].T
        td_OB = meta_opt_q - option_targets

        per_opt_loss = jnp.sum(elig * (0.5 * td_OB ** 2), axis=1) / (jnp.sum(elig, axis=1) + 1e-8)
        total_loss = jnp.sum(per_opt_loss)

        q_mean_t = jnp.sum(elig * meta_opt_q, axis=1) / (jnp.sum(elig, axis=1) + 1e-8)
        if use_value_rescale:
            q_mean_raw = jnp.sum(elig * inv_value_rescale(meta_opt_q), axis=1) / (
                jnp.sum(elig, axis=1) + 1e-8
            )
        else:
            q_mean_raw = q_mean_t

        elig_any = jnp.mean((jnp.sum(elig, axis=0) > 0).astype(jnp.float32))
        elig_count = jnp.sum(elig, axis=1)

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


@partial(jax.jit, static_argnums=range(4, 10))
def update_intra_option_meta_shared_dual(
    train_state,
    transitions,          # flattened primitive batch
    ext_option_targets,   # [O,B]
    int_option_targets,   # [O,B]
    n_prim_actions: int,
    use_enc_bn: bool,
    use_q_bn: bool,
    use_value_rescale: bool,
    int_coef: float,
    deterministic_match: bool = True,
):
    """
    Dual intra-option meta update:
      updates q_ext_state and q_int_state option rows A:(A+O)

    train_state must contain:
      enc_state, q_ext_state, q_int_state, opt_enc_state, opt_q_states
    """
    obs = jnp.transpose(transitions.obs, (0, 2, 3, 1))
    act = transitions.actions

    A = n_prim_actions
    O = ext_option_targets.shape[0]

    opt_enc_vars = {"params": train_state.opt_enc_state.params}
    if use_enc_bn:
        opt_enc_vars["batch_stats"] = train_state.opt_enc_state.batch_stats
    opt_emb = train_state.opt_enc_state.apply_fn(opt_enc_vars, obs, train=False)

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
        return opt_q_apply(vars_q, opt_emb, train=False)

    opt_q_OBA = jax.vmap(
        one_opt_head, in_axes=(0, 0 if use_q_bn else None)
    )(opt_q_params, opt_q_bs)

    if deterministic_match:
        greedy_OB = jnp.argmax(opt_q_OBA, axis=-1)
        elig = (greedy_OB == act[None, :]).astype(jnp.float32)
    else:
        pi_OBA = jax.nn.softmax(opt_q_OBA, axis=-1)
        elig = jnp.take_along_axis(
            pi_OBA, act[None, :, None], axis=-1
        ).squeeze(-1)

    def loss_fn(enc_params, qext_params, qint_params):
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
            q_ext_vals, qext_mut = train_state.q_ext_state.apply_fn(
                {"params": qext_params, "batch_stats": train_state.q_ext_state.batch_stats},
                emb, train=True, mutable=["batch_stats"]
            )
            q_int_vals, qint_mut = train_state.q_int_state.apply_fn(
                {"params": qint_params, "batch_stats": train_state.q_int_state.batch_stats},
                emb, train=True, mutable=["batch_stats"]
            )
            new_qext_bs = qext_mut["batch_stats"]
            new_qint_bs = qint_mut["batch_stats"]
        else:
            q_ext_vals = train_state.q_ext_state.apply_fn({"params": qext_params}, emb, train=True)
            q_int_vals = train_state.q_int_state.apply_fn({"params": qint_params}, emb, train=True)
            new_qext_bs = None
            new_qint_bs = None

        meta_opt_q_ext = q_ext_vals[:, A:A + O].T
        meta_opt_q_int = q_int_vals[:, A:A + O].T

        td_ext = meta_opt_q_ext - ext_option_targets
        td_int = meta_opt_q_int - int_option_targets

        per_opt_ext_loss = jnp.sum(elig * (0.5 * td_ext ** 2), axis=1) / (jnp.sum(elig, axis=1) + 1e-8)
        per_opt_int_loss = jnp.sum(elig * (0.5 * td_int ** 2), axis=1) / (jnp.sum(elig, axis=1) + 1e-8)
        per_opt_total_loss = per_opt_ext_loss + int_coef * per_opt_int_loss
        total_loss = jnp.sum(per_opt_total_loss)

        q_ext_mean_t = jnp.sum(elig * meta_opt_q_ext, axis=1) / (jnp.sum(elig, axis=1) + 1e-8)
        q_int_mean_t = jnp.sum(elig * meta_opt_q_int, axis=1) / (jnp.sum(elig, axis=1) + 1e-8)

        if use_value_rescale:
            q_ext_mean_raw = jnp.sum(elig * inv_value_rescale(meta_opt_q_ext), axis=1) / (jnp.sum(elig, axis=1) + 1e-8)
            q_int_mean_raw = jnp.sum(elig * inv_value_rescale(meta_opt_q_int), axis=1) / (jnp.sum(elig, axis=1) + 1e-8)
        else:
            q_ext_mean_raw = q_ext_mean_t
            q_int_mean_raw = q_int_mean_t

        elig_any = jnp.mean((jnp.sum(elig, axis=0) > 0).astype(jnp.float32))
        elig_count = jnp.sum(elig, axis=1)

        aux = (
            (new_enc_bs, new_qext_bs, new_qint_bs),
            per_opt_ext_loss,
            per_opt_int_loss,
            q_ext_mean_t,
            q_ext_mean_raw,
            q_int_mean_t,
            q_int_mean_raw,
            elig_any,
            elig_count,
        )
        return total_loss, aux

    (loss, (
        (new_enc_bs, new_qext_bs, new_qint_bs),
        per_opt_ext_loss,
        per_opt_int_loss,
        q_ext_mean_t,
        q_ext_mean_raw,
        q_int_mean_t,
        q_int_mean_raw,
        elig_any,
        elig_count,
    )), grads = jax.value_and_grad(
        loss_fn, argnums=(0, 1, 2), has_aux=True
    )(
        train_state.enc_state.params,
        train_state.q_ext_state.params,
        train_state.q_int_state.params,
    )

    new_enc_state = train_state.enc_state.apply_gradients(grads=grads[0])
    new_qext_state = train_state.q_ext_state.apply_gradients(grads=grads[1])
    new_qint_state = train_state.q_int_state.apply_gradients(grads=grads[2])

    if use_enc_bn and new_enc_bs is not None:
        new_enc_state = new_enc_state.replace(batch_stats=new_enc_bs)
    if use_q_bn and new_qext_bs is not None:
        new_qext_state = new_qext_state.replace(batch_stats=new_qext_bs)
    if use_q_bn and new_qint_bs is not None:
        new_qint_state = new_qint_state.replace(batch_stats=new_qint_bs)

    train_state = train_state.replace(
        enc_state=new_enc_state,
        q_ext_state=new_qext_state,
        q_int_state=new_qint_state,
        grad_steps=train_state.grad_steps + 1,
    )
    return train_state, (
        per_opt_ext_loss,
        per_opt_int_loss,
        q_ext_mean_t,
        q_ext_mean_raw,
        q_int_mean_t,
        q_int_mean_raw,
        elig_any,
        elig_count,
    )


@partial(jax.jit, static_argnums=range(3, 8))
def update_intra_option_meta_shared_exclude_exec(
    train_state,
    transitions,
    option_targets,
    n_prim_actions: int,
    use_enc_bn: bool,
    use_q_bn: bool,
    use_value_rescale: bool,
    deterministic_match: bool = True,
):
    """
    Backward-compatible alias currently behaving the same as the inclusive version.
    """
    return update_intra_option_meta_shared(
        train_state=train_state,
        transitions=transitions,
        option_targets=option_targets,
        n_prim_actions=n_prim_actions,
        use_enc_bn=use_enc_bn,
        use_q_bn=use_q_bn,
        use_value_rescale=use_value_rescale,
        deterministic_match=deterministic_match,
    )


@partial(jax.jit, static_argnums=range(4, 10))
def update_intra_option_meta_shared_exclude_exec_dual(
    train_state,
    transitions,
    ext_option_targets,
    int_option_targets,
    n_prim_actions: int,
    use_enc_bn: bool,
    use_q_bn: bool,
    use_value_rescale: bool,
    int_coef: float,
    deterministic_match: bool = True,
):
    """
    Backward-compatible alias currently behaving the same as the inclusive dual version.
    """
    return update_intra_option_meta_shared_dual(
        train_state=train_state,
        transitions=transitions,
        ext_option_targets=ext_option_targets,
        int_option_targets=int_option_targets,
        n_prim_actions=n_prim_actions,
        use_enc_bn=use_enc_bn,
        use_q_bn=use_q_bn,
        use_value_rescale=use_value_rescale,
        int_coef=int_coef,
        deterministic_match=deterministic_match,
    )