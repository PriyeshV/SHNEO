import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
from src.mains.algorithms.PQN import (value_rescale, inv_value_rescale,
                                      compute_td_lambda_targets, compute_transformed_td_lambda_targets)


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def eigenvalues_to_option_durations_skip_first(
    eigs: jnp.ndarray,          # [D] or [1,D]
    n_base_options: int,
    bidir_options: bool,
    min_dur: int,
    max_dur: int,
    eps: float = 1e-6,
):
    eigs = jnp.reshape(eigs, (-1,))
    eigs = jnp.nan_to_num(eigs, nan=0.0, posinf=0.0, neginf=0.0)
    eigs = jnp.clip(eigs, 0.0, jnp.inf)

    # skip first eigendimension
    usable = eigs[1:1 + n_base_options]
    lam = jnp.maximum(usable, eps)
    dur = jnp.ceil(1.0 / lam).astype(jnp.int32)
    dur = jnp.clip(dur, min_dur, max_dur)

    if bidir_options:
        dur = jnp.concatenate([dur, dur], axis=0)
    return dur


@partial(jax.jit, static_argnums=range(4, 9))
def compute_options_lambda_targets(
    train_state,
    transitions,
    gamma,
    td_lambda,
    use_opt_bn,
    use_lap_bn,
    n_base_options,
    bidir_options,
    use_value_rescale,
):
    """
    Returns:
        lambda_targets: [O, T, E]
    """
    T, E, C, H, W = transitions.obs.shape

    # ------------------------------------------------------------
    # Laplacian features for intrinsic rewards
    # ------------------------------------------------------------
    obs_nhwc = jnp.transpose(transitions.obs, (0, 1, 3, 4, 2))
    next_obs_nhwc = jnp.transpose(transitions.next_obs, (0, 1, 3, 4, 2))
    all_states = jnp.concatenate([obs_nhwc, next_obs_nhwc[-1:]], axis=0)  # [T+1,E,H,W,C]
    flat = all_states.reshape(((T + 1) * E, H, W, C))

    lap_vars = {"params": train_state.lap_state.params}
    if use_lap_bn:
        lap_vars["batch_stats"] = train_state.lap_state.batch_stats
        phis_all, _ = train_state.lap_state.apply_fn(
            lap_vars, flat, train=False, mutable=["batch_stats"]
        )
    else:
        phis_all = train_state.lap_state.apply_fn(lap_vars, flat, train=False)

    # If you want reference behavior, remove this skip-first later.
    phis_all = phis_all[:, 1:1 + n_base_options]

    phis = phis_all[:T * E].reshape((T, E, n_base_options))
    next_phis = phis_all[E:(T + 1) * E].reshape((T, E, n_base_options))

    if bidir_options:
        r_all = jnp.concatenate([next_phis - phis, phis - next_phis], axis=-1)  # [T,E,O]
    else:
        r_all = next_phis - phis

    # Your current version zeros terminal intrinsic reward
    valid = 1.0 - transitions.dones.astype(jnp.float32)
    r_all = r_all * valid[..., None]

    # ------------------------------------------------------------
    # Option Q-values along rollout
    # ------------------------------------------------------------
    q_all = transitions.option_q_vals                 # [T,E,O,A]
    dones = transitions.dones                         # [T,E]

    # ------------------------------------------------------------
    # Fresh bootstrap from s_T
    # ------------------------------------------------------------
    next_obs_last = jnp.transpose(transitions.next_obs[-1], (0, 2, 3, 1))  # [E,H,W,C]

    enc_states = train_state.opt_enc_states
    q_states = train_state.opt_q_states

    enc_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in enc_states])
    q_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in q_states])

    enc_apply = enc_states[0].apply_fn
    q_apply = q_states[0].apply_fn

    if use_opt_bn:
        enc_bs = jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in enc_states])

        def encode_one(p_enc, bs_enc):
            return enc_apply(
                {"params": p_enc, "batch_stats": bs_enc},
                next_obs_last,
                train=False,
            )

        embs = jax.vmap(encode_one, in_axes=(0, 0))(enc_params, enc_bs)  # [O,E,D]
    else:
        def encode_one(p_enc):
            return enc_apply(
                {"params": p_enc},
                next_obs_last,
                train=False,
            )

        embs = jax.vmap(encode_one, in_axes=0)(enc_params)  # [O,E,D]

    if use_opt_bn:
        q_bs = jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in q_states])

        def q_one(p_q, bs_q, emb):
            return q_apply(
                {"params": p_q, "batch_stats": bs_q},
                emb,
                train=False,
            )

        q_boot = jax.vmap(q_one, in_axes=(0, 0, 0))(q_params, q_bs, embs)  # [O,E,A]
    else:
        def q_one(p_q, emb):
            return q_apply(
                {"params": p_q},
                emb,
                train=False,
            )

        q_boot = jax.vmap(q_one, in_axes=(0, 0))(q_params, embs)  # [O,E,A]

    max_q_last = jnp.max(q_boot, axis=-1)  # [O,E]

    # ------------------------------------------------------------
    # Per-option lambda targets
    # ------------------------------------------------------------
    r_all_o = jnp.transpose(r_all, (2, 0, 1))     # [O,T,E]
    q_all_o = jnp.transpose(q_all, (2, 0, 1, 3))  # [O,T,E,A]

    if use_value_rescale:
        target_fn = partial(
            compute_transformed_td_lambda_targets,
            gamma=gamma,
            td_lambda=td_lambda,
        )
    else:
        target_fn = partial(
            compute_td_lambda_targets,
            gamma=gamma,
            td_lambda=td_lambda,
        )

    def _for_one_option(q_vals_o, r_o, next_q_o):
        return target_fn(next_q_o, q_vals_o, r_o, dones)

    lambda_targets = jax.vmap(_for_one_option, in_axes=(0, 0, 0))(q_all_o, r_all_o, max_q_last)
    return lambda_targets


@partial(jax.jit, static_argnums=range(4, 9))
def compute_options_lambda_targets_shared(
    train_state,
    transitions,
    gamma,
    td_lambda,
    use_opt_bn,
    use_lap_bn,
    n_base_options,
    bidir_options,
    use_value_rescale,
):
    """
    Returns:
        lambda_targets: [O, T, E]
    """
    T, E, C, H, W = transitions.obs.shape

    # ------------------------------------------------------------
    # Laplacian features for intrinsic rewards
    # ------------------------------------------------------------
    obs_nhwc = jnp.transpose(transitions.obs, (0, 1, 3, 4, 2))
    next_obs_nhwc = jnp.transpose(transitions.next_obs, (0, 1, 3, 4, 2))
    all_states = jnp.concatenate([obs_nhwc, next_obs_nhwc[-1:]], axis=0)
    flat = all_states.reshape(((T + 1) * E, H, W, C))

    lap_vars = {"params": train_state.lap_state.params}
    if use_lap_bn:
        lap_vars["batch_stats"] = train_state.lap_state.batch_stats
        phis_all, _ = train_state.lap_state.apply_fn(
            lap_vars, flat, train=False, mutable=["batch_stats"]
        )
    else:
        phis_all = train_state.lap_state.apply_fn(lap_vars, flat, train=False)

    phis_all = phis_all[:, 1:1 + n_base_options]

    phis = phis_all[:T * E].reshape((T, E, n_base_options))
    next_phis = phis_all[E:(T + 1) * E].reshape((T, E, n_base_options))

    if bidir_options:
        r_all = jnp.concatenate([next_phis - phis, phis - next_phis], axis=-1)
    else:
        r_all = next_phis - phis

    valid = 1.0 - transitions.dones.astype(jnp.float32)
    r_all = r_all * valid[..., None]

    # ------------------------------------------------------------
    # Option Q-values along rollout
    # ------------------------------------------------------------
    q_all = transitions.option_q_vals
    dones = transitions.dones

    # ------------------------------------------------------------
    # Fresh bootstrap from s_T using shared option encoder
    # ------------------------------------------------------------
    next_obs_last = jnp.transpose(transitions.next_obs[-1], (0, 2, 3, 1))  # [E,H,W,C]

    enc_vars = {"params": train_state.opt_enc_state.params}
    if use_opt_bn:
        enc_vars["batch_stats"] = train_state.opt_enc_state.batch_stats
    emb = train_state.opt_enc_state.apply_fn(enc_vars, next_obs_last, train=False)  # [E,D]

    q_states = train_state.opt_q_states
    q_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in q_states])
    q_bs = jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in q_states]) if use_opt_bn else None
    q_apply = q_states[0].apply_fn

    def q_one(p_q, bs_q):
        vars_q = {"params": p_q}
        if use_opt_bn:
            vars_q["batch_stats"] = bs_q
        return q_apply(vars_q, emb, train=False)  # [E,A]

    q_boot = jax.vmap(q_one, in_axes=(0, 0 if use_opt_bn else None))(q_params, q_bs)  # [O,E,A]
    max_q_last = jnp.max(q_boot, axis=-1)  # [O,E]

    # ------------------------------------------------------------
    # Per-option lambda targets
    # ------------------------------------------------------------
    r_all_o = jnp.transpose(r_all, (2, 0, 1))     # [O,T,E]
    q_all_o = jnp.transpose(q_all, (2, 0, 1, 3))  # [O,T,E,A]

    if use_value_rescale:
        target_fn = partial(
            compute_transformed_td_lambda_targets,
            gamma=gamma,
            td_lambda=td_lambda,
        )
    else:
        target_fn = partial(
            compute_td_lambda_targets,
            gamma=gamma,
            td_lambda=td_lambda,
        )

    def _for_one_option(q_vals_o, r_o, next_q_o):
        return target_fn(next_q_o, q_vals_o, r_o, dones)

    lambda_targets = jax.vmap(_for_one_option, in_axes=(0, 0, 0))(q_all_o, r_all_o, max_q_last)
    return lambda_targets


@partial(jax.jit, static_argnums=(3, 4, 5))
def update_option_networks(
    train_state,
    transitions,
    opt_lambda_targets,   # [O,B]
    use_enc_bn,
    use_q_bn,
    use_value_rescale,
):
    """
    Per-option TrainState version.
    Returns:
        train_state, (losses[O], q_mean_t[O], q_mean_raw[O])
    """
    obs = jnp.transpose(transitions.obs, (0, 2, 3, 1))  # [B,H,W,C]
    act = transitions.actions                            # [B]

    enc_states = train_state.opt_enc_states
    q_states = train_state.opt_q_states
    O = len(enc_states)

    enc_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in enc_states])
    q_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in q_states])

    enc_bs = jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in enc_states]) if use_enc_bn else None
    q_bs = jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in q_states]) if use_q_bn else None

    enc_apply = enc_states[0].apply_fn
    q_apply = q_states[0].apply_fn

    def batched_loss_fn(enc_p, enc_b, q_p, q_b, targets):
        def per_opt(p_enc, bs_enc, p_q, bs_q, tgt):
            if use_enc_bn:
                vars_e = {"params": p_enc, "batch_stats": bs_enc}
                emb, mut_e = enc_apply(vars_e, obs, train=True, mutable=["batch_stats"])
                new_bs_e = mut_e["batch_stats"]
            else:
                emb = enc_apply({"params": p_enc}, obs, train=True)
                new_bs_e = None

            if use_q_bn:
                vars_q = {"params": p_q, "batch_stats": bs_q}
                qv, mut_q = q_apply(vars_q, emb, train=True, mutable=["batch_stats"])
                new_bs_q = mut_q["batch_stats"]
            else:
                qv = q_apply({"params": p_q}, emb, train=True)
                new_bs_q = None

            preds = jnp.take_along_axis(qv, act[:, None], axis=-1).squeeze(-1)  # [B]
            loss_o = 0.5 * jnp.mean(jnp.square(preds - tgt))
            q_mean_t = jnp.mean(preds)
            q_mean_raw = jnp.mean(inv_value_rescale(preds)) if use_value_rescale else q_mean_t
            return loss_o, (q_mean_t, q_mean_raw, new_bs_e, new_bs_q)

        losses, aux = jax.vmap(
            per_opt,
            in_axes=(0,
                     0 if use_enc_bn else None,
                     0,
                     0 if use_q_bn else None,
                     0)
        )(enc_p, enc_b, q_p, q_b, targets)

        q_mean_t, q_mean_raw, new_bs_e, new_bs_q = aux
        total_loss = jnp.sum(losses)
        return total_loss, (losses, q_mean_t, q_mean_raw, new_bs_e, new_bs_q)

    (total_loss, (losses, q_mean_t, q_mean_raw, new_bs_e, new_bs_q)), (grads_e, grads_q) = \
        jax.value_and_grad(batched_loss_fn, argnums=(0, 2), has_aux=True)(
            enc_params, enc_bs, q_params, q_bs, opt_lambda_targets
        )

    enc_grads_list = [jax.tree_map(lambda x: x[i], grads_e) for i in range(O)]
    q_grads_list = [jax.tree_map(lambda x: x[i], grads_q) for i in range(O)]

    if use_enc_bn:
        bs_e_list = [jax.tree_map(lambda x: x[i], new_bs_e) for i in range(O)]
    if use_q_bn:
        bs_q_list = [jax.tree_map(lambda x: x[i], new_bs_q) for i in range(O)]

    new_opt_enc_states = []
    for i, s in enumerate(enc_states):
        s2 = s.apply_gradients(grads=enc_grads_list[i])
        if use_enc_bn:
            s2 = s2.replace(batch_stats=bs_e_list[i])
        new_opt_enc_states.append(s2)

    new_opt_q_states = []
    for i, s in enumerate(q_states):
        s2 = s.apply_gradients(grads=q_grads_list[i])
        if use_q_bn:
            s2 = s2.replace(batch_stats=bs_q_list[i])
        new_opt_q_states.append(s2)

    train_state = train_state.replace(
        opt_enc_states=new_opt_enc_states,
        opt_q_states=new_opt_q_states,
    )
    return train_state, (losses, q_mean_t, q_mean_raw)


@partial(jax.jit, static_argnums=(3, 4, 5))
def update_option_networks_shared(
    train_state,
    transitions,
    opt_lambda_targets,   # [O,B]
    use_enc_bn,
    use_q_bn,
    use_value_rescale,
):
    """
    Shared encoder + separate Q-heads.

    Returns:
        train_state, (losses[O], q_mean_t[O], q_mean_raw[O])
    """
    obs = jnp.transpose(transitions.obs, (0, 2, 3, 1))  # [B,H,W,C]
    act = transitions.actions                            # [B]

    q_states = train_state.opt_q_states
    O = len(q_states)

    q_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in q_states])
    q_bs = jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in q_states]) if use_q_bn else None
    q_apply = q_states[0].apply_fn

    def loss_fn(enc_params, q_params_stacked):
        # shared encoder
        if use_enc_bn:
            enc_vars = {
                "params": enc_params,
                "batch_stats": train_state.opt_enc_state.batch_stats,
            }
            emb, enc_mut = train_state.opt_enc_state.apply_fn(
                enc_vars, obs, train=True, mutable=["batch_stats"]
            )
            new_enc_bs = enc_mut["batch_stats"]
        else:
            emb = train_state.opt_enc_state.apply_fn({"params": enc_params}, obs, train=True)
            new_enc_bs = None

        def per_opt(p_q, bs_q, tgt):
            if use_q_bn:
                vars_q = {"params": p_q, "batch_stats": bs_q}
                qv, q_mut = q_apply(vars_q, emb, train=True, mutable=["batch_stats"])
                new_bs_q = q_mut["batch_stats"]
            else:
                qv = q_apply({"params": p_q}, emb, train=True)
                new_bs_q = None

            preds = jnp.take_along_axis(qv, act[:, None], axis=-1).squeeze(-1)  # [B]
            loss_o = 0.5 * jnp.mean(jnp.square(preds - tgt))
            q_mean_t = jnp.mean(preds)
            q_mean_raw = jnp.mean(inv_value_rescale(preds)) if use_value_rescale else q_mean_t
            return loss_o, (q_mean_t, q_mean_raw, new_bs_q)

        losses, aux = jax.vmap(
            per_opt,
            in_axes=(0, 0 if use_q_bn else None, 0)
        )(q_params_stacked, q_bs, opt_lambda_targets)

        q_mean_t, q_mean_raw, new_q_bs = aux
        total_loss = jnp.sum(losses)
        return total_loss, (new_enc_bs, losses, q_mean_t, q_mean_raw, new_q_bs)

    (total_loss, (new_enc_bs, losses, q_mean_t, q_mean_raw, new_q_bs)), (g_enc, g_q) = \
        jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            train_state.opt_enc_state.params,
            q_params,
        )

    # update shared encoder
    new_opt_enc_state = train_state.opt_enc_state.apply_gradients(grads=g_enc)
    if use_enc_bn:
        new_opt_enc_state = new_opt_enc_state.replace(batch_stats=new_enc_bs)

    # update each Q-head
    q_grads_list = [jax.tree_map(lambda x: x[i], g_q) for i in range(O)]
    if use_q_bn:
        q_bs_list = [jax.tree_map(lambda x: x[i], new_q_bs) for i in range(O)]

    new_opt_q_states = []
    for i, s in enumerate(q_states):
        s2 = s.apply_gradients(grads=q_grads_list[i])
        if use_q_bn:
            s2 = s2.replace(batch_stats=q_bs_list[i])
        new_opt_q_states.append(s2)

    train_state = train_state.replace(
        opt_enc_state=new_opt_enc_state,
        opt_q_states=new_opt_q_states,
    )
    return train_state, (losses, q_mean_t, q_mean_raw)


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def update_option_networks_shared_dual(
    train_state,
    transitions,
    opt_ext_lambda_targets,   # [O,B]
    opt_int_lambda_targets,   # [O,B]
    use_enc_bn,
    use_q_bn,
    use_value_rescale,
    int_coef,
):
    """
    Shared encoder + separate dual Q-heads per option.

    Returns:
        train_state, (losses[O], q_sel_mean_t[O], q_sel_mean_raw[O])
    """
    obs = jnp.transpose(transitions.obs, (0, 2, 3, 1))  # [B,H,W,C]
    act = transitions.actions                            # [B]

    qext_states = train_state.opt_q_ext_states
    qint_states = train_state.opt_q_int_states
    O = len(qext_states)

    qext_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in qext_states])
    qint_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in qint_states])

    qext_bs = jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in qext_states]) if use_q_bn else None
    qint_bs = jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in qint_states]) if use_q_bn else None

    qext_apply = qext_states[0].apply_fn
    qint_apply = qint_states[0].apply_fn

    def loss_fn(enc_params, qext_params_stacked, qint_params_stacked):
        if use_enc_bn:
            enc_vars = {
                "params": enc_params,
                "batch_stats": train_state.opt_enc_state.batch_stats,
            }
            emb, enc_mut = train_state.opt_enc_state.apply_fn(
                enc_vars, obs, train=True, mutable=["batch_stats"]
            )
            new_enc_bs = enc_mut["batch_stats"]
        else:
            emb = train_state.opt_enc_state.apply_fn({"params": enc_params}, obs, train=True)
            new_enc_bs = None

        def per_opt(p_qext, bs_qext, p_qint, bs_qint, tgt_ext, tgt_int):
            if use_q_bn:
                vars_qext = {"params": p_qext, "batch_stats": bs_qext}
                q_ext, qext_mut = qext_apply(vars_qext, emb, train=True, mutable=["batch_stats"])
                new_bs_qext = qext_mut["batch_stats"]

                vars_qint = {"params": p_qint, "batch_stats": bs_qint}
                q_int, qint_mut = qint_apply(vars_qint, emb, train=True, mutable=["batch_stats"])
                new_bs_qint = qint_mut["batch_stats"]
            else:
                q_ext = qext_apply({"params": p_qext}, emb, train=True)
                q_int = qint_apply({"params": p_qint}, emb, train=True)
                new_bs_qext = None
                new_bs_qint = None

            pred_ext = jnp.take_along_axis(q_ext, act[:, None], axis=-1).squeeze(-1)  # [B]
            pred_int = jnp.take_along_axis(q_int, act[:, None], axis=-1).squeeze(-1)  # [B]

            ext_loss = 0.5 * jnp.mean(jnp.square(pred_ext - tgt_ext))
            int_loss = 0.5 * jnp.mean(jnp.square(pred_int - tgt_int))
            loss_o = ext_loss + int_coef * int_loss

            if use_value_rescale:
                pred_sel_raw = inv_value_rescale(pred_ext) + int_coef * inv_value_rescale(pred_int)
                pred_sel = value_rescale(pred_sel_raw)
                q_sel_mean_t = jnp.mean(pred_sel)
                q_sel_mean_raw = jnp.mean(pred_sel_raw)
            else:
                pred_sel = pred_ext + int_coef * pred_int
                q_sel_mean_t = jnp.mean(pred_sel)
                q_sel_mean_raw = q_sel_mean_t

            return loss_o, (q_sel_mean_t, q_sel_mean_raw, new_bs_qext, new_bs_qint)

        losses, aux = jax.vmap(
            per_opt,
            in_axes=(0, 0 if use_q_bn else None, 0, 0 if use_q_bn else None, 0, 0)
        )(qext_params_stacked, qext_bs, qint_params_stacked, qint_bs, opt_ext_lambda_targets, opt_int_lambda_targets)

        q_sel_mean_t, q_sel_mean_raw, new_qext_bs, new_qint_bs = aux
        total_loss = jnp.sum(losses)
        return total_loss, (new_enc_bs, losses, q_sel_mean_t, q_sel_mean_raw, new_qext_bs, new_qint_bs)

    (total_loss, (new_enc_bs, losses, q_sel_mean_t, q_sel_mean_raw, new_qext_bs, new_qint_bs)), (g_enc, g_qext, g_qint) = \
        jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)(
            train_state.opt_enc_state.params,
            qext_params,
            qint_params,
        )

    new_opt_enc_state = train_state.opt_enc_state.apply_gradients(grads=g_enc)
    if use_enc_bn:
        new_opt_enc_state = new_opt_enc_state.replace(batch_stats=new_enc_bs)

    qext_grads_list = [jax.tree_map(lambda x: x[i], g_qext) for i in range(O)]
    qint_grads_list = [jax.tree_map(lambda x: x[i], g_qint) for i in range(O)]

    if use_q_bn:
        qext_bs_list = [jax.tree_map(lambda x: x[i], new_qext_bs) for i in range(O)]
        qint_bs_list = [jax.tree_map(lambda x: x[i], new_qint_bs) for i in range(O)]

    new_opt_qext_states = []
    new_opt_qint_states = []

    for i in range(O):
        s_ext = qext_states[i].apply_gradients(grads=qext_grads_list[i])
        s_int = qint_states[i].apply_gradients(grads=qint_grads_list[i])

        if use_q_bn:
            s_ext = s_ext.replace(batch_stats=qext_bs_list[i])
            s_int = s_int.replace(batch_stats=qint_bs_list[i])

        new_opt_qext_states.append(s_ext)
        new_opt_qint_states.append(s_int)

    train_state = train_state.replace(
        opt_enc_state=new_opt_enc_state,
        opt_q_ext_states=new_opt_qext_states,
        opt_q_int_states=new_opt_qint_states,
    )
    return train_state, (losses, q_sel_mean_t, q_sel_mean_raw)


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def update_option_networks_shared_dual_IS(
    train_state,
    transitions,
    opt_ext_lambda_targets,   # [O,B]
    opt_int_lambda_targets,   # [O,B]
    opt_valid_mask,           # [O,B]  1 where this option/sample should train
    use_enc_bn,
    use_q_bn,
    use_value_rescale,
    int_coef,
):
    """
    Shared encoder + separate dual Q-heads per option.

    This version is correct for executed-option-only option training.

    Args:
        transitions.obs:            [B,C,H,W]
        transitions.actions:        [B]
        opt_ext_lambda_targets:     [O,B]
        opt_int_lambda_targets:     [O,B]
        opt_valid_mask:             [O,B]
            opt_valid_mask[o,b] = 1 iff sample b should contribute to option o.
            For executed-option-only Retrace / TD updates, this is typically 1
            only for the option that actually executed on that primitive step.

    Returns:
        train_state,
        (
            losses[O],         per-option masked loss
            q_sel_mean_t[O],   per-option transformed selected-Q mean on valid samples
            q_sel_mean_raw[O], per-option raw selected-Q mean on valid samples
        )
    """
    obs = jnp.transpose(transitions.obs, (0, 2, 3, 1))  # [B,H,W,C]
    act = transitions.actions                            # [B]

    qext_states = train_state.opt_q_ext_states
    qint_states = train_state.opt_q_int_states
    O = len(qext_states)

    qext_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in qext_states])
    qint_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in qint_states])

    qext_bs = (
        jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in qext_states])
        if use_q_bn else None
    )
    qint_bs = (
        jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in qint_states])
        if use_q_bn else None
    )

    qext_apply = qext_states[0].apply_fn
    qint_apply = qint_states[0].apply_fn

    def loss_fn(enc_params, qext_params_stacked, qint_params_stacked):
        # ------------------------------------------------------------
        # Shared encoder forward
        # ------------------------------------------------------------
        if use_enc_bn:
            enc_vars = {
                "params": enc_params,
                "batch_stats": train_state.opt_enc_state.batch_stats,
            }
            emb, enc_mut = train_state.opt_enc_state.apply_fn(
                enc_vars, obs, train=True, mutable=["batch_stats"]
            )
            new_enc_bs = enc_mut["batch_stats"]
        else:
            emb = train_state.opt_enc_state.apply_fn({"params": enc_params}, obs, train=True)
            new_enc_bs = None

        # ------------------------------------------------------------
        # Per-option loss
        # ------------------------------------------------------------
        def per_opt(p_qext, bs_qext, p_qint, bs_qint, tgt_ext, tgt_int, valid_mask_o):
            """
            tgt_ext, tgt_int, valid_mask_o: [B]
            """
            if use_q_bn:
                vars_qext = {"params": p_qext, "batch_stats": bs_qext}
                q_ext, qext_mut = qext_apply(vars_qext, emb, train=True, mutable=["batch_stats"])
                new_bs_qext = qext_mut["batch_stats"]

                vars_qint = {"params": p_qint, "batch_stats": bs_qint}
                q_int, qint_mut = qint_apply(vars_qint, emb, train=True, mutable=["batch_stats"])
                new_bs_qint = qint_mut["batch_stats"]
            else:
                q_ext = qext_apply({"params": p_qext}, emb, train=True)
                q_int = qint_apply({"params": p_qint}, emb, train=True)
                new_bs_qext = None
                new_bs_qint = None

            pred_ext = jnp.take_along_axis(q_ext, act[:, None], axis=-1).squeeze(-1)  # [B]
            pred_int = jnp.take_along_axis(q_int, act[:, None], axis=-1).squeeze(-1)  # [B]

            valid = valid_mask_o.astype(jnp.float32)  # [B]
            denom = jnp.sum(valid) + 1e-8

            # --------------------------------------------------------
            # Masked TD losses
            #
            # Only samples with valid=1 contribute to the option loss.
            # This is essential for executed-option-only targets.
            # --------------------------------------------------------
            ext_sq = jnp.square(pred_ext - tgt_ext)
            int_sq = jnp.square(pred_int - tgt_int)

            ext_loss = 0.5 * jnp.sum(ext_sq * valid) / denom
            int_loss = 0.5 * jnp.sum(int_sq * valid) / denom
            loss_o = ext_loss + int_coef * int_loss

            # --------------------------------------------------------
            # Diagnostics on valid samples only
            # --------------------------------------------------------
            if use_value_rescale:
                pred_sel_raw = inv_value_rescale(pred_ext) + int_coef * inv_value_rescale(pred_int)
                pred_sel = value_rescale(pred_sel_raw)
            else:
                pred_sel_raw = pred_ext + int_coef * pred_int
                pred_sel = pred_sel_raw

            q_sel_mean_t = jnp.sum(pred_sel * valid) / denom
            q_sel_mean_raw = jnp.sum(pred_sel_raw * valid) / denom

            return loss_o, (q_sel_mean_t, q_sel_mean_raw, new_bs_qext, new_bs_qint)

        losses, aux = jax.vmap(
            per_opt,
            in_axes=(
                0,
                0 if use_q_bn else None,
                0,
                0 if use_q_bn else None,
                0,
                0,
                0,
            ),
        )(
            qext_params_stacked,
            qext_bs,
            qint_params_stacked,
            qint_bs,
            opt_ext_lambda_targets,
            opt_int_lambda_targets,
            opt_valid_mask,
        )

        q_sel_mean_t, q_sel_mean_raw, new_qext_bs, new_qint_bs = aux

        # Sum per-option losses. Since each option loss is already normalized
        # by its own valid count, this avoids frequent options dominating.
        total_loss = jnp.sum(losses)

        return total_loss, (
            new_enc_bs,
            losses,
            q_sel_mean_t,
            q_sel_mean_raw,
            new_qext_bs,
            new_qint_bs,
        )

    (
        total_loss,
        (new_enc_bs, losses, q_sel_mean_t, q_sel_mean_raw, new_qext_bs, new_qint_bs),
    ), (g_enc, g_qext, g_qint) = jax.value_and_grad(
        loss_fn,
        argnums=(0, 1, 2),
        has_aux=True,
    )(
        train_state.opt_enc_state.params,
        qext_params,
        qint_params,
    )

    # ------------------------------------------------------------
    # Apply shared encoder gradients
    # ------------------------------------------------------------
    new_opt_enc_state = train_state.opt_enc_state.apply_gradients(grads=g_enc)
    if use_enc_bn:
        new_opt_enc_state = new_opt_enc_state.replace(batch_stats=new_enc_bs)

    # ------------------------------------------------------------
    # Apply per-option head gradients
    # ------------------------------------------------------------
    qext_grads_list = [jax.tree_map(lambda x: x[i], g_qext) for i in range(O)]
    qint_grads_list = [jax.tree_map(lambda x: x[i], g_qint) for i in range(O)]

    if use_q_bn:
        qext_bs_list = [jax.tree_map(lambda x: x[i], new_qext_bs) for i in range(O)]
        qint_bs_list = [jax.tree_map(lambda x: x[i], new_qint_bs) for i in range(O)]

    new_opt_qext_states = []
    new_opt_qint_states = []

    for i in range(O):
        s_ext = qext_states[i].apply_gradients(grads=qext_grads_list[i])
        s_int = qint_states[i].apply_gradients(grads=qint_grads_list[i])

        if use_q_bn:
            s_ext = s_ext.replace(batch_stats=qext_bs_list[i])
            s_int = s_int.replace(batch_stats=qint_bs_list[i])

        new_opt_qext_states.append(s_ext)
        new_opt_qint_states.append(s_int)

    train_state = train_state.replace(
        opt_enc_state=new_opt_enc_state,
        opt_q_ext_states=new_opt_qext_states,
        opt_q_int_states=new_opt_qint_states,
    )

    return train_state, (losses, q_sel_mean_t, q_sel_mean_raw)

