from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def _num_constraints(d: int) -> float:
    """Number of entries in a ``d x d`` lower-triangular matrix."""
    return d * (d + 1) / 2.0


def _valid_prefix_mask(masks: jnp.ndarray) -> jnp.ndarray:
    """Keep transitions through the first terminal and drop later padding."""
    masks = jnp.asarray(masks, dtype=bool)
    n_steps = masks.shape[1]
    done_indicator = jnp.logical_not(masks)
    done_exists = jnp.any(done_indicator, axis=1)
    first_done_index = jnp.where(
        done_exists,
        jnp.argmax(done_indicator, axis=1),
        n_steps - 1,
    )
    return jnp.arange(n_steps)[None, :] <= first_done_index[:, None]


@partial(jax.jit, static_argnums=tuple(range(11, 23)))
def multi_step_ALLO(
    key,
    phis_all,
    future_phis_all,
    actions,
    sample_mask,
    masks,
    barrier_coeffs,
    betas,
    weights=None,
    novelty_values: jnp.ndarray | None = None,
    zeta_values: jnp.ndarray | None = None,
    use_novelty: bool = False,
    use_zeta: bool = False,
    normalize_zeta: bool = True,
    gamma=0.99,
    state_action=False,
    uni_policy=False,
    n_steps=1,
    use_dual=True,
    use_weights=False,
    average_loss=True,
    rms_norm=False,
    barrier_dims=1,
):
    """Compute the multi-step graph-drawing and ALLO constraint losses.

    ``average_loss=True`` uniformly divides every lower-triangular constraint
    term by ``D * (D + 1) / 2``.  The same convention is used by the dual and
    barrier parameter updates below.
    """
    if state_action:
        batch_size, n_actions, dims = phis_all.shape
        batch_ids = jnp.arange(batch_size)
        phi = phis_all[batch_ids, actions[:, 0], :]

        if uni_policy:
            future_phis_pos = future_phis_all
        else:
            action_indices = actions[:, 1:, None, None]
            future_phis_pos = jnp.take_along_axis(
                future_phis_all, action_indices, axis=2
            ).squeeze(2)
    else:
        phi = phis_all
        batch_size, dims = phi.shape
        future_phis_pos = future_phis_all

    pos_loss = compute_multi_step_graph_drawing_loss(
        phi,
        future_phis_pos,
        sample_mask,
        masks,
        weights=weights,
        novelty_values=novelty_values,
        zeta_values=zeta_values,
        use_novelty=use_novelty,
        use_zeta=use_zeta,
        normalize_zeta=normalize_zeta,
        gamma=gamma,
        uni_policy=uni_policy,
        n_steps=n_steps,
        use_weights=use_weights,
        average_loss=average_loss,
    )

    # Construct a pool of valid current/future representations and draw two
    # batches independently with replacement.  A row permutation alone would
    # leave the Gram matrix unchanged and therefore would not provide an
    # independent estimator for the ALLO product.
    key_actions_fut, key_actions_cur, key_ortho1, key_ortho2 = (
        jax.random.split(key, 4)
    )
    available_steps = future_phis_all.shape[1]

    if state_action:
        sampled_future_actions = jax.random.randint(
            key_actions_fut, (batch_size, available_steps), 0, n_actions
        )
        future_candidates = jnp.take_along_axis(
            future_phis_all,
            sampled_future_actions[:, :, None, None],
            axis=2,
        ).squeeze(2)
    else:
        future_candidates = future_phis_all

    if state_action:
        random_actions = jax.random.randint(
            key_actions_cur, (batch_size,), 0, n_actions
        )
        current_candidates = phis_all[
            jnp.arange(batch_size), random_actions, :
        ]
    else:
        current_candidates = phi

    sample_mask_bool = jnp.asarray(sample_mask, dtype=bool)
    valid_future = _valid_prefix_mask(masks) & sample_mask_bool[:, None]
    state_pool = jnp.concatenate(
        [current_candidates[:, None, :], future_candidates], axis=1
    ).reshape((-1, dims))
    valid_pool = jnp.concatenate(
        [sample_mask_bool[:, None], valid_future], axis=1
    ).reshape((-1,))

    # A training batch should contain at least one valid row.  The fallback
    # keeps categorical sampling numerically defined for an all-invalid batch.
    valid_pool = jnp.where(
        jnp.any(valid_pool), valid_pool, jnp.ones_like(valid_pool)
    )
    logits = jnp.where(valid_pool, 0.0, -jnp.inf)
    indices1 = jax.random.categorical(
        key_ortho1, logits, shape=(batch_size,)
    )
    indices2 = jax.random.categorical(
        key_ortho2, logits, shape=(batch_size,)
    )
    ortho_batch1 = state_pool[indices1]
    ortho_batch2 = state_pool[indices2]

    _linear_error, quadratic_error, error_matrix = (
        compute_orthogonality_error(
            ortho_batch1,
            ortho_batch2,
            average_loss=average_loss,
            rms_norm=rms_norm,
        )
    )

    constraint_scale = (
        1.0 / _num_constraints(dims) if average_loss else 1.0
    )
    quadratic_errors = error_matrix["quadratic_errors"]

    if barrier_dims == 1:
        barrier_loss = (
            jax.lax.stop_gradient(barrier_coeffs[0, 0])
            * quadratic_error
        )
    else:
        barrier_diagonal = jax.lax.stop_gradient(jnp.diag(barrier_coeffs))
        quadratic_error_by_row = quadratic_errors.sum(axis=1)
        barrier_loss = constraint_scale * jnp.dot(
            barrier_diagonal, quadratic_error_by_row
        )

    dual_loss = jnp.asarray(0.0, dtype=phi.dtype)
    if use_dual:
        dual_loss = constraint_scale * (
            jax.lax.stop_gradient(betas) * error_matrix["errors"]
        ).sum()

    ortho_weight = (
        jnp.asarray(1.0, dtype=phi.dtype)
        if not use_weights
        else jnp.asarray(weights, dtype=phi.dtype).mean()
    )
    loss = pos_loss + ortho_weight * (barrier_loss + dual_loss)

    weighted_errors = {
        name: ortho_weight * value for name, value in error_matrix.items()
    }

    # The third metric is the actual dual-weighted contribution.  The raw
    # linear constraint error remains available in weighted_errors["errors"].
    metrics = (pos_loss, quadratic_error, dual_loss)
    return loss, metrics, (ortho_weight, weighted_errors)


@partial(jax.jit, static_argnums=range(7, 15))
def compute_multi_step_graph_drawing_loss(
    phi,
    target_phis,
    sample_mask,
    masks,
    weights=None,
    novelty_values: jnp.ndarray | None = None,
    zeta_values: jnp.ndarray | None = None,
    use_novelty: bool = True,
    use_zeta: bool = False,
    normalize_zeta: bool = True,
    gamma=0.99,
    uni_policy=False,
    n_steps=1,
    use_weights=False,
    average_loss=True,
):
    """Compute batch-mean graph smoothness and optional zeta sink losses.

    ``uni_policy`` and ``average_loss`` remain in the signature for drop-in
    compatibility.  Action averaging is inferred from ``target_phis.ndim``;
    batch aggregation is always a mean, as required for a Monte Carlo estimate
    of the graph-drawing expectation.
    """
    del uni_policy, average_loss

    batch_size = phi.shape[0]
    dtype = phi.dtype
    eps = jnp.asarray(1e-8, dtype=dtype)

    if target_phis.shape[1] != n_steps:
        raise ValueError(
            "target_phis.shape[1] must equal n_steps: "
            f"{target_phis.shape[1]} != {n_steps}"
        )

    sample_mask = jnp.asarray(sample_mask, dtype=bool)
    masks = jnp.asarray(masks, dtype=bool)

    valid_steps = _valid_prefix_mask(masks)
    valid_steps_f = valid_steps.astype(dtype)

    if use_novelty and novelty_values is not None:
        novelty = jnp.maximum(
            jnp.asarray(novelty_values, dtype=dtype), 0.0
        )
    else:
        novelty = jnp.ones((batch_size, n_steps), dtype=dtype)

    discount = jnp.power(
        jnp.asarray(gamma, dtype=dtype),
        jnp.arange(n_steps, dtype=dtype),
    )[None, :]
    time_weights = discount * valid_steps_f * novelty

    if use_weights:
        if weights is None:
            raise ValueError("weights must be provided when use_weights=True")
        time_weights *= jnp.asarray(weights, dtype=dtype)

    time_weights /= time_weights.sum(axis=1, keepdims=True) + eps

    # Give valid rows mean weight one.  When novelty is enabled, this also
    # emphasizes rows containing a larger novelty value.
    row_strength = jnp.max(novelty * valid_steps_f, axis=1)
    row_strength *= sample_mask.astype(dtype)
    row_weights = (
        row_strength / (row_strength.sum() + eps) * batch_size
    )

    if target_phis.ndim == 4:
        target_nodes = target_phis.mean(axis=2)
    else:
        target_nodes = target_phis

    # Match the conventional graph Dirichlet-energy factor used by ALLO.
    squared_difference = 0.5 * (phi[:, None, :] - target_nodes) ** 2
    per_row_smoothness = (
        time_weights[:, :, None] * squared_difference
    ).sum(axis=(1, 2))
    per_row_smoothness *= sample_mask.astype(dtype) * row_weights
    smoothness_loss = per_row_smoothness.mean()

    sink_loss = jnp.asarray(0.0, dtype=dtype)
    if use_zeta:
        if zeta_values is None:
            raise ValueError("zeta_values must be provided when use_zeta=True")

        zeta = jnp.maximum(jnp.asarray(zeta_values, dtype=dtype), 0.0)
        if zeta.shape[1] != n_steps + 1:
            raise ValueError(
                "zeta_values.shape[1] must equal n_steps + 1: "
                f"{zeta.shape[1]} != {n_steps + 1}"
            )

        if normalize_zeta:
            valid_zeta = jnp.concatenate(
                [sample_mask[:, None], valid_steps], axis=1
            )
            valid_zeta &= sample_mask[:, None]
            valid_zeta_f = valid_zeta.astype(dtype)
            zeta_mean = (zeta * valid_zeta_f).sum() / (
                valid_zeta_f.sum() + eps
            )
            zeta /= zeta_mean + eps

        phi0_squared_norm = (phi * phi).sum(axis=-1)
        sink_current = (
            zeta[:, 0] * sample_mask.astype(dtype) * phi0_squared_norm
        )

        target_squared_norm = (target_nodes * target_nodes).sum(axis=-1)
        future_zeta = zeta[:, 1:] * valid_steps_f
        future_zeta *= sample_mask.astype(dtype)[:, None]
        valid_future_count = valid_steps_f.sum(axis=1) + eps
        sink_future = (
            future_zeta * target_squared_norm
        ).sum(axis=1) / valid_future_count

        sink_loss = ((sink_current + sink_future) * row_weights).mean()

    return smoothness_loss + sink_loss


@partial(jax.jit, static_argnums=(2, 3))
def compute_orthogonality_error(
    phi_u,
    phi_v,
    average_loss=True,
    rms_norm=False,
):
    """Return linear and unbiased two-sample quadratic ALLO errors."""
    if phi_u.shape != phi_v.shape:
        raise ValueError(
            f"phi_u and phi_v must have equal shapes: "
            f"{phi_u.shape} != {phi_v.shape}"
        )
    if rms_norm:
        raise NotImplementedError(
            "rms_norm was previously unused and has no defined ALLO behavior"
        )

    batch_size, dims = phi_u.shape
    identity = jnp.eye(dims, dtype=phi_u.dtype)

    gram_u = (
        phi_u.T @ jax.lax.stop_gradient(phi_u)
    ) / batch_size
    gram_v = (
        phi_v.T @ jax.lax.stop_gradient(phi_v)
    ) / batch_size

    error_u = jnp.tril(gram_u - identity)
    error_v = jnp.tril(gram_v - identity)
    linear_errors = 0.5 * (error_u + error_v)
    quadratic_errors = error_u * error_v

    constraint_scale = (
        1.0 / _num_constraints(dims) if average_loss else 1.0
    )
    linear_loss = constraint_scale * linear_errors.sum()
    quadratic_loss = constraint_scale * quadratic_errors.sum()

    errors = {
        "errors": linear_errors,
        "quadratic_errors": quadratic_errors,
    }
    return linear_loss, quadratic_loss, errors


@partial(jax.jit, static_argnums=range(2, 8))
def update_ALLO_parameters(
    lap_state,
    curr_errors,
    update_rates,
    barrier_args,
    dual_args,
    dual_a_min=-500,
    average_loss=True,
    barrier_dims=1,
):
    """Update moving constraint estimates, barriers, and dual variables."""
    moving_errors = {
        "errors": lap_state.errors,
        "quadratic_errors": lap_state.quadratic_errors,
    }
    # The encoder sees the signed two-sample product.  The barrier controller
    # sees only positive per-batch violations, exactly as in ALLO.  Applying
    # the floor before the EMA avoids cancellation by negative sampling noise.
    update_errors = {
        "errors": curr_errors["errors"],
        "quadratic_errors": jnp.maximum(
            jnp.tril(curr_errors["quadratic_errors"]), 0.0
        ),
    }
    new_errors, _ = update_error_estimates(
        moving_errors,
        update_errors,
        barrier_args.on,
        dual_args.on,
        update_rates.error,
        update_rates.q_error,
    )

    if barrier_args.on:
        new_quadratic_errors = new_errors["quadratic_errors"]
        new_barrier_coeffs = update_barrier_coefficients(
            lap_state.barrier_coeffs,
            new_quadratic_errors,
            barrier_lr=barrier_args.lr,
            use_clip=barrier_args.clip,
            average_loss=average_loss,
            barrier_dims=barrier_dims,
        )
        lap_state = lap_state.replace(
            quadratic_errors=new_quadratic_errors,
            barrier_coeffs=new_barrier_coeffs,
        )

    if dual_args.on:
        new_linear_errors = new_errors["errors"]
        new_betas = update_duals(
            new_linear_errors,
            lap_state.betas,
            dual_lr=dual_args.lr,
            a_min=dual_a_min,
            use_clip=dual_args.clip,
            average_loss=average_loss,
        )
        lap_state = lap_state.replace(
            errors=new_linear_errors,
            betas=new_betas,
        )

    return lap_state


@partial(jax.jit, static_argnums=range(2, 9))
def update_barrier_coefficients(
    barrier_coefficients,
    quadratic_error_matrix,
    barrier_lr=1.0,
    a_min=0.0,
    a_max=10000.0,
    clip_value=1.0,
    use_clip=False,
    average_loss=True,
    barrier_dims=1,
):
    """Monotonically increase barrier coefficients from positive violations.

    The raw two-sample quadratic estimator remains unclipped in the network
    loss.  Clamping happens only here, so sampling noise cannot decrease the
    barrier coefficient.
    """
    dims = quadratic_error_matrix.shape[0]
    constraint_scale = (
        1.0 / _num_constraints(dims) if average_loss else 1.0
    )
    positive_errors = jnp.maximum(
        jnp.tril(quadratic_error_matrix), 0.0
    )

    if barrier_dims == 1:
        update_signal = constraint_scale * positive_errors.sum()
        raw_update = barrier_lr * update_signal
        if use_clip:
            raw_update = jnp.clip(raw_update, 0.0, clip_value)

        new_barrier = jnp.clip(
            barrier_coefficients + raw_update,
            a_min,
            a_max,
        )
        return jnp.tril(new_barrier)

    update_signal = constraint_scale * positive_errors.sum(axis=1)
    raw_update = barrier_lr * update_signal
    if use_clip:
        raw_update = jnp.clip(raw_update, 0.0, clip_value)

    barrier_diagonal = jnp.diag(barrier_coefficients)
    new_diagonal = jnp.clip(
        barrier_diagonal + raw_update,
        a_min,
        a_max,
    )
    return jnp.diag(new_diagonal)


@partial(jax.jit, static_argnums=range(2, 8))
def update_duals(
    error_matrix,
    betas,
    dual_lr=1e-4,
    a_min=-500,
    a_max=0,
    clip_value=0.1,
    use_clip=False,
    average_loss=True,
):
    """Perform dual ascent using the loss's constraint normalization."""
    dims = error_matrix.shape[0]
    updates = jnp.tril(error_matrix)
    if average_loss:
        updates /= _num_constraints(dims)

    if use_clip:
        updates = jnp.clip(updates, -clip_value, clip_value)

    updated_duals = jnp.clip(
        betas + dual_lr * updates,
        a_min=a_min,
        a_max=a_max,
    )
    return jnp.tril(updated_duals)


@partial(jax.jit, static_argnums=range(2, 5))
def update_error_estimates(
    moving_errors,
    errors,
    use_barrier,
    use_dual,
    err_rate,
    q_err_rate,
):
    """Update exponential moving estimates of active constraint errors."""
    updates = {}
    error_norms = {"errors": 0.0, "quadratic_errors": 0.0}
    update_rates = {
        "errors": err_rate,
        "quadratic_errors": q_err_rate,
    }

    for error_type in ("errors", "quadratic_errors"):
        inactive = (
            error_type == "errors" and not use_dual
        ) or (
            error_type == "quadratic_errors" and not use_barrier
        )
        if inactive:
            continue

        old_value = moving_errors[error_type]
        error_norms[error_type] = jnp.linalg.norm(old_value)

        is_uninitialized = jnp.isclose(
            error_norms[error_type],
            0.0,
            rtol=1e-10,
            atol=1e-13,
        )
        configured_rate = update_rates[error_type]
        update_rate = is_uninitialized + (
            1.0 - is_uninitialized
        ) * configured_rate
        updates[error_type] = old_value + update_rate * (
            errors[error_type] - old_value
        )

    return updates, error_norms


@partial(jax.jit, static_argnums=(2,))
def compute_batch_eigenvalues(phi, phi_next, eps=1e-8):
    """Estimate per-dimension Rayleigh quotients from a transition batch."""
    numerator = 0.5 * jnp.mean((phi - phi_next) ** 2, axis=0)
    denominator = jnp.mean(
        0.5 * (phi**2 + phi_next**2), axis=0
    )
    return numerator / (denominator + eps)


@partial(jax.jit, static_argnums=(2,))
def update_eigenvalues(lambda_current, lambda_batch, alpha=0.1):
    """Exponential moving average of eigenvalue estimates."""
    return (1.0 - alpha) * lambda_current + alpha * lambda_batch


@partial(jax.jit, static_argnums=(2,))
def get_laplacian_metrics(curr_hidden, neg_hidden, average_loss=True):
    """Return representation distance and mean per-row representation norm."""
    squared_distance = (curr_hidden - neg_hidden) ** 2
    negative_distance = (
        squared_distance.mean(axis=0).sum()
        if average_loss
        else squared_distance.sum()
    )
    row_norm = jnp.linalg.norm(curr_hidden, ord=2, axis=1).mean()
    return negative_distance, row_norm
