import jax
import jax.numpy as jnp
from functools import partial
from src.utils.jax_utils import tril_row_normalization


@partial(jax.jit, static_argnums=tuple(range(11, 23)))  # updated: include use_count, etc.
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

    # NEW -------------------------
    novelty_values: jnp.ndarray = None,  # [B, n_steps]  novelty(s_k), k=1..n_steps
    zeta_values: jnp.ndarray = None,  # [B, n_steps+1] zeta(s_0..s_n_steps)
    use_novelty: bool = False,
    use_zeta: bool = False,
    normalize_zeta: bool = True,
    # -----------------------------

    gamma=0.99, state_action=False, uni_policy=False, n_steps=1,
    use_dual=True, use_weights=False, average_loss=True,
    rms_norm=False, barrier_dims=1,
):

    # ---------- Build POS term inputs ----------
    if state_action:
        B, A, D = phis_all.shape
        batch_ids = jnp.arange(B)

        phi = phis_all[batch_ids, actions[:, 0], :]  # [B, D]

        if uni_policy:
            future_phis_pos = future_phis_all        # [B, N, A, D]
        else:
            idx = actions[:, 1:][:, :, None, None]   # [B, N, 1, 1]
            future_phis_pos = jnp.take_along_axis(future_phis_all, idx, axis=2).squeeze(2)
    else:
        phi = phis_all                               # [B, D]
        B, D = phi.shape
        future_phis_pos = future_phis_all            # [B, N, D]

    pos_loss = compute_multi_step_graph_drawing_loss(
        phi, future_phis_pos, sample_mask, masks,
        weights=weights,

        novelty_values=novelty_values,
        zeta_values=zeta_values,
        use_novelty=use_novelty,
        use_zeta=use_zeta,
        normalize_zeta=normalize_zeta,

        gamma=gamma, uni_policy=uni_policy, n_steps=n_steps,
        use_weights=use_weights, average_loss=average_loss
    )

    # ---------- Ortho batches (unchanged) ----------
    key_time, key_perm1, key_actions_fut, key_perm2, key_actions_cur = jax.random.split(key, 5)
    N = future_phis_all.shape[1]

    if state_action:
        sampled_fut = jax.random.randint(key_actions_fut, (B, N), 0, A)
        fut = jnp.take_along_axis(future_phis_all, sampled_fut[:, :, None, None], axis=2).squeeze(2)
    else:
        fut = future_phis_all

    rand_steps = jax.random.randint(key_time, (B,), 0, N)
    rand_fut   = fut[jnp.arange(B), rand_steps, :]
    perm1      = jax.random.permutation(key_perm1, B)
    ortho_batch1 = rand_fut[perm1, :]

    if state_action:
        rand_a = jax.random.randint(key_actions_cur, (B,), 0, A)
        curr   = phis_all[jnp.arange(B), rand_a, :]
    else:
        curr = phi
    perm2 = jax.random.permutation(key_perm2, B)
    ortho_batch2 = curr[perm2, :]

    linear_error, quadratic_error, error_matrix = compute_orthogonality_error(
        ortho_batch1, ortho_batch2, average_loss, rms_norm
    )

    if barrier_dims == 1:
        ortho_normal_loss = jax.lax.stop_gradient(barrier_coeffs[0, 0]) * quadratic_error
    else:
        b_diag = jax.lax.stop_gradient(jnp.diag(barrier_coeffs))
        ortho_normal_loss = jnp.dot(b_diag, tril_row_normalization(error_matrix['quadratic_errors'], D).sum(axis=1))

    dual_loss = 0.0
    if use_dual:
        if average_loss:
            Z = (D * (D + 1)) / 2
            dual_loss = (jax.lax.stop_gradient(betas) * error_matrix['errors'] / Z).sum()
        else:
            dual_loss = (jax.lax.stop_gradient(betas) * error_matrix['errors']).sum()
    ortho_normal_loss += dual_loss

    ortho_weight = 1.0 if not use_weights else weights.mean()
    loss = pos_loss + ortho_weight * ortho_normal_loss

    if use_weights:
        for k in error_matrix.keys():
            error_matrix[k] = ortho_weight * error_matrix[k]

    return loss, (pos_loss, quadratic_error, -1 * linear_error), (ortho_weight, error_matrix)


@partial(jax.jit, static_argnums=range(7, 15))
def compute_multi_step_graph_drawing_loss(
    phi, target_phis, sample_mask, masks, weights=None,

    novelty_values: jnp.ndarray = None,   # [B, n_steps]
    zeta_values: jnp.ndarray = None,      # [B, n_steps+1]
    use_novelty: bool = True,
    use_zeta: bool = False,
    normalize_zeta: bool = True,

    gamma=0.99, uni_policy=False, n_steps=1,
    use_weights=False, average_loss=True
):
    B = phi.shape[0]
    eps = 1e-8

    # ------------------------------------------------------------
    # done-masking to only include transitions until first done
    # ------------------------------------------------------------
    done_indicator = ~masks                               # [B, n_steps]
    done_exists = jnp.any(done_indicator, axis=1)         # [B]
    first_done_idx = jnp.where(done_exists,
                               jnp.argmax(done_indicator, axis=1),
                               n_steps - 1)               # [B]
    time_indices = jnp.arange(n_steps)[None, :]           # [1, n_steps]
    new_mask = time_indices <= first_done_idx[:, None]    # [B, n_steps]

    # ------------------------------------------------------------
    # novelty matrix (per-step)
    # ------------------------------------------------------------
    if (not use_novelty) or (novelty_values is None):
        nov = jnp.ones((B, n_steps), dtype=phi.dtype)
    else:
        nov = jnp.asarray(novelty_values, dtype=phi.dtype)     # [B, n_steps]
        nov = jnp.maximum(nov, 0.0)

    nov = nov * new_mask.astype(phi.dtype)  # kill invalid steps early

    # ------------------------------------------------------------
    # per-time weights within each env: gamma^k * mask * novelty * IS
    # normalize across time within row
    # ------------------------------------------------------------
    base_time = jnp.power(gamma, jnp.arange(n_steps, dtype=phi.dtype))[None, :]  # [1, n_steps]
    base_time = base_time * new_mask.astype(phi.dtype)                           # [B, n_steps]
    base_time = base_time * nov                                                  # [B, n_steps]

    if use_weights:
        base_time = base_time * jnp.asarray(weights, dtype=phi.dtype)            # [B, n_steps]

    w_time = base_time / (jnp.sum(base_time, axis=1, keepdims=True) + eps)       # [B, n_steps]

    # ------------------------------------------------------------
    # per-env (row) weights across envs:
    # row_max / max(row_max), then make it a prob-dist with mean=1
    # ------------------------------------------------------------
    row_max = jnp.max(nov, axis=1) * sample_mask.astype(phi.dtype)               # [B]
    row_max = jnp.maximum(row_max, 0.0)

    row_w = row_max / (jnp.max(row_max) + eps)                                   # [B] in [0,1]

    # "probability distribution" across envs BUT keep overall scale stable:
    # sum(row_w_prob) = B  => mean(row_w_prob) = 1
    row_w = row_w / (jnp.sum(row_w) + eps) * B                                   # [B]

    # ------------------------------------------------------------
    # Prepare target nodes (collapse action axis if uni_policy)
    # ------------------------------------------------------------
    if target_phis.ndim == 4:
        targ_nodes = jnp.mean(target_phis, axis=2)   # [B, n_steps, D]
    else:
        targ_nodes = target_phis                     # [B, n_steps, D]

    # ------------------------------------------------------------
    # 1) Smoothness term (time-weighted AND row-weighted)
    # ------------------------------------------------------------
    diff_sq = (phi[:, None, :] - targ_nodes) ** 2                                # [B, n_steps, D]
    diff_sq = diff_sq * new_mask[:, :, None].astype(phi.dtype)                   # mask
    # time-weight then sum over time and D => per-row scalar
    per_row_smooth = jnp.sum(w_time[:, :, None] * diff_sq, axis=1)               # [B, D]
    per_row_smooth = jnp.sum(per_row_smooth, axis=-1)                            # [B]
    per_row_smooth = per_row_smooth * sample_mask.astype(phi.dtype)              # [B]
    per_row_smooth = per_row_smooth * row_w                                      # [B]

    smooth_loss = jnp.mean(per_row_smooth) if average_loss else jnp.sum(per_row_smooth)

    # ------------------------------------------------------------
    # 2) Zeta sink term (row-weighted ONLY, no novelty)
    # ------------------------------------------------------------
    if use_zeta:
        z = jnp.asarray(zeta_values, dtype=phi.dtype)                            # [B, n_steps+1]
        if normalize_zeta:
            z = z / (jnp.mean(z) + eps)

        # anchor
        phi0_sq = jnp.sum(phi * phi, axis=-1)                                    # [B]
        sink0 = (z[:, 0] * sample_mask.astype(phi.dtype)) * phi0_sq              # [B]

        # futures
        targ_sq = jnp.sum(targ_nodes * targ_nodes, axis=-1)                      # [B, n_steps]
        zf = z[:, 1:] * new_mask.astype(phi.dtype)                               # [B, n_steps]
        zf = zf * sample_mask.astype(phi.dtype)[:, None]                         # [B, n_steps]

        denom = jnp.sum(new_mask, axis=1) + eps
        sink_future = jnp.sum(zf * targ_sq, axis=1) / denom                       # [B]

        sink_vec = sink0 + sink_future                                            # [B]
        sink_vec = sink_vec * row_w                                               # row-weighted
        sink_loss = jnp.mean(sink_vec) if average_loss else jnp.sum(sink_vec)
    else:
        sink_loss = 0.0

    return smooth_loss + sink_loss


@partial(jax.jit, static_argnums=(2, 3))
def compute_orthogonality_error(phi_u, phi_v, average_loss=True, rms_norm=False):
    # Combine phi_u and phi_v into one tensor of shape [B, n+1, d]
    all_phis = jnp.concatenate([phi_u[:, None, :], phi_v[:, None, :]], axis=1)  # [B, n+1, d]
    B, T, d = all_phis.shape  # T = n+1

    # Compute the batch-averaged inner product for each time step.
    M = jnp.einsum('btd,bte->tde', all_phis, jax.lax.stop_gradient(all_phis))  # Shape: [T, d, d]
    E = jnp.tril(M - B * jnp.eye(d)) / B  # if rms_norm else jnp.tril(M/B - jnp.eye(d))
    avg_err = E.mean(axis=0)
    # quad_err = (E ** 2).mean(axis=0)
    # since these two batches are assumed to be from independent draws, but the loss goes negative without clipping
    quad_err = jnp.maximum(jnp.prod(E, axis=0), 0)

    if average_loss:
        # loss     = tril_row_normalization(avg_err,  d).sum(axis=1).mean()
        quad_loss= tril_row_normalization(quad_err, d).sum(axis=1).mean()
        Z = (d*(d+1))/2
        loss = (avg_err / Z).sum()
        # quad_loss = (quad_err / Z).sum()
        return loss, quad_loss, {'errors': avg_err, 'quadratic_errors': quad_err}

    return avg_err.sum(), quad_err.sum(), {'errors': avg_err, 'quadratic_errors': quad_err}


# @partial(jax.jit, static_argnums=(2, 3))
# def compute_orthogonality_error(phi_u, phi_v, average_loss=True, rms_norm=False):
#     def _lower_tri(m, keep_diag=True):
#         return jnp.tril(m, 0 if keep_diag else -1)
#
#     def _gram_err(phi):
#         B, d = phi.shape
#         G = (phi.T @ jax.lax.stop_gradient(phi)) / B
#         return _lower_tri(G - jnp.eye(d), keep_diag=True)  # include diag
#
#     E1 = _gram_err(phi_u)       # [d,d]
#     E2 = _gram_err(phi_v)       # [d,d]
#
#     # Split diagonal vs off-diagonal
#     d = E1.shape[0]
#     diag_mask = jnp.eye(d, dtype=E1.dtype)
#     off_mask  = _lower_tri(jnp.ones_like(E1)) - diag_mask
#
#     # Diagonal: use sum of squares (always ≥0)
#     diag1 = E1 * diag_mask
#     diag2 = E2 * diag_mask
#     diag_sq = 0.5 * (diag1**2 + diag2**2)         # option (b) on diag
#
#     off_prod = (E1 * E2) * off_mask
#     quad_err = diag_sq + off_prod
#     avg_err  = 0.5 * (E1 + E2)
#
#     if average_loss:
#         loss     = tril_row_normalization(avg_err,  d).sum(axis=1).mean()
#         quad_loss= tril_row_normalization(quad_err, d).sum(axis=1).mean()
#         return loss, quad_loss, {'errors': avg_err, 'quadratic_errors': quad_err}
#
#     return avg_err.sum(), quad_err.sum(), {'errors': avg_err, 'quadratic_errors': quad_err}


@partial(jax.jit, static_argnums=range(2, 8))
def update_ALLO_parameters(lap_state, curr_errors, update_rates, barrier_args, dual_args, dual_a_min=-500,
                           average_loss=True, barrier_dims=1):
    moving_error_estimates = {'errors': lap_state.errors, 'quadratic_errors': lap_state.quadratic_errors}
    new_error_estimates, error_norms = update_error_estimates(moving_error_estimates, curr_errors,
                                                              barrier_args.on, dual_args.on,
                                                              update_rates.error, update_rates.q_error,)
    if barrier_args.on:
        new_quad_errors = new_error_estimates['quadratic_errors']
        new_barrier_coeffs = update_barrier_coefficients(lap_state.barrier_coeffs, new_quad_errors,
                                                         barrier_lr=barrier_args.lr, use_clip=barrier_args.clip,
                                                         average_loss=average_loss, barrier_dims=barrier_dims)
        lap_state = lap_state.replace(quadratic_errors=new_error_estimates['quadratic_errors'],
                                      barrier_coeffs=new_barrier_coeffs)

    if dual_args.on:
        new_betas = update_duals(new_error_estimates['errors'], lap_state.betas,
                                 dual_lr=dual_args.lr, a_min=dual_a_min,
                                 use_clip=dual_args.clip, average_loss=average_loss)
        lap_state = lap_state.replace(errors=new_error_estimates['errors'], betas=new_betas)

    return lap_state


@partial(jax.jit, static_argnums=range(2, 9))
def update_barrier_coefficients(barrier_coefficients, quadratic_error_matrix,
                                barrier_lr=1, a_min=0.0, a_max=10000.0,
                                clip_value=1.0, use_clip=False, average_loss=True, barrier_dims=1):
    D = quadratic_error_matrix.shape[0]
    q_vec = tril_row_normalization(quadratic_error_matrix, D).sum(axis=1)  # [D]

    if barrier_dims == 1:
        quadratic_error = q_vec.mean() if average_loss else q_vec.sum()

        raw_update = barrier_lr * quadratic_error  # [D]
        if use_clip:
            raw_update = jnp.clip(raw_update, -clip_value, clip_value)

        new_barrier = jnp.clip(barrier_coefficients + raw_update, a_min, a_max)
        return jnp.tril(new_barrier)

    else:
        if average_loss:
            q_vec = q_vec / D

        raw_update = barrier_lr * q_vec  # [D]
        if use_clip:
            raw_update = jnp.clip(raw_update, -clip_value, clip_value)

        b_diag = jnp.diag(barrier_coefficients)             # [D]
        new_diag = jnp.clip(b_diag + raw_update, a_min, a_max)  # [D]
        return jnp.diag(new_diag)


@partial(jax.jit, static_argnums=range(2, 8))
def update_duals(error_matrix, betas, dual_lr=1e-4, a_min=-500, a_max=0, clip_value=0.1,
                 use_clip=False, average_loss=True):
    # if average_loss:
    #     D = error_matrix.shape[0]
    #     error_matrix = tril_row_normalization(error_matrix, D) #/ D
    #     # Z = (D * (D + 1)) / 2
    #     # error_matrix = error_matrix / Z

    updates = error_matrix
    if use_clip:
        updates = jnp.clip(updates, -clip_value, clip_value)

    updated_duals = betas + dual_lr * updates
    # Positional (min, max): jax removed the a_min/a_max keywords in 0.4.30.
    updated_duals = jnp.clip(updated_duals, a_min, a_max)
    return jnp.tril(updated_duals)


@partial(jax.jit, static_argnums=range(2, 5))
def update_error_estimates(moving_errors, errors, use_barrier, use_dual, err_rate, q_err_rate):
    updates = {}
    error_norm = {'errors': 0.0, 'quadratic_errors': 0.0}
    update_rates = {'errors': err_rate, 'quadratic_errors': q_err_rate}

    for error_type in ['errors', 'quadratic_errors']:
        if (error_type == 'errors' and not use_dual) or (error_type == 'quadratic_errors' and not use_barrier):
            continue

        # Get old moving error estimates
        old = moving_errors[error_type]
        error_norm[error_type] = jnp.linalg.norm(old)

        # Set update rate to 1 in the first iteration
        non_init_update_rate = update_rates[error_type]
        init_coeff = jnp.isclose(error_norm[error_type], 0.0, rtol=1e-10, atol=1e-13)
        update_rate = init_coeff + (1 - init_coeff) * non_init_update_rate

        # Update error estimates
        update = old + update_rate * (errors[error_type] - old)
        updates[error_type] = update

    # error_norm[error_type] = jnp.linalg.norm(errors[error_type])
    return updates, error_norm


################################################################
@partial(jax.jit, static_argnums=(2, ))
def compute_batch_eigenvalues(phi, phi_next, eps=1e-8):
    # Numerator: average squared difference between embeddings of consecutive states.
    num = 0.5 * jnp.mean((phi - phi_next) ** 2, axis=0)
    # Denominator: average squared norm of embeddings from both states.
    denom = jnp.mean(0.5 * (phi ** 2 + phi_next ** 2), axis=0)

    # Rayleigh quotient: if eigenvectors are not normalized, this ratio corrects for it.
    lambda_batch = num / (denom + eps)
    return lambda_batch


@partial(jax.jit, static_argnums=(2))
def update_eigenvalues(lambda_current, lambda_batch, alpha=0.1):
    return (1 - alpha) * lambda_current + alpha * lambda_batch


@partial(jax.jit, static_argnums=(2,))
def get_laplacian_metrics(curr_hidden, neg_hidden, average_loss=True):
    # neg_dist = compute_multi_step_graph_drawing_loss(curr_hidden, neg_hidden, average_loss=average_loss)
    dist = (curr_hidden - neg_hidden)**2
    neg_dist = dist.mean(0).sum() if average_loss else dist.sum()
    row_norm = jnp.linalg.norm(curr_hidden, ord=2, axis=1).mean()
    return neg_dist, row_norm