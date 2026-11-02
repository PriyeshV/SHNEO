# mains/algorithms/NEO.py
from functools import partial
import jax
import jax.numpy as jnp
from flax import struct

from src.mains.algorithms.PQN import (
    compute_td_lambda_targets,
    compute_transformed_td_lambda_targets,
)


@struct.dataclass
class NEOScaleState:
    max_inv_ema: jnp.ndarray
    mean_scale_ema: jnp.ndarray


@partial(jax.jit, static_argnums=range(2, 7))
def neo_scale_from_state_calibrated(
    inv_count: jnp.ndarray,
    neo_state: NEOScaleState,
    delta: float,
    k: float,
    target_mean: float = 0.3,
    tau_mean: float = 1e-3,
    eps: float = 1e-8,
):
    denom = jnp.maximum(neo_state.max_inv_ema, eps)
    x = inv_count / denom
    x = jnp.clip(x, 0.0, 1.0)

    z = delta * jnp.power(x, 1.0 / k)

    z_mean = jnp.mean(z)
    desired_scale = target_mean / (z_mean + eps)
    desired_scale = jnp.clip(desired_scale, 0.1, 10.0)

    new_scale = (1.0 - tau_mean) * neo_state.mean_scale_ema + tau_mean * desired_scale
    z_cal = jnp.clip(z * new_scale, 0.0, delta)

    new_state = neo_state.replace(mean_scale_ema=new_scale)
    return z_cal, new_state


def _neo_eigen_rewards(train_state, transitions, n_options):
    """Evaluate magnitude-based NEO rewards with the original rollout mask.
    ``abs(phi(s')) - abs(phi(s))``.
    """
    T, E, C, H, W = transitions.obs.shape

    obs_nhwc = jnp.transpose(transitions.obs, (0, 1, 3, 4, 2))
    next_obs_nhwc = jnp.transpose(transitions.next_obs, (0, 1, 3, 4, 2))
    all_states = jnp.concatenate(
        [obs_nhwc, next_obs_nhwc[-1:]],
        axis=0,
    )
    flat_states = all_states.reshape(((T + 1) * E, H, W, C))

    lap_vars = {"params": train_state.lap_state.params}
    if train_state.lap_state.batch_stats is not None:
        lap_vars["batch_stats"] = train_state.lap_state.batch_stats

    if train_state.lap_state.batch_stats is not None:
        phis_all, _ = train_state.lap_state.apply_fn(
            lap_vars,
            flat_states,
            train=False,
            mutable=["batch_stats"],
        )
    else:
        phis_all = train_state.lap_state.apply_fn(
            lap_vars,
            flat_states,
            train=False,
        )

    # The source Laplacian generally has no constant first eigenfunction.
    phis_all = phis_all[:, :n_options]

    phis = phis_all[: T * E].reshape((T, E, n_options))
    next_phis = phis_all[E : (T + 1) * E].reshape(
        (T, E, n_options)
    )

    raw_eig_rewards = jnp.abs(next_phis) - jnp.abs(phis)

    # Original cumulative mask: keep the prefix through the first terminal
    # transition (including that transition) in each environment's rollout.
    dones_f = transitions.dones.astype(jnp.float32)
    not_done = 1.0 - dones_f
    prefix = jnp.concatenate(
        [jnp.ones((1, E), dtype=jnp.float32), not_done[:-1]],
        axis=0,
    )
    alive_before = jnp.cumprod(prefix, axis=0)
    mask = jnp.clip(alive_before + dones_f, 0.0, 1.0)

    eig_rewards = raw_eig_rewards * mask[..., None]
    return eig_rewards, mask


def _option_bootstrap(
    train_state,
    transitions,
    q_states,
    use_opt_bn,
):
    """Return max_a Q_o(s_T, a) for every option and environment."""
    next_obs_last = jnp.transpose(
        transitions.next_obs[-1],
        (0, 2, 3, 1),
    )

    enc_vars = {"params": train_state.opt_enc_state.params}
    if use_opt_bn:
        enc_vars["batch_stats"] = train_state.opt_enc_state.batch_stats
    embedding = train_state.opt_enc_state.apply_fn(
        enc_vars,
        next_obs_last,
        train=False,
    )

    q_params = jax.tree_util.tree_map(
        lambda *params: jnp.stack(params),
        *[state.params for state in q_states],
    )
    q_batch_stats = (
        jax.tree_util.tree_map(
            lambda *stats: jnp.stack(stats),
            *[state.batch_stats for state in q_states],
        )
        if use_opt_bn
        else None
    )
    q_apply = q_states[0].apply_fn

    def apply_one(params, batch_stats):
        variables = {"params": params}
        if use_opt_bn:
            variables["batch_stats"] = batch_stats
        return q_apply(variables, embedding, train=False)

    q_boot = jax.vmap(
        apply_one,
        in_axes=(0, 0 if use_opt_bn else None),
    )(q_params, q_batch_stats)
    return jnp.max(q_boot, axis=-1)


def _target_function(gamma, td_lambda, use_value_rescale):
    target_fn = (
        compute_transformed_td_lambda_targets
        if use_value_rescale
        else compute_td_lambda_targets
    )
    return partial(target_fn, gamma=gamma, td_lambda=td_lambda)


@partial(jax.jit, static_argnums=range(2, 9))
def compute_neo_options_lambda_targets_shared(
    train_state,
    transitions,
    gamma,
    td_lambda,
    use_opt_bn,
    n_options,
    use_value_rescale,
    rnd_coef,
    stall_coef,
):
    """Compute single-head NEO TD(lambda) targets.

    The NEO component is magnitude based.  RND augmentation and the stall
    penalty are project-specific extensions; set their coefficients to zero for
    the closest baseline to original NEO.

    Returns:
        lambda_targets: [O, T, E]
        eig_rewards:    [T, E, O]
        stall_mean:     scalar
    """
    eig_rewards, mask = _neo_eigen_rewards(
        train_state,
        transitions,
        n_options,
    )

    rnd_rewards = (
        transitions.rnd_bonus_n[:, :, None]
        * mask[..., None]
    )

    # This penalty is not part of original NEO.  Keep the threshold near the
    # observed one-step feature-difference scale and disable with stall_coef=0.
    stall_threshold = jnp.float32(0.02)
    stall = jax.nn.relu(
        stall_threshold - jnp.abs(eig_rewards)
    ) / (stall_threshold + 1e-8)
    stall = stall * mask[..., None]

    option_rewards = (
        eig_rewards
        + jnp.float32(rnd_coef) * rnd_rewards
        - jnp.float32(stall_coef) * stall
    )
    stall_mean = jnp.sum(stall) / (
        jnp.sum(mask) * jnp.float32(n_options) + 1e-8
    )

    max_q_last = _option_bootstrap(
        train_state,
        transitions,
        train_state.opt_q_states,
        use_opt_bn,
    )

    rewards_by_option = jnp.transpose(option_rewards, (2, 0, 1))
    q_by_option = jnp.transpose(
        transitions.option_q_vals,
        (2, 0, 1, 3),
    )
    target_fn = _target_function(gamma, td_lambda, use_value_rescale)

    def targets_for_one_option(q_values, rewards, bootstrap):
        # The target routine handles episode termination through dones.
        return target_fn(bootstrap, q_values, rewards, transitions.dones)

    lambda_targets = jax.vmap(
        targets_for_one_option,
        in_axes=(0, 0, 0),
    )(q_by_option, rewards_by_option, max_q_last)

    return lambda_targets, eig_rewards, stall_mean


@partial(jax.jit, static_argnums=range(2, 9))
def compute_neo_options_lambda_targets_shared_dual(
    train_state,
    transitions,
    gamma,
    td_lambda,
    use_opt_bn,
    n_options,
    use_value_rescale,
    stall_coef,
    rnd_coef,
):
    """Compute separate eigen-potential and RND TD(lambda) targets.

    ``rnd_coef`` is retained for call-site compatibility.  The RND target is
    intentionally unscaled here; its coefficient is applied when the two value
    heads are combined for action selection and learning.

    Returns:
        ext_lambda_targets: [O, T, E]
        int_lambda_targets: [O, T, E]
        eig_rewards:        [T, E, O]
        stall_mean:         scalar
    """
    del rnd_coef

    eig_rewards, mask = _neo_eigen_rewards(
        train_state,
        transitions,
        n_options,
    )

    # Retain the original active dual-path threshold for a controlled change.
    stall_threshold = jnp.float32(1.0)
    stall = jax.nn.relu(
        stall_threshold - jnp.abs(eig_rewards)
    ) / (stall_threshold + 1e-8)
    stall = stall * mask[..., None]

    eigen_rewards = eig_rewards - jnp.float32(stall_coef) * stall
    rnd_rewards = (
        transitions.rnd_bonus_n[:, :, None]
        * mask[..., None]
    )
    rnd_rewards = jnp.broadcast_to(rnd_rewards, eig_rewards.shape)

    stall_mean = jnp.sum(stall) / (
        jnp.sum(mask) * jnp.float32(n_options) + 1e-8
    )

    max_qext_last = _option_bootstrap(
        train_state,
        transitions,
        train_state.opt_q_ext_states,
        use_opt_bn,
    )
    max_qint_last = _option_bootstrap(
        train_state,
        transitions,
        train_state.opt_q_int_states,
        use_opt_bn,
    )

    eigen_rewards_by_option = jnp.transpose(eigen_rewards, (2, 0, 1))
    rnd_rewards_by_option = jnp.transpose(rnd_rewards, (2, 0, 1))
    qext_by_option = jnp.transpose(
        transitions.option_q_ext_vals,
        (2, 0, 1, 3),
    )
    qint_by_option = jnp.transpose(
        transitions.option_q_int_vals,
        (2, 0, 1, 3),
    )
    target_fn = _target_function(gamma, td_lambda, use_value_rescale)

    def targets_for_one_option(q_values, rewards, bootstrap):
        return target_fn(bootstrap, q_values, rewards, transitions.dones)

    ext_lambda_targets = jax.vmap(
        targets_for_one_option,
        in_axes=(0, 0, 0),
    )(qext_by_option, eigen_rewards_by_option, max_qext_last)
    int_lambda_targets = jax.vmap(
        targets_for_one_option,
        in_axes=(0, 0, 0),
    )(qint_by_option, rnd_rewards_by_option, max_qint_last)

    return (
        ext_lambda_targets,
        int_lambda_targets,
        eig_rewards,
        stall_mean,
    )


@partial(jax.jit, static_argnums=(1, 2, 3))
def source_eigenvalues_to_option_durations(
    eigenvalues,
    n_options,
    min_duration,
    max_duration,
    eps=1e-6,
):
    """Map source-Laplacian eigenvalues to option durations.

    Unlike the ordinary random-walk Laplacian, the source Laplacian generally
    has no constant first eigenvector.  Therefore, this mapping starts at index
    zero, matching the eigenfunctions used to construct the option rewards.
    """
    eigenvalues = jnp.reshape(eigenvalues, (-1,))
    eigenvalues = jnp.nan_to_num(
        eigenvalues,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    usable = jnp.maximum(eigenvalues[:n_options], eps)
    durations = jnp.ceil(1.0 / usable).astype(jnp.int32)
    return jnp.clip(durations, min_duration, max_duration)
