# mains/algorithms/Laplacian.py
import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
from src.losses.laplacian import multi_step_ALLO, update_ALLO_parameters, update_eigenvalues


@jax.jit
def compute_batch_eigenvalues_seq(phis, valid_transitions, eps=1e-8):
    """
    phis: [B,T,D]
    valid_transitions: [B,T-1]
    """
    phi = phis[:, :-1, :]
    phi1 = phis[:, 1:, :]

    w = valid_transitions[..., None]
    wsum = jnp.sum(w, axis=(0, 1)) + eps

    num = jnp.sum(w * (phi - phi1) ** 2, axis=(0, 1)) / wsum
    denom = jnp.sum(w * 0.5 * (phi ** 2 + phi1 ** 2), axis=(0, 1)) / wsum
    return num / (denom + eps)


@partial(jax.jit, static_argnums=(3, 4,))
def update_laplacian_network(key, train_state, lap_batch, args, n_actions):
    """
    Compatible with PQN_EO.py call site.

    lap_batch.obs: [B, N, C, H, W]
    lap_batch.dones: [B, N]
    lap_batch.actions: [B, N]
    lap_batch.action_probs: unused here
    """
    gamma = args.env.gamma
    use_bar = args.laplacian.barrier.on
    use_dual = args.laplacian.dual.on
    average_loss = args.laplacian.average_loss
    use_bn = args.laplacian.batch_norm

    lap_state = train_state.lap_state

    B, N = lap_batch.obs.shape[:2]
    obs_nhwc = jnp.transpose(lap_batch.obs, (0, 1, 3, 4, 2))
    flat = obs_nhwc.reshape((B * N,) + obs_nhwc.shape[2:])  # [(B*N),H,W,C]

    valid_mask = jnp.logical_not(lap_batch.dones)
    valid_trans = (valid_mask[:, :-1] & valid_mask[:, 1:]).astype(jnp.float32)

    def laplacian_loss_fn(lap_params):
        lap_vars = {"params": lap_params}
        if use_bn:
            lap_vars["batch_stats"] = lap_state.batch_stats
            phis_flat, updates = lap_state.apply_fn(
                lap_vars, flat, train=True, mutable=["batch_stats"]
            )
        else:
            phis_flat = lap_state.apply_fn(lap_vars, flat, train=True)
            updates = None

        D = phis_flat.shape[-1]
        phis = phis_flat.reshape(B, N, D)

        mask = valid_mask[:, 0]
        future_masks = valid_mask[:, 1:]

        ph0 = phis[:, 0, :]
        fut = phis[:, 1:, :]
        n_future = fut.shape[1]

        eig_loss, (pos_loss, ortho_loss, dual_loss), (_avg_w, err_mat) = multi_step_ALLO(
            key,
            ph0,
            fut,
            None,
            mask,
            future_masks,
            lap_state.barrier_coeffs,
            lap_state.betas,
            gamma=gamma,
            state_action=False,
            n_steps=n_future,
            use_dual=use_dual,
            use_weights=False,
            weights=None,
            average_loss=average_loss,
        )

        lambda_batch = compute_batch_eigenvalues_seq(phis, valid_trans)
        lambda_batch = jax.lax.stop_gradient(lambda_batch)

        return eig_loss, (updates, pos_loss, ortho_loss, dual_loss, err_mat, lambda_batch)

    (eig_loss, (updates, pos_loss, ortho_loss, dual_loss, err_mat, lambda_batch)), grads = \
        jax.value_and_grad(laplacian_loss_fn, has_aux=True)(lap_state.params)

    new_lap_state = lap_state.apply_gradients(grads=grads)

    if use_bn:
        new_lap_state = new_lap_state.replace(batch_stats=updates["batch_stats"])

    if use_bar or use_dual:
        new_lap_state = update_ALLO_parameters(
            new_lap_state,
            err_mat,
            update_rates=args.laplacian.rate,
            barrier_args=args.laplacian.barrier,
            dual_args=args.laplacian.dual,
            average_loss=average_loss,
        )

    alpha_ev = 0.1
    ev_old = jnp.reshape(new_lap_state.eigen_values, (-1,))
    ev_new = update_eigenvalues(ev_old, lambda_batch, alpha=alpha_ev)

    # keep eigen_values shape same as setup: [D]
    ev_new = jnp.reshape(ev_new, new_lap_state.eigen_values.shape)
    new_lap_state = new_lap_state.replace(eigen_values=ev_new)

    train_state = train_state.replace(lap_state=new_lap_state)
    return train_state, (eig_loss, pos_loss, ortho_loss, dual_loss)


@partial(jax.jit, static_argnums=(3, 4))
def update_neo_laplacian_network(key, train_state, transitions, args, n_actions):
    gamma = args.env.gamma
    use_bn = args.laplacian.batch_norm
    use_barrier = args.laplacian.barrier.on
    use_dual = args.laplacian.dual.on

    lap_state = train_state.lap_state
    dones = transitions.dones
    actions = transitions.actions
    # action_probs = transitions.action_probs if imp_samp else None
    action_probs = None

    counts_all = transitions.counts.astype(jnp.float32)      # [B,N]
    novelty_all = transitions.novelty.astype(jnp.float32)    # [B,N]

    obs = jnp.transpose(transitions.obs, (0, 1, 3, 4, 2))    # [B,N,H,W,C]
    B, N, H, W, C = obs.shape
    obs_flat = obs.reshape(B * N, H, W, C)

    masks = jnp.logical_not(dones)
    mask = masks[:, 0]
    future_masks = masks[:, 1:]
    valid_trans = (masks[:, :-1] & masks[:, 1:]).astype(jnp.float32)

    weights = None

    def laplacian_loss_fn(lap_params):
        updates = None
        lap_vars = {'params': lap_params}
        mutable_keys = []

        if use_bn:
            lap_vars['batch_stats'] = lap_state.batch_stats
            mutable_keys.append('batch_stats')

        if mutable_keys:
            phis, updates = lap_state.apply_fn(lap_vars, obs_flat, train=True, mutable=mutable_keys)
        else:
            phis = lap_state.apply_fn(lap_vars, obs_flat, train=True)

        phis = phis.reshape(B, N, args.laplacian.dims)
        ph0 = phis[:, 0, :]
        future_phis = phis[:, 1:, :]

        loss, (pos_loss, ortho_loss, dual_loss), (avg_weight, new_err_mat) = multi_step_ALLO(
            key=key,
            phis_all=ph0,
            future_phis_all=future_phis,
            actions=actions,
            sample_mask=mask,
            masks=future_masks,
            barrier_coeffs=lap_state.barrier_coeffs,
            betas=lap_state.betas,
            weights=weights,
            use_novelty=args.neo.lap_novel_weight,
            novelty_values=novelty_all[:, 1:],
            zeta_values=counts_all,
            use_zeta=True,
            normalize_zeta=False,
            gamma=gamma,
            state_action=False,
            uni_policy=False,
            n_steps=future_masks.shape[1],
            use_dual=use_dual,
            use_weights=False,
            average_loss=args.laplacian.average_loss,
            rms_norm=getattr(args.laplacian, "rms_norm", False),
            barrier_dims=1,
        )

        lambda_batch = compute_batch_eigenvalues_seq(phis, valid_trans)
        return loss, (updates, pos_loss, ortho_loss, dual_loss, new_err_mat, lambda_batch)

    (eigen_loss, (updates, pos_loss, ortho_loss, dual_loss, curr_err_mat, lambda_batch)), grads = \
        jax.value_and_grad(laplacian_loss_fn, has_aux=True)(lap_state.params)

    new_lap_state = lap_state.apply_gradients(grads=grads)

    if use_bn:
        new_lap_state = new_lap_state.replace(batch_stats=updates['batch_stats'])

    if use_barrier or use_dual:
        new_lap_state = update_ALLO_parameters(
            new_lap_state,
            curr_err_mat,
            update_rates=args.laplacian.rate,
            barrier_args=args.laplacian.barrier,
            dual_args=args.laplacian.dual,
            average_loss=args.laplacian.average_loss,
        )

    ev_old = jnp.reshape(new_lap_state.eigen_values, (-1,))
    ev_new = update_eigenvalues(ev_old, lambda_batch, alpha=0.1)
    ev_new = jnp.reshape(ev_new, new_lap_state.eigen_values.shape)
    new_lap_state = new_lap_state.replace(eigen_values=ev_new)

    train_state = train_state.replace(lap_state=new_lap_state)
    return train_state, (eigen_loss, pos_loss, ortho_loss, dual_loss)

