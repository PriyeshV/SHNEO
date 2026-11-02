import jax
import jax.numpy as jnp
from flax import struct
from functools import partial


@jax.jit
def value_rescale(x: jnp.ndarray, eps: float = 1e-3) -> jnp.ndarray:
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + eps * x


@jax.jit
def inv_value_rescale(y: jnp.ndarray, eps: float = 1e-3) -> jnp.ndarray:
    ay = jnp.abs(y)
    s = jnp.sign(y)
    inside = 1.0 + 4.0 * eps * (ay + 1.0 + eps)
    t = (jnp.sqrt(inside) - 1.0) / (2.0 * eps)
    return s * (t * t - 1.0)


@partial(jax.jit, static_argnums=(4, 5))
def compute_td_lambda_targets(next_q, q_vals, rewards, dones, gamma, td_lambda):
    G_last = rewards[-1] + gamma * (1 - dones[-1]) * next_q
    q_last = jnp.max(q_vals[-1], axis=-1)

    def _get_target(carry, data):
        G_next, q_next = carry
        r_t, q_t, done_t = data

        B_t = r_t + gamma * (1 - done_t) * q_next
        delta_t = G_next - q_next
        G_t = B_t + gamma * td_lambda * delta_t
        G_t = (1 - done_t) * G_t + done_t * r_t
        q_prev = jnp.max(q_t, axis=-1)
        return (G_t, q_prev), G_t

    (_, _), targets = jax.lax.scan(
        _get_target,
        (G_last, q_last),
        jax.tree_util.tree_map(lambda x: x[:-1], (rewards, q_vals, dones)),
        reverse=True,
    )
    targets = jnp.concatenate([targets, G_last[jnp.newaxis]], axis=0)
    return targets


@partial(jax.jit, static_argnums=(4, 5))
def compute_transformed_td_lambda_targets(next_q_t, q_vals_t, rewards, dones, gamma, td_lambda):
    next_q = inv_value_rescale(next_q_t)
    q_last = inv_value_rescale(jnp.max(q_vals_t[-1], axis=-1))

    G_last_raw = rewards[-1] + gamma * (1.0 - dones[-1]) * next_q
    G_last_t = value_rescale(G_last_raw)

    def _get_target(carry, data):
        G_next_raw, q_next_raw = carry
        r_t, q_t_t, done_t = data

        q_prev_raw = inv_value_rescale(jnp.max(q_t_t, axis=-1))
        B_t_raw = r_t + gamma * (1.0 - done_t) * q_next_raw
        delta_t_raw = G_next_raw - q_next_raw
        G_t_raw = B_t_raw + gamma * td_lambda * delta_t_raw
        G_t_raw = (1.0 - done_t) * G_t_raw + done_t * r_t

        return (G_t_raw, q_prev_raw), value_rescale(G_t_raw)

    (_, _), targets_t = jax.lax.scan(
        _get_target,
        (G_last_raw, q_last),
        jax.tree_util.tree_map(lambda x: x[:-1], (rewards, q_vals_t, dones)),
        reverse=True,
    )
    targets_t = jnp.concatenate([targets_t, G_last_t[jnp.newaxis]], axis=0)
    return targets_t


@partial(jax.jit, static_argnums=(2, 3, 4))
def learn_phase(carry, minibatch_and_target, use_enc_bn, use_q_bn, use_value_rescale):
    train_state, _key = carry
    minibatch, target = minibatch_and_target
    obs = jnp.transpose(minibatch.obs, (0, 2, 3, 1))

    def _loss_fn(enc_params, q_params):
        if use_enc_bn:
            embeddings, enc_bn_update = train_state.enc_state.apply_fn(
                {'params': enc_params, 'batch_stats': train_state.enc_state.batch_stats}, obs, train=True,
                mutable=['batch_stats'])
        else:
            embeddings, enc_bn_update = train_state.enc_state.apply_fn({'params': enc_params}, obs, train=True), None
        if use_q_bn:
            (q_values, q_bn_update) = train_state.q_state.apply_fn(
                {'params': q_params, 'batch_stats': train_state.q_state.batch_stats},
                embeddings, train=True, mutable=['batch_stats'])
        else:
            q_values, q_bn_update = train_state.q_state.apply_fn({'params': q_params}, embeddings), None

        q_preds = jnp.take_along_axis(q_values, jnp.expand_dims(minibatch.actions, axis=-1), axis=-1).squeeze(axis=-1)
        td_loss = 0.5 * jnp.square(q_preds - target).mean()

        q_mean_t = q_preds.mean()
        q_mean_raw = inv_value_rescale(q_preds).mean() if use_value_rescale else q_mean_t
        return td_loss, ((enc_bn_update, q_bn_update), (q_mean_t, q_mean_raw))


    (loss, ((enc_bn_updates, q_bn_updates), q_stats)), grads = (jax.value_and_grad(_loss_fn, has_aux=True, argnums=(0, 1))
                                                              (train_state.enc_state.params, train_state.q_state.params))
    grads_enc_w, grads_q_w = grads

    # apply those gradients to get **new** sub‐states -- only shallow copies/references made
    new_enc_state = train_state.enc_state.apply_gradients(grads=grads_enc_w)
    new_q_state = train_state.q_state.apply_gradients(grads=grads_q_w)
    if use_enc_bn:
        new_enc_state = new_enc_state.replace(batch_stats=enc_bn_updates['batch_stats'])
    if use_q_bn:
        new_q_state = new_q_state.replace(batch_stats=q_bn_updates['batch_stats'])

    train_state = train_state.replace(enc_state=new_enc_state, q_state=new_q_state, grad_steps=train_state.grad_steps+1)
    return (train_state, _key), (loss, q_stats)


@partial(jax.jit, static_argnums=(4, 5, 6))
def get_dual_q_values(obs, enc_state, q_ext_state, q_int_state, use_enc_bn, use_qext_bn, use_qint_bn):
    obs = jnp.transpose(obs, (0, 2, 3, 1))

    enc_vars = {'params': enc_state.params, 'batch_stats': enc_state.batch_stats} if use_enc_bn else {'params': enc_state.params}
    embeddings = enc_state.apply_fn(enc_vars, obs)

    qext_vars = {'params': q_ext_state.params, 'batch_stats': q_ext_state.batch_stats} if use_qext_bn else {'params': q_ext_state.params}
    q_ext = q_ext_state.apply_fn(qext_vars, embeddings)

    qint_vars = {'params': q_int_state.params, 'batch_stats': q_int_state.batch_stats} if use_qint_bn else {'params': q_int_state.params}
    q_int = q_int_state.apply_fn(qint_vars, embeddings)

    return q_ext, q_int


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def learn_phase_dual(carry, minibatch_and_targets, use_enc_bn, use_q_bn, use_value_rescale, int_coef):
    train_state, _key = carry
    minibatch, ext_target, int_target = minibatch_and_targets
    obs = jnp.transpose(minibatch.obs, (0, 2, 3, 1))

    def _loss_fn(enc_params, qext_params, qint_params):
        if use_enc_bn:
            embeddings, enc_bn_update = train_state.enc_state.apply_fn(
                {'params': enc_params, 'batch_stats': train_state.enc_state.batch_stats},
                obs, train=True, mutable=['batch_stats']
            )
        else:
            embeddings, enc_bn_update = train_state.enc_state.apply_fn(
                {'params': enc_params}, obs, train=True
            ), None

        if use_q_bn:
            q_ext_vals, qext_bn_update = train_state.q_ext_state.apply_fn(
                {'params': qext_params, 'batch_stats': train_state.q_ext_state.batch_stats},
                embeddings, train=True, mutable=['batch_stats']
            )
            q_int_vals, qint_bn_update = train_state.q_int_state.apply_fn(
                {'params': qint_params, 'batch_stats': train_state.q_int_state.batch_stats},
                embeddings, train=True, mutable=['batch_stats']
            )
        else:
            q_ext_vals, qext_bn_update = train_state.q_ext_state.apply_fn(
                {'params': qext_params}, embeddings
            ), None
            q_int_vals, qint_bn_update = train_state.q_int_state.apply_fn(
                {'params': qint_params}, embeddings
            ), None

        aidx = jnp.expand_dims(minibatch.actions, axis=-1)
        q_ext_pred = jnp.take_along_axis(q_ext_vals, aidx, axis=-1).squeeze(axis=-1)
        q_int_pred = jnp.take_along_axis(q_int_vals, aidx, axis=-1).squeeze(axis=-1)

        ext_loss = 0.5 * jnp.square(q_ext_pred - ext_target).mean()
        int_loss = 0.5 * jnp.square(q_int_pred - int_target).mean()
        total_loss = ext_loss + int_coef * int_loss

        q_ext_mean_t = q_ext_pred.mean()
        q_int_mean_t = q_int_pred.mean()

        if use_value_rescale:
            q_ext_mean_raw = inv_value_rescale(q_ext_pred).mean()
            q_int_mean_raw = inv_value_rescale(q_int_pred).mean()
            q_sel_mean_raw = (inv_value_rescale(q_ext_pred) + int_coef * inv_value_rescale(q_int_pred)).mean()
            q_sel_mean_t = value_rescale(
                inv_value_rescale(q_ext_pred) + int_coef * inv_value_rescale(q_int_pred)
            ).mean()
        else:
            q_ext_mean_raw = q_ext_mean_t
            q_int_mean_raw = q_int_mean_t
            q_sel_mean_raw = (q_ext_pred + int_coef * q_int_pred).mean()
            q_sel_mean_t = q_sel_mean_raw

        aux = (
            (enc_bn_update, qext_bn_update, qint_bn_update),
            (ext_loss, int_loss,
             q_ext_mean_t, q_ext_mean_raw, q_int_mean_t, q_int_mean_raw, q_sel_mean_t, q_sel_mean_raw)
        )
        return total_loss, aux

    (loss, ((enc_bn_updates, qext_bn_updates, qint_bn_updates), stats)), grads = (
        jax.value_and_grad(_loss_fn, has_aux=True, argnums=(0, 1, 2))(
            train_state.enc_state.params,
            train_state.q_ext_state.params,
            train_state.q_int_state.params,
        )
    )
    g_enc, g_qext, g_qint = grads

    new_enc_state = train_state.enc_state.apply_gradients(grads=g_enc)
    new_qext_state = train_state.q_ext_state.apply_gradients(grads=g_qext)
    new_qint_state = train_state.q_int_state.apply_gradients(grads=g_qint)

    if use_enc_bn:
        new_enc_state = new_enc_state.replace(batch_stats=enc_bn_updates['batch_stats'])
    if use_q_bn:
        new_qext_state = new_qext_state.replace(batch_stats=qext_bn_updates['batch_stats'])
        new_qint_state = new_qint_state.replace(batch_stats=qint_bn_updates['batch_stats'])

    train_state = train_state.replace(
        enc_state=new_enc_state,
        q_ext_state=new_qext_state,
        q_int_state=new_qint_state,
        grad_steps=train_state.grad_steps + 1,
    )
    return (train_state, _key), (loss, stats)