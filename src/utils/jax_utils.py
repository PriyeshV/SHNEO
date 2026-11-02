import jax
import chex
import optax
import jax.numpy as jnp
from functools import partial
from typing import Iterable, Callable, Optional
from src.utils.utils import *


@jax.jit
def td_mse_loss(q_pred, actions, next_q_value):
    q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions]  # (batch_size,)
    loss = ((q_pred - next_q_value) ** 2)
    return loss, q_pred


@jax.jit
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    # Convert t and duration to float32 for accurate division.
    t_float = jnp.array(t, dtype=jnp.float32)
    duration_float = jnp.array(duration, dtype=jnp.float32)
    # Compute the fraction of progress, capped at 1.0.
    fraction = jnp.minimum(t_float / duration_float, 1.0)
    # Linearly interpolate between start_e and end_e.
    return start_e + fraction * (end_e - start_e)


def learning_rate_fn(init_lr, warmup_epochs, num_epochs, alpha, steps_per_epoch=1):
    """Creates learning rate schedule."""
    warmup_fn = optax.constant_schedule(init_lr)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=init_lr,
                                            decay_steps=cosine_epochs * steps_per_epoch,
                                            alpha=alpha)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn],
                                       boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn


def strict_tril_row_mean(matrix):
    # Get the strictly lower triangular part (exclude diagonal)
    lower_triangular = jnp.abs(jnp.tril(matrix, k=-1))
    num_elements = jnp.count_nonzero(lower_triangular)
    mean = lower_triangular.sum() / (num_elements + 1e-6)
    return mean


def strict_triu_row_mean(matrix):
    # Get the strictly upper triangular part (exclude diagonal)
    upper_triangular = jnp.abs(jnp.triu(matrix, k=1))
    num_elements = jnp.count_nonzero(upper_triangular)
    mean = upper_triangular.sum() / (num_elements + 1e-6)
    return mean


@partial(jax.jit, static_argnums=(1,))
def tril_row_mean(matrix, dims):
    counts = jnp.expand_dims(jnp.arange(1, dims+1), axis=1)  # shape [dims, 1]
    return matrix.sum(1)/counts


@partial(jax.jit, static_argnums=(1,))
def tril_row_normalization(matrix, dims):
    counts = jnp.expand_dims(jnp.arange(1, dims+1), axis=1)  # shape [dims, 1]
    return matrix/counts  # broadcasts along rows/columns


@partial(jax.jit, static_argnums=(1, 2,))
def get_first_k_rows_fbx(batch, batch_size, prioritized):
    ids = jnp.arange(batch_size)
    subset_experience = batch.experience.replace(observation=batch.experience.observation[ids],
                                                 next_observation=batch.experience.next_observation[ids],
                                                 action=batch.experience.action[ids],
                                                 reward=batch.experience.reward[ids],
                                                 done=batch.experience.done[ids])
    if prioritized:
        return batch.replace(experience=subset_experience,
                             priorities=batch.priorities[ids], indices=batch.indices[ids])
    else:
        return batch.replace(experience=subset_experience)


@partial(jax.jit, static_argnums=(1, 2, 3))
def get_first_k_rows_n_step_fbx(batch, batch_size, prioritized, n_step):
    ids = jnp.arange(batch_size)
    subset_experience = batch.experience.replace(
        observation=batch.experience.observation[ids, :n_step, ...],
        action=batch.experience.action[ids, :n_step, ...],
        reward=batch.experience.reward[ids, :n_step, ...],
        done=batch.experience.done[ids, :n_step, ...]
    )
    if prioritized:
        return batch.replace(
            experience=subset_experience,
            priorities=batch.priorities[ids],
            indices=batch.indices[ids]
        )
    else:
        return batch.replace(experience=subset_experience)


@partial(jax.jit, static_argnums=range(2, 7))
def qema_update(prev, x, tau=1e-3, q=0.99, eps=1e-8, monotone=False, decay=0.0):
    x = jnp.reshape(x, (-1,))
    x = jnp.maximum(x, eps)

    xs = jnp.sort(x)
    n = xs.shape[0]
    idx = jnp.clip(jnp.asarray(jnp.floor(q * (n - 1)), jnp.int32), 0, n - 1)
    qv = jnp.maximum(xs[idx], eps)

    new = (1.0 - tau) * prev + tau * qv

    if decay > 0.0:
        new = (1.0 - decay) * new

    new = jnp.maximum(new, eps)

    if monotone:
        new = jnp.maximum(prev, new)

    return new


# ============================================================
# Running mean / variance utilities
# ============================================================

@chex.dataclass
class RunningMeanStdState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray  # scalar float32


def rms_init(shape, eps: float = 1e-4) -> RunningMeanStdState:
    return RunningMeanStdState(
        mean=jnp.zeros(shape, dtype=jnp.float32),
        var=jnp.ones(shape, dtype=jnp.float32),
        count=jnp.array(eps, dtype=jnp.float32),
    )


@jax.jit
def rms_update_from_moments(
    rms: RunningMeanStdState,
    batch_mean: jnp.ndarray,
    batch_var: jnp.ndarray,
    batch_count: jnp.ndarray,
):
    # Parallel variance update as in OpenAI baselines / original file
    delta = batch_mean - rms.mean
    tot_count = rms.count + batch_count

    new_mean = rms.mean + delta * (batch_count / tot_count)
    m_a = rms.var * rms.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * (rms.count * batch_count / tot_count)
    new_var = m2 / tot_count

    return RunningMeanStdState(mean=new_mean, var=new_var, count=tot_count)


def rms_update_from_batch(rms: RunningMeanStdState, x: jnp.ndarray) -> RunningMeanStdState:
    """
    computes batch moments and then uses rms_update_from_moments.
    """
    x = x.astype(jnp.float32)
    tail_shape = rms.mean.shape
    x = x.reshape((-1,) + tail_shape)

    batch_mean = jnp.mean(x, axis=0)
    batch_var = jnp.var(x, axis=0)
    batch_count = jnp.array(x.shape[0], dtype=jnp.float32)

    return rms_update_from_moments(rms, batch_mean, batch_var, batch_count)



