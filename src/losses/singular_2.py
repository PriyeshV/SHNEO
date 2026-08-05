from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray


@partial(jax.jit, static_argnums=(2,))
def compute_sigma_hat(f: Array, g: Array, eps: float = 0.0) -> Array:
    """
    Mini-batch estimate of Σ:
        Σ̂ = (1/b) Σ_i f(s_i) g(s'_i)^T

    Args:
      f: [B, K] left features for s
      g: [B, K] right features for s'
    Returns:
      sigma_hat: [K, K]
    """
    # [K, K] = [K, B] @ [B, K]
    B = f.shape[0]
    sigma = (f.T @ g) / B
    return sigma


def _offdiag_mask(k: int, dtype=jnp.float32) -> Array:
    return jnp.ones((k, k), dtype=dtype) - jnp.eye(k, dtype=dtype)


def _skip_index_mask(k: int, skip_idx: int | None, dtype=jnp.float32) -> Array:
    if skip_idx is None:
        return jnp.ones((k, k), dtype=dtype)
    v = jnp.ones((k,), dtype=dtype).at[skip_idx].set(0.0)
    return v[:, None] * v[None, :]


@jax.jit
def svd_off_objective_double_sample(sigma1: Array, sigma2: Array, mask_off: Array) -> Array:
    s1 = sigma1 * mask_off
    s2 = sigma2 * mask_off
    return jnp.sum(s1 * jax.lax.stop_gradient(s2) + s2 * jax.lax.stop_gradient(s1)) / jnp.maximum(
        jnp.sum(mask_off), 1.0
    )


@jax.jit
def weighted_diag(diag: Array, weights: Array, skip_idx: int | None) -> Array:
    if skip_idx is None:
        return jnp.sum(weights * diag) / jnp.maximum(jnp.sum(weights), 1.0)
    m = jnp.ones_like(diag).at[skip_idx].set(0.0)
    diag = diag * m
    weights = weights * m
    return jnp.sum(weights * diag) / jnp.maximum(jnp.sum(weights), 1.0)


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def svd_loss_fn(
    f1: Array, g1: Array,
    f2: Array, g2: Array,
    lambda_r: float = 10.0,
    laplacian: bool = False,
    beta_sb: float = 1.0,
    beta_order: float = 1.0,
    skip_idx: int | None = None,
) -> Tuple[Array, Dict[str, Array]]:
    B, K = f1.shape

    skip_idx = 0  #if laplacian else -1

    sigma1 = (f1.T @ g1) / B
    sigma2 = (f2.T @ g2) / B

    mask_off = _offdiag_mask(K, dtype=sigma1.dtype) * _skip_index_mask(K, skip_idx, dtype=sigma1.dtype)
    off_main = svd_off_objective_double_sample(sigma1, sigma2, mask_off)

    # if laplacian:
    #     sigma1 = sigma1 @ sigma1.T  # [K, K], PSD

    d1 = jnp.diag(sigma1)
    if skip_idx is None:
        diag_mean = jnp.mean(d1)
    else:
        m = jnp.ones_like(d1).at[skip_idx].set(0.0)
        diag_mean = jnp.sum(d1 * m) / jnp.maximum(jnp.sum(m), 1.0)

    # if laplacian:
    #     diag_mean = jnp.abs(diag_mean)

    # one-sided symmetry breaker
    sigma_sb1 = (f1.T @ jax.lax.stop_gradient(g1)) / B
    sigma_sb2 = (f2.T @ jax.lax.stop_gradient(g2)) / B
    off_sb = 0.5 * (jnp.sum((sigma_sb1 * mask_off) ** 2) / jnp.maximum(jnp.sum(mask_off), 1.0) +
                    jnp.sum((sigma_sb2 * mask_off) ** 2) / jnp.maximum(jnp.sum(mask_off), 1.0))

    # ordering: use strictly positive weights if you want all dims ordered
    w = jnp.linspace(1.0, 0.1, K, dtype=sigma1.dtype)  # NOTE: nonzero tail
    order_term = 0.5 *(weighted_diag(jnp.diag(sigma_sb1), w, skip_idx) +
                       weighted_diag(jnp.diag(sigma_sb2), w, skip_idx))
    order_term *= 0.

    loss = lambda_r * off_main  # + beta_sb * off_sb
    loss = loss + diag_mean if laplacian else loss - diag_mean
    # loss = loss + beta_order * order_term if laplacian else loss - beta_order * order_term

    metrics: Dict[str, Array] = {
        "svd/off_main": off_main,
        "svd/off_sb": off_sb,
        "svd/diag_mean": diag_mean,
        "svd/order_term": order_term,
        "svd/loss": loss,
        "svd/skip_idx": jnp.asarray(-1 if skip_idx is None else skip_idx),
    }
    return loss, metrics


@partial(jax.jit, static_argnums=())
def update_running_sigma_diag(
    running_diag: Array,
    sigma_hat: Array,
    alpha: float = 0.01,
) -> Array:
    """
    Optional: maintain a running estimate of diag(Σ̃) as suggested for exploration,
    keeping off-diagonal elements as zero.

    Args:
      running_diag: [K] running estimate of diag
      sigma_hat:    [K, K] current mini-batch sigma
      alpha: EMA step
    Returns:
      new_running_diag: [K]
    """
    diag = jnp.diag(sigma_hat)
    return (1.0 - alpha) * running_diag + alpha * diag


@partial(jax.jit, static_argnums=())
def svd_lambda_from_running_diag(running_diag: Array, eps: float = 1e-8) -> Array:
    """
    Λ̃ := Σ̃^2 in the paper, with Σ̃ treated as diagonal in practice.
    If Σ̃ has diagonal d, then Λ̃ has diagonal d^2.

    Returns:
      lambda_diag: [K] (nonnegative)
    """
    return jnp.square(running_diag) + eps


@partial(jax.jit, static_argnums=())
def svd_weighted_norm_sq(f: Array, lambda_diag: Array, eps: float = 1e-8) -> Array:
    """
    Computes ||f(s)||_{Λ^{-1}} = f^T Λ^{-1} f when Λ is diagonal.

    Args:
      f: [B, K]
      lambda_diag: [K] diagonal of Λ (positive)
    Returns:
      norm_sq: [B]
    """
    inv = 1.0 / (lambda_diag + eps)
    return jnp.sum((f ** 2) * inv[None, :], axis=-1)