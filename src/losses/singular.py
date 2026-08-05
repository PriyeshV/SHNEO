from __future__ import annotations

from functools import partial
from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp


Array = jnp.ndarray


def _offdiag_mask(k: int, dtype=jnp.float32) -> Array:
    """Mask with 1 on off-diagonal entries, 0 on diagonal."""
    return (jnp.ones((k, k), dtype=dtype) - jnp.eye(k, dtype=dtype))


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


@partial(jax.jit, static_argnums=())
def svd_diag_objective(sigma_hat: Array) -> Array:
    """
    L_diag(Σ̂) = (1/K) Σ_i Σ̂[i,i]
    """
    return jnp.mean(jnp.diag(sigma_hat))


@partial(jax.jit, static_argnums=())
def svd_off_objective_naive(sigma_hat: Array) -> Array:
    """
    Naive off-diagonal objective from Eq. (10):
      L_off(Σ̂) = (1/(K^2-K)) Σ_{i!=j} (Σ̂[i,j])^2

    Note: gradients of this term are biased when computed from a single mini-batch.
    (Use svd_off_objective_double_sample for training.)
    """
    K = sigma_hat.shape[0]
    mask = _offdiag_mask(K, dtype=sigma_hat.dtype)
    off_sq_sum = jnp.sum((sigma_hat ** 2) * mask)
    return off_sq_sum / (K * K - K)


@partial(jax.jit, static_argnums=())
def svd_off_objective_double_sample(sigma1: Array, sigma2: Array) -> Array:
    """
    Double-sampling / stop-gradient unbiased estimator (Eq. 14):

      L̂_off(θ) = (1/(K^2-K)) Σ_{i!=j} [
          Σ̂_1[i,j] * sg(Σ̂_2[i,j]) + Σ̂_2[i,j] * sg(Σ̂_1[i,j])
      ]

    This is the term you should use inside the SVD loss.

    Args:
      sigma1: Σ̂ from batch-1   [K, K]
      sigma2: Σ̂ from batch-2   [K, K]
    Returns:
      off_loss_hat: scalar
    """
    K = sigma1.shape[0]
    mask = _offdiag_mask(K, dtype=sigma1.dtype)

    s1 = sigma1 * mask
    s2 = sigma2 * mask

    # stop-gradient operator
    sg_s1 = jax.lax.stop_gradient(s1)
    sg_s2 = jax.lax.stop_gradient(s2)

    # (sum over off-diagonals) / (K^2-K)
    term = jnp.sum(s1 * sg_s2 + s2 * sg_s1)
    return term / (K * K - K)


@partial(jax.jit, static_argnums=(2, 3))
def svd_loss_from_sigmas(
    sigma1: Array,
    sigma2: Array,
    lambda_r: float,
    laplacian: bool = False,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Total SVD loss:
      L = -L_diag(Σ̂_1) + λ_r * L̂_off(Σ̂_1, Σ̂_2)

    (Diagonal term can be estimated from one batch; its gradient is unbiased.)
    """
    diag_obj = svd_diag_objective(sigma1)
    off_hat = svd_off_objective_double_sample(sigma1, sigma2)

    loss = lambda_r * off_hat
    if not laplacian:
        loss -= diag_obj
    else:
        loss += diag_obj

    metrics = {
        "svd/diag_obj": diag_obj,
        "svd/off_hat": off_hat,
        "svd/loss": loss,
        # Useful diagnostics (not necessarily the training term):
        "svd/off_naive_b1": svd_off_objective_naive(sigma1),
        "svd/off_naive_b2": svd_off_objective_naive(sigma2),
        "svd/trace": jnp.trace(sigma1),
    }
    return loss, metrics


@partial(jax.jit, static_argnums=(4, 5, 6))
def svd_loss_fn(
    f1: Array,
    g1: Array,
    f2: Array,
    g2: Array,
    lambda_r: float = 1.0,
    eps: float = 0.0,
    laplacian: bool = False,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Main entry point for training.

    You must provide TWO independent batches of one-step transitions:
      batch-1: (s, s') -> (f1, g1)
      batch-2: (s, s') -> (f2, g2)

    Shapes:
      f1, g1, f2, g2: [B, K]

    Returns:
      loss, metrics
    """
    sigma1 = compute_sigma_hat(f1, g1, eps=eps)
    sigma2 = compute_sigma_hat(f2, g2, eps=eps)
    return svd_loss_from_sigmas(sigma1, sigma2, lambda_r=lambda_r, laplacian=laplacian)


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