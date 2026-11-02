from typing import Tuple, Any
from functools import partial

import chex
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from src.utils.jax_utils import RunningMeanStdState, rms_init, rms_update_from_moments, rms_update_from_batch


# Compatibility alias used by newer files
RMS = RunningMeanStdState


# ============================================================
# Forward-filter state for discounted intrinsic returns
# ============================================================
@struct.dataclass
class RewardForwardFilterState:
    rewems: jnp.ndarray  # [N_env]
    gamma: jnp.ndarray   # scalar float32


def rff_init(n_envs: int, gamma: float) -> RewardForwardFilterState:
    return RewardForwardFilterState(rewems=jnp.zeros((n_envs,), dtype=jnp.float32),
                                    gamma=jnp.array(gamma, dtype=jnp.float32))


@jax.jit
def rff_update(st: RewardForwardFilterState, r_t: jnp.ndarray):
    rewems = st.rewems * st.gamma + r_t
    return st.replace(rewems=rewems), rewems


# ============================================================
# RND networks (NHWC)
# ============================================================
class RNDTarget(nn.Module):
    @nn.compact
    def __call__(self, x):  # [B,H,W,1]
        x = nn.Conv(32, (8, 8), (4, 4))(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(64, (4, 4), (2, 2))(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(64, (3, 3), (1, 1))(x)
        x = nn.leaky_relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        return x


class RNDPredictor(nn.Module):
    @nn.compact
    def __call__(self, x):  # [B,H,W,1]
        x = nn.Conv(32, (8, 8), (4, 4))(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(64, (4, 4), (2, 2))(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(64, (3, 3), (1, 1))(x)
        x = nn.leaky_relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        return x


# ============================================================
# Train-time state
# ============================================================
@struct.dataclass
class RNDState:
    predictor: TrainState
    target_params: FrozenDict
    obs_rms: RunningMeanStdState
    rew_rms: RunningMeanStdState
    rff: RewardForwardFilterState
    key: jax.Array
    rnd_bonus_max_ema: jnp.ndarray

    int_gamma: jnp.ndarray
    update_proportion: jnp.ndarray

    @property
    def pred_state(self):
        return self.predictor


# ============================================================
# Initialization
# ============================================================
def rnd_setup(
    key,
    lr: float,
    max_grad_norm: float,
    n_envs: int,
    int_gamma: float,
    update_proportion: float = 0.25,
    obs_shape=(84, 84, 1)
):
    """
    This preserves the ORIGINAL PQN_RND behavior:
    - optimizer has grad clipping
    - Adam eps=1e-5
    - persistent per-env forward filter state
    """
    target_net = RNDTarget()
    pred_net = RNDPredictor()

    # dummy = jnp.zeros((1, 84, 84, 1), dtype=jnp.float32)  # NHWC
    dummy = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
    k1, k2, key = jax.random.split(key, 3)
    target_vars = target_net.init(k1, dummy)
    pred_vars = pred_net.init(k2, dummy)

    predictor = TrainState.create(
        apply_fn=jax.jit(pred_net.apply),
        params=pred_vars["params"],
        tx=optax.chain(optax.clip_by_global_norm(max_grad_norm),
                       optax.adam(lr, eps=1e-5)))

    return RNDState(
        predictor=predictor,
        target_params=target_vars["params"],
        obs_rms=rms_init((1, *obs_shape)),
        rew_rms=rms_init(()),
        rff=rff_init(n_envs, int_gamma),
        key=key,
        rnd_bonus_max_ema=jnp.array(1.0, dtype=jnp.float32),
        int_gamma=jnp.array(int_gamma, dtype=jnp.float32),
        update_proportion=jnp.array(update_proportion, dtype=jnp.float32),
    )


# ============================================================
# Inference utilities
# ============================================================

def whiten_last_frame(obs_nchw: jnp.ndarray, obs_rms: RunningMeanStdState) -> jnp.ndarray:
    """
    obs_nchw: [B,C,H,W] -> [B,H,W,1]
    Uses the last stacked frame, matching the original reference logic.
    """
    x = obs_nchw[:, -1, :, :].astype(jnp.float32)[..., None]
    x = (x - obs_rms.mean) / jnp.sqrt(obs_rms.var + 1e-8)
    return jnp.clip(x, -5.0, 5.0)


@jax.jit
def rnd_normalize_obs(rnd: RNDState, x_hwc1: jnp.ndarray):
    x = (x_hwc1 - rnd.obs_rms.mean) / jnp.sqrt(rnd.obs_rms.var + 1e-8)
    return jnp.clip(x, -5.0, 5.0)


@jax.jit
def rnd_intrinsic_reward(rnd: RNDState, x_hwc1: jnp.ndarray):
    """
    Original PQN_RND behavior:
    normalized obs -> raw half-MSE novelty.
    """
    x = rnd_normalize_obs(rnd, x_hwc1)
    pred = rnd.predictor.apply_fn({"params": rnd.predictor.params}, x)
    targ = RNDTarget().apply({"params": rnd.target_params}, x)
    err = 0.5 * jnp.sum(jnp.square(targ - pred), axis=-1)
    return err


# ============================================================
# RMS updates
# Preserves original PQN_RND behavior.
# ============================================================

@jax.jit
def rnd_update_obs_rms(rnd: RNDState, frames_hwc1: jnp.ndarray):
    batch_mean = frames_hwc1.mean(axis=0, keepdims=True)
    batch_var = frames_hwc1.var(axis=0, keepdims=True)
    batch_count = jnp.array(frames_hwc1.shape[0], jnp.float32)

    obs_rms = rms_update_from_moments(rnd.obs_rms, batch_mean, batch_var, batch_count)
    return rnd.replace(obs_rms=obs_rms)


@jax.jit
def rnd_update_rew_rms(rnd: RNDState, int_rewards_TN: jnp.ndarray):
    """
    ORIGINAL behavior:
    - use persistent per-env forward filter state rnd.rff
    - update rew_rms from those forward-filtered discounted intrinsic returns
    """
    int_rewards_TN = int_rewards_TN.astype(jnp.float32)

    def step(carry, r_t):
        (rff,) = carry
        rff, rewems = rff_update(rff, r_t)
        return (rff,), rewems

    (rff_new,), rewems_TN = jax.lax.scan(step, (rnd.rff,), int_rewards_TN)

    rewems_all = rewems_TN.reshape((-1,))
    mean = rewems_all.mean()
    var = rewems_all.var()
    count = jnp.array(rewems_all.size, jnp.float32)

    rew_rms = rms_update_from_moments(rnd.rew_rms, mean, var, count)

    rnd = rnd.replace(rff=rff_new, rew_rms=rew_rms)
    normed = int_rewards_TN / jnp.sqrt(rew_rms.var + 1e-8)
    return rnd, normed


# ============================================================
# Predictor training
# Preserves original PQN_RND behavior.
# ============================================================

@jax.jit
def rnd_predictor_loss_and_update(rnd: RNDState, frames_hwc1: jnp.ndarray, update_proportion: float):
    """
    frames_hwc1: [B,H,W,1] float32
    returns: new_rnd, forward_loss_scalar
    """
    key, sub = jax.random.split(rnd.key)

    def loss_fn(pred_params):
        x = rnd_normalize_obs(rnd, frames_hwc1)
        pred = rnd.predictor.apply_fn({"params": pred_params}, x)
        targ = RNDTarget().apply({"params": rnd.target_params}, x)
        per_ex = jnp.mean(jnp.square(pred - jax.lax.stop_gradient(targ)), axis=-1)
        mask = (jax.random.uniform(sub, shape=per_ex.shape) < update_proportion).astype(jnp.float32)
        loss = jnp.sum(per_ex * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(rnd.predictor.params)
    predictor = rnd.predictor.apply_gradients(grads=grads)
    rnd = rnd.replace(predictor=predictor, key=key)
    return rnd, loss


# def rnd_predictor_train_step(rnd: RNDState, rng, x_hwc1: jnp.ndarray):
#     """
#     Compatibility helper used by some HNEO/reference-style code.
#
#     Matches the original predictor-training semantics, but also returns
#     auxiliary stats.
#     """
#     upd_prop = rnd.update_proportion
#     pred_apply = rnd.predictor.apply_fn
#     tgt_params = rnd.target_params
#
#     def loss_fn(params, rng_local):
#         l2, half_mse = rnd_forward_error(params, tgt_params, pred_apply, x_hwc1)
#         mask = jax.random.bernoulli(rng_local, p=upd_prop, shape=half_mse.shape).astype(x_hwc1.dtype)
#         denom = jnp.maximum(jnp.sum(mask), 1.0)
#         fl = jnp.sum(half_mse * mask) / denom
#         return fl, {"fwd_l2_mean": jnp.mean(l2)}
#
#     rng2, rngm = jax.random.split(rng)
#     (loss_val, aux_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(rnd.predictor.params, rngm)
#
#     new_pred = rnd.predictor.apply_gradients(grads=grads)
#     new_rnd = rnd.replace(predictor=new_pred, key=rng2)
#
#     aux_out = {
#         "fwd_loss": loss_val,
#         "fwd_l2_mean": aux_dict["fwd_l2_mean"],
#     }
#     return new_rnd, aux_out, rng2


@partial(jax.jit, static_argnums=(2, 3))
def rnd_update_once_per_epoch(rnd: RNDState, flat_storage, update_proportion: float, burnin: bool):
    """
    Original PQN_RND behavior:
    - if burnin: skip predictor update
    - otherwise one masked predictor update over rollout frames

    Expects:
      flat_storage.rnd_frame: [B,H,W,1]
    """
    def _do(rnd_in):
        frames = flat_storage.rnd_frame
        rnd_out, fwd_loss = rnd_predictor_loss_and_update(rnd_in, frames, update_proportion)
        return rnd_out, fwd_loss

    def _skip(rnd_in):
        return rnd_in, jnp.array(0.0, dtype=jnp.float32)

    return jax.lax.cond(burnin, _skip, _do, rnd)


# ============================================================
# Post-rollout stats update
# Kept for HNEO/reference compatibility, but implemented so that it
# still preserves original persistent-RFF reward normalization behavior.
# ============================================================

@jax.jit
def rnd_stats_post_rollout_update(
    train_state,
    transitions,
    warmup_updates: int,
    use_ema: bool = False,
    ema_alpha: float = 1e-3,
) -> Tuple[Any, jnp.ndarray]:
    next_obs = transitions.next_obs
    t_env, e, c, h, w = next_obs.shape

    last_frames = next_obs[:, :, -1, :, :][..., None]      # [T,E,H,W,1]
    last_frames_flat = last_frames.reshape((t_env * e, h, w, 1)).astype(jnp.float32)

    def _ema_update(op):
        obs_rms_old, x = op
        b_mean = jnp.mean(x, axis=0)
        b_var = jnp.var(x, axis=0)
        mean = (1.0 - ema_alpha) * obs_rms_old.mean + ema_alpha * b_mean
        var = (1.0 - ema_alpha) * obs_rms_old.var + ema_alpha * b_var
        return RunningMeanStdState(mean=mean, var=var, count=obs_rms_old.count + x.shape[0])

    def _welford_update(op):
        obs_rms_old, x = op
        return rms_update_from_batch(obs_rms_old, x)

    def _no_update(op):
        obs_rms_old, _ = op
        return obs_rms_old

    upd_during_warmup = _ema_update if use_ema else _welford_update

    pred = (jnp.asarray(train_state.n_updates, jnp.int32) < jnp.asarray(warmup_updates, jnp.int32))

    obs_rms = jax.lax.cond(
        pred,
        upd_during_warmup,
        _no_update,
        operand=(train_state.rnd_state.obs_rms, last_frames_flat),
    )

    # Compute raw half-MSE using the possibly updated obs RMS
    x_wh = (last_frames_flat - obs_rms.mean) / jnp.sqrt(obs_rms.var + 1e-8)
    x_wh = jnp.clip(x_wh, -5.0, 5.0)

    _, raw_half_mse_flat = rnd_forward_error(
        pred_params=train_state.rnd_state.predictor.params,
        target_params=train_state.rnd_state.target_params,
        pred_apply=train_state.rnd_state.predictor.apply_fn,
        x_bhw1=x_wh,
    )
    raw_half_mse = raw_half_mse_flat.reshape((t_env, e))  # [T,E]

    # ORIGINAL persistent forward-filter behavior
    def step(carry, r_t):
        (rff,) = carry
        rff, rewems = rff_update(rff, r_t)
        return (rff,), rewems

    (rff_new,), rewems_TN = jax.lax.scan(step, (train_state.rnd_state.rff,), raw_half_mse)

    rewems_all = rewems_TN.reshape((-1,))
    mean = rewems_all.mean()
    var = rewems_all.var()
    count = jnp.array(rewems_all.size, jnp.float32)

    rew_rms = rms_update_from_moments(train_state.rnd_state.rew_rms, mean, var, count)

    train_state = train_state.replace(
        rnd_state=train_state.rnd_state.replace(
            obs_rms=obs_rms,
            rew_rms=rew_rms,
            rff=rff_new,
        )
    )
    return train_state, last_frames_flat