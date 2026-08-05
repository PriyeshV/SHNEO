# ==============================================================================
# HDQN + EigenOptions (Laplacian) + PER
# - Single primitive sample drives: meta-from-prim, laplacian update, option updates.
# - Separate meta buffer for SMDP transitions (also PER).
# - Uses grayscale observations (uint8, 1 channel) for efficiency.
# ==============================================================================

import math
import os, time, random, traceback, imageio
from copy import deepcopy
from functools import partial
from typing import Any, Union, List

import numpy as np
import jax
import jax.numpy as jnp
import chex
import flax
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from src.config.customgrids.laplacian import *
args = convert_args_to_dict(parse_args())

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from src.utils.jax_utils import *
from src.utils.grid_agent import GridAgent

import src.fbx_buffers.trajectory_buffer as trajectory_buffer
import src.fbx_buffers.prioritised_trajectory_buffer as prioritised_trajectory_buffer

from src.networks.networks import FFNetwork, Conv_Encoder
from src.losses.laplacian import multi_step_ALLO, update_ALLO_parameters, update_eigenvalues

print("Available Jax Devices:", jax.devices())


# ==============================================================================
# TrainStates / Data containers
# ==============================================================================
class DQN_Enc_State(TrainState):
    target_params: flax.core.FrozenDict
    batch_stats: Union[None, flax.core.FrozenDict] = None
    target_batch_stats: Union[None, flax.core.FrozenDict] = None


class DQN_FFN_State(TrainState):
    target_params: flax.core.FrozenDict
    batch_stats: Union[None, flax.core.FrozenDict] = None
    target_batch_stats: Union[None, flax.core.FrozenDict] = None


@chex.dataclass(frozen=True)
class RunningMaxAbs:
    max_abs: chex.Array   # [O] (or [n_lap])


@chex.dataclass(frozen=True)
class PrimStep:
    # Stored in replay buffer with leading axes [add_batch=1, seq=1] at add time.
    observation: chex.Array  # [H,W,1] uint8
    action: chex.Array       # int32
    reward: chex.Array       # float32
    done: chex.Array         # bool
    action_prob: chex.Array  # float32  mu(a_t | s_t)  (behavior prob of primitive action)

    # NEW: meta action currently being executed at this primitive step
    # - primitive step: meta == action in [0..A-1]
    # - option step: meta == A + option_index in [A..A+O-1]
    meta_action: chex.Array  # int32


@chex.dataclass(frozen=True)
class MetaStep:
    observation: chex.Array  # start obs [H,W,1] uint8
    action: chex.Array       # meta action (0..A+O-1)
    reward: chex.Array       # discounted return over option duration
    done: chex.Array         # termination flag at option end
    duration: chex.Array     # option duration (k)
    behav_prob: chex.Array   # behavior prob at option start (stored for future off-policy; unused here)


class Lap_State(TrainState):
    # ALLO bookkeeping
    betas: Any
    errors: Any
    quadratic_errors: Any
    barrier_coeffs: Any
    eigen_values: Any
    batch_stats: Union[FrozenDict, None]


@chex.dataclass(frozen=True)
class RunningMeanStd:
    mean: chex.Array   # [O]
    var: chex.Array    # [O]   population variance
    count: chex.Array  # [O]


class Train_State(flax.struct.PyTreeNode):
    enc_state: DQN_Enc_State
    meta_q_state: DQN_FFN_State

    opt_enc_state: DQN_Enc_State
    opt_q_states: List[DQN_FFN_State]

    lap_state: Lap_State
    int_rms: RunningMeanStd
    phi_maxabs: RunningMaxAbs


@jax.jit
def rms_update_per_option(rms: RunningMeanStd, x_bo: chex.Array) -> RunningMeanStd:
    """
    x_bo: [B,O] intrinsic n-step returns (or rewards) per option.

    Parallel (Welford merge) update done elementwise over O.
    """
    x = x_bo.astype(jnp.float32)
    B = x.shape[0]
    batch_count = jnp.full((x.shape[1],), jnp.float32(B))  # [O]

    batch_mean = jnp.mean(x, axis=0)  # [O]
    batch_var  = jnp.var(x, axis=0)   # [O] population var

    delta = batch_mean - rms.mean     # [O]
    tot_count = rms.count + batch_count

    new_mean = rms.mean + delta * (batch_count / tot_count)

    m_a = rms.var * rms.count
    m_b = batch_var * batch_count
    M2  = m_a + m_b + (delta * delta) * (rms.count * batch_count / tot_count)
    new_var = M2 / tot_count

    return RunningMeanStd(mean=new_mean, var=new_var, count=tot_count)


@jax.jit
def rms_normalize_per_option(rms: RunningMeanStd, x_bo: chex.Array, eps: float = 1e-8) -> chex.Array:
    """
    Variance normalization like RND: divide by std (no mean subtraction).
    """
    return x_bo / jnp.sqrt(rms.var[None, :] + eps)  # broadcast over B


# ==============================================================================
# Utilities: grayscale + tree slicing + PER IS weights
# ==============================================================================
def rgb_to_gray_uint8(obs_rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) -> (H,W,1) uint8"""
    obs = obs_rgb.astype(np.float32)
    gray = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray[..., None]


@partial(jax.jit, static_argnums=1)
def slice_seq(exp, T: int):
    """Truncate sequence/time axis for a replay-batch pytree.
    exp.* fields are assumed shaped [B, Seq, ...]; returns [B, T, ...]
    """
    return jax.tree_util.tree_map(lambda x: x[:, :T, ...], exp)


@jax.jit
def compute_is_weights(probs, running_index, capacity, seq_len, beta):
    """Standard PER importance weights given sampling probabilities P(i)."""
    # Approx number of valid items: running_index - (seq_len-1), clipped to [1, capacity]
    n_valid = jnp.clip(running_index - (seq_len - 1), a_min=1)
    n_valid = jnp.minimum(n_valid, capacity)

    w = (1.0 / (n_valid * probs + 1e-8)) ** beta
    return w / (jnp.max(w) + 1e-8)


@jax.jit
def get_buffer_key(key):
    return jax.random.split(key, 2)


# ==============================================================================
# Replay buffers
# ==============================================================================
def setup_prim_rb(args):
    # We sample long enough to support BOTH TD n-step and Laplacian n-step.
    seq_len = safe_max(args.algo.n_step, args.laplacian.n_step) + 1

    buffer_args = dict(
        max_size=args.buffer.size,
        period=1,
        sample_sequence_length=seq_len,
        add_batch_size=1,
        min_length_time_axis=args.algo.starts,
        sample_batch_size=args.buffer.q_batch_size,
    )
    if args.buffer.prioritized:
        buffer_args["priority_exponent"] = 0.6
        fbx_buffer = prioritised_trajectory_buffer.make_prioritised_trajectory_buffer
    else:
        fbx_buffer = trajectory_buffer.make_trajectory_buffer

    rb = fbx_buffer(**buffer_args)
    rb = rb.replace(
        init=jax.jit(rb.init),
        can_sample=jax.jit(rb.can_sample),
        add=jax.jit(rb.add, donate_argnums=0),
        sample=jax.jit(rb.sample),
    )
    if args.buffer.prioritized:
        rb = rb.replace(set_priorities=jax.jit(rb.set_priorities, donate_argnums=0))

    dummy = PrimStep(
        observation=jnp.ones((args.env.screen_size, args.env.screen_size, 1), dtype=jnp.uint8),
        action=jnp.int32(0),
        reward=jnp.float32(0.0),
        done=jnp.bool_(True),
        action_prob=jnp.float32(1.0),
        meta_action=jnp.int32(0),
    )
    rb_state = rb.init(dummy)
    return rb, rb_state


def setup_meta_rb(args):
    # SMDP buffer stores option-level transitions: we need m_step+1 to bootstrap
    buffer_args = dict(
        max_size=args.buffer.size,
        period=1,
        sample_sequence_length=args.algo.m_step + 1,
        add_batch_size=1,
        min_length_time_axis=math.ceil(args.algo.starts / args.algo.opt_dur),
        sample_batch_size=args.buffer.q_batch_size,
    )
    if args.buffer.prioritized:
        buffer_args["priority_exponent"] = 0.6
        fbx_buffer = prioritised_trajectory_buffer.make_prioritised_trajectory_buffer
    else:
        fbx_buffer = trajectory_buffer.make_trajectory_buffer

    rb = fbx_buffer(**buffer_args)
    rb = rb.replace(
        init=jax.jit(rb.init),
        can_sample=jax.jit(rb.can_sample),
        add=jax.jit(rb.add, donate_argnums=0),
        sample=jax.jit(rb.sample),
    )
    if args.buffer.prioritized:
        rb = rb.replace(set_priorities=jax.jit(rb.set_priorities, donate_argnums=0))

    dummy = MetaStep(
        observation=jnp.ones((args.env.screen_size, args.env.screen_size, 1), dtype=jnp.uint8),
        action=jnp.int32(0),
        reward=jnp.float32(0.0),
        done=jnp.bool_(True),
        duration=jnp.int32(1),
        behav_prob=jnp.float32(1.0),
    )
    rb_state = rb.init(dummy)
    return rb, rb_state


# ==============================================================================
# Setup networks + states
# ==============================================================================
def setup(key, env, args, n_actions, n_options):
    assert n_options > 0, "This cleaned script is HDQN-only; set n_options > 0."

    key, q_key = jax.random.split(key, 2)

    # Shared encoder for meta-Q
    enc_net = Conv_Encoder(
        name="HDQN:Enc",
        min_scale=1,
        batch_norm=args.laplacian.batch_norm,
        layer_norm=args.laplacian.layer_norm,
        layer_dims=list(args.net.c_dims),
        activations=[nn.relu] * len(args.net.c_dims),
    )
    meta_q_net = FFNetwork(
        name="HDQN:MetaQ",
        stop_grad=False,
        ip_batch_norm=False,
        batch_norm=args.net.batch_norm,
        layer_norm=args.net.layer_norm,
        layer_dims=list(args.net.q_dims) + [n_actions + n_options],
        activations=[nn.relu] * len(args.net.q_dims) + [None],
    )

    # Per-option encoder + Q
    opt_enc_net = Conv_Encoder(
        name="HDQN:OptEnc",
        min_scale=1,
        batch_norm=args.laplacian.batch_norm,
        layer_norm=args.laplacian.layer_norm,
        layer_dims=list(args.net.c_dims) + [args.laplacian.dims],
        activations=[nn.relu] * len(args.net.c_dims) + [nn.relu],
    )
    opt_q_net = FFNetwork(
        name="HDQN:OptQ",
        stop_grad=False,
        ip_batch_norm=False,
        batch_norm=args.net.batch_norm,
        layer_norm=args.net.layer_norm,
        layer_dims=list(args.net.q_dims) + [n_actions],
        activations=[nn.relu] * len(args.net.q_dims) + [None],
    )

    # Laplacian encoder for eigenfunctions
    use_lap_bn = False
    lap_net = Conv_Encoder(
        name="HDQN:LapEnc",
        min_scale=1,
        batch_norm=use_lap_bn,
        layer_norm=True,
        layer_dims=list(args.net.c_dims) + [args.laplacian.dims],
        activations=[nn.relu] * len(args.net.c_dims) + [None],
    )

    # Dummy obs for init/tabulate
    obs, _ = env.reset(seed=args.exp.seed)
    obs = rgb_to_gray_uint8(obs)
    obs = jnp.expand_dims(obs, axis=0)  # [1,H,W,1]
    dummy_emb = jnp.ones((1, args.net.c_dims[-1]), dtype=jnp.float32)

    print(enc_net.tabulate(key, obs))
    print(meta_q_net.tabulate(key, dummy_emb))

    # Optimizers
    enc_optim  = optax.chain(optax.clip_by_global_norm(args.net.max_grad_norm),
                             optax.adam(args.net.lr.ffn, eps=1.5e-4))
    meta_optim = optax.chain(optax.clip_by_global_norm(args.net.max_grad_norm),
                             optax.adam(args.net.lr.ffn, eps=1.5e-4))
    opt_optim  = optax.chain(optax.clip_by_global_norm(args.net.max_grad_norm),
                             optax.adam(args.net.lr.ffn, eps=1.5e-4))
    # lap_optim  = optax.chain(optax.clip_by_global_norm(args.laplacian.max_grad_norm),
    #                          optax.adam(args.laplacian.lr, eps=1.5e-4))

    train_env_steps = max(1, args.algo.tot_steps - args.algo.starts)
    num_updates = max(1, train_env_steps // args.algo.train_freq)
    end_frac = getattr(args, "lr_end_frac", 1e-6)  # add to config if you want

    def linear_lr_schedule(init_lr: float, end_lr: float, num_updates: int):
        # clip_count makes it flat at end_lr after num_updates
        return optax.linear_schedule(
            init_value=init_lr,
            end_value=end_lr,
            transition_steps=num_updates,
            transition_begin=0,
        )

    def sched(init_lr):
        return linear_lr_schedule(init_lr, init_lr * end_frac, num_updates)

    lap_lr = sched(args.laplacian.lr)
    lap_optim = optax.chain(optax.clip_by_global_norm(args.laplacian.max_grad_norm),
                            optax.adamw(lap_lr, weight_decay=1e-4, eps=1.5e-4))

    # Shared enc + meta-Q
    enc_vars = enc_net.init(q_key, obs, train=False)
    enc_bs = enc_vars["batch_stats"] if args.laplacian.batch_norm else None
    enc_state = DQN_Enc_State.create(
        apply_fn=enc_net.apply, tx=enc_optim,
        params=enc_vars["params"], batch_stats=enc_bs,
        target_params=enc_vars["params"], target_batch_stats=enc_bs
    )

    q_vars = meta_q_net.init(q_key, dummy_emb, train=False)
    q_bs = q_vars["batch_stats"] if args.net.batch_norm else None
    meta_q_state = DQN_FFN_State.create(
        apply_fn=meta_q_net.apply, tx=meta_optim,
        params=q_vars["params"], batch_stats=q_bs,
        target_params=q_vars["params"], target_batch_stats=q_bs
    )

    # -------------------------------
    # Options: SHARED encoder + per-option Q heads
    # -------------------------------
    k = q_key
    k, k_enc = jax.random.split(k, 2)

    # Shared option encoder (single TrainState)
    v_enc = opt_enc_net.init(k_enc, obs, train=False)
    bs_enc = v_enc["batch_stats"] if args.laplacian.batch_norm else None
    opt_enc_state = DQN_Enc_State.create(
        apply_fn=opt_enc_net.apply, tx=opt_optim,
        params=v_enc["params"], batch_stats=bs_enc,
        target_params=v_enc["params"], target_batch_stats=bs_enc
    )

    # Per-option Q heads (O TrainStates)
    opt_q_states = []
    for _ in range(n_options):
        k, k_q = jax.random.split(k, 2)
        dummy_lap_emb = jnp.ones((1, args.laplacian.dims), dtype=jnp.float32)
        v_q = opt_q_net.init(k_q, dummy_lap_emb, train=False)
        bs_q = v_q["batch_stats"] if args.net.batch_norm else None
        s_q = DQN_FFN_State.create(
            apply_fn=opt_q_net.apply, tx=opt_optim,
            params=v_q["params"], batch_stats=bs_q,
            target_params=v_q["params"], target_batch_stats=bs_q
        )
        opt_q_states.append(s_q)

    # Lap encoder
    lv = lap_net.init(q_key, obs, train=False)
    lap_bs = lv["batch_stats"] if use_lap_bn else None
    lap_state = Lap_State.create(
        apply_fn=lap_net.apply, tx=lap_optim,
        params=lv["params"], batch_stats=lap_bs,
        barrier_coeffs=jnp.tril(jnp.ones((1, 1))) * args.laplacian.barrier.value,
        betas=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
        eigen_values=jnp.zeros((1, args.laplacian.dims)),
        errors=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
        quadratic_errors=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
    )

    O = n_options
    if not args.algo.intr_norm:
        init_rms = RunningMeanStd(
            mean=jnp.zeros((O,), jnp.float32),
            var=jnp.ones((O,), jnp.float32),
            count=jnp.zeros((O,), jnp.float32),
        )
    else:  # per_option_rms
        init_rms = RunningMeanStd(
            mean=jnp.zeros((O,), jnp.float32),
            var=jnp.ones((O,), jnp.float32),
            count=jnp.full((O,), 1e-4, jnp.float32),
        )

    if not args.algo.phi_norm:
        init_maxabs = RunningMaxAbs(max_abs=jnp.ones((1,), jnp.float32))  # dummy
    else:
        # phi_maxabs should match the dimensionality you normalize (diffs), i.e., n_lap
        n_lap = n_options // 2 if getattr(args.algo, "bidir_opt", False) else n_options
        init_maxabs = RunningMaxAbs(max_abs=jnp.ones((n_lap,), jnp.float32))

    return Train_State(
        enc_state=enc_state,
        meta_q_state=meta_q_state,
        opt_enc_state=opt_enc_state,
        opt_q_states=opt_q_states,
        lap_state=lap_state,
        int_rms=init_rms,
        phi_maxabs=init_maxabs
    )


# ==============================================================================
# Meta Q helpers (online/target)
# ==============================================================================
@partial(jax.jit, static_argnums=(3, 4))
def get_meta_q(obs, enc_state, q_state, use_enc_bn, use_q_bn):
    e_vars = {"params": enc_state.params}
    if use_enc_bn and enc_state.batch_stats is not None:
        e_vars["batch_stats"] = enc_state.batch_stats
    emb = enc_state.apply_fn(e_vars, obs, train=False)

    q_vars = {"params": q_state.params}
    if use_q_bn and q_state.batch_stats is not None:
        q_vars["batch_stats"] = q_state.batch_stats
    return q_state.apply_fn(q_vars, emb, train=False)  # [B, A+O]


@partial(jax.jit, static_argnums=(3, 4))
def get_meta_q_target(obs, enc_state, q_state, use_enc_bn, use_q_bn):
    e_vars = {"params": enc_state.target_params}
    if use_enc_bn and enc_state.target_batch_stats is not None:
        e_vars["batch_stats"] = enc_state.target_batch_stats
    emb = enc_state.apply_fn(e_vars, obs, train=False)

    q_vars = {"params": q_state.target_params}
    if use_q_bn and q_state.target_batch_stats is not None:
        q_vars["batch_stats"] = q_state.target_batch_stats
    return q_state.apply_fn(q_vars, emb, train=False)


# ==============================================================================
# Option Q helpers
# Requirement: "Make compute_option_qs_any to reuse compute_option_q_one"
#
# We do this by factoring the common computation into a low-level function
# that takes (enc_vars, q_vars) dictionaries. Then:
#   - compute_option_q_one builds the vars and calls it.
#   - compute_option_qs_any vmaps over stacked vars and calls it.
# ==============================================================================
@partial(jax.jit, static_argnums=(1, 2))
def compute_option_q_from_vars(obs, enc_apply_fn, q_apply_fn, enc_vars, q_vars):
    """Core option forward: obs -> emb -> Q. Reused by one and any."""
    emb = enc_apply_fn(enc_vars, obs, train=False)
    return q_apply_fn(q_vars, emb, train=False)  # [B, A]


@partial(jax.jit, static_argnums=(3, 4, 5))
def compute_option_q_one(obs, opt_enc_state, opt_q_state, use_enc_bn, use_q_bn, use_target):
    """
    Shared option encoder + one option head.
    obs: [B,H,W,C]
    returns: [B,A]
    """
    enc_params = opt_enc_state.target_params if use_target else opt_enc_state.params
    q_params   = opt_q_state.target_params  if use_target else opt_q_state.params

    enc_vars = {"params": enc_params}
    if use_enc_bn:
        enc_vars["batch_stats"] = opt_enc_state.target_batch_stats if use_target else opt_enc_state.batch_stats

    q_vars = {"params": q_params}
    if use_q_bn:
        q_vars["batch_stats"] = opt_q_state.target_batch_stats if use_target else opt_q_state.batch_stats

    emb = opt_enc_state.apply_fn(enc_vars, obs, train=False)          # [B,D]
    qv  = opt_q_state.apply_fn(q_vars, emb, train=False)              # [B,A]
    return qv


@partial(jax.jit, static_argnums=(3, 4, 5))
def compute_option_qs_any(obs, opt_enc_state, opt_q_states, use_enc_bn, use_q_bn, use_target):
    """
    Shared option encoder + ALL option heads.
    returns: [O,B,A]
    """
    # ----- shared embedding -----
    enc_params = opt_enc_state.target_params if use_target else opt_enc_state.params
    enc_vars = {"params": enc_params}
    if use_enc_bn:
        enc_vars["batch_stats"] = opt_enc_state.target_batch_stats if use_target else opt_enc_state.batch_stats
    emb = opt_enc_state.apply_fn(enc_vars, obs, train=False)  # [B,D]

    # ----- stack head params/stats -----
    def pick_q_params(s): return s.target_params if use_target else s.params
    def pick_q_bs(s):     return s.target_batch_stats if use_target else s.batch_stats

    q_apply = opt_q_states[0].apply_fn
    q_params = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[pick_q_params(s) for s in opt_q_states])

    if use_q_bn:
        q_bs = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[pick_q_bs(s) for s in opt_q_states])
    else:
        q_bs = None

    def one_head(qp, qb):
        q_vars = {"params": qp}
        if use_q_bn:
            q_vars["batch_stats"] = qb
        return q_apply(q_vars, emb, train=False)  # [B,A]

    return jax.vmap(one_head, in_axes=(0, 0 if use_q_bn else None))(q_params, q_bs)  # [O,B,A]


# ==============================================================================
# Target updates (Polyak / hard)
# ==============================================================================
@partial(jax.jit, static_argnums=0)
def soft_update(tau, enc_state, q_state):
    enc_state = enc_state.replace(target_params=optax.incremental_update(enc_state.params, enc_state.target_params, tau))
    q_state   = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, tau))

    if (enc_state.batch_stats is not None) and (enc_state.target_batch_stats is not None):
        enc_state = enc_state.replace(target_batch_stats=optax.incremental_update(enc_state.batch_stats,
                                                                                  enc_state.target_batch_stats, tau))
    if (q_state.batch_stats is not None) and (q_state.target_batch_stats is not None):
        q_state = q_state.replace(target_batch_stats=optax.incremental_update(q_state.batch_stats,
                                                                              q_state.target_batch_stats, tau))
    return enc_state, q_state


@jax.jit
def soft_update_only(state, tau):
    state = state.replace(target_params=optax.incremental_update(state.params, state.target_params, tau))
    if (state.batch_stats is not None) and (state.target_batch_stats is not None):
        state = state.replace(target_batch_stats=optax.incremental_update(state.batch_stats, state.target_batch_stats, tau))
    return state


# ==============================================================================
# MAX ABS updates (Polyak / hard)
# ==============================================================================
@jax.jit
def maxabs_update_ema(state: RunningMaxAbs, phi_bto: chex.Array, alpha: float = 0.001):
    """
    phi_bto: [B,T,O] or [B,T,n_lap] float32
    Track EMA of max(|phi|) per option dimension.
    """
    batch_max = jnp.max(jnp.abs(phi_bto), axis=(0, 1))  # [O]
    new_max = (1.0 - alpha) * state.max_abs + alpha * batch_max
    # Optional lower bound to avoid divide-by-0:
    new_max = jnp.maximum(new_max, 1e-6)
    return RunningMaxAbs(max_abs=new_max)


@jax.jit
def maxabs_normalize(state: RunningMaxAbs, phi_bto: chex.Array, eps: float = 1e-8):
    return phi_bto / (state.max_abs[None, None, :] + eps)


# ==============================================================================
# TD target construction (variable-k n-step from a fixed-length sampled sequence)
# ==============================================================================
@partial(jax.jit, static_argnums=(1, 2))
def compute_n_step_from_td_batch(exp, gamma: float, n_step: int):
    B = exp.reward.shape[0]

    future_done = exp.done[:, 1:n_step]  # [B, n_step-1]

    prefix = jnp.concatenate([
        jnp.ones_like(exp.done[:, :1], dtype=jnp.float32),
        (1.0 - exp.done[:, :1].astype(jnp.float32)) * jnp.cumprod(1.0 - future_done.astype(jnp.float32), axis=1)],
        axis=1)

    mask = jnp.clip(prefix + exp.done[:, :n_step].astype(jnp.float32), 0.0, 1.0)  # [B, n_step]

    discounts = jnp.power(jnp.float32(gamma), jnp.arange(n_step, dtype=jnp.float32))[None, :]
    G = jnp.sum(exp.reward[:, :n_step] * discounts * mask, axis=1)  # [B]

    k_steps = jnp.sum(mask, axis=1).astype(jnp.int32)  # [B]
    idx = jnp.arange(B)

    s0 = exp.observation[:, 0, ...]
    sk = exp.observation[idx, k_steps, ...]
    done_last = exp.done[idx, k_steps - 1]
    a0 = exp.action[:, 0]

    # NEW: executing meta-action at time 0 of the sampled sequence
    meta0 = exp.meta_action[:, 0]  # [B]

    mu_seq = exp.action_prob[:, :n_step]
    mu_seq = jnp.clip(mu_seq, 1e-12, 1.0)
    log_mu_prod = jnp.sum(mask * jnp.log(mu_seq), axis=1)
    mu_prod = jnp.exp(log_mu_prod)

    # return meta0 too
    return s0, sk, done_last, a0, G, k_steps, mu_seq, mu_prod, meta0


@partial(jax.jit, static_argnums=(3, 4, 6, 7, 8, 9))
def update_ioq_options_from_prim_excluding_exec(
    td_data,
    per_is_w,              # [B] or None
    train_state,
    n_prim_actions: int,   # A
    n_options: int,        # O
    opt_dur_table: chex.Array,  # [O] int32/float32 durations (CURRENT table)
    use_enc_bn: bool,
    use_q_bn: bool,
    use_ddqn: bool,
    deterministic_match: bool,
):
    """
    Off-policy IOQ update for meta-Q option entries only (A..A+O-1),
    excluding the currently executing option for each sample.

    td_data = (s0, sk, done_last, a0, G, k_steps, mu_seq, mu_prod, meta0)
    """
    enc_s = train_state.enc_state
    q_s   = train_state.meta_q_state

    s0, sk, done_last, a0, G, k_steps, mu_seq, _, meta0 = td_data
    B = G.shape[0]
    A = n_prim_actions
    O = n_options
    M = A + O

    # ---------- bootstrap pieces ----------
    kf = k_steps.astype(jnp.float32)                     # [B]
    gamma_k = (jnp.float32(args.env.gamma) ** kf)        # [B]
    boot_mask = 1.0 - done_last.astype(jnp.float32)      # [B]

    q_sk_targ_all = get_meta_q_target(sk, enc_s, q_s, use_enc_bn, use_q_bn)  # [B,M]

    # v_term = max over ALL meta-actions (primitives + options)
    if use_ddqn:
        q_sk_online_all = get_meta_q(sk, enc_s, q_s, use_enc_bn, use_q_bn)   # [B,M]
        u_star = jnp.argmax(q_sk_online_all, axis=-1)                        # [B]
        v_term = q_sk_targ_all[jnp.arange(B), u_star]                        # [B]
    else:
        v_term = jnp.max(q_sk_targ_all, axis=-1)                             # [B]

    # continuation values for options at sk
    q_cont_opt = q_sk_targ_all[:, A:A+O].T                                   # [O,B]

    # ---------- eligibility: option policy matches observed primitive action ----------
    qs0_opt = compute_option_qs_any(
        s0, train_state.opt_enc_state, train_state.opt_q_states,
        use_enc_bn, use_q_bn, use_target=False
    )

    if deterministic_match:
        a_star = jnp.argmax(qs0_opt, axis=-1)                                # [O,B]
        elig_opt = (a_star == a0[None, :]).astype(jnp.float32)               # [O,B]
    else:
        # soft eligibility: pi_o(a0|s0)
        pi0 = jax.nn.softmax(qs0_opt, axis=-1)                               # [O,B,A]
        elig_opt = jnp.take_along_axis(pi0, a0[None, :, None], axis=-1).squeeze(-1)  # [O,B]

    # ---------- EXCLUDE executing option ----------
    # meta0 in [0..A+O-1], option index is meta0-A when meta0>=A
    exec_opt = meta0 - A                                                     # [B]
    is_exec_opt = (meta0 >= A)                                               # [B]
    same_as_exec = (jnp.arange(O)[:, None] == exec_opt[None, :]) & is_exec_opt[None, :]
    elig_opt = elig_opt * (1.0 - same_as_exec.astype(jnp.float32))           # [O,B]

    # ---------- β per option (variable duration) ----------
    # beta_{o,b} = clip(k_steps[b] / dur[o], 0, 1)
    dur = jnp.clip(opt_dur_table.astype(jnp.float32), 1.0, 1e9)              # [O]
    beta_ob = jnp.clip(kf[None, :] / dur[:, None], 0.0, 1.0)                 # [O,B]

    mix_opt = (1.0 - beta_ob) * q_cont_opt + beta_ob * v_term[None, :]       # [O,B]
    y_opt = G[None, :] + boot_mask[None, :] * gamma_k[None, :] * mix_opt     # [O,B]
    y_opt = jax.lax.stop_gradient(y_opt)

    def loss_fn(enc_params, q_params):
        # encoder
        if use_enc_bn and (enc_s.batch_stats is not None):
            emb, up = enc_s.apply_fn(
                {"params": enc_params, "batch_stats": enc_s.batch_stats},
                s0, train=True, mutable=["batch_stats"]
            )
            enc_up = up["batch_stats"]
        else:
            emb = enc_s.apply_fn({"params": enc_params}, s0, train=True)
            enc_up = None

        # meta head
        if use_q_bn and (q_s.batch_stats is not None):
            qv, up = q_s.apply_fn(
                {"params": q_params, "batch_stats": q_s.batch_stats},
                emb, train=True, mutable=["batch_stats"]
            )
            q_up = up["batch_stats"]
        else:
            qv = q_s.apply_fn({"params": q_params}, emb, train=True)
            q_up = None

        q_pred_opt = qv[:, A:A+O].T                                          # [O,B]
        td = q_pred_opt - y_opt                                              # [O,B]
        per_loss = td ** 2

        w = elig_opt                                                         # [O,B]
        if per_is_w is not None:
            w = w * per_is_w[None, :]

        denom = jnp.sum(w) + 1e-8
        loss = jnp.sum(w * per_loss) / denom

        # PER priority per sample: mean abs TD over eligible options
        abs_td = jnp.abs(td)
        new_prio = jnp.sum(w * abs_td, axis=0) / (jnp.sum(w, axis=0) + 1e-8)  # [B]

        elig_any = jnp.mean((jnp.sum(w, axis=0) > 0).astype(jnp.float32))
        q_mean = jnp.mean(q_pred_opt)
        return loss, (enc_up, q_up, q_mean, new_prio, elig_any)

    (loss, (enc_up, q_up, q_mean, new_prio, elig_any)), grads = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(enc_s.params, q_s.params)

    enc_new = enc_s.apply_gradients(grads=grads[0])
    q_new   = q_s.apply_gradients(grads=grads[1])

    if use_enc_bn and (enc_up is not None):
        enc_new = enc_new.replace(batch_stats=enc_up)
    if use_q_bn and (q_up is not None):
        q_new = q_new.replace(batch_stats=q_up)

    new_ts = train_state.replace(enc_state=enc_new, meta_q_state=q_new)
    return new_ts, (loss, q_mean, new_prio, elig_any)


# ==============================================================================
# Primtive update from primitive n-step (PER weights passed in)
# From Primitive Action Data, we update only the primitive actions of the meta-controller
# ==============================================================================
@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def update_primitives_from_prim(
    td_data,
    is_w,
    train_state,
    n_prim_actions: int,   # A
    use_enc_bn: bool,
    use_q_bn: bool,
    use_ddqn: bool,
):
    """
    Primitive-action update (1-step or n-step TD) on the *primitive rows* of the meta-Q head.
    Bootstrap is over ALL meta-actions (A+O), but the regression target is applied only to Q_meta(s, a0) for a0 in [0..A-1].
    """
    enc_s = train_state.enc_state
    q_s   = train_state.meta_q_state

    # td_data now has meta0 as last element
    s0, sk, done_last, a0, G, k_steps, _, _, _meta0 = td_data
    kf = k_steps.astype(jnp.float32)

    # Bootstrap over ALL meta-actions
    q_next_all = get_meta_q_target(sk, enc_s, q_s, use_enc_bn, use_q_bn)  # [B, A+O]

    if use_ddqn:
        q_eval_all = get_meta_q(sk, enc_s, q_s, use_enc_bn, use_q_bn)      # [B, A+O]
        next_u = jnp.argmax(q_eval_all, axis=-1)                           # [B]
        q_boot = q_next_all[jnp.arange(q_next_all.shape[0]), next_u]       # [B]
    else:
        q_boot = jnp.max(q_next_all, axis=-1)                              # [B]

    gamma_k = (jnp.float32(args.env.gamma) ** kf)
    boot_mask = 1.0 - done_last.astype(jnp.float32)
    target = jax.lax.stop_gradient(G + boot_mask * gamma_k * q_boot)       # [B]

    def loss_fn(ep, qp):
        # encoder
        if use_enc_bn and (enc_s.batch_stats is not None):
            emb, up = enc_s.apply_fn({"params": ep, "batch_stats": enc_s.batch_stats},
                                     s0, train=True, mutable=["batch_stats"])
            enc_up = up["batch_stats"]
        else:
            emb = enc_s.apply_fn({"params": ep}, s0, train=True)
            enc_up = None

        # meta-Q head
        if use_q_bn and (q_s.batch_stats is not None):
            qv, up = q_s.apply_fn({"params": qp, "batch_stats": q_s.batch_stats},
                                  emb, train=True, mutable=["batch_stats"])
            q_up = up["batch_stats"]
        else:
            qv = q_s.apply_fn({"params": qp}, emb, train=True)
            q_up = None

        # IMPORTANT: only regress primitive entry corresponding to executed primitive action a0
        prim_q = qv[:, :n_prim_actions]                                    # [B, A]
        q_pred = jnp.take_along_axis(prim_q, a0[:, None], axis=-1).squeeze(-1)  # [B]
        td = q_pred - target
        per_loss = td ** 2

        loss = jnp.mean(is_w * per_loss) if is_w is not None else jnp.mean(per_loss)
        new_prio = jnp.abs(td) + 1e-8
        q_mean = jnp.mean(q_pred)
        return loss, (enc_up, q_up, q_mean, new_prio)

    (loss, (enc_up, q_up, q_mean, new_prio)), grads = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(enc_s.params, q_s.params)

    enc_new = enc_s.apply_gradients(grads=grads[0])
    q_new   = q_s.apply_gradients(grads=grads[1])

    if use_enc_bn and (enc_up is not None):
        enc_new = enc_new.replace(batch_stats=enc_up)
    if use_q_bn and (q_up is not None):
        q_new = q_new.replace(batch_stats=q_up)

    return train_state.replace(enc_state=enc_new, meta_q_state=q_new), (loss, q_mean, new_prio)


# ==============================================================================
# Meta-Q update from meta SMDP buffer (PER weights passed in)
# ==============================================================================
@partial(jax.jit, static_argnums=(4, 5))
def update_meta_from_meta(exp, is_w, enc_state, q_state, use_enc_bn: bool, use_q_bn: bool):
    # one-step SMDP: s0=exp.observation[:,0], sp=exp.observation[:,1]
    s0   = exp.observation[:, 0, ...]
    sp   = exp.observation[:, 1, ...]
    a0   = exp.action[:, 0]
    R    = exp.reward[:, 0]
    done = exp.done[:, 0].astype(jnp.float32)
    dur  = exp.duration[:, 0].astype(jnp.float32)

    v_next = jnp.max(get_meta_q_target(sp, enc_state, q_state, use_enc_bn, use_q_bn), axis=-1)
    target = jax.lax.stop_gradient(R + (1.0 - done) * (args.env.gamma ** dur) * v_next)

    def loss_fn(ep, qp):
        if use_enc_bn and (enc_state.batch_stats is not None):
            emb, up = enc_state.apply_fn({"params": ep, "batch_stats": enc_state.batch_stats},
                                         s0, train=True, mutable=["batch_stats"])
            enc_up = up["batch_stats"]
        else:
            emb = enc_state.apply_fn({"params": ep}, s0, train=True)
            enc_up = None

        if use_q_bn and (q_state.batch_stats is not None):
            qv, up = q_state.apply_fn({"params": qp, "batch_stats": q_state.batch_stats},
                                      emb, train=True, mutable=["batch_stats"])
            q_up = up["batch_stats"]
        else:
            qv = q_state.apply_fn({"params": qp}, emb, train=True)
            q_up = None

        q_pred = qv[jnp.arange(qv.shape[0]), a0]
        td = q_pred - target
        per_loss = td ** 2

        loss = jnp.mean(is_w * per_loss) if is_w is not None else jnp.mean(per_loss)
        new_prio = jnp.abs(td) + 1e-8
        q_mean = jnp.mean(q_pred)

        greedy_meta = jnp.argmax(qv, axis=-1)  # [B]
        greedy_is_opt = (greedy_meta >= n_actions)  # [B]
        greedy_opt_frac = jnp.mean(greedy_is_opt.astype(jnp.float32))

        # behavior option usage from replay actions
        behav_is_opt = (a0 >= n_actions)
        behav_opt_frac = jnp.mean(behav_is_opt.astype(jnp.float32))

        return loss, (enc_up, q_up, q_mean, new_prio, greedy_opt_frac, behav_opt_frac)

    (loss, (enc_up, q_up, q_mean, new_prio, greed_opt_frac, behav_opt_frac)), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
        enc_state.params, q_state.params
    )

    enc_state = enc_state.apply_gradients(grads=grads[0])
    q_state   = q_state.apply_gradients(grads=grads[1])
    if use_enc_bn and (enc_up is not None):
        enc_state = enc_state.replace(batch_stats=enc_up)
    if use_q_bn and (q_up is not None):
        q_state = q_state.replace(batch_stats=q_up)

    return enc_state, q_state, loss, q_mean, new_prio, greed_opt_frac, behav_opt_frac


@partial(jax.jit, static_argnums=range(3, 9))
def compute_option_nstep_targets_norm(
    train_state,
    exp,
    int_rms: RunningMeanStd,
    gamma: float,
    n_options: int,
    bi_dir_opt: bool,
    use_enc_bn: bool,
    use_q_bn: bool,
    use_ddqn: bool,
    eps: float = 1e-8,
):
    """
    Returns:
      targets: [O,B]
      eig_rewards_mean: scalar (your diag)
      new_int_rms: RunningMeanStd with mean/var/count shape [O]
    """
    B, T = exp.observation.shape[:2]
    O = n_options
    use_lap_bn = False

    # --- Laplacian embeddings for states 0..T (need T+1, last repeats)
    nhwc = exp.observation
    all_states = jnp.concatenate([nhwc, nhwc[:, -1:, ...]], axis=1)   # [B,T+1,...]
    flat = all_states.reshape((B * (T + 1),) + nhwc.shape[2:])

    lap_vars = {"params": train_state.lap_state.params}
    if use_lap_bn and train_state.lap_state.batch_stats is not None:
        lap_vars["batch_stats"] = train_state.lap_state.batch_stats
        phis_flat, _ = train_state.lap_state.apply_fn(lap_vars, flat, train=False, mutable=["batch_stats"])
    else:
        phis_flat = train_state.lap_state.apply_fn(lap_vars, flat, train=False)

    n_lap = O // 2 if bi_dir_opt else O
    phis_flat = phis_flat[:, :n_lap]
    phis = phis_flat.reshape(B, T + 1, n_lap)                         # [B,T+1,n_lap]

    # Normalize embeddings
    # Update running maxabs on phis (option dims)
    new_maxabs = maxabs_update_ema(train_state.phi_maxabs, phis, alpha=0.01)
    if args.algo.phi_norm:
        phis = maxabs_normalize(new_maxabs, phis, eps=1e-8)

    diffs = phis[:, 1:, :] - phis[:, :-1, :]  # [B,T,n_lap]
    # if args.algo.phi_norm:
    #     new_maxabs = maxabs_update_ema(train_state.phi_maxabs, diffs, alpha=0.001)  # state max_abs is [n_lap]
    #     diffs = maxabs_normalize(new_maxabs, diffs, eps=1e-8)

    # diffs and raw intrinsic reward
    r_all_raw = jnp.concatenate([diffs, -diffs], axis=-1) if bi_dir_opt else diffs  # [B,T,O] or [B,T,n_lap]

    # r_all = r_all_raw
    # ------------------------------------------------------------
    # Done-safe mask for per-step intrinsic rewards (your original logic)
    # ------------------------------------------------------------
    not_done = 1.0 - exp.done[:, :T].astype(jnp.float32)  # [B,T]
    prefix = jnp.concatenate([jnp.ones((B, 1), jnp.float32), not_done[:, :-1]], axis=1)
    alive_before = jnp.cumprod(prefix, axis=1)
    mask = jnp.clip(alive_before + exp.done[:, :T].astype(jnp.float32), 0.0, 1.0)  # [B,T]

    # ------------------------------------------------------------
    # Option-specific stall penalty (done-safe)
    # ------------------------------------------------------------
    stall_thr = jnp.float32(getattr(args.algo, "stall_thr", 0.02))
    stall_coef = jnp.float32(getattr(args.algo, "stall_coef", 0.01))

    # stall per base dimension (n_lap): [B,T,n_lap]
    stall_dim = jax.nn.relu(stall_thr - jnp.abs(diffs)) / (stall_thr + 1e-8)

    # mask out invalid/terminal steps
    stall_dim = stall_dim * mask[..., None]  # [B,T,n_lap]

    # map to options: [B,T,O]
    if bi_dir_opt:
        stall_o = jnp.concatenate([stall_dim, stall_dim], axis=-1)  # [B,T,2*n_lap] = [B,T,O]
    else:
        stall_o = stall_dim  # [B,T,O] where O=n_lap

    # final intrinsic per-step reward, masked + stall
    # r_all = (r_all_raw * mask[..., None]) - stall_coef * stall_o  # [B,T,O]
    r_all = r_all_raw * mask[..., None]

    # optional scalar diagnostic
    stall_val = jnp.sum(stall_o) / (jnp.sum(mask) * jnp.float32(O) + 1e-8)

    not_done = 1.0 - exp.done[:, :T].astype(jnp.float32)              # [B,T]
    prefix = jnp.concatenate([jnp.ones((B, 1), jnp.float32), not_done[:, :-1]], axis=1)
    alive_before = jnp.cumprod(prefix, axis=1)
    mask = jnp.clip(alive_before + exp.done[:, :T].astype(jnp.float32), 0.0, 1.0)

    discounts = jnp.power(jnp.float32(gamma), jnp.arange(T, dtype=jnp.float32))[None, :]
    n_ret = jnp.einsum("bto,bt->bo", r_all, discounts * mask)         # [B,O]

    if args.algo.intr_norm:
        new_rms = rms_update_per_option(int_rms, n_ret)                   # mean/var/count are [O]
        n_ret_norm = rms_normalize_per_option(new_rms, n_ret, eps=eps)    # [B,O]
    else:
        new_rms = int_rms
        n_ret_norm = n_ret

    # --- bootstrap
    boot_mask = 1.0 - exp.done.any(axis=1).astype(jnp.float32)        # [B]
    gamma_n = jnp.float32(gamma) ** T

    sp = exp.observation[:, -1, ...]
    qs_t = compute_option_qs_any(
        sp, train_state.opt_enc_state, train_state.opt_q_states,
        use_enc_bn, use_q_bn, use_target=True
    )                                                                 # [O,B,A]

    if use_ddqn:
        qs_o = compute_option_qs_any(
            sp, train_state.opt_enc_state, train_state.opt_q_states,
            use_enc_bn, use_q_bn, use_target=False
        )
        na = jnp.argmax(qs_o, axis=-1)                                # [O,B]
        q_sp = jnp.take_along_axis(qs_t, na[..., None], axis=-1).squeeze(-1)  # [O,B]
    else:
        q_sp = jnp.max(qs_t, axis=-1)                                 # [O,B]

    targets = n_ret_norm.T + (gamma_n * boot_mask)[None, :] * q_sp    # [O,B]
    eig_rewards_mean = jnp.abs(r_all).mean()

    return targets, eig_rewards_mean, new_rms, new_maxabs, jnp.mean(stall_o)


@partial(jax.jit, static_argnums=(4, 5))
def update_option_networks(train_state, obs0, act0, opt_targets, use_enc_bn: bool, use_q_bn: bool):
    """
    Shared option encoder + per-option Q-head regression to intrinsic targets.
    opt_targets: [O,B]
    """
    opt_targets = jax.lax.stop_gradient(opt_targets)

    enc_s = train_state.opt_enc_state
    q_states = train_state.opt_q_states
    O = len(q_states)

    # Stack head params/stats
    q_params = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s.params for s in q_states])
    q_bs = (
        jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s.batch_stats for s in q_states])
        if use_q_bn else None
    )
    q_apply = q_states[0].apply_fn

    def loss_fn(enc_params, q_params_all):
        # ---- shared encoder forward (train=True) ----
        if use_enc_bn and (enc_s.batch_stats is not None):
            emb, up = enc_s.apply_fn(
                {"params": enc_params, "batch_stats": enc_s.batch_stats},
                obs0, train=True, mutable=["batch_stats"]
            )
            enc_up = up["batch_stats"]
        else:
            emb = enc_s.apply_fn({"params": enc_params}, obs0, train=True)
            enc_up = None

        # ---- per-head forward + TD ----
        def per_head(qp, qb, tgt_ob):
            if use_q_bn:
                qv, up = q_apply(
                    {"params": qp, "batch_stats": qb},
                    emb, train=True, mutable=["batch_stats"]
                )
                qb_up = up["batch_stats"]
            else:
                qv = q_apply({"params": qp}, emb, train=True)
                qb_up = None

            pred = jnp.take_along_axis(qv, act0[:, None], axis=-1).squeeze(-1)  # [B]
            loss = jnp.mean((pred - tgt_ob) ** 2)
            return loss, (jnp.mean(pred), qb_up)

        losses, (q_means, qb_up_all) = jax.vmap(
            per_head,
            in_axes=(0, 0 if use_q_bn else None, 0),
        )(q_params_all, q_bs, opt_targets)

        total = jnp.sum(losses)
        return total, (enc_up, losses, q_means, qb_up_all)

    (total, (enc_up, losses, q_means, qb_up_all)), (g_enc, g_q) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(enc_s.params, q_params)

    # ---- apply grads ----
    new_enc = enc_s.apply_gradients(grads=g_enc)
    if use_enc_bn and (enc_up is not None):
        new_enc = new_enc.replace(batch_stats=enc_up)

    new_q_states = []
    for i in range(O):
        gi = jax.tree_util.tree_map(lambda x: x[i], g_q)
        sQ = q_states[i].apply_gradients(grads=gi)
        if use_q_bn and (qb_up_all is not None):
            sQ = sQ.replace(batch_stats=jax.tree_util.tree_map(lambda x: x[i], qb_up_all))
        new_q_states.append(sQ)

    train_state = train_state.replace(opt_enc_state=new_enc, opt_q_states=new_q_states)
    return train_state, (losses, q_means)


# ==============================================================================
# Laplacian update (state-only ALLO)
# ==============================================================================
@partial(jax.jit, static_argnums=(4,))
def update_laplacian_from_prim(key, train_state, lap_obs_seq, lap_valid_mask, _n_actions):
    gamma      = args.env.gamma
    use_lap_bn = False
    use_bar    = args.laplacian.barrier.on
    use_dual   = args.laplacian.dual.on

    lap_state = train_state.lap_state
    B, N = lap_obs_seq.shape[:2]
    # The first state is the anchor, leaving N - 1 future differences/masks.
    n_future_steps = N - 1

    mask         = lap_valid_mask[:, 0]
    future_masks = lap_valid_mask[:, 1:]
    flat_obs     = lap_obs_seq.reshape((B * N,) + lap_obs_seq.shape[2:])

    # transition validity for eigenvalues: [B, N-1] float32 in {0,1}
    valid_trans = (lap_valid_mask[:, :-1] & lap_valid_mask[:, 1:]).astype(jnp.float32)

    # curvature coefficient (start small: 1e-3; try 3e-3, 1e-2)
    lam_curv = getattr(args.laplacian, "curv_coef", 0.)

    def loss_fn(lap_params, batch_stats):
        vars_ = {"params": lap_params}
        if use_lap_bn and batch_stats is not None:
            vars_["batch_stats"] = batch_stats
            phis_flat, upd = lap_state.apply_fn(vars_, flat_obs, train=True, mutable=["batch_stats"])
            new_bs = upd["batch_stats"]
        else:
            phis_flat = lap_state.apply_fn(vars_, flat_obs, train=True)
            new_bs = None

        D = phis_flat.shape[-1]
        phis = phis_flat.reshape(B, N, D)  # [B,N,D]

        # ----- ALLO loss -----
        ph0  = phis[:, 0, :]
        fut  = phis[:, 1:, :]
        allo_loss, (pos_loss, ortho_loss, dual_loss), (_avg_w, err_mat) = multi_step_ALLO(
            key, ph0, fut, None, mask, future_masks,
            lap_state.barrier_coeffs, lap_state.betas,
            gamma=gamma, state_action=False, n_steps=n_future_steps,
            use_dual=use_dual, use_weights=False, weights=None,
            average_loss=args.laplacian.average_loss,
        )

        # ----- curvature (second-difference) penalty (scale-invariant) -----
        if N >= 3:
            d1 = phis[:, 1:, :] - phis[:, :-1, :]  # [B, N-1, D]
            d2 = phis[:, 2:, :] - 2.0 * phis[:, 1:-1, :] + phis[:, :-2, :]  # [B, N-2, D]

            # valid_trans: edges (t -> t+1), length N-1
            v1 = valid_trans  # [B, N-1]
            v01 = valid_trans[:, :-1]  # [B, N-2]
            v12 = valid_trans[:, 1:]  # [B, N-2]
            v2 = v01 * v12  # [B, N-2]

            # energies
            eps_curv = 1e-8
            denom1 = jnp.sum(v1) * jnp.float32(D) + eps_curv
            denom2 = jnp.sum(v2) * jnp.float32(D) + eps_curv

            d1_energy = jnp.sum(v1[..., None] * (d1 ** 2)) / denom1  # scalar
            d2_energy = jnp.sum(v2[..., None] * (d2 ** 2)) / denom2  # scalar

            # relative curvature (dimensionless)
            curv_loss = d2_energy / (d1_energy + eps_curv)
        else:
            curv_loss = 0.0

        # ----- eigenvalue estimate (stop-grad) -----
        lambda_batch = compute_batch_eigenvalues_seq(phis, valid_trans)  # [D]
        lambda_batch = jax.lax.stop_gradient(lambda_batch)
        loss = allo_loss + lam_curv * curv_loss
        return loss, (new_bs, pos_loss, ortho_loss, dual_loss, curv_loss, err_mat, lambda_batch)

    (total_loss, (bs_updated, pos_loss, ortho_loss, dual_loss, curv_loss, err_mat, lambda_batch)), grads = (
        jax.value_and_grad(loss_fn, has_aux=True)(lap_state.params, lap_state.batch_stats)
    )

    new_lap = lap_state.apply_gradients(grads=grads)
    if use_lap_bn and (bs_updated is not None):
        new_lap = new_lap.replace(batch_stats=bs_updated)

    if use_bar or use_dual:
        new_lap = update_ALLO_parameters(
            new_lap, err_mat,
            update_rates=args.laplacian.rate,
            barrier_args=args.laplacian.barrier,
            dual_args=args.laplacian.dual,
            average_loss=args.laplacian.average_loss,
        )

    alpha_ev = getattr(args.laplacian, "eig_alpha", 0.1)
    ev_old = new_lap.eigen_values.reshape(-1)
    ev_new = update_eigenvalues(ev_old, lambda_batch, alpha=alpha_ev)
    new_lap = new_lap.replace(eigen_values=ev_new[None, :])

    train_state = train_state.replace(lap_state=new_lap)

    # return curv_loss too so you can log it and confirm it decreases
    return train_state, (total_loss, pos_loss, ortho_loss, dual_loss, curv_loss, lambda_batch)


def eigenvalues_to_option_durations(
    eigs, n_options, bi_dir_opt=False,
    min_dur=1, max_dur=200, eps=1e-6
):
    """
    eigs: [D] (or [1,D]) eigenvalues estimate
    n_options <= D (for non-bidir), and n_options even if bidir (your convention).
    Returns durations: [n_options] int32 where dur_i = ceil(1 / eig_i) (clipped).
    """
    eigs = np.asarray(eigs, dtype=np.float32).reshape(-1)
    eigs = np.nan_to_num(eigs, nan=0.0, posinf=0.0, neginf=0.0)
    eigs = np.clip(eigs, 0.0, None)

    if bi_dir_opt:
        n_lap = n_options // 2
        lam = np.maximum(eigs[:n_lap], eps)         # only first n_lap eigenvalues
        dur = np.ceil(1.0 / lam).astype(np.int32)
        dur = np.clip(dur, min_dur, max_dur)
        dur = np.concatenate([dur, dur], axis=0)    # same duration for +/- options
    else:
        lam = np.maximum(eigs[:n_options], eps)
        dur = np.ceil(1.0 / lam).astype(np.int32)
        dur = np.clip(dur, min_dur, max_dur)

    return dur.astype(np.int32)


@jax.jit
def compute_batch_eigenvalues_seq(phis, valid_transitions, eps=1e-8):
    """
    phis: [B, T, D]
    valid_transitions: [B, T-1] float32 in {0,1}  (1 => (t->t+1) is valid)
    Returns: lambda_batch [D]
    """
    phi  = phis[:, :-1, :]    # [B,T-1,D]
    phi1 = phis[:, 1:, :]     # [B,T-1,D]

    w = valid_transitions[..., None]  # [B,T-1,1]
    wsum = jnp.sum(w, axis=(0,1)) + eps  # scalar

    num   = jnp.sum(w * (phi - phi1) ** 2, axis=(0,1)) / wsum
    denom = jnp.sum(w * 0.5 * (phi ** 2 + phi1 ** 2), axis=(0,1)) / wsum

    return num / (denom + eps)  # [D]


def meta_primitive_probs(q_meta: jnp.ndarray, eps: float, mu: float, A: int, O: int) -> jnp.ndarray:
    """
    Returns mu_meta(a|s) over primitive actions only, shape [A].
    Matches your stored ap formula structure:
        ap(meta) = base[meta] + (1-eps)*softmax(q_meta)[meta]
    but returns it for all primitive actions a=0..A-1.
    """
    # exploration mass assigned to primitives (uniform over A)
    base_prim = (eps * (1.0 - mu)) / max(1, A)
    # greedy/non-exploration mass: softmax over meta actions (A+O)
    sm = jax.nn.softmax(q_meta, axis=-1)  # [A+O]
    return base_prim + (1.0 - eps) * sm[:A]  # [A]


def option_action_probs(q_opt: jnp.ndarray) -> jnp.ndarray:
    """
    Returns pi_o(a|s) for an option, shape [A].
    Even if you EXECUTE argmax, we STORE softmax prob of that argmax action.
    """
    return jax.nn.softmax(q_opt, axis=-1)  # [A]


@partial(jax.jit, static_argnums=(2, 3,))
def sample_random_option_filtered(key: jax.Array, q_meta: jnp.ndarray, A: int, bidir_opt: bool):
    """
    Sample a meta-action in [A, A+O) under your filtered option-exploration rule.

    Rules:
      - Non-bidir: eligible options are those with q_opt[o] > 0; sample uniformly among them.
      - Bidir (O=2*n_lap): treat options as (+d, -d) pairs:
          * eligible dims are those with max(q_plus[d], q_minus[d]) > 0
          * sample dim uniformly among eligible dims
          * choose direction = argmax(q_plus[d], q_minus[d])
    Fallback:
      - if no eligible options/dims, sample uniformly among all options (or all dims in bidir, then pick best dir).

    Returns:
      meta: int32 in [A, A+O)
      p_explore: float32 = P(meta | explore-options branch)  (i.e., conditional on u2 < mu)
    """
    q_opt = q_meta[A:]                  # [O]
    O = q_opt.shape[0]
    eps = jnp.float32(1e-8)

    if bidir_opt:
        n_lap = O // 2
        q_plus  = q_opt[:n_lap]         # [n_lap]
        q_minus = q_opt[n_lap:]         # [n_lap]

        q_pair = jnp.maximum(q_plus, q_minus)                 # [n_lap]
        elig = (q_pair > 0.0).astype(jnp.float32)             # [n_lap]
        n_elig = jnp.sum(elig)

        # P(dim | explore-options)
        p_dim = jnp.where(
            n_elig > 0.0,
            elig / (n_elig + eps),
            jnp.ones((n_lap,), jnp.float32) / jnp.float32(n_lap),
        )

        d = jax.random.choice(key, a=jnp.arange(n_lap), p=p_dim)  # int32 scalar

        choose_plus = (q_plus[d] >= q_minus[d])
        opt_idx = jnp.where(choose_plus, d, d + n_lap)            # [0..O-1]
        meta = jnp.int32(A) + jnp.int32(opt_idx)

        # Because direction is deterministic given d, P(meta | explore-options) = P(d)
        p_explore = p_dim[d]
        return meta, p_explore

    else:
        elig = (q_opt > 0.0).astype(jnp.float32)             # [O]
        n_elig = jnp.sum(elig)

        p_opt = jnp.where(
            n_elig > 0.0,
            elig / (n_elig + eps),
            jnp.ones((O,), jnp.float32) / jnp.float32(O),
        )

        opt_idx = jax.random.choice(key, a=jnp.arange(O), p=p_opt)
        meta = jnp.int32(A) + jnp.int32(opt_idx)
        p_explore = p_opt[opt_idx]
        return meta, p_explore


# ==============================================================================
# Main HDQN learning loop
# ==============================================================================
def hdqn_learning(train_state, rb_p_state, rb_m_state, key, Env, RBp, RBm, n_actions, n_options):
    buffer_device = jax.devices(args.buffer.device)[0]

    use_enc_bn = args.laplacian.batch_norm
    use_q_bn   = args.net.batch_norm

    total_steps = args.algo.tot_steps
    A, O = n_actions, n_options
    A_meta = A + O

    opt_dur_table = np.full((O,), args.algo.opt_dur, dtype=np.int32)
    opt_dur_min = getattr(args.algo, "opt_dur_min", 1)
    opt_dur_max = getattr(args.algo, "opt_dur_max", 100)  # or something larger if you want

    # ε, μ schedules (μ controls how often random exploration picks an option)
    eps_sched = partial(linear_schedule, args.algo.eps.start, args.algo.eps.end,
                        int(args.algo.eps.frac * (total_steps + args.algo.starts)))
    mu_sched = partial(linear_schedule, 0.5, 0.5, total_steps)

    start_training = False
    obs, _ = Env.reset(seed=args.exp.seed)

    # Option bookkeeping for meta buffer
    last_term = True
    opt_count = 0
    start_obs = obs
    start_meta = -1
    start_ap = 1.0
    cum_r, cum_d = 0.0, 1.0

    # ---- Execution trace bookkeeping (for mu(a|s) under options) ----
    curr_meta = -1  # current meta action being executed (frozen while option runs)
    curr_meta_prob = 1.0  # mu_meta(curr_meta | s_start) at option start (your ap)
    curr_opt_sel = -1  # option index if curr_meta is option, else -1
    cont_prob = 1.0  # approx P(option has not terminated yet) up to *this* step

    test_results = []
    eps_step, eps_return = 1, 0
    for global_step in range(total_steps + 1):
        key, k1, k2, k3 = jax.random.split(key, 4)
        eps = float(eps_sched(global_step))
        mu  = float(mu_sched(global_step))

        # preprocess obs
        obs_g = rgb_to_gray_uint8(obs)
        obs_b = jnp.expand_dims(obs_g, 0)  # [1,H,W,1]

        if last_term:
            q_meta = get_meta_q(obs_b, train_state.enc_state, train_state.meta_q_state, use_enc_bn, use_q_bn)[0]
            greedy_meta = int(jnp.argmax(q_meta))

            u1 = float(jax.random.uniform(k1))
            u2 = float(jax.random.uniform(k2))

            if u1 < eps:
                if u2 < mu:
                    # filtered option exploration
                    meta_j, p_explore = sample_random_option_filtered(k3, q_meta, A, args.algo.bidir_opt)
                    meta = int(meta_j)
                    ap = float(eps * mu * float(p_explore))
                else:
                    # primitive exploration: uniform over primitives
                    meta = int(jax.random.randint(k3, (), 0, A))
                    ap = float(eps * (1.0 - mu) * (1.0 / max(1, A)))
            else:
                # greedy (deterministic argmax)
                meta = greedy_meta
                ap = float(1.0 - eps)

            # freeze meta + prob for this segment
            curr_meta = meta
            curr_meta_prob = ap
            curr_opt_sel = (meta - A) if (meta >= A) else -1
            cont_prob = 1.0

        else:
            # ----- continue previously selected option -----
            meta = curr_meta

        # Start of an option segment (or primitive step treated as length-1 option)
        if last_term:
            start_obs, start_meta, start_ap = obs_g, meta, ap
            opt_count, cum_r, cum_d = 0, 0.0, 1.0

        # --- low-level action
        if meta < A:
            action = meta
            curr_term = True  # primitive "option" terminates immediately
            action_prob = jnp.float32(curr_meta_prob)  # mu(a|s) from meta policy at this step
        else:
            sel = curr_opt_sel  # == meta - A

            # option Q-values for current state
            q_opt = compute_option_q_one(
                obs_b,
                train_state.opt_enc_state, train_state.opt_q_states[sel],
                use_enc_bn, use_q_bn, use_target=False
            )[0]  # [A]
            action = int(jnp.argmax(q_opt))

            # option primitive policy prob for the chosen (argmax) action
            # (treat Q as logits; optionally divide by temperature if you have one)
            pi_opt = jax.nn.softmax(q_opt)[action]

            # behavior prob of the primitive action under "option execution"
            action_prob = jnp.float32(curr_meta_prob) * jnp.float32(cont_prob) * jnp.float32(pi_opt)

            # your *actual* termination condition is 1/lambda
            curr_opt_dur = int(opt_dur_table[sel])
            beta_t = jnp.clip(jnp.float32(opt_count + 1) / jnp.float32(curr_opt_dur), 0.0, 1.0)
            curr_term = (opt_count + 1 >= curr_opt_dur)

            # update continuation probability for the NEXT step if we continue
            cont_prob = float(jnp.float32(cont_prob) * (1.0 - beta_t))

        # --- env step
        next_obs, reward, terminated, _, _ = Env.step(action)
        done = bool(terminated)
        if done:
            curr_term = True

        # --- add primitive transition to prim buffer
        with jax.default_device(buffer_device):
            ts = PrimStep(
                observation=jnp.expand_dims(obs_g, (0, 1)),              # [1,1,H,W,1]
                action=jnp.expand_dims(jnp.int32(action), (0, 1)),
                reward=jnp.expand_dims(jnp.float32(reward), (0, 1)),
                done=jnp.expand_dims(jnp.bool_(done), (0, 1)),
                action_prob=jnp.expand_dims(jnp.float32(action_prob), (0, 1)),
                meta_action=jnp.expand_dims(jnp.int32(curr_meta), (0, 1)),
            )
            rb_p_state = RBp.add(rb_p_state, ts)

        # --- accumulate SMDP reward and write meta buffer on termination
        cum_r += cum_d * reward
        cum_d *= args.env.gamma
        opt_count += 1

        if curr_term:
            if start_meta >= A:  # Meta-buffer only updates options (no meta-primitives here) -- useful for SMDP updates
                with jax.default_device(buffer_device):
                    ms = MetaStep(
                        observation=jnp.expand_dims(start_obs, (0, 1)),
                        action=jnp.expand_dims(jnp.int32(start_meta), (0, 1)),
                        reward=jnp.expand_dims(jnp.float32(cum_r), (0, 1)),
                        done=jnp.expand_dims(jnp.bool_(done), (0, 1)),
                        duration=jnp.expand_dims(jnp.int32(opt_count), (0, 1)),
                        behav_prob=jnp.expand_dims(jnp.float32(start_ap), (0, 1)))
                    rb_m_state = RBm.add(rb_m_state, ms)
            last_term, opt_count, cum_r, cum_d = True, 0, 0.0, 1.0
        else:
            last_term = False

        # --- episode bookkeeping
        eps_return += reward
        if terminated or eps_step >= args.algo.max_eps_len:
            if args.wb.on and global_step >= args.algo.starts:
                wandb.log({'values/episodic_return': eps_return,
                           'values/episodic_length': eps_step,
                           'values/epsilon': eps}, step=global_step)
            print(f"Step {global_step} | EpLen {eps_step} | Return {eps_return}")
            next_obs, _ = Env.reset()
            eps_step, eps_return = 0, 0
            last_term = True

        obs = next_obs
        eps_step += 1

        # --- learning start
        if not start_training and (global_step >= args.algo.starts and (RBp.can_sample(rb_p_state) and RBm.can_sample(rb_m_state))):
            print("Learning starts")
            start_training = True
            start_time = time.time()

        # --- training step
        if start_training and (global_step % args.algo.train_freq == 0):
            beta = linear_schedule(args.buffer.beta, 0.1, total_steps, global_step - args.algo.starts)

            # ==========================================================
            # (A) Sample ONCE from primitive buffer (PER or uniform)
            # ==========================================================
            with jax.default_device(buffer_device):
                key, sample_key = get_buffer_key(key)
                prim = RBp.sample(rb_p_state, sample_key)  # has .experience, .indices, .probabilities

            exp_full = prim.experience
            T_full = exp_full.reward.shape[1]

            # Slice for TD and Lap (both derived from same exp_full)
            T_td  = args.algo.n_step + 1
            T_lap = args.laplacian.n_step + 1
            td_exp  = slice_seq(exp_full, T_td)
            lap_exp = slice_seq(exp_full, T_lap)

            # PER IS weights for TD update (based on TD seq_len)
            td_is_w = None
            if args.buffer.prioritized:
                td_is_w = compute_is_weights(prim.probabilities,
                                             rb_p_state.running_index, rb_p_state.sum_tree_state.capacity,
                                             seq_len=T_td, beta=beta)

            # ==========================================================
            # (B) Meta-from-primitive TD update (returns new priorities)
            # ==========================================================
            td_data = compute_n_step_from_td_batch(td_exp, args.env.gamma, args.algo.n_step)

            train_state, (prim_td_loss, prim_q_mean, new_prio_prim) = update_primitives_from_prim(
                td_data, td_is_w, train_state,
                n_prim_actions=A,
                use_enc_bn=use_enc_bn,
                use_q_bn=use_q_bn,
                use_ddqn=True,
            )

            # IOQ: update options except executing option, β uses opt_dur_table
            train_state, (ioq_loss, ioq_q_mean, new_prio_ioq, ioq_elig_any) = (
                update_ioq_options_from_prim_excluding_exec(
                    td_data=td_data,
                    per_is_w=td_is_w,
                    train_state=train_state,
                    n_prim_actions=A,
                    n_options=O,
                    opt_dur_table=jnp.asarray(opt_dur_table),  # <-- durations table you maintain
                    use_enc_bn=use_enc_bn,
                    use_q_bn=use_q_bn,
                    use_ddqn=True,
                    deterministic_match=True,  # or False for soft pi eligibility
                )
            )

            # priorities updated from IOQ TDs
            if args.buffer.prioritized:
                with jax.default_device(buffer_device):
                    rb_p_state = RBp.set_priorities(rb_p_state, prim.indices, jnp.maximum(new_prio_prim, new_prio_ioq))

            # ==========================================================
            # (C) Laplacian update (state-only) from lap_exp
            # ==========================================================
            lap_obs_seq = lap_exp.observation
            lap_valid   = jnp.logical_not(lap_exp.done)
            train_state, (eig_loss, eig_pos, eig_ort, eig_dual, curv_loss, _) = update_laplacian_from_prim(
                key, train_state, lap_obs_seq, lap_valid, A)

            if args.algo.eig_dur:
                eigs = np.asarray(train_state.lap_state.eigen_values[0], dtype=np.float32)[:n_options]  # [D]
                opt_dur_table = eigenvalues_to_option_durations(eigs,
                                                                n_options=O, bi_dir_opt=args.algo.bidir_opt,
                                                                min_dur=opt_dur_min, max_dur=opt_dur_max,eps=1e-6)
            else:
                opt_dur_table = np.full((O,), args.algo.opt_dur, dtype=np.int32)
            # ==========================================================
            # (D) Option networks update using intrinsic n-step targets
            # ==========================================================
            opt_targets, eig_rewards_mean, new_rms, new_maxabs, stall_vals = compute_option_nstep_targets_norm(
                train_state, td_exp, train_state.int_rms, args.env.gamma,
                n_options=O, bi_dir_opt=args.algo.bidir_opt,
                use_enc_bn=use_enc_bn, use_q_bn=use_q_bn, use_ddqn=True, eps=1e-8)
            train_state = train_state.replace(int_rms=new_rms, phi_maxabs=new_maxabs)

            obs0 = td_exp.observation[:, 0, ...]
            act0 = td_exp.action[:, 0]
            train_state, (opt_td_losses, opt_q_means) = update_option_networks(
                train_state, obs0, act0, opt_targets, use_enc_bn, use_q_bn
            )

            # ==========================================================
            # (E) Meta-from-meta (SMDP) update (if meta buffer ready)
            # ==========================================================
            # smdp_loss, smdp_q_mean, smdp_new_prio, greed_opt_frac, behav_opt_frac = 0, 0, None, 0, 0
            # if RBm.can_sample(rb_m_state):
            with jax.default_device(buffer_device):
                key, mk = get_buffer_key(key)
                mb = RBm.sample(rb_m_state, mk)

            m_is_w = None
            if args.buffer.prioritized:
                m_is_w = compute_is_weights(
                    mb.probabilities,
                    rb_m_state.running_index,
                    rb_m_state.sum_tree_state.capacity,
                    seq_len=args.algo.m_step + 1,
                    beta=beta,
                )

            enc_new, meta_q_new, smdp_loss, smdp_q_mean, smdp_new_prio, greed_opt_frac, behav_opt_frac = (
                update_meta_from_meta(mb.experience, m_is_w,
                                      train_state.enc_state, train_state.meta_q_state,
                                      use_enc_bn, use_q_bn))
            train_state = train_state.replace(enc_state=enc_new, meta_q_state=meta_q_new)

            if args.buffer.prioritized:
                with jax.default_device(buffer_device):
                    rb_m_state = RBm.set_priorities(rb_m_state, mb.indices, smdp_new_prio)

            if args.wb.on and (global_step % 1000 == 0):
                metrics = {
                    # Exploration / PER
                    "values/epsilon": float(eps),
                    "values/mu": float(mu),
                    "values/beta": float(beta),

                    # Intra-Option Q-learning
                    "losses/IntraOption": float(ioq_loss),
                    "values/IntraOption-Q": float(ioq_q_mean),
                    "values/IntraOption-elig-any": float(ioq_elig_any),

                    # Meta-from-primitive TD
                    "losses/primitive": float(prim_td_loss),
                    "values/q_prim_mean": float(prim_q_mean),

                    # Laplacian / ALLO
                    "eigen/total": float(eig_loss),
                    "eigen/pos": float(eig_pos),
                    "eigen/ortho": float(eig_ort),
                    "eigen/dual": float(eig_dual),
                    "eigen/curv": float(curv_loss),

                    "Diag/Eigen_Rew": float(eig_rewards_mean),
                    "Diag/greedy_opt_frac": float(greed_opt_frac),
                    "Diag/behav_opt_frac": float(behav_opt_frac),

                    # Options (aggregate across options)
                    "options/td_mean": float(jnp.mean(opt_td_losses)),
                    "options/q_mean": float(jnp.mean(opt_q_means)),
                    "options/td_max": float(jnp.max(opt_td_losses)),
                    "options/q_max": float(jnp.max(opt_q_means)),
                    "options/stall_val": float(jnp.mean(stall_vals)),

                    "options/dur_min": int(opt_dur_table.min()),
                    "options/dur_max": int(opt_dur_table.max()),
                    "options/dur_mean": float(opt_dur_table.mean()),

                    "laplacian/eig_min": float(jnp.min(eigs)),
                    "laplacian/eig_max": float(jnp.max(eigs)),
                    "laplacian/eig_mean": float(jnp.mean(eigs)),

                    'values/SPS': int((global_step - args.algo.starts) / (time.time() - start_time))
                }

                # Meta-from-meta only exists if that branch ran
                if RBm.can_sample(rb_m_state):
                    metrics.update({
                        "losses/meta_from_meta": float(smdp_loss),
                        "values/meta_q_meta_mean": float(smdp_q_mean),
                    })

                wandb.log(metrics, step=global_step)

            # ==========================================================
            # (F) Target net updates
            # ==========================================================
            if (args.algo.tau == 1 and global_step % args.algo.tgt_upd_freq == 0) or (args.algo.tau < 1):
                enc_new, meta_q_new = soft_update(args.algo.tau, train_state.enc_state, train_state.meta_q_state)
                train_state = train_state.replace(enc_state=enc_new, meta_q_state=meta_q_new)

                # Option target updates
                train_state = train_state.replace(
                    opt_enc_state=soft_update_only(train_state.opt_enc_state, args.algo.tau),
                    opt_q_states=[soft_update_only(s, args.algo.tau) for s in train_state.opt_q_states],
                )

            # ...
            if (global_step >= total_steps * 0.5) and (global_step % 5_000 == 0):
                key, reached_goal, test_ret, test_len, test_stats = evaluate_policy_hdqn(
                    key=key,
                    train_state=train_state,
                    Env=Env,
                    global_step=global_step,
                    n_actions=A,
                    n_options=O,
                    use_enc_bn=use_enc_bn,
                    use_q_bn=use_q_bn,
                    max_eps_len=args.algo.max_eps_len,
                )
                test_results.append(reached_goal)

                n_success = int(np.asarray(test_results, np.int32).sum())
                success_rate = n_success / len(test_results)

                print(f"[EVAL] step={global_step} ret={test_ret:.1f} len={test_len} "
                      f"succ={n_success}/{len(test_results)}={success_rate:.3f} "
                      f"opt_frac={test_stats['opt_step_frac']:.2f} opt_picks={test_stats['opt_picks']}")

                if args.wb.on:
                    wandb.log({
                        "test/episodic_return": float(test_ret),
                        "test/episodic_length": int(test_len),
                        "test/success": float(reached_goal),
                        "test/success_rate": float(success_rate),
                        "test/opt_step_frac": float(test_stats["opt_step_frac"]),
                        "test/opt_picks": int(test_stats["opt_picks"]),
                    }, step=global_step)

            video_every = max(1, int(0.25 * total_steps))
            if (global_step % video_every == 0) and (global_step >= args.algo.starts):
                record_all_eigenoption_videos(
                    train_state=train_state,
                    Env=Env,
                    global_step=global_step,
                    n_actions=A,
                    n_options=O,
                    use_enc_bn=use_enc_bn,
                    use_q_bn=use_q_bn,
                    opt_dur_table=opt_dur_table,
                    fps=6,
                    use_target=False,
                )

    return train_state, rb_p_state, rb_m_state, key


def evaluate_policy_hdqn(
    key,
    train_state,
    Env,                 # your base env (will deepcopy)
    global_step: int,
    n_actions: int,
    n_options: int,
    use_enc_bn: bool,
    use_q_bn: bool,
    max_eps_len: int,
):
    """
    Greedy evaluation for HDQN with options.

    Meta policy: argmax MetaQ(s) over A+O.
    - If meta < A: execute primitive action for 1 step.
    - Else: execute option 'meta-A' for up to opt_dur steps (or until done),
            selecting primitive actions greedily from that option's Q.

    Returns:
        key, reached_goal(bool), eps_return(float), eps_len(int), stats(dict)
    """
    env = deepcopy(Env)
    obs, _ = env.reset(seed=args.exp.seed)

    eps_return = 0.0
    eps_len = 0

    # stats
    opt_steps = 0
    opt_picks = 0

    # inside evaluate_policy_hdqn, after you have n_options
    if args.algo.eig_dur:
        eigs = np.asarray(train_state.lap_state.eigen_values[0], np.float32)[:n_options]
        opt_dur_table = eigenvalues_to_option_durations(
            eigs, n_options=n_options, bi_dir_opt=args.algo.bidir_opt,
            min_dur=getattr(args.algo, "opt_dur_min", 1),
            max_dur=getattr(args.algo, "opt_dur_max", 100),
            eps=1e-6)
    else:
        opt_dur_table = np.full((n_options,), args.algo.opt_dur, dtype=np.int32)

    while eps_len < max_eps_len:
        # preprocess obs
        obs_g = rgb_to_gray_uint8(obs)
        obs_b = jnp.expand_dims(obs_g, 0)  # [1,H,W,1]

        # greedy meta
        q_meta = get_meta_q(obs_b, train_state.enc_state, train_state.meta_q_state, use_enc_bn, use_q_bn)[0]
        meta = int(jnp.argmax(q_meta))

        if meta < n_actions:
            # primitive
            action = meta
            obs, reward, terminated, _, _ = env.step(action)
            eps_return += float(reward)
            eps_len += 1
            if terminated:
                break
        else:
            # option execution
            sel = meta - n_actions
            opt_picks += 1

            curr_opt_dur = int(opt_dur_table[sel])
            for _ in range(curr_opt_dur):
                obs_g = rgb_to_gray_uint8(obs)
                obs_b = jnp.expand_dims(obs_g, 0)

                q_opt = compute_option_q_one(
                    obs_b,
                    train_state.opt_enc_state,
                    train_state.opt_q_states[sel],
                    use_enc_bn, use_q_bn,
                    use_target=False,   # eval uses online; you can flip to True if you prefer target
                )[0]  # [A]
                action = int(jnp.argmax(q_opt))

                obs, reward, terminated, _, _ = env.step(action)
                eps_return += float(reward)
                eps_len += 1
                opt_steps += 1

                if terminated or (eps_len >= max_eps_len):
                    break

            if terminated or (eps_len >= max_eps_len):
                break

    reached_goal = bool(terminated)
    stats = dict(
        opt_picks=opt_picks,
        opt_steps=opt_steps,
        opt_step_frac=(opt_steps / max(1, eps_len)),
    )
    return key, reached_goal, eps_return, eps_len, stats


def _get_rgb_frame(env, obs):
    frame = obs
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def record_all_eigenoption_videos(
    train_state,
    Env,                     # your training env instance (we deepcopy it)
    global_step: int,
    n_actions: int,
    n_options: int,
    use_enc_bn: bool,
    use_q_bn: bool,
    opt_dur_table: np.ndarray,   # [O] int32 durations (already computed from eigenvalues)
    out_dir: str = "wandb_option_videos",
    fps: int = 6,
    use_target: bool = False,    # usually False for “current policy”
):
    """
    Logs O videos: option_00 ... option_{O-1}, each executed greedily from the same start state.
    """
    if (not args.wb.on) or (n_options <= 0):
        return

    os.makedirs(out_dir, exist_ok=True)

    for o in range(n_options):
        env = deepcopy(Env)
        obs, _ = env.reset(seed=args.exp.seed)

        frames = []
        # max_len = int(opt_dur_table[o])
        max_len = 50

        for t in range(max_len):
            # record current frame BEFORE stepping
            frames.append(_get_rgb_frame(env, obs))

            # choose greedy primitive action under option o
            obs_g = rgb_to_gray_uint8(obs) if (isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[-1] == 3) else obs
            obs_b = jnp.expand_dims(obs_g, 0)

            q_opt = compute_option_q_one(
                obs_b,
                train_state.opt_enc_state,
                train_state.opt_q_states[o],
                use_enc_bn, use_q_bn,
                use_target=use_target,
            )[0]
            a = int(jnp.argmax(q_opt))

            obs, r, terminated, _, _ = env.step(a)
            if bool(terminated):
                # capture terminal frame too (optional)
                frames.append(_get_rgb_frame(env, obs))
                break

        # ---- log to W&B ----
        vid = np.stack(frames, axis=0)  # [T,H,W,3]
        # wandb.Video expects [T,C,H,W]
        wandb.log(
            {
                f"Options_{global_step:09d}/option_{o:02d}":
                    wandb.Video(vid.transpose(0, 3, 1, 2), fps=fps, format="gif")
            },
            step=global_step,
        )

        # Optional: also save gif locally
        gif_path = os.path.join(out_dir, f"step_{global_step:09d}_option_{o:02d}.gif")
        imageio.mimsave(gif_path, frames, fps=fps)

        env.close()

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    EnvWrap = GridAgent(args)
    Env = EnvWrap.env
    n_actions = Env.action_space.n
    n_options = args.algo.n_options * 2 if args.algo.bidir_opt else args.algo.n_options
    assert n_options > 0, "HDQN-only: set args.algo.n_options > 0."

    if args.wb.on:
        create_wb_logger(args)
        wandb.save('src/networks/networks.py')

    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    key = jax.random.PRNGKey(args.exp.seed)

    # Buffers on buffer_device
    with jax.default_device(jax.devices(args.buffer.device)[0]):
        RBp, RBp_state = setup_prim_rb(args)
        RBm, RBm_state = setup_meta_rb(args)

    # Networks on GPU
    with jax.default_device(jax.devices("gpu")[0]):
        try:
            train_state = setup(key, Env, args, n_actions, n_options)
            train_state, RBp_state, RBm_state, key = hdqn_learning(
                train_state, RBp_state, RBm_state, key, Env, RBp, RBm, n_actions, n_options
            )
        except Exception:
            traceback.print_exc()
        finally:
            Env.close()
            jax.clear_backends()
