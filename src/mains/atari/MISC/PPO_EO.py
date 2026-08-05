# ==============================================================================
# PPO-EO.py
# ------------------------------------------------------------------------------
# PPO (Actor-Critic) + EigenOptions (EO)
#
# Design goals :
#   1) PPO architecture is the "main" optimizer for the META-controller:
#        meta-actions = primitive actions + options  (size = A + O)
#      We treat options as SMDP actions with duration k, and do SMDP-GAE.
#
#   2) Laplacian encoder (ALLO) is trained online from *primitive* rollouts.
#      Its eigenfunctions define intrinsic rewards for options:
#        r^o_t = phi_o(s_{t+1}) - phi_o(s_t)   (and bidirectional variant)
#
#   3) Three learning signals each update:
#      (i)  Primitive learning: meta-PPO updates on meta-transitions where duration=1
#           (primitive meta-actions).  (This naturally uses primitive rollouts because
#           those meta steps are generated from primitive segments.)
#
#      (ii) SMDP learning for executed options: meta-PPO updates on meta-transitions
#           where the meta-action is an option and duration=k>1.
#
#      (iii) Intra-option actor-critic learning for NON-executed options using
#           primitive rollouts (off-policy intra-option learning) with
#           importance sampling ratios.
#
#   4) Options use a *shared option encoder* distinct from the main encoder.
#      Options differ ONLY in their actor/value heads.
#
#   5) Reuse network architecture from ppo_networks.py (AtariConvEnc / Actor / Critic)
#      but extend Actor/Critic for the meta and option settings.
#
# Notes on storage:
#   - Primitive storage: length T_prim = args.env.n_steps, per-env per-step.
#   - Meta storage: variable number of meta decisions per rollout, but bounded
#     by T_prim (worst case: all primitives). We implement a fixed-shape
#     MetaStorage with per-env pointers (like a per-rollout "ring", but no wrap).
#
# Why NOT MetaRingBuffer here?
#   - MetaRingBuffer (as in PQN-EO) is mainly useful if you:
#       (a) do off-policy bootstrapped learning across updates, OR
#       (b) need a persistent buffer for meta transitions.
#   - PPO is on-policy, so we only need meta transitions from the current rollout.
#     A per-rollout fixed-shape MetaStorage is enough and simpler.
#
# ==============================================================================

import math
import os
import time
import random
import traceback
from functools import partial
from typing import Any, Optional, Tuple

import envpool
import numpy as np

import jax
import jax.numpy as jnp
import chex
import flax
from flax import struct
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

# ----------------------------------------------------------------------
# Config + local utilities (same pattern as PPO.py)
# ----------------------------------------------------------------------
from src.config.atari.ppo_laplacian import *  # expects args = parse_args() etc.
args = convert_args_to_dict(parse_args())

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from src.utils.jax_utils import *
from src.utils.purejax_atari_wrappers import JaxLogEnvPoolWrapper

# Reuse trunk architecture from your PPO networks file
from src.networks.ppo_networks import AtariConvEnc, Actor, Critic

# Laplacian / ALLO utilities (same ones you use elsewhere)
from src.losses.laplacian import multi_step_ALLO, update_ALLO_parameters, update_eigenvalues


# ------------------------------------------------------------------------------
# Environment helpers (copied from PPO.py)
# ------------------------------------------------------------------------------
def make_env():
    env_kwargs = dict(
        episodic_life=True,
        reward_clip=True,
        repeat_action_probability=0.0,
        frame_skip=4,
        noop_max=30,
        full_action_space=False,
    )
    env = envpool.make(
        args.env.id + "-v5",
        env_type="gym",
        num_envs=args.env.n_total_envs,
        seed=args.exp.seed,
        **env_kwargs,
    )
    env.num_envs = args.env.n_total_envs
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    env.name = args.env.id + "-v5"
    return env


def make_eval_env(n_eval_envs: int):
    env_kwargs = dict(
        episodic_life=True,
        reward_clip=True,
        repeat_action_probability=0.0,
        frame_skip=4,
        noop_max=30,
        full_action_space=False,
    )
    env = envpool.make(
        args.env.id + "-v5",
        env_type="gym",
        num_envs=n_eval_envs,
        seed=args.exp.seed + 999,
        **env_kwargs,
    )
    env.num_envs = n_eval_envs
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    env.name = args.env.id + "-v5"
    return JaxLogEnvPoolWrapper(env)


# ------------------------------------------------------------------------------
# Shape helpers
# ------------------------------------------------------------------------------
@jax.jit
def chw_to_nhwc(obs_chw: jnp.ndarray) -> jnp.ndarray:
    """(N,C,H,W)->(N,H,W,C)"""
    return jnp.transpose(obs_chw, (0, 2, 3, 1))


@jax.jit
def nhwc_to_chw(obs_nhwc: jnp.ndarray) -> jnp.ndarray:
    """(N,H,W,C)->(N,C,H,W)"""
    return jnp.transpose(obs_nhwc, (0, 3, 1, 2))


# ==============================================================================
# RunningMeanStd for per-option intrinsic normalization (from DQN-EO.py)
# ==============================================================================
@chex.dataclass(frozen=True)
class RunningMeanStd:
    mean: chex.Array   # [O]
    var: chex.Array    # [O]
    count: chex.Array  # [O]


@jax.jit
def rms_update_per_option(rms: RunningMeanStd, x_bo: chex.Array) -> RunningMeanStd:
    """
    x_bo: [B,O] intrinsic returns per option.
    Welford merge done elementwise over O.
    """
    x = x_bo.astype(jnp.float32)
    B = x.shape[0]
    batch_count = jnp.full((x.shape[1],), jnp.float32(B))

    batch_mean = jnp.mean(x, axis=0)
    batch_var  = jnp.var(x, axis=0)

    delta = batch_mean - rms.mean
    tot_count = rms.count + batch_count

    new_mean = rms.mean + delta * (batch_count / tot_count)

    m_a = rms.var * rms.count
    m_b = batch_var * batch_count
    M2  = m_a + m_b + (delta * delta) * (rms.count * batch_count / tot_count)
    new_var = M2 / tot_count

    return RunningMeanStd(mean=new_mean, var=new_var, count=tot_count)


@jax.jit
def rms_normalize_per_option(rms: RunningMeanStd, x_bo: chex.Array, eps: float = 1e-8) -> chex.Array:
    """Variance normalization (no mean subtraction)"""
    return x_bo / jnp.sqrt(rms.var[None, :] + eps)

# ==============================================================================
# Networks: MetaActor/MetaCritic and OptionActor/OptionCritic
# ------------------------------------------------------------------------------
# We reuse the same simple Dense heads used in ppo_networks.py:
#   - Actor: Dense(action_dim), orthogonal(0.01)
#   - Critic: Dense(1), orthogonal(1.0)
#
# For options:
#   - shared OPTION encoder f_opt(s) (separate params from main enc)
#   - per-option actor heads: logits_o = Actor(A)  (params axis=O)
#   - per-option value heads: v_o = Critic()       (params axis=O)
#
# We implement option heads via nn.vmap to keep a single TrainState and avoid
# Python lists of TrainStates (compilation + overhead).
# ==============================================================================


class OptionActorHead(nn.Module):
    """Per-option policy head over primitive actions."""
    action_dim: int  # A

    @nn.compact
    def __call__(self, x, train: bool = False):
        # reuse the same init scale as PPO actor head (0.01)
        return nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01),
                        bias_init=nn.initializers.constant(0.0))(x)


class OptionCriticHead(nn.Module):
    """Per-option value head (scalar)."""
    @nn.compact
    def __call__(self, x, train: bool = False):
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0),
                        bias_init=nn.initializers.constant(0.0))(x)


def build_vmap_option_heads(n_options: int, n_actions: int):
    """
    Returns:
      VmapOptActor: Module mapping (B,D) -> (O,B,A)
      VmapOptCritic: Module mapping (B,D) -> (O,B)
    """
    VmapOptActor = nn.vmap(
        OptionActorHead,
        variable_axes={"params": 0},     # params have leading axis O
        split_rngs={"params": True},
        in_axes=None,                   # input shared for all options
        out_axes=0,                     # output has leading axis O
        axis_size=n_options,
    )
    VmapOptCritic = nn.vmap(
        OptionCriticHead,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=0,
        axis_size=n_options,
    )
    return VmapOptActor(action_dim=n_actions), VmapOptCritic()


# ==============================================================================
# Laplacian state (ALLO) - modeled after your DQN-EO and PQN-EO code
# ==============================================================================
class LapState(TrainState):
    betas: Any
    errors: Any
    quadratic_errors: Any
    barrier_coeffs: Any
    eigen_values: Any  # [1,D] EMA
    batch_stats: Optional[flax.core.FrozenDict] = None


# ==============================================================================
# Agent state
# ==============================================================================
@struct.dataclass
class AgentState:
    # Main encoder + meta heads (PPO)
    enc: TrainState
    meta_actor: TrainState
    meta_critic: TrainState

    # Option encoder (shared across options, separate from enc)
    opt_enc: TrainState

    # Option heads (params have leading axis O)
    opt_actor: TrainState
    opt_critic: TrainState

    # Laplacian encoder state (ALLO)
    lap: LapState

    # Intrinsic normalization
    int_rms: RunningMeanStd

    # Counters
    timesteps: jnp.int32 = jnp.int32(0)
    n_updates: jnp.int32 = jnp.int32(0)
    grad_steps: jnp.int32 = jnp.int32(0)
    next_eval_step: jnp.int32 = jnp.int32(0)


# ==============================================================================
# Storage: primitive and meta
# ------------------------------------------------------------------------------
# PrimitiveStorage: per-step data for Laplacian + option learning
# MetaStorage: per meta-decision (SMDP step) data for PPO update
# ==============================================================================
@struct.dataclass
class PrimitiveStorage:
    # per-step (T,N,...) arrays
    obs: jnp.ndarray        # uint8 (T,N,C,H,W)
    actions: jnp.ndarray    # int32 (T,N) primitive action executed
    rewards: jnp.ndarray    # float32 (T,N) extrinsic env reward
    dones: jnp.ndarray      # bool (T,N) done after step

    # meta action executing at this primitive step
    meta_action: jnp.ndarray  # int32 (T,N) in [0..A+O-1]

    # behavior probability of primitive action under the behavior low-level policy μ(a|s)
    # - If executing primitive meta-action: μ=1 for the chosen a (=meta)
    # - If executing option: μ = π_o(a|s,o) (we sample from option actor so this is well-defined)
    mu_prim: jnp.ndarray      # float32 (T,N)


@struct.dataclass
class MetaStorage:
    """
    Fixed-shape per-rollout buffer for meta decisions.

    We store at most T_prim meta steps per env (worst-case all primitives).
    ptr[e] counts how many meta entries were written for env e.
    """
    ptr: jnp.ndarray          # (N,) int32 in [0..T_prim]
    start_obs: jnp.ndarray    # (T_prim,N,C,H,W) uint8
    actions: jnp.ndarray      # (T_prim,N) int32 meta action u
    logprobs: jnp.ndarray     # (T_prim,N) float32 log π(u|s_start) under behavior (on-policy PPO)
    values: jnp.ndarray       # (T_prim,N) float32 V_meta(s_start)
    rewards: jnp.ndarray      # (T_prim,N) float32 SMDP return (discounted sum of extrinsic rewards)
    dones: jnp.ndarray        # (T_prim,N) bool done at segment end
    durations: jnp.ndarray    # (T_prim,N) int32 duration k
    end_obs: jnp.ndarray      # (T_prim,N,C,H,W) uint8 terminal obs of segment


def make_empty_meta_storage(T: int, N: int, obs_shape_chw: Tuple[int,int,int]) -> MetaStorage:
    C,H,W = obs_shape_chw
    return MetaStorage(
        ptr=jnp.zeros((N,), dtype=jnp.int32),
        start_obs=jnp.zeros((T, N, C, H, W), dtype=jnp.uint8),
        actions=jnp.zeros((T, N), dtype=jnp.int32),
        logprobs=jnp.zeros((T, N), dtype=jnp.float32),
        values=jnp.zeros((T, N), dtype=jnp.float32),
        rewards=jnp.zeros((T, N), dtype=jnp.float32),
        dones=jnp.ones((T, N), dtype=jnp.bool_),  # default done=True (masked later)
        durations=jnp.ones((T, N), dtype=jnp.int32),
        end_obs=jnp.zeros((T, N, C, H, W), dtype=jnp.uint8),
    )


# ==============================================================================
# Schedules (reuse PPO.py pattern)
# ==============================================================================
def linear_schedule(count):
    frac = 1.0 - (count // (args.algo.n_mini_batches * args.algo.upd_epochs)) / args.algo.num_iterations
    return args.net.lr * frac


# ==============================================================================
# Setup: initialize all networks and TrainStates
# ==============================================================================
def setup(key, args, obs_chw0, n_actions: int):
    """
    Initialize:
      - main encoder + meta actor/critic
      - option encoder + vmap option actor/critic
      - laplacian encoder state
      - per-option RMS
    """
    # -------------------------
    # option counts
    # -------------------------
    n_base_opt = args.algo.n_options
    bi_dir_opt = args.algo.bidir_options
    n_options = (2 * n_base_opt) if bi_dir_opt else n_base_opt
    assert n_options > 0, "PPO-EO expects args.algo.n_options > 0"

    A = n_actions
    M = A + n_options

    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    # -------------------------
    # Main PPO encoder + meta heads
    # -------------------------
    main_enc = AtariConvEnc(
        use_batch_norm=False, use_layer_norm=False,
        mlp_dims=args.net.c_dims, mlp_activations=[nn.relu],
    )
    meta_actor = Actor(action_dim=M)   # logits over A+O
    meta_critic = Critic()             # V(s)

    # -------------------------
    # Option encoder (separate params) + option heads (vmap)
    # -------------------------
    opt_enc = AtariConvEnc(
        use_batch_norm=False, use_layer_norm=False,
        mlp_dims=args.net.c_dims, mlp_activations=[nn.relu],
    )
    opt_actor_mod, opt_critic_mod = build_vmap_option_heads(n_options=n_options, n_actions=A)

    # -------------------------
    # Laplacian encoder
    # Use the same AtariConvEnc trunk but last dim = args.laplacian.dims and last activation None.
    # We keep BN off here because your PPO networks file doesn't implement BN in ConvNet by default
    # (but you can flip if your implementation supports it).
    # -------------------------
    lap_dims = args.laplacian.dims
    lap_enc = AtariConvEnc(
        use_batch_norm=False, use_layer_norm=False,
        mlp_dims=list(args.net.c_dims) + [lap_dims],
        mlp_activations=[nn.relu] * len(args.net.c_dims) + [None],
    )

    # -------------------------
    # Dummy inputs
    # PPO stores CHW; encoder expects NHWC
    # -------------------------
    obs0 = chw_to_nhwc(obs_chw0[:1])  # (1,H,W,C)
    dummy_emb = jnp.ones((1, args.net.c_dims[-1]), dtype=jnp.float32)

    # -------------------------
    # Init params
    # -------------------------
    main_enc_vars = main_enc.init(k1, obs0, train=False)
    meta_actor_vars = meta_actor.init(k2, dummy_emb, train=False)
    meta_critic_vars = meta_critic.init(k3, dummy_emb, train=False)

    opt_enc_vars = opt_enc.init(k4, obs0, train=False)

    # For option heads we need an embedding; input is shared -> [B,D]
    opt_actor_vars = opt_actor_mod.init(k5, dummy_emb, train=False)
    opt_critic_vars = opt_critic_mod.init(k6, dummy_emb, train=False)

    lap_vars = lap_enc.init(k1, obs0, train=False)

    # -------------------------
    # Optimizers
    # -------------------------
    def _adam(lr, g_norm):
        return optax.chain(
            optax.clip_by_global_norm(g_norm),
            optax.inject_hyperparams(optax.adam)(
                eps=1e-5,
                learning_rate=linear_schedule if args.algo.anneal_lr else lr,
            ),
        )

    enc_tx = _adam(args.net.lr, args.net.max_grad_norm)
    meta_tx = _adam(args.net.lr, args.net.max_grad_norm)
    opt_enc_tx = _adam(args.net.lr, args.net.max_grad_norm)
    opt_head_tx = _adam(args.net.lr, args.net.max_grad_norm)
    lap_tx = _adam(args.laplacian.lr, args.laplacian.max_grad_norm)

    # -------------------------
    # TrainStates
    # -------------------------
    main_enc_state = TrainState.create(apply_fn=jax.jit(main_enc.apply), params=main_enc_vars["params"], tx=enc_tx)
    meta_actor_state = TrainState.create(apply_fn=jax.jit(meta_actor.apply), params=meta_actor_vars["params"], tx=meta_tx)
    meta_critic_state = TrainState.create(apply_fn=jax.jit(meta_critic.apply), params=meta_critic_vars["params"], tx=meta_tx)

    opt_enc_state = TrainState.create(apply_fn=jax.jit(opt_enc.apply), params=opt_enc_vars["params"], tx=opt_enc_tx)
    opt_actor_state = TrainState.create(apply_fn=jax.jit(opt_actor_mod.apply), params=opt_actor_vars["params"], tx=opt_head_tx)
    opt_critic_state = TrainState.create(apply_fn=jax.jit(opt_critic_mod.apply), params=opt_critic_vars["params"], tx=opt_head_tx)

    # Lap state (ALLO bookkeeping mirrors DQN-EO)
    lap_state = LapState.create(
        apply_fn=jax.jit(lap_enc.apply),
        params=lap_vars["params"],
        tx=lap_tx,
        barrier_coeffs=jnp.tril(jnp.ones((1, 1))) * args.laplacian.barrier.value,
        betas=jnp.zeros((lap_dims, lap_dims)),
        eigen_values=jnp.zeros((1, lap_dims), dtype=jnp.float32),
        errors=jnp.zeros((lap_dims, lap_dims)),
        quadratic_errors=jnp.zeros((lap_dims, lap_dims)),
        batch_stats=None,
    )

    # intrinsic RMS
    init_rms = RunningMeanStd(
        mean=jnp.zeros((n_options,), dtype=jnp.float32),
        var=jnp.ones((n_options,), dtype=jnp.float32),
        count=jnp.full((n_options,), 1e-4, dtype=jnp.float32),
    )

    eval_interval = args.algo.tot_steps // 25

    return AgentState(
        enc=main_enc_state,
        meta_actor=meta_actor_state,
        meta_critic=meta_critic_state,
        opt_enc=opt_enc_state,
        opt_actor=opt_actor_state,
        opt_critic=opt_critic_state,
        lap=lap_state,
        int_rms=init_rms,
        next_eval_step=jnp.int32(eval_interval),
    )


# ==============================================================================
# Action sampling helpers
# ==============================================================================
@jax.jit
def categorical_sample(logits: jnp.ndarray, key: chex.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample categorical actions and return (actions, logp_of_sample).
    logits: [B, K]
    actions: [B]
    logp: [B]
    """
    key, sub = jax.random.split(key)
    u = jax.random.uniform(sub, shape=logits.shape)
    a = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=-1)
    logp = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), a]
    return a, logp


@jax.jit
def greedy(logits: jnp.ndarray) -> jnp.ndarray:
    return jnp.argmax(logits, axis=-1)


# ==============================================================================
# Forward passes: main/meta and option networks
# ==============================================================================
@jax.jit
def encode_main(agent: AgentState, obs_chw: jnp.ndarray, train: bool) -> jnp.ndarray:
    obs = chw_to_nhwc(obs_chw)
    return agent.enc.apply_fn({"params": agent.enc.params}, obs, train=train)


@jax.jit
def encode_opt(agent: AgentState, obs_chw: jnp.ndarray, train: bool) -> jnp.ndarray:
    obs = chw_to_nhwc(obs_chw)
    return agent.opt_enc.apply_fn({"params": agent.opt_enc.params}, obs, train=train)


@jax.jit
def meta_logits_and_value(agent: AgentState, obs_chw: jnp.ndarray, train: bool):
    h = encode_main(agent, obs_chw, train=train)
    logits = agent.meta_actor.apply_fn({"params": agent.meta_actor.params}, h, train=train)
    v = agent.meta_critic.apply_fn({"params": agent.meta_critic.params}, h, train=train).squeeze(-1)
    return logits, v


@jax.jit
def option_logits_and_values(agent: AgentState, obs_chw: jnp.ndarray, train: bool):
    """
    Returns:
      opt_logits: [O,B,A]
      opt_values: [O,B]
    """
    h = encode_opt(agent, obs_chw, train=train)  # [B,D]
    opt_logits = agent.opt_actor.apply_fn({"params": agent.opt_actor.params}, h, train=train)  # [O,B,A]
    opt_values = agent.opt_critic.apply_fn({"params": agent.opt_critic.params}, h, train=train).squeeze(-1)  # [O,B]
    return opt_logits, opt_values


# ==============================================================================
# Option duration from eigenvalues
# ==============================================================================
@partial(jax.jit, static_argnums=range(1, 6))
def eigenvalues_to_option_durations(
    eigs: jnp.ndarray,          # [D] or [1,D]
    n_options: int,
    bi_dir_opt: bool,
    min_dur: int,
    max_dur: int,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    JIT-safe duration mapping:
      dur = ceil(1 / max(eig, eps)), clipped to [min_dur, max_dur]
    If bi_dir_opt: compute for first O/2 and duplicate.
    """
    eigs = jnp.reshape(eigs, (-1,))
    eigs = jnp.nan_to_num(eigs, nan=0.0, posinf=0.0, neginf=0.0)
    eigs = jnp.clip(eigs, 0.0, jnp.inf)

    if bi_dir_opt:
        n_lap = n_options // 2
        lam = jnp.maximum(eigs[:n_lap], eps)
        dur = jnp.ceil(1.0 / lam).astype(jnp.int32)
        dur = jnp.clip(dur, min_dur, max_dur)
        dur = jnp.concatenate([dur, dur], axis=0)
    else:
        lam = jnp.maximum(eigs[:n_options], eps)
        dur = jnp.ceil(1.0 / lam).astype(jnp.int32)
        dur = jnp.clip(dur, min_dur, max_dur)

    return dur


# ==============================================================================
# Laplacian helpers: eigenvalue estimate (same formula as DQN-EO)
# ==============================================================================
@jax.jit
def compute_batch_eigenvalues_seq(phis, valid_transitions, eps=1e-8):
    """
    phis: [B, T, D]
    valid_transitions: [B, T-1] float32
    """
    phi  = phis[:, :-1, :]
    phi1 = phis[:, 1:, :]

    w = valid_transitions[..., None]
    wsum = jnp.sum(w, axis=(0, 1)) + eps

    num   = jnp.sum(w * (phi - phi1) ** 2, axis=(0, 1)) / wsum
    denom = jnp.sum(w * 0.5 * (phi ** 2 + phi1 ** 2), axis=(0, 1)) / wsum
    return num / (denom + eps)


# ==============================================================================
# Laplacian update from primitive rollout slices
# ==============================================================================
@partial(jax.jit, static_argnums=(4,))
def update_laplacian_from_prim(key, agent: AgentState, lap_obs_seq_chw, lap_valid_mask, n_steps: int):
    """
    lap_obs_seq_chw: [B, N, C, H, W] where N = n_steps (>=2)
    lap_valid_mask:  [B, N] bool (valid states)
    We compute ALLO loss on embeddings [B,N,D] and update lap params.
    """
    # gamma = args.env.gamma
    gamma = 0.99
    use_bar = args.laplacian.barrier.on
    use_dual= args.laplacian.dual.on
    average_loss = args.laplacian.average_loss

    lap_state = agent.lap
    B, N = lap_obs_seq_chw.shape[:2]

    # Flatten for a single encoder call
    flat = chw_to_nhwc(lap_obs_seq_chw.reshape((B * N,) + lap_obs_seq_chw.shape[2:]))

    # transition validity for eigenvalues: [B, N-1]
    valid_trans = (lap_valid_mask[:, :-1] & lap_valid_mask[:, 1:]).astype(jnp.float32)

    def loss_fn(lap_params):
        # Forward (no BN here, matches this script's lap_enc choice)
        phis_flat = lap_state.apply_fn({"params": lap_params}, flat, train=True)  # [(B*N),D]
        D = phis_flat.shape[-1]
        phis = phis_flat.reshape(B, N, D)

        # ALLO expects first state embedding + future embeddings
        mask = lap_valid_mask[:, 0]
        future_masks = lap_valid_mask[:, 1:]

        ph0 = phis[:, 0, :]
        fut = phis[:, 1:, :]

        loss, (pos_loss, ortho_loss, dual_loss), (_avg_w, err_mat) = multi_step_ALLO(
            key, ph0, fut, None, mask, future_masks,
            lap_state.barrier_coeffs, lap_state.betas,
            gamma=gamma, state_action=False, n_steps=N,
            use_dual=use_dual, use_weights=False, weights=None,
            average_loss=average_loss,
        )

        lambda_batch = compute_batch_eigenvalues_seq(phis, valid_trans)  # [D]
        lambda_batch = jax.lax.stop_gradient(lambda_batch)

        return loss, (pos_loss, ortho_loss, dual_loss, err_mat, lambda_batch)

    (eig_loss, (pos_loss, ortho_loss, dual_loss, err_mat, lambda_batch)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(lap_state.params)

    new_lap = lap_state.apply_gradients(grads=grads)

    if use_bar or use_dual:
        new_lap = update_ALLO_parameters(
            new_lap, err_mat,
            update_rates=args.laplacian.rate,
            barrier_args=args.laplacian.barrier,
            dual_args=args.laplacian.dual,
            average_loss=average_loss,
        )

    # EMA eigenvalues
    alpha_ev = 0.1
    ev_old = new_lap.eigen_values.reshape(-1)
    ev_new = update_eigenvalues(ev_old, lambda_batch, alpha=alpha_ev)
    new_lap = new_lap.replace(eigen_values=ev_new[None, :])

    agent = agent.replace(lap=new_lap)
    return agent, (eig_loss, pos_loss, ortho_loss, dual_loss, lambda_batch)


# ==============================================================================
# Intrinsic reward computation from Laplacian embeddings
# ------------------------------------------------------------------------------
# Given a primitive rollout (T,N,...) we compute:
#   - phi(s_t) for all t
#   - r_t^o = phi_o(s_{t+1}) - phi_o(s_t)  (or bidir)
# and build n-step returns for option learning.
# ==============================================================================
@partial(jax.jit, static_argnums=range(2, 6))
def compute_intrinsic_rewards_TNO(
    agent: AgentState,
    prim: PrimitiveStorage,
    n_options: int,
    bi_dir_opt: bool,
    eps: float = 1e-8,
):
    """
    Returns:
      r_int:   [T,N,O] intrinsic reward (variance-normalized per option)
      done:    [T,N]   done float32
      new_rms:  RunningMeanStd
    """
    T, N = prim.obs.shape[:2]

    obs_seq = prim.obs                                           # [T,N,C,H,W]
    obs_tp1 = jnp.concatenate([obs_seq[1:], obs_seq[-1:]], axis=0)

    flat0 = chw_to_nhwc(obs_seq.reshape((T * N,) + obs_seq.shape[2:]))
    flat1 = chw_to_nhwc(obs_tp1.reshape((T * N,) + obs_tp1.shape[2:]))

    phi0 = agent.lap.apply_fn({"params": agent.lap.params}, flat0, train=False)  # [T*N,D]
    phi1 = agent.lap.apply_fn({"params": agent.lap.params}, flat1, train=False)

    n_lap = (n_options // 2) if bi_dir_opt else n_options
    phi0 = phi0[:, :n_lap]
    phi1 = phi1[:, :n_lap]

    diffs = phi1 - phi0  # [T*N, n_lap]
    if bi_dir_opt:
        r_flat = jnp.concatenate([diffs, -diffs], axis=-1)        # [T*N,O]
    else:
        r_flat = diffs                                            # [T*N,O]

    # RMS update + variance normalize
    new_rms = rms_update_per_option(agent.int_rms, r_flat)
    r_flat = rms_normalize_per_option(new_rms, r_flat, eps=eps)

    r_int = r_flat.reshape((T, N, n_options))                      # [T,N,O]
    done  = prim.dones.astype(jnp.float32)                         # [T,N]
    return r_int, done, new_rms


# ==============================================================================
# Option intrinsic targets + minibatch update
# ------------------------------------------------------------------------------
# Goal:
#   - Compute intrinsic OFF-POLICY V-trace-style targets/advantages on the *ordered*
#     primitive rollout (so GAE/trace recursion is well-defined).
#   - Then train option policies/values in shuffled minibatches, PPO-style, using
#     stored old logπ (pre-update) and the precomputed targets.
#
# Shapes convention used here (minibatch-friendly):
#   - Flatten time/env: B = T * N
#   - Store per-option quantities as [B, O]
# ==============================================================================
@jax.jit
def _opt_vtrace_scan(carry, inp, gamma: float):
    """
    carry:
      vs_next: [O, N]   (V-trace value target at next time step)
    inp (per time step t):
      r_t    : [O, N]
      done_t : [N]
      v_t    : [O, N]
      v_tp1  : [O, N]
      rho_t  : [O, N]   (rho_bar)
      c_t    : [O, N]
    returns:
      vs_t   : [O, N]
      (vs_t, adv_t) where adv_t is policy advantage for actor
    """
    vs_next = carry
    r_t, done_t, v_t, v_tp1, rho_t, c_t = inp

    nonterm = 1.0 - done_t[None, :]  # [1,N] broadcast to [O,N]

    # corrected one-step TD
    delta = rho_t * (r_t + gamma * nonterm * v_tp1 - v_t)  # [O,N]

    # V-trace value target
    vs_t = v_t + delta + gamma * nonterm * c_t * (vs_next - v_tp1)

    # V-trace policy advantage (uses vs_{t+1})
    adv_t = rho_t * (r_t + gamma * nonterm * vs_next - v_t)

    return vs_t, (vs_t, adv_t)


@partial(jax.jit, static_argnums=(4, 5))
def compute_option_intrinsic_targets(
    agent: AgentState,
    prim: PrimitiveStorage,
    r_int_TNO: jnp.ndarray,   # [T,N,O]
    done_TN: jnp.ndarray,     # [T,N]
    n_actions: int,           # A
    n_options: int,           # O
    gamma: float,
    gae_lambda: float,        # unused here (V-trace has c_t), keep for API
    rho_clip: float = 1.0,
    c_clip: float = 1.0,
    eps: float = 1e-12,
):
    T, N = prim.actions.shape
    O = n_options
    B = T * N

    obs_t = prim.obs                                                 # [T,N,C,H,W]
    obs_tp1 = jnp.concatenate([prim.obs[1:], prim.obs[-1:]], axis=0)  # [T,N,...]
    a_TN = prim.actions.astype(jnp.int32)                             # [T,N]
    mu_TN = jnp.clip(prim.mu_prim.astype(jnp.float32), eps, 1.0)      # [T,N]
    done_TN = done_TN.astype(jnp.float32)

    # r_int: want [T,O,N] for scan
    r_time = jnp.transpose(r_int_TNO.astype(jnp.float32), (0, 2, 1))   # [T,O,N]

    def per_t_step(_carry, inp):
        obsN, obs1N, aN = inp                                          # obsN:[N,...], aN:[N]

        # shared option encoder on s_t and s_{t+1}, each batch N
        h0 = agent.opt_enc.apply_fn({"params": agent.opt_enc.params}, chw_to_nhwc(obsN), train=False)   # [N,D]
        h1 = agent.opt_enc.apply_fn({"params": agent.opt_enc.params}, chw_to_nhwc(obs1N), train=False)  # [N,D]

        # option heads: logits [O,N,A], values [O,N]
        logits0 = agent.opt_actor.apply_fn({"params": agent.opt_actor.params}, h0, train=False)         # [O,N,A]
        v0 = agent.opt_critic.apply_fn({"params": agent.opt_critic.params}, h0, train=False).squeeze(-1)  # [O,N]
        v1 = agent.opt_critic.apply_fn({"params": agent.opt_critic.params}, h1, train=False).squeeze(-1)  # [O,N]

        logp_all = jax.nn.log_softmax(logits0, axis=-1)                # [O,N,A]

        # log π_o(a_t|s_t) for executed primitive action aN, for all options
        logpi_ON = jnp.take_along_axis(logp_all, aN[None, :, None], axis=-1).squeeze(-1)  # [O,N]
        return _carry, (v0, v1, logpi_ON)

    _, (v_time, v1_time, logpi_time) = jax.lax.scan(
        per_t_step,
        None,
        (obs_t, obs_tp1, a_TN),
        length=T,
    )
    # v_time: [T,O,N], v1_time: [T,O,N], logpi_time: [T,O,N]

    pi_time = jnp.exp(logpi_time)                                     # [T,O,N]
    rho_time = pi_time / mu_TN[:, None, :]                            # [T,O,N]
    rho_bar_time = jnp.clip(rho_time, 0.0, rho_clip)                  # [T,O,N]
    c_time = jnp.clip(rho_time, 0.0, c_clip)                          # [T,O,N]

    # V-trace reverse scan expects carry vs_next [O,N]
    vs_last_ON = v1_time[-1]                                          # [O,N]

    scan_inp = (r_time, done_TN, v_time, v1_time, rho_bar_time, c_time)

    def scan_step(carry, inp):
        r_t_ON, done_t_N, v_t_ON, v_tp1_ON, rho_t_ON, c_t_ON = inp
        return _opt_vtrace_scan(carry, (r_t_ON, done_t_N, v_t_ON, v_tp1_ON, rho_t_ON, c_t_ON), gamma=gamma)

    _, (vs_time, adv_time) = jax.lax.scan(scan_step, vs_last_ON, scan_inp, reverse=True)
    # vs_time, adv_time are [T,O,N]

    # flatten to [B,O]
    vs_BO = jnp.reshape(jnp.transpose(vs_time, (0, 2, 1)), (B, O))       # [T,N,O] -> [B,O]
    adv_BO = jnp.reshape(jnp.transpose(adv_time, (0, 2, 1)), (B, O))     # [B,O]
    logpi_old_BO = jnp.reshape(jnp.transpose(logpi_time, (0, 2, 1)), (B, O))

    return {
        "obs": obs_t.reshape((B,) + obs_t.shape[2:]),                    # [B,C,H,W]
        "act": a_TN.reshape((B,)),                                       # [B]
        "logpi_old": logpi_old_BO,                                      # [B,O]
        "vs_target": vs_BO,                                             # [B,O]
        "adv_target": adv_BO,                                           # [B,O]
    }, (jnp.mean(rho_bar_time), jnp.mean(c_time))


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def update_option_intrinsic_minibatch(
    agent: AgentState,
    mb: dict,            # minibatch from make_minibatches
    n_actions: int,      # A
    n_options: int,      # O
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    norm_adv: bool,
):
    """
    One SGD step on a minibatch of option intrinsic targets.

    mb fields (expected):
      obs        : [B,C,H,W]
      act        : [B]
      logpi_old  : [B,O]
      vs_target  : [B,O]
      adv_target : [B,O]

    Update params:
      - opt_enc (shared)
      - opt_actor (vmap heads, params axis O)
      - opt_critic (vmap heads, params axis O)

    PPO-style clipped objective, averaged over (batch, options):
      ratio_o = exp(logpi_new - logpi_old)
      L_pi = E_{b,o}[ max( -A * ratio,  -A * clip(ratio) ) ]
      L_v  = 0.5 * E_{b,o}[ (V - V_target)^2 ]
      L_ent = - E_{b,o}[ entropy(pi_o(.|s_b)) ]
    """
    B = mb["act"].shape[0]
    A = n_actions
    O = n_options

    adv_BO = mb["adv_target"].astype(jnp.float32)  # [B,O]
    if norm_adv:
        # normalize globally across both batch and options
        m = jnp.mean(adv_BO)
        v = jnp.mean(jnp.square(adv_BO - m))
        adv_BO = (adv_BO - m) / jnp.sqrt(v + 1e-8)

    vs_targ_BO = mb["vs_target"].astype(jnp.float32)      # [B,O]
    logpi_old_BO = mb["logpi_old"].astype(jnp.float32)    # [B,O]
    a_B = mb["act"].astype(jnp.int32)                     # [B]

    def loss_fn(opt_enc_p, opt_act_p, opt_crt_p):
        # forward
        h = agent.opt_enc.apply_fn({"params": opt_enc_p}, chw_to_nhwc(mb["obs"]), train=True)  # [B,D]
        logits = agent.opt_actor.apply_fn({"params": opt_act_p}, h, train=True)                # [O,B,A]
        v_pred = agent.opt_critic.apply_fn({"params": opt_crt_p}, h, train=True).squeeze(-1)   # [O,B]

        logp_all = jax.nn.log_softmax(logits, axis=-1)  # [O,B,A]
        logpi_OB = jnp.take_along_axis(logp_all, a_B[None, :, None], axis=-1).squeeze(-1)  # [O,B]
        logpi_BO = jnp.transpose(logpi_OB, (1, 0))  # [B,O]

        # ratios
        logratio = logpi_BO - logpi_old_BO
        ratio = jnp.exp(logratio)  # [B,O]

        # policy loss (PPO clip) averaged over B,O
        pg1 = -adv_BO * ratio
        pg2 = -adv_BO * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        pg_loss = jnp.mean(jnp.maximum(pg1, pg2))

        # critic loss: V(s) vs V-trace target
        v_BO = jnp.transpose(v_pred, (1, 0))  # [B,O]
        v_loss = 0.5 * jnp.mean(jnp.square(v_BO - jax.lax.stop_gradient(vs_targ_BO)))

        # entropy over actions, average over O,B
        p = jax.nn.softmax(logits, axis=-1)               # [O,B,A]
        ent_OB = -(p * logp_all).sum(axis=-1)             # [O,B]
        ent = jnp.mean(ent_OB)

        total = pg_loss + vf_coef * v_loss - ent_coef * ent

        aux = (
            pg_loss,
            v_loss,
            ent,
            jnp.mean(ratio),   # average ratio across B,O
        )
        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1, 2))(
        agent.opt_enc.params, agent.opt_actor.params, agent.opt_critic.params
    )
    g_enc, g_act, g_crt = grads

    agent = agent.replace(
        opt_enc=agent.opt_enc.apply_gradients(grads=g_enc),
        opt_actor=agent.opt_actor.apply_gradients(grads=g_act),
        opt_critic=agent.opt_critic.apply_gradients(grads=g_crt),
        grad_steps=agent.grad_steps + 1,
    )
    return agent, (loss, *aux)


# ==============================================================================
# Meta (PPO) advantage computation for SMDP transitions
# ------------------------------------------------------------------------------
# SMDP-GAE:
#   delta_t = R_t + (1-done) * gamma^k * V(s_{t+1}) - V(s_t)
#   A_t = delta_t + (1-done) * gamma^k * lambda * A_{t+1}
# ==============================================================================
@jax.jit
def smdp_gae_scan(carry, inp, gae_lambda: float):
    adv_next = carry
    done, gamma_k, v_end, v_start, R = inp  # v_end is V(s'_i)
    nonterm = 1.0 - done.astype(jnp.float32)
    delta = R + nonterm * gamma_k * v_end - v_start
    adv = delta + nonterm * gamma_k * gae_lambda * adv_next
    return adv, adv


@partial(jax.jit, static_argnums=(2, 3, 4))
def compute_smdp_gae(agent: AgentState, meta: MetaStorage, gamma: float, gae_lambda: float, T_max: int):
    """
    Correct SMDP-GAE:
      delta_i = R_i + (1-d_i) * gamma^{k_i} * V(end_obs_i) - V(start_obs_i)
      A_i = delta_i + (1-d_i) * gamma^{k_i} * lambda * A_{i+1}
    """

    T, N = meta.actions.shape

    # V(end_obs_i)
    end_obs_flat = meta.end_obs.reshape((T * N,) + meta.end_obs.shape[2:])
    _, v_end_flat = meta_logits_and_value(agent, end_obs_flat, train=False)
    v_end = v_end_flat.reshape(T, N)

    done = meta.dones.astype(jnp.float32)                 # [T,N]
    k = meta.durations.astype(jnp.float32)                # [T,N]
    gamma_k = (jnp.float32(gamma) ** k)                   # [T,N]

    v_start = meta.values                                 # [T,N] stored at segment start (on-policy)
    R = meta.rewards                                      # [T,N]

    adv0 = jnp.zeros((N,), dtype=jnp.float32)

    scan_inp = (done, gamma_k, v_end, v_start, R)
    _, adv = jax.lax.scan(
        partial(smdp_gae_scan, gae_lambda=gae_lambda),
        adv0,
        scan_inp,
        reverse=True,
    )

    rets = adv + v_start
    return adv, rets


# =======
# Meta-Primitive Updates from primitive transitions
# =======
@jax.jit
def gae_scan(carry, inp, gae_lambda: float):
    adv_next = carry
    done, v_next, v, r = inp
    nonterm = 1.0 - done.astype(jnp.float32)
    delta = r + nonterm * jnp.float32(args.env.gamma) * v_next - v
    adv = delta + nonterm * jnp.float32(args.env.gamma) * gae_lambda * adv_next
    return adv, adv


@jax.jit
def compute_primitive_gae(agent: AgentState, prim: PrimitiveStorage, last_obs_chw: jnp.ndarray):
    """
    Returns (adv, ret, logp_old, v_old) all shaped [T,N].
    adv/ret are valid for *all* primitive steps; you will mask to meta-primitives later.
    """
    T, N = prim.rewards.shape

    # V(s_t) for all primitive steps
    obs_flat = prim.obs.reshape((T * N,) + prim.obs.shape[2:])
    _logits, v_flat = meta_logits_and_value(agent, obs_flat, train=False)   # v_flat: [T*N]
    v = v_flat.reshape(T, N)

    # V(s_{t+1}) using shifted obs and final bootstrap from last_obs_chw
    obs_tp1 = jnp.concatenate([prim.obs[1:], last_obs_chw[None, ...]], axis=0)  # [T,N,...]
    obs_tp1_flat = obs_tp1.reshape((T * N,) + obs_tp1.shape[2:])
    _logits2, v_tp1_flat = meta_logits_and_value(agent, obs_tp1_flat, train=False)
    v_next = v_tp1_flat.reshape(T, N)

    done = prim.dones.astype(jnp.float32)
    r = prim.rewards.astype(jnp.float32)

    adv0 = jnp.zeros((N,), dtype=jnp.float32)
    scan_inp = (done, v_next, v, r)
    _, adv = jax.lax.scan(partial(gae_scan, gae_lambda=args.algo.gae_lambda), adv0, scan_inp, reverse=True)
    ret = adv + v

    # old logprob for PPO: recompute under current policy at data-collection time
    # (must be computed BEFORE you update meta params in this update_step)
    logits_flat, _ = meta_logits_and_value(agent, obs_flat, train=False)  # [T*N, M]
    logp_all = jax.nn.log_softmax(logits_flat, axis=-1)
    a_flat = prim.meta_action.reshape((T * N,))
    logp_old = logp_all[jnp.arange(T * N), a_flat].reshape(T, N)

    return adv, ret, logp_old, v


@jax.jit
def _vtrace_scan(carry, inp, gamma: float):
    """
    carry: vs_next [N]
    inp per time t:
      r_t     [N]
      done_t  [N]
      v_t     [N]
      v_tp1   [N]
      rho_t   [N]  (rho_bar)
      c_t     [N]
    """
    vs_next = carry
    r_t, done_t, v_t, v_tp1, rho_t, c_t = inp
    nonterm = 1.0 - done_t

    # TD error corrected by rho
    delta = rho_t * (r_t + gamma * nonterm * v_tp1 - v_t)

    # V-trace value target
    vs_t = v_t + delta + gamma * nonterm * c_t * (vs_next - v_tp1)

    # V-trace policy advantage (uses vs_{t+1})
    adv_t = rho_t * (r_t + gamma * nonterm * vs_next - v_t)

    return vs_t, (vs_t, adv_t)


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def compute_meta_prim_vtrace_targets(
    agent: AgentState,
    prim: PrimitiveStorage,
    last_obs_chw: jnp.ndarray,   # [N,C,H,W]
    n_actions: int,              # A (unused here but keep API)
    gamma: float,
    rho_clip: float,
    c_clip: float,
    eps: float = 1e-12,
):
    """
    Time-scan version:
      - encoder + heads are evaluated per time step on batch N (not T*N)
      - returns flattened dict for minibatching, but avoids huge conv fwd on B
    """
    T, N = prim.actions.shape
    B = T * N

    obs_t = prim.obs                                               # [T,N,C,H,W]
    obs_tp1 = jnp.concatenate([prim.obs[1:], last_obs_chw[None]], axis=0)  # [T,N,...]

    a_TN  = prim.actions.astype(jnp.int32)                          # [T,N]
    mu_TN = jnp.clip(prim.mu_prim.astype(jnp.float32), eps, 1.0)    # [T,N]
    r_TN  = prim.rewards.astype(jnp.float32)                        # [T,N]
    d_TN  = prim.dones.astype(jnp.float32)                          # [T,N]

    def per_t_step(_carry, inp):
        obsN, obs1N, aN = inp                                       # obsN:[N,...], aN:[N]
        # forward on s_t
        h = agent.enc.apply_fn({"params": agent.enc.params}, chw_to_nhwc(obsN), train=False)            # [N,D]
        logits = agent.meta_actor.apply_fn({"params": agent.meta_actor.params}, h, train=False)         # [N,M]
        v = agent.meta_critic.apply_fn({"params": agent.meta_critic.params}, h, train=False).squeeze(-1)  # [N]

        logp_all = jax.nn.log_softmax(logits, axis=-1)               # [N,M]
        logpi = logp_all[jnp.arange(N), aN]                          # [N]

        # forward on s_{t+1} for bootstrap value
        h1 = agent.enc.apply_fn({"params": agent.enc.params}, chw_to_nhwc(obs1N), train=False)          # [N,D]
        v1 = agent.meta_critic.apply_fn({"params": agent.meta_critic.params}, h1, train=False).squeeze(-1)  # [N]
        return _carry, (v, v1, logpi)

    _, (v_TN, v1_TN, logpi_TN) = jax.lax.scan(
        per_t_step,
        None,
        (obs_t, obs_tp1, a_TN),
        length=T,
    )
    # v_TN, v1_TN, logpi_TN are [T,N]

    pi_TN = jnp.exp(logpi_TN)
    rho_TN = pi_TN / mu_TN
    rho_bar_TN = jnp.clip(rho_TN, 0.0, rho_clip)
    c_TN = jnp.clip(rho_TN, 0.0, c_clip)

    # bootstrap vs_{T} = V(s_{T}) (use last v1_TN[-1])
    vs_last_N = v1_TN[-1]                                           # [N]

    scan_inp = (r_TN, d_TN, v_TN, v1_TN, rho_bar_TN, c_TN)
    _, (vs_TN, adv_TN) = jax.lax.scan(
        lambda carry, inp: _vtrace_scan(carry, inp, gamma=gamma),
        vs_last_N,
        scan_inp,
        reverse=True,
    )

    batch = {
        "obs": obs_t.reshape((B,) + obs_t.shape[2:]),                # [B,C,H,W]
        "act_prim": a_TN.reshape((B,)),                              # [B]
        "logpi_old": logpi_TN.reshape((B,)),                         # [B]
        "v_old": v_TN.reshape((B,)),                                 # [B]
        "vs_target": vs_TN.reshape((B,)),
        "adv_target": adv_TN.reshape((B,)),
    }
    stats = (jnp.mean(rho_bar_TN), jnp.mean(c_TN))
    return batch, stats


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def update_meta_primitives_offpolicy_minibatch(
    agent: AgentState,
    mb,                     # minibatch dict
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    norm_adv: bool,
):
    B = mb["act_prim"].shape[0]

    adv = mb["adv_target"]
    if norm_adv:
        adv = (adv - jnp.mean(adv)) / jnp.sqrt(jnp.var(adv) + 1e-8)

    def loss_fn(enc_p, act_p, crt_p):
        h = agent.enc.apply_fn({"params": enc_p}, chw_to_nhwc(mb["obs"]), train=True)
        logits = agent.meta_actor.apply_fn({"params": act_p}, h, train=True)                 # [B,M]
        v = agent.meta_critic.apply_fn({"params": crt_p}, h, train=True).squeeze(-1)         # [B]

        logp_all = jax.nn.log_softmax(logits, axis=-1)                                       # [B,M]
        logpi = logp_all[jnp.arange(B), mb["act_prim"]]                                      # [B]

        # PPO-style ratio against stored old logpi (computed pre-update in compute step)
        logratio = logpi - mb["logpi_old"]
        ratio = jnp.exp(logratio)

        # policy loss (clipped)
        pg1 = -adv * ratio
        pg2 = -adv * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        pg_loss = jnp.mean(jnp.maximum(pg1, pg2))

        # value loss to V-trace target
        v_targ = mb["vs_target"]
        v_loss = 0.5 * jnp.mean(jnp.square(v - jax.lax.stop_gradient(v_targ)))

        # entropy
        p = jax.nn.softmax(logits, axis=-1)
        ent = jnp.mean(-(p * logp_all).sum(axis=-1))

        total = pg_loss + vf_coef * v_loss - ent_coef * ent
        aux = (pg_loss, v_loss, ent, jnp.mean(ratio))
        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1, 2))(
        agent.enc.params, agent.meta_actor.params, agent.meta_critic.params
    )
    g_enc, g_act, g_crt = grads

    agent = agent.replace(
        enc=agent.enc.apply_gradients(grads=g_enc),
        meta_actor=agent.meta_actor.apply_gradients(grads=g_act),
        meta_critic=agent.meta_critic.apply_gradients(grads=g_crt),
        grad_steps=agent.grad_steps + 1,
    )
    return agent, (loss, *aux)

# ==============================================================================
# PPO loss on meta transitions
# ==============================================================================
@jax.jit
def _wm(mask: jnp.ndarray, x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """
    Weighted mean over leading dimension.
    mask: [B] float32 in {0,1} (or [B] weights)
    x:    [B] or [B,K]
    """
    w = mask
    while w.ndim < x.ndim:
        w = w[..., None]
    denom = jnp.sum(w) + eps
    return jnp.sum(w * x) / denom


@jax.jit
def _wvar(mask: jnp.ndarray, x: jnp.ndarray, mean: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """
    Weighted variance over leading dimension.
    mean: scalar or [K] matching x's trailing dims.
    """
    w = mask
    while w.ndim < x.ndim:
        w = w[..., None]
    denom = jnp.sum(w) + eps
    return jnp.sum(w * jnp.square(x - mean)) / denom


@partial(jax.jit, static_argnums=range(2, 7))
def ppo_update_on_meta(
    agent: AgentState,
    batch,                 # dict of fixed-size arrays, includes batch["mask"]
    args,
    n_actions: int,
    n_options: int,
    use_clip_vloss: bool,
    ent_coef: float,
    vf_coef: float,
):
    """
    Weighted PPO for padded / fixed-shape batches.

    Key idea:
      - We NEVER do boolean indexing inside jit.
      - We keep B fixed (here B = T_prim * N_env).
      - batch["mask"] is 1 for valid meta entries, 0 for padded entries.
      - All losses are computed as weighted means using mask.

    This makes PPO JIT-safe while preserving correctness.
    """

    B = batch["actions"].shape[0]
    mask = batch["mask"].astype(jnp.float32)  # [B]

    def loss_fn(enc_p, act_p, crt_p):
        # ------------------------------------------------------------
        # Forward pass
        # ------------------------------------------------------------
        h = agent.enc.apply_fn({"params": enc_p}, chw_to_nhwc(batch["obs"]), train=True)
        logits = agent.meta_actor.apply_fn({"params": act_p}, h, train=True)          # [B,M]
        v = agent.meta_critic.apply_fn({"params": crt_p}, h, train=True).squeeze(-1) # [B]

        logp_all = jax.nn.log_softmax(logits, axis=-1)                                # [B,M]
        logp = logp_all[jnp.arange(B), batch["actions"]]                               # [B]

        logratio = logp - batch["logprobs"]                                            # [B]
        ratio = jnp.exp(logratio)                                                      # [B]

        # ------------------------------------------------------------
        # Advantage (optionally weighted-normalized)
        # ------------------------------------------------------------
        adv = batch["advantages"]                                                      # [B]
        if args.algo.norm_adv:
            adv_mean = _wm(mask, adv)
            adv_var  = _wvar(mask, adv, adv_mean)
            adv = (adv - adv_mean) / jnp.sqrt(adv_var + 1e-8)

        # ------------------------------------------------------------
        # Policy gradient loss (weighted)
        # ------------------------------------------------------------
        pg1 = -adv * ratio
        pg2 = -adv * jnp.clip(ratio, 1.0 - args.algo.clip_coef, 1.0 + args.algo.clip_coef)
        pg_loss = _wm(mask, jnp.maximum(pg1, pg2))

        # ------------------------------------------------------------
        # Value loss (weighted)
        # ------------------------------------------------------------
        v_target = batch["returns"]                                                    # [B]
        v_old    = batch["values"]                                                     # [B]

        v_loss_unclipped = 0.5 * jnp.square(v - v_target)                              # [B]
        if use_clip_vloss:
            v_clipped = v_old + jnp.clip(v - v_old, -args.algo.clip_coef, args.algo.clip_coef)
            v_loss_clipped = 0.5 * jnp.square(v_clipped - v_target)
            v_loss = _wm(mask, jnp.maximum(v_loss_unclipped, v_loss_clipped))
        else:
            v_loss = _wm(mask, v_loss_unclipped)

        # ------------------------------------------------------------
        # Entropy (weighted)
        # ------------------------------------------------------------
        # Standard categorical entropy: -sum p log p
        p = jax.nn.softmax(logits, axis=-1)
        entropy_per = -(p * logp_all).sum(axis=-1)                                     # [B]
        entropy = _wm(mask, entropy_per)

        # ------------------------------------------------------------
        # Approx KL (weighted) and total loss
        # ------------------------------------------------------------
        approx_kl_per = ((ratio - 1.0) - logratio)                                     # [B]
        approx_kl = _wm(mask, approx_kl_per)

        loss = pg_loss - ent_coef * entropy + vf_coef * v_loss
        return loss, (pg_loss, v_loss, entropy, jax.lax.stop_gradient(approx_kl))

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1, 2))(
        agent.enc.params, agent.meta_actor.params, agent.meta_critic.params
    )
    g_enc, g_act, g_crt = grads

    agent = agent.replace(
        enc=agent.enc.apply_gradients(grads=g_enc),
        meta_actor=agent.meta_actor.apply_gradients(grads=g_act),
        meta_critic=agent.meta_critic.apply_gradients(grads=g_crt),
        grad_steps=agent.grad_steps + 1,
    )
    return agent, (loss, *aux)

# ==============================================================================
# Option actor-critic updates
# ------------------------------------------------------------------------------
@jax.jit
def _intr_gae_scan(carry, inp, gamma: float, lam: float):
    """
    carry: adv_next [O,N]
    inp: (done_t [N], delta_hat_t [O,N], c_t [O,N])
    """
    adv_next = carry
    done_t, delta_hat_t, c_t = inp
    nonterm = 1.0 - done_t[None, :]  # [1,N] -> broadcast
    adv_t = delta_hat_t + (gamma * lam) * nonterm * c_t * adv_next
    return adv_t, adv_t

# ==============================================================================
# Rollout: step function that executes options and builds BOTH storages
# ------------------------------------------------------------------------------
# Key idea:
#   - Primitive storage is written every primitive step.
#   - Meta storage is written only when a segment terminates (primitive meta-action
#     or option termination).
#
# We keep per-env segment accumulators in carry:
#   - seg_active_meta (current meta action)
#   - seg_start_obs, seg_start_logp, seg_start_value
#   - seg_cumR (discounted), seg_cumG (current discount), seg_len
# ==============================================================================
@partial(jax.jit, static_argnums=range(3, 8))
def env_take_step(
    carry,
    _unused,
    agent: AgentState,
    env,
    n_actions: int,
    n_options: int,
    opt_dur_table: jnp.ndarray,   # [O] int32
    bi_dir_opt: bool,
    ):
    """
    carry = (
      obs_chw, env_state, key,
      seg_meta, seg_start_obs, seg_start_logp, seg_start_value,
      seg_cumR, seg_cumG, seg_len, seg_active,
      meta_storage
    )
    """
    (obs, env_state, key,
     seg_meta, seg_start_obs, seg_start_logp, seg_start_value,
     seg_cumR, seg_cumG, seg_len, seg_active,
     meta_store) = carry

    N = obs.shape[0]
    A = n_actions
    O = n_options
    M = A + O

    key, k_meta, k_opt = jax.random.split(key, 3)

    # ------------------------------------------------------------
    # (1) If no active segment, sample new meta action
    # ------------------------------------------------------------
    def _sample_new():
        logits, v = meta_logits_and_value(agent, obs, train=False)  # [N,M], [N]
        u, logp = categorical_sample(logits, k_meta)                # [N], [N]
        return u, logp, v, logits

    def _reuse_old():
        # dummy values; will be ignored
        logits, v = meta_logits_and_value(agent, obs, train=False)
        logp = jnp.zeros((N,), dtype=jnp.float32)
        return seg_meta, seg_start_logp, seg_start_value, logits

    meta_u, meta_logp, meta_v, meta_logits = jax.lax.cond(
        jnp.any(~seg_active),
        lambda: _sample_new(),
        lambda: _reuse_old()
    )
    # The cond above is "global"; but we need per-env logic.
    # We implement per-env selection with where:
    logits, v = meta_logits_and_value(agent, obs, train=False)
    u_new, logp_new = categorical_sample(logits, k_meta)

    meta_u = jnp.where(seg_active, seg_meta, u_new)
    # If starting new segment, store logp/value at start; else keep previous
    meta_logp_start = jnp.where(seg_active, seg_start_logp, logp_new)
    meta_v_start = jnp.where(seg_active, seg_start_value, v)

    seg_start_obs = jnp.where(seg_active[:, None, None, None], seg_start_obs, obs)
    seg_meta = meta_u
    seg_start_logp = meta_logp_start
    seg_start_value = meta_v_start

    # Reset accumulators on new segment
    new_seg = ~seg_active
    seg_cumR = jnp.where(new_seg, 0.0, seg_cumR)
    seg_cumG = jnp.where(new_seg, 1.0, seg_cumG)
    seg_len  = jnp.where(new_seg, 0, seg_len)

    # ------------------------------------------------------------
    # (2) Low-level action execution
    # ------------------------------------------------------------
    is_prim = seg_meta < A
    sel_opt = (seg_meta - A).clip(0, O - 1)

    # Option low-level policy
    opt_logits, _opt_v = option_logits_and_values(agent, obs, train=False)  # [O,N,A]
    # Gather logits for selected option: [N,A]
    opt_logits_sel = opt_logits[sel_opt, jnp.arange(N), :]

    # Sample primitive action:
    # - if primitive meta-action: action = meta, mu=1
    # - if option meta-action: sample from option logits, mu = pi_o(a|s)
    a_opt, logp_opt = categorical_sample(opt_logits_sel, k_opt)  # [N], [N]
    pi_opt = jnp.exp(logp_opt)

    action = jnp.where(is_prim, seg_meta, a_opt)
    mu_prim = jnp.where(is_prim, 1.0, pi_opt)

    # ------------------------------------------------------------
    # (3) Env step
    # ------------------------------------------------------------
    next_obs, next_env_state, r_ext, done, infos = env.step(env_state, action)

    # ------------------------------------------------------------
    # (4) Update segment accumulators
    #   cumR += cumG * r_ext
    #   cumG *= gamma
    #   seg_len += 1
    # ------------------------------------------------------------
    seg_cumR = seg_cumR + seg_cumG * r_ext
    seg_cumG = seg_cumG * jnp.float32(args.env.gamma)
    seg_len  = seg_len + 1

    # ------------------------------------------------------------
    # (5) Termination rule
    #   - primitive meta terminates immediately
    #   - option terminates when seg_len == opt_dur_table[o]
    #   - or env done
    # ------------------------------------------------------------
    dur = opt_dur_table[sel_opt].astype(jnp.int32)
    opt_term = (seg_len >= dur)
    term = is_prim | opt_term | done

    # ------------------------------------------------------------
    # (6) Write meta transition when term=True (per env)
    # ------------------------------------------------------------
    ptr = meta_store.ptr  # [N]
    # For each env e, write at time index ptr[e] if term[e]
    t_idx = ptr
    e_idx = jnp.arange(N)

    def write_field(arr, val, broadcast_val=False):
        # arr has shape [T,N,...]
        # val has shape [N,...] or scalar
        if broadcast_val:
            val_e = jnp.broadcast_to(val, (N,) + arr.shape[2:])
        else:
            val_e = val
        return arr.at[t_idx, e_idx].set(jnp.where(term[:, None] if arr.ndim > 2 else term, val_e, arr[t_idx, e_idx]))

    # But .at with per-element boolean is tricky; easiest is do masked update using where then set.
    # We'll do "set always" then restore old where not term.
    def masked_set(arr, val):
        cur = arr[t_idx, e_idx]
        new = val
        # broadcast term to cur shape
        m = term
        while m.ndim < cur.ndim:
            m = m[..., None]
        out = jnp.where(m, new, cur)
        return arr.at[t_idx, e_idx].set(out)

    meta_store = meta_store.replace(
        start_obs=masked_set(meta_store.start_obs, seg_start_obs),
        actions=masked_set(meta_store.actions, seg_meta),
        logprobs=masked_set(meta_store.logprobs, seg_start_logp),
        values=masked_set(meta_store.values, seg_start_value),
        rewards=masked_set(meta_store.rewards, seg_cumR.astype(jnp.float32)),
        dones=masked_set(meta_store.dones, done),
        durations=masked_set(meta_store.durations, seg_len.astype(jnp.int32)),
        end_obs=masked_set(meta_store.end_obs, next_obs),
        ptr=ptr + term.astype(jnp.int32),
    )

    # Reset segment active flags on termination
    seg_active = jnp.where(term, False, True)

    # ------------------------------------------------------------
    # (7) Build primitive step output for PrimitiveStorage
    # ------------------------------------------------------------
    prim_t = (
        obs, action, r_ext, done, seg_meta, mu_prim
    )

    new_carry = (
        next_obs, next_env_state, key,
        seg_meta, seg_start_obs, seg_start_logp, seg_start_value,
        seg_cumR, seg_cumG, seg_len, seg_active,
        meta_store,
    )
    return new_carry, (prim_t, infos)


# ==============================================================================
# Minibatch maker utilities
# ==============================================================================
def flatten_TN(x):
    return x.reshape((-1,) + x.shape[2:])


def make_minibatches(rng, n_mini_batches: int, data_tree):
    # Infer B from a leaf (all leaves share leading dimension B)
    leaf0 = jax.tree_util.tree_leaves(data_tree)[0]
    B = leaf0.shape[0]  # <-- static at compile time

    perm = jax.random.permutation(rng, B)

    def take(x):
        x = x[perm]
        return x.reshape((n_mini_batches, -1) + x.shape[1:])

    return jax.tree_util.tree_map(take, data_tree)


# ==============================================================================
# Main update_step: rollout + (Lap update) + (Option update) + (Meta PPO update)
# ==============================================================================
# ==============================================================================
# Clean update_step() for PPO-EO with:
#   (A) Rollout + primitive storage + meta storage
#   (B) Laplacian (ALLO) update from primitive rollout
#   (C) Option low-level policies update on intrinsic rewards
#       - MUST be minibatched
#   (D) Meta-primitives update on extrinsic rewards (off-policy, ALL primitive steps)
#       - V-trace targets/advantages, minibatched
#   (E) Meta-options update on extrinsic rewards (on-policy PPO on meta buffer)
#       - minibatched with mask over valid+option entries
# ==============================================================================

@partial(jax.jit, static_argnums=range(2, 7))
def update_step(
    runner_state,
    _unused,
    args,
    train_env,
    eval_env,
    start_time: float,
    eval_episodes: int,
):
    """
    runner_state = (agent, (obs, env_state), (eval_obs, eval_state), key, opt_dur_table)

    IMPORTANT invariants:
      - PrimitiveStorage.actions  : executed primitive action a_t (always in [0..A-1])
      - PrimitiveStorage.mu_prim  : behavior prob of executed primitive a_t
            * if meta picked primitive => mu=1
            * if option executed       => mu=pi_option(a_t|s_t)
      - MetaStorage.actions      : meta action u (in [0..A+O-1]) recorded ONLY at segment ends
      - MetaStorage.rewards      : discounted sum of EXTRINSIC rewards over the segment
      - MetaStorage.durations    : segment length k
    """
    agent, train_state, eval_state, key, opt_dur_table = runner_state
    obs, env_state = train_state

    N = obs.shape[0]
    A = train_env.single_action_space.n
    n_base_opt = args.algo.n_options
    bi_dir_opt = args.algo.bidir_options
    O = (2 * n_base_opt) if bi_dir_opt else n_base_opt
    T_prim = args.env.n_steps

    # --------------------------------------------------------------------------
    # (1) Rollout T_prim primitive steps:
    #     - fills PrimitiveStorage for ALL primitive steps
    #     - fills MetaStorage only when segment terminates (primitive meta-action
    #       terminates immediately; option terminates after duration or env done)
    # --------------------------------------------------------------------------
    meta_store0 = make_empty_meta_storage(T_prim, N, obs_shape_chw=obs.shape[1:])

    seg_active0 = jnp.zeros((N,), dtype=jnp.bool_)
    seg_meta0 = jnp.zeros((N,), dtype=jnp.int32)
    seg_start_obs0 = obs
    seg_start_logp0 = jnp.zeros((N,), dtype=jnp.float32)
    seg_start_value0 = jnp.zeros((N,), dtype=jnp.float32)
    seg_cumR0 = jnp.zeros((N,), dtype=jnp.float32)
    seg_cumG0 = jnp.ones((N,), dtype=jnp.float32)
    seg_len0 = jnp.zeros((N,), dtype=jnp.int32)

    carry0 = (
        obs, env_state, key,
        seg_meta0, seg_start_obs0, seg_start_logp0, seg_start_value0,
        seg_cumR0, seg_cumG0, seg_len0, seg_active0,
        meta_store0,
    )

    step_fn = partial(
        env_take_step,
        agent=agent,
        env=train_env,
        n_actions=A,
        n_options=O,
        opt_dur_table=opt_dur_table,
        bi_dir_opt=bi_dir_opt,
    )

    (obsF, env_stateF, keyF,
     _seg_metaF, _seg_start_obsF, _seg_start_logpF, _seg_start_valueF,
     _seg_cumRF, _seg_cumGF, _seg_lenF, _seg_activeF,
     meta_storeF), (prim_pack, infos) = jax.lax.scan(step_fn, carry0, None, length=T_prim)

    obs_seq, act_seq, r_seq, d_seq, meta_seq, mu_seq = prim_pack  # each [T,N,...]

    prim = PrimitiveStorage(
        obs=obs_seq,
        actions=act_seq,       # executed primitive action a_t
        rewards=r_seq,         # extrinsic reward
        dones=d_seq,
        meta_action=meta_seq,  # meta action u_t (constant during option execution)
        mu_prim=mu_seq,        # behavior prob of executed primitive action
    )
    meta = meta_storeF

    T, N = prim.rewards.shape
    B = T * N

    # --------------------------------------------------------------------------
    # (2) Laplacian update (ALLO) on primitive rollout slices
    #     - uses sequences of states; updates lap encoder + barrier/dual vars
    # --------------------------------------------------------------------------
    lap_n_step = 32
    N_lap = min(lap_n_step + 1, T_prim)  # safety

    lap_obs_seq = prim.obs[:N_lap]                        # [N_lap,N,C,H,W]
    lap_obs_seq = jnp.transpose(lap_obs_seq, (1, 0, 2, 3, 4))  # [N, N_lap, C,H,W]
    lap_valid = jnp.logical_not(jnp.transpose(prim.dones[:N_lap], (1, 0)))     # [N, N_lap]

    keyF, lap_key = jax.random.split(keyF)
    agent, (eig_loss, eig_pos, eig_ort, eig_dual, _lambda_batch) = update_laplacian_from_prim(
        lap_key, agent, lap_obs_seq, lap_valid, n_steps=N_lap
    )

    # Update option durations (from eigenvalues EMA or fixed)
    if args.algo.eigdur_options:
        eigs = agent.lap.eigen_values[0]
        opt_dur_table = eigenvalues_to_option_durations(
            eigs,
            n_options=O,
            bi_dir_opt=bi_dir_opt,
            min_dur=args.algo.opt_dur_min,
            max_dur=args.algo.opt_dur_max,
            eps=1e-6,
        )
    else:
        opt_dur_table = jnp.full((O,), jnp.int32(args.algo.opt_dur), dtype=jnp.int32)

    # --------------------------------------------------------------------------
    # (3) Intrinsic rewards for options (per primitive step), then OPTION update
    #     NOTE: you requested minibatching for intrinsic option learning.
    #
    #     We compute r_int_TNO once, then do multiple epochs/minibatches.
    # --------------------------------------------------------------------------
    r_int_TNO, done_TN, new_rms = compute_intrinsic_rewards_TNO(
        agent, prim, n_options=O, bi_dir_opt=bi_dir_opt, eps=1e-8
    )
    agent = agent.replace(int_rms=new_rms)

    # Build a minibatchable "option dataset" over primitive steps.
    # We'll flatten time/env, and the per-option training fn will reconstitute shapes as needed.
    # (This keeps make_minibatches generic.)
    opt_dataset = {
        "obs": prim.obs.reshape((B,) + prim.obs.shape[2:]),          # [B,C,H,W]
        "obs_tp1": jnp.concatenate([prim.obs[1:], obsF[None, ...]], axis=0).reshape((B,) + prim.obs.shape[2:]),
        "act": prim.actions.reshape((B,)),                           # executed primitive action a_t
        "mu": jnp.clip(prim.mu_prim.reshape((B,)), 1e-12, 1.0),      # behavior prob of a_t
        "done": done_TN.reshape((B,)),                               # [B]
        "r_int": r_int_TNO.reshape((B, O)),                          # [B,O] intrinsic reward per option
        "T": jnp.asarray(T, jnp.int32),                              # scalars for reshaping if needed
        "N": jnp.asarray(N, jnp.int32),
    }

    # We need an OPTION minibatch update primitive that consumes a minibatch (fixed size)
    # and applies grads. For cleanliness, implement it as a small wrapper around your
    # sequence-based update_option_policies_intrinsic_gae.
    #
    # Easiest consistent approach:
    #   - reconstruct [T_mb, N_mb] from the minibatch isn't possible (it is a random set).
    # So, for minibatching we do the intrinsic update as a *per-transition* off-policy AC:
    #   - advantage via 1-step or V-trace-like recursion is not meaningful on shuffled samples.
    #
    # Since you explicitly requested "trained in minibatches" AND "GAE-lambda",
    # the only consistent way is:
    #   (a) compute sequence advantages/returns ON THE FULL ROLLOUT (ordered),
    #   (b) then shuffle transitions and SGD on those precomputed targets.
    #
    # We'll do that: compute (adv_o[t,n], ret_o[t,n]) once, then minibatch regression.

    # --- compute intrinsic option targets/advantages on full rollout (ordered) ---
    # We'll reuse your update_option_policies_intrinsic_gae logic, but split it into:
    #    compute_option_intrinsic_targets(...)  -> returns adv/ret/logpi_old/rho_bar
    #    update_option_intrinsic_minibatch(...) -> SGD on those targets
    #
    # If you don't already have these split functions, implement them similarly to
    # the meta-primitive V-trace split (compute targets once, then minibatch update).
    #
    # Here we assume you have:
    #   opt_targets = compute_option_intrinsic_targets(agent, prim, r_int_TNO, done_TN, A, O, gamma, lam, rho_clip, c_clip)
    # and:
    #   agent, stats = update_option_intrinsic_minibatch(agent, mb, ...)
    #
    # (If you want, I can provide these two exact functions next.)

    opt_targets, (opt_rho_mean, opt_c_mean) = compute_option_intrinsic_targets(
        agent=agent,
        prim=prim,
        r_int_TNO=r_int_TNO,
        done_TN=done_TN,
        n_actions=A,
        n_options=O,
        gamma=args.env.gamma,
        gae_lambda=args.algo.gae_lambda,
        rho_clip=1.0,
        c_clip=1.0,
    )
    # opt_targets should be flattened to length B, with per-option info folded in:
    #   obs: [B,C,H,W]
    #   act: [B]
    #   mu:  [B]
    #   logpi_old: [O,B] or [B,O] depending on your choice
    #   adv: [O,B] or [B,O]
    #   ret: [O,B] or [B,O]
    # We'll assume opt_targets uses [B,O] layout for simplicity.

    keyF, sub = jax.random.split(keyF)
    opt_mbs = make_minibatches(sub, args.algo.n_mini_batches, opt_targets)

    def opt_one_epoch(carry, _):
        ag = carry
        def opt_one_mb(ag2, mb):
            ag2, st = update_option_intrinsic_minibatch(  # <<< YOU ADD THIS (see note above)
                ag2, mb,
                n_actions=A,
                n_options=O,
                clip_coef=args.algo.clip_coef,
                ent_coef=0.01,        # your option entropy coef
                vf_coef=0.5,          # your option vf coef
                norm_adv=args.algo.norm_adv,
            )
            return ag2, st
        ag, st = jax.lax.scan(opt_one_mb, ag, opt_mbs)
        return ag, st

    agent, opt_stats = jax.lax.scan(opt_one_epoch, agent, None, length=args.algo.upd_epochs)
    # opt_stats: [epochs, minibatches, ...]
    opt_total_loss, opt_actor_loss, opt_v_loss, opt_ent, opt_ratio = [x.mean() for x in opt_stats]

    # --------------------------------------------------------------------------
    # (4) Meta-primitives update (extrinsic) on ALL primitive steps (off-policy)
    #     - compute V-trace targets/advantages on ordered rollout
    #     - then minibatch SGD with PPO-style ratio clipping against stored logpi_old
    # --------------------------------------------------------------------------
    # prim_targets, (prim_rho_mean, prim_c_mean) = compute_meta_prim_vtrace_targets(
    #     agent=agent,
    #     prim=prim,
    #     last_obs_chw=obsF,
    #     n_actions=A,
    #     gamma=args.env.gamma,
    #     rho_clip=1.0,
    #     c_clip=1.0,
    # )
    #
    # keyF, sub = jax.random.split(keyF)
    # prim_mbs = make_minibatches(sub, args.algo.n_mini_batches, prim_targets)
    #
    # def prim_one_epoch(carry, _):
    #     ag = carry
    #     def prim_one_mb(ag2, mb):
    #         ag2, st = update_meta_primitives_offpolicy_minibatch(
    #             ag2, mb,
    #             clip_coef=args.algo.clip_coef,
    #             ent_coef=args.algo.ent_coef,
    #             vf_coef=args.algo.vf_coef,
    #             norm_adv=args.algo.norm_adv,
    #         )
    #         return ag2, st
    #     ag, st = jax.lax.scan(prim_one_mb, ag, prim_mbs)
    #     return ag, st
    #
    # agent, prim_stats = jax.lax.scan(prim_one_epoch, agent, None, length=args.algo.upd_epochs)
    # prim_loss, prim_pg, prim_v, prim_ent, prim_ratio = [x.mean() for x in prim_stats]
    prim_loss = prim_pg = prim_v = prim_ent = prim_ratio = prim_rho_mean = prim_c_mean = jnp.nan

    # --------------------------------------------------------------------------
    # (5) Meta-options update (extrinsic) on meta buffer entries where:
    #       - valid entry (t < ptr[e])
    #       - meta action is an option (u >= A)
    #     This is on-policy PPO on SMDP transitions (mask-weighted).
    # --------------------------------------------------------------------------
    Tm = T_prim
    t_ids = jnp.arange(Tm)[:, None]
    valid = t_ids < meta.ptr[None, :]                      # [T,N] bool

    # SMDP-GAE targets on meta buffer
    adv_m, ret_m = compute_smdp_gae(agent, meta, gamma=args.env.gamma, gae_lambda=args.algo.gae_lambda, T_max=Tm)

    # is_opt_meta = (meta.actions >= A)
    # mask_opt = (valid & is_opt_meta).reshape((-1,)).astype(jnp.float32)  # [Tm*N]
    mask_opt = valid.reshape((-1,)).astype(jnp.float32)

    def flat_all(x):
        return x.reshape((-1,) + x.shape[2:])

    meta_batch = {
        "obs": flat_all(meta.start_obs),
        "actions": flat_all(meta.actions),
        "logprobs": flat_all(meta.logprobs),
        "values": flat_all(meta.values),
        "advantages": flat_all(adv_m),
        "returns": flat_all(ret_m),
        "mask": mask_opt,
    }
    Bm = meta_batch["actions"].shape[0]

    def do_meta_ppo_updates(agent_in, key_in):
        def one_epoch(carry, _):
            ag, rng = carry
            rng, sub = jax.random.split(rng)
            mbs = make_minibatches(sub, args.algo.n_mini_batches, meta_batch)

            def one_mb(ag2, mb):
                ag2, st = ppo_update_on_meta(
                    ag2, mb, args,
                    n_actions=A, n_options=O,
                    use_clip_vloss=args.algo.clip_vloss,
                    ent_coef=args.algo.ent_coef,
                    vf_coef=args.algo.vf_coef,
                )
                return ag2, st

            ag, st = jax.lax.scan(one_mb, ag, mbs)
            return (ag, rng), st

        key2, upd_key = jax.random.split(key_in)
        (agent_out, _), st = jax.lax.scan(one_epoch, (agent_in, upd_key), None, length=args.algo.upd_epochs)
        return agent_out, st, key2

    agent, meta_stats, keyF = do_meta_ppo_updates(agent, keyF)
    meta_loss, meta_pg, meta_v, meta_ent, meta_kl = [x.mean() for x in meta_stats]

    # --------------------------------------------------------------------------
    # (6) Counters + metrics
    # --------------------------------------------------------------------------
    agent = agent.replace(
        timesteps=agent.timesteps + args.env.n_steps * args.env.n_total_envs,
        n_updates=agent.n_updates + 1,
    )

    metrics = {
        "env_step": agent.timesteps,
        "update_steps": agent.n_updates,
        "grad_steps": agent.grad_steps,

        # meta-options PPO (extrinsic)
        "ppo_meta_opt/loss": meta_loss,
        "ppo_meta_opt/pg": meta_pg,
        "ppo_meta_opt/v": meta_v,
        "ppo_meta_opt/ent": meta_ent,
        "ppo_meta_opt/kl": meta_kl,
        "ppo_meta_opt/mask_frac": jnp.mean(mask_opt),

        # meta-primitives off-policy V-trace (extrinsic)
        "meta_prim_off/loss": prim_loss,
        "meta_prim_off/pg": prim_pg,
        "meta_prim_off/v": prim_v,
        "meta_prim_off/ent": prim_ent,
        "meta_prim_off/ratio": prim_ratio,
        "meta_prim_off/rho_mean": prim_rho_mean,
        "meta_prim_off/c_mean": prim_c_mean,

        # Laplacian
        "lap/total": eig_loss,
        "lap/pos": eig_pos,
        "lap/ortho": eig_ort,
        "lap/dual": eig_dual,
        "lap/eig_mean": jnp.mean(agent.lap.eigen_values),

        # Options (intrinsic)
        "opt_intr/loss": opt_total_loss,
        "opt_intr/pg": opt_actor_loss,
        "opt_intr/v": opt_v_loss,
        "opt_intr/ent": opt_ent,
        "opt_intr/int_rms_var_mean": jnp.mean(agent.int_rms.var),

        # durations
        "opt/dur_min": jnp.min(opt_dur_table),
        "opt/dur_max": jnp.max(opt_dur_table),
        "opt/dur_mean": jnp.mean(opt_dur_table.astype(jnp.float32)),
    }

    # envpool infos
    metrics.update({k: v.mean() for k, v in infos.items()})
    metrics.update({"Max_Return": infos["returned_episode_returns"].max()})

    metrics["opt_intr/ratio"] = opt_ratio
    metrics["opt_intr/rho_mean"] = opt_rho_mean
    metrics["opt_intr/c_mean"] = opt_c_mean

    # --------------------------------------------------------------------------
    # (7) Optional eval (unchanged)
    # --------------------------------------------------------------------------
    do_eval = agent.timesteps >= agent.next_eval_step

    def _eval_fn(carry):
        ag, ev_state, met = carry
        ev_state2, ev_met = run_eval_until_episodes(ag, ev_state, eval_env, n_episodes=eval_episodes)
        eval_interval = jnp.int32(args.algo.tot_steps // 25)
        ag = ag.replace(next_eval_step=ag.next_eval_step + eval_interval)
        met = {**met, **ev_met}
        return ag, ev_state2, met

    def _no_eval(carry):
        ag, ev_state, met = carry
        met = {**met,
               "eval/episodes": jnp.array(0, jnp.int32),
               "eval/avg_return": jnp.array(0.0, jnp.float32),
               "eval/avg_ep_len": jnp.array(0.0, jnp.float32)}
        return ag, ev_state, met

    agent, eval_state, metrics = jax.lax.cond(do_eval, _eval_fn, _no_eval, (agent, eval_state, metrics))

    # --------------------------------------------------------------------------
    # (8) WandB callbacks (same pattern as before, omitted here for brevity)
    # --------------------------------------------------------------------------
    # train_log: everything except eval/*
    train_log = {k: v for k, v in metrics.items() if not k.startswith("eval/")}

    eval_log = {
        "env_step": metrics["env_step"],
        "eval/episodes": metrics["eval/episodes"],
        "eval/avg_return": metrics["eval/avg_return"],
        "eval/avg_ep_len": metrics["eval/avg_ep_len"],
        # add a marker so you can see *exactly* which steps had eval
        "eval/logged": jnp.asarray(1, jnp.int32),
    }

    def _wandb_train_cb(m):
        # host-side SPS
        elapsed = time.time() - start_time
        sps = float(m["env_step"]) / max(elapsed, 1e-8)
        m = dict(m)
        m["SPS"] = sps
        wandb.log(m, step=int(m["env_step"]))

    def _wandb_eval_cb(m):
        import wandb
        wandb.log(dict(m), step=int(m["env_step"]))

    # train: every update
    _ = jax.lax.cond(
        jnp.asarray(args.wb.on),
        lambda m: jax.debug.callback(_wandb_train_cb, m),
        lambda m: None,
        train_log,
    )

    # eval: only when do_eval
    _ = jax.lax.cond(
        jnp.asarray(args.wb.on) & do_eval,
        lambda m: jax.debug.callback(_wandb_eval_cb, m),
        lambda m: None,
        eval_log,
    )

    new_runner_state = (agent, (obsF, env_stateF), eval_state, keyF, opt_dur_table)
    return new_runner_state, metrics


def wandb_log_callback(m: dict):
    wandb.log(m, step=int(m["env_step"]))


# ==============================================================================
# Evaluation: greedy meta action, greedy option low-level
# ==============================================================================
@jax.jit
def greedy_meta_action(agent: AgentState, obs_chw):
    logits, _ = meta_logits_and_value(agent, obs_chw, train=False)
    return jnp.argmax(logits, axis=-1)


@jax.jit
def greedy_option_action(agent: AgentState, obs_chw, opt_idx: jnp.ndarray):
    opt_logits, _ = option_logits_and_values(agent, obs_chw, train=False)  # [O,N,A]
    # opt_idx: [N]
    N = obs_chw.shape[0]
    logits_sel = opt_logits[opt_idx, jnp.arange(N), :]                      # [N,A]
    return jnp.argmax(logits_sel, axis=-1)


@partial(jax.jit, static_argnums=(2, 3, 4))
def run_eval_until_episodes(agent: AgentState, eval_state, eval_env, n_episodes: int, max_steps: int = 10_000):
    obs, env_state = eval_state
    N = obs.shape[0]
    A = eval_env.single_action_space.n

    # durations from eigenvalues EMA
    n_base_opt = args.algo.n_options
    bi_dir_opt = args.algo.bidir_options
    O = (2 * n_base_opt) if bi_dir_opt else n_base_opt
    if args.algo.eigdur_options:
        eigs = agent.lap.eigen_values[0]
        opt_dur_table = eigenvalues_to_option_durations(
            eigs,
            n_options=O,
            bi_dir_opt=bi_dir_opt,
            min_dur=args.algo.opt_dur_min,
            max_dur=args.algo.opt_dur_max,
            eps=1e-6,
        )
    else:
        opt_dur_table = jnp.full((O,), jnp.int32(args.algo.opt_dur), dtype=jnp.int32)

    ep_count = jnp.array(0, jnp.int32)
    ret_sum = jnp.array(0.0, jnp.float32)
    len_sum = jnp.array(0.0, jnp.float32)
    steps = jnp.array(0, jnp.int32)

    # option execution counters
    opt_steps = jnp.array(0, jnp.int32)
    opt_picks = jnp.array(0, jnp.int32)
    opt_left = jnp.zeros((N,), dtype=jnp.int32)
    cur_opt = jnp.full((N,), -1, dtype=jnp.int32)

    def cond(carry):
        _, _, epc, _, _, st, *_ = carry
        return (epc < n_episodes) & (st < max_steps)

    def body(carry):
        obs, env_state, epc, rsum, lsum, st, opt_left, cur_opt, opt_steps, opt_picks = carry

        # If currently executing an option, continue; else pick new greedy meta
        pick_new = (opt_left <= 0)

        meta = greedy_meta_action(agent, obs)
        is_opt = meta >= A
        o_idx = (meta - A).clip(0, O - 1)
        # If pick_new, set new option; else keep cur_opt
        new_cur_opt = jnp.where(pick_new & is_opt, o_idx, cur_opt)
        # duration
        dur = jnp.asarray(opt_dur_table, jnp.int32)[new_cur_opt.clip(0, O-1)]
        new_opt_left = jnp.where(pick_new & is_opt, dur, opt_left)

        # Choose primitive action:
        # if executing option => greedy under that option head
        a_opt = greedy_option_action(agent, obs, new_cur_opt.clip(0, O-1))
        a = jnp.where((new_opt_left > 0) & is_opt, a_opt, meta.clip(0, A-1))

        next_obs, next_state, r, d, info = eval_env.step(env_state, a)

        # update option counters
        took_opt_step = ((new_opt_left > 0) & is_opt).astype(jnp.int32)
        opt_steps = opt_steps + took_opt_step.sum()
        opt_picks = opt_picks + (pick_new & is_opt).astype(jnp.int32).sum()
        new_opt_left = jnp.where((new_opt_left > 0) & is_opt, new_opt_left - 1, 0)

        done_mask = info["returned_episode"].astype(jnp.float32)
        done_ret = info["returned_episode_returns"].astype(jnp.float32)
        done_len = info["returned_episode_lengths"].astype(jnp.float32)

        epc = epc + done_mask.sum().astype(jnp.int32)
        rsum = rsum + (done_mask * done_ret).sum()
        lsum = lsum + (done_mask * done_len).sum()
        st = st + 1

        # reset option state on done envs
        new_opt_left = jnp.where(d, 0, new_opt_left)
        new_cur_opt = jnp.where(d, -1, new_cur_opt)

        return (next_obs, next_state, epc, rsum, lsum, st, new_opt_left, new_cur_opt, opt_steps, opt_picks)

    carry0 = (obs, env_state, ep_count, ret_sum, len_sum, steps, opt_left, cur_opt, opt_steps, opt_picks)
    obsF, env_stateF, epcF, rsumF, lsumF, stF, *_rest = jax.lax.while_loop(cond, body, carry0)

    epc_f = jnp.maximum(epcF.astype(jnp.float32), 1.0)
    avg_ret = rsumF / epc_f
    avg_len = lsumF / epc_f

    eval_state2 = (obsF, env_stateF)
    eval_metrics = {
        "eval/episodes": epcF,
        "eval/avg_return": avg_ret,
        "eval/avg_ep_len": avg_len,
    }
    return eval_state2, eval_metrics


# ==============================================================================
# Training loop wrapper
# ==============================================================================
def ppo_eo_learning(key, train_env, eval_env, agent: AgentState):
    train_obs, train_env_state = train_env.reset()
    eval_obs, eval_env_state = eval_env.reset()

    # initial option durations: constant opt_dur
    n_base_opt = args.algo.n_options
    bi_dir_opt = args.algo.bidir_options
    O = (2 * n_base_opt) if bi_dir_opt else n_base_opt

    # initial option durations
    n_base_opt = args.algo.n_options
    bi_dir_opt = args.algo.bidir_options
    O = (2 * n_base_opt) if bi_dir_opt else n_base_opt

    # Initialize option duration table
    opt_dur_table = jnp.full((O,), jnp.int32(args.algo.opt_dur), dtype=jnp.int32)

    runner_state = (agent, (train_obs, train_env_state), (eval_obs, eval_env_state), key, opt_dur_table)

    start_time = time.time()
    step_fn = partial(
        update_step,
        args=args,
        train_env=train_env,
        eval_env=eval_env,
        start_time=start_time,
        eval_episodes=1,
    )
    runner_state, metrics = jax.lax.scan(step_fn, runner_state, None, length=args.algo.num_iterations)
    return runner_state, metrics


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    Env = make_env()
    n_actions = Env.single_action_space.n

    if args.wb.on:
        create_wb_logger(args)
        # Save code artifacts
        try:
            import wandb
            wandb.save("src/networks/ppo_networks.py", policy="now")
        except Exception:
            pass

    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    key, setup_key, algo_key = jax.random.split(jax.random.PRNGKey(args.exp.seed), 3)

    with jax.default_device(jax.devices("gpu")[0]):
        try:
            TrainEnv = JaxLogEnvPoolWrapper(make_env())
            EvalEnv = make_eval_env(n_eval_envs=1)

            init_obs, _ = TrainEnv.reset()
            agent = setup(setup_key, args, init_obs, n_actions=n_actions)

            ppo_eo_learning(algo_key, TrainEnv, EvalEnv, agent)

        except Exception:
            traceback.print_exc()
            if args.wb.on:
                try:
                    import wandb
                    wandb.finish(exit_code=1)
                except Exception:
                    pass
        finally:
            Env.close()
            try:
                close_logger() if args.wb.on else None
            except Exception:
                pass
            jax.clear_backends()
