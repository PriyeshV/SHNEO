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
from src.networks.ppo_networks import AtariConvEnc, Actor, Critic, QHead

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
# Networks: PPO Meta-Controller + PQN Option-Controller
# ------------------------------------------------------------------------------
# Meta-controller (PPO):
#   - main encoder f(s)
#   - actor head outputs logits over meta-actions: (primitive actions + options)
#   - critic head outputs V(s)
#
# Option-controller (PQN-style Q-learning on intrinsic rewards):
#   - separate option encoder f_opt(s) (shared across all options)
#   - per-option Q heads Q_o(s, a) over primitive actions (params axis = O via nn.vmap)
#
# Key design choices for options:
#   - Off-policy TD(λ) learning on intrinsic rewards r_int^o(s,s')
#   - NO target network (PQN style); LayerNorm enabled for stability
#   - Behavior during rollout for options is GREEDY w.r.t. Q_o(s, ·)
#     (no intra-option exploration; once an option is selected it executes to termination)
# ==============================================================================

def build_vmap_option_q_heads(n_options: int, n_actions: int):
    """
    Returns a Module mapping (B,D) -> (O,B,A) where A is primitive actions.
    """
    VmapOptQ = nn.vmap(
        QHead,
        variable_axes={"params": 0},     # params have leading axis O
        split_rngs={"params": True},
        in_axes=None,                   # shared input embedding
        out_axes=0,                     # leading axis O
        axis_size=n_options,
    )
    return VmapOptQ(action_dim=n_actions)




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

    # Option PQN: shared option encoder + per-option Q heads
    opt_q_enc: TrainState
    opt_q_head: TrainState  # params leading axis O, outputs Q over primitive actions

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

    # behavior probability of executed primitive action μ(a_t|s_t)
    # - If meta-action is primitive u=a:      μ = π_meta(u=a | s_t)
    # - If meta-action is option u=A+o:       μ = π_meta(u=A+o | s_t) * π_opt(a_t | s_t,o)
    #   where π_opt is softmax over Q_o(s_t,·)/opt_tau (even if action executed greedily).
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


def _get_lr_from_opt_state(opt_state):
    # Works with: optax.chain(clip, inject_hyperparams(adam)(...))
    # Typically opt_state = (clip_state, injected_adam_state)
    try:
        return opt_state[1].hyperparams["learning_rate"]
    except Exception:
        return None


def _current_lr(agent_state, args):
    lr = _get_lr_from_opt_state(agent_state.enc.opt_state)
    if lr is None:
        # fallback: use grad_steps as the "count" (same convention as CleanRL)
        lr = linear_schedule(agent_state.grad_steps) if args.algo.anneal_lr else args.net.lr
    return lr

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
    opt_q_enc = AtariConvEnc(
        use_batch_norm=False, use_layer_norm=True,
        mlp_dims=args.net.c_dims, mlp_activations=[nn.relu],
    )
    opt_q_head_mod = build_vmap_option_q_heads(n_options=n_options, n_actions=A)

    # -------------------------
    # Laplacian encoder
    # Use the same AtariConvEnc trunk but last dim = args.laplacian.dims and last activation None.
    # We keep BN off here because your PPO networks file doesn't implement BN in ConvNet by default
    # (but you can flip if your implementation supports it).
    # -------------------------
    lap_dims = args.laplacian.dims
    lap_enc = AtariConvEnc(
        use_batch_norm=False, use_layer_norm=True,
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

    opt_q_enc_vars = opt_q_enc.init(k4, obs0, train=False)
    opt_q_head_vars = opt_q_head_mod.init(k5, dummy_emb, train=False)

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

    opt_q_enc_state = TrainState.create(apply_fn=jax.jit(opt_q_enc.apply), params=opt_q_enc_vars["params"], tx=opt_enc_tx)
    opt_q_head_state = TrainState.create(apply_fn=jax.jit(opt_q_head_mod.apply), params=opt_q_head_vars["params"], tx=opt_head_tx)

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
        opt_q_enc=opt_q_enc_state,
        opt_q_head=opt_q_head_state,
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


@jax.jit
def logprob_meta_action(logits_M: jnp.ndarray, u: jnp.ndarray, tau: float = 1.0):
    """
    logits_M: [N,M] or [B,M]
    u:        [N] or [B] int32 (meta-action index in [0..M-1])
    returns log π_meta(u|s) under softmax(logits/tau)
    """
    logp_all = jax.nn.log_softmax(logits_M / jnp.float32(tau), axis=-1)
    idx = jnp.arange(logits_M.shape[0])
    return logp_all[idx, u]

@jax.jit
def logprob_option_action_from_q(q_A: jnp.ndarray, a: jnp.ndarray, tau: float = 1.0):
    """
    q_A: [N,A] option Q-values for the active option at current state
    a:   [N] executed primitive action
    returns log π_opt(a|s,o) where π_opt = softmax(q/tau)
    """
    logp_all = jax.nn.log_softmax(q_A / jnp.float32(tau), axis=-1)
    idx = jnp.arange(q_A.shape[0])
    return logp_all[idx, a]


@partial(jax.jit, static_argnums=range(3, 7))
def flat_action_logprob_from_logits_and_q(
    meta_logits: jnp.ndarray,     # [B, A+O]
    q_opts: jnp.ndarray,          # [O, B, A]
    a: jnp.ndarray,               # [B]
    A: int,
    meta_tau: float = 1.0,
    opt_tau: float = 1.0,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Returns log π_flat(a|s) where:
      π_flat(a|s) = π_meta(a|s) + Σ_o π_meta(A+o|s) π_opt(a|s,o)
    with π_meta = softmax(meta_logits/meta_tau)
         π_opt  = softmax(q_o/opt_tau)
    """
    B = meta_logits.shape[0]
    O = q_opts.shape[0]

    pi_meta = jax.nn.softmax(meta_logits / jnp.float32(meta_tau), axis=-1)  # [B, A+O]
    pi_meta_prim = pi_meta[:, :A]                                           # [B, A]
    pi_meta_opt  = pi_meta[:, A:A+O]                                        # [B, O]

    # π_opt(o, a | s): gather prob of chosen a under each option o
    logp_opt_all = jax.nn.log_softmax(q_opts / jnp.float32(opt_tau), axis=-1)  # [O,B,A]
    logp_opt_a = jnp.take_along_axis(
        logp_opt_all, a[None, :, None], axis=-1
    ).squeeze(-1)                                                             # [O,B]
    p_opt_a = jnp.exp(logp_opt_a).T                                            # [B,O]

    p_meta_a = pi_meta_prim[jnp.arange(B), a]                                  # [B]
    p_mix_a  = jnp.sum(pi_meta_opt * p_opt_a, axis=-1)                         # [B]

    p_flat_a = jnp.clip(p_meta_a + p_mix_a, eps, 1.0)
    return jnp.log(p_flat_a)


# ==============================================================================
# Forward passes: main/meta and option networks
# ==============================================================================
@jax.jit
def encode_main(agent: AgentState, obs_chw: jnp.ndarray, train: bool) -> jnp.ndarray:
    obs = chw_to_nhwc(obs_chw)
    return agent.enc.apply_fn({"params": agent.enc.params}, obs, train=train)


@jax.jit
def meta_logits_and_value(agent: AgentState, obs_chw: jnp.ndarray, train: bool):
    h = encode_main(agent, obs_chw, train=train)
    logits = agent.meta_actor.apply_fn({"params": agent.meta_actor.params}, h, train=train)
    v = agent.meta_critic.apply_fn({"params": agent.meta_critic.params}, h, train=train).squeeze(-1)
    return logits, v


@jax.jit
def option_qs(agent: AgentState, obs_chw: jnp.ndarray, train: bool):
    """
    obs_chw: [B,C,H,W]
    returns:
      q_opts: [O,B,A]
    """
    obs = chw_to_nhwc(obs_chw)
    h = agent.opt_q_enc.apply_fn({"params": agent.opt_q_enc.params}, obs, train=train)   # [B,D]
    q_opts = agent.opt_q_head.apply_fn({"params": agent.opt_q_head.params}, h, train=train)  # [O,B,A]
    return q_opts


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
    n_actions: int,              # A
    gamma: float,
    rho_clip: float,
    c_clip: float,
    eps: float = 1e-12,
):
    """
    V-trace on primitive steps using flat-policy IS:
      π = π_flat(a|s),  μ = μ_flat(a|s)
    where π_flat(a|s) = π_meta(a|s) + Σ_o π_meta(A+o|s) π_opt(a|s,o).
    """
    T, N = prim.actions.shape
    B = T * N
    A = n_actions

    obs_t   = prim.obs
    obs_tp1 = jnp.concatenate([prim.obs[1:], last_obs_chw[None]], axis=0)

    a_TN   = prim.actions.astype(jnp.int32)                             # [T,N]
    mu_TN  = jnp.clip(prim.mu_prim.astype(jnp.float32), eps, 1.0)       # [T,N]
    logmu_TN = jnp.log(mu_TN)                                           # [T,N]

    r_TN = prim.rewards.astype(jnp.float32)                             # [T,N]
    d_TN = prim.dones.astype(jnp.float32)                               # [T,N]

    meta_tau = getattr(args.algo, "meta_tau", 1.0)
    opt_tau  = getattr(args.algo, "opt_tau", 1.0)

    def per_t_step(_carry, inp):
        obsN, obs1N, aN = inp

        # ---- trunk ----
        h = agent.enc.apply_fn({"params": agent.enc.params}, chw_to_nhwc(obsN), train=False)     # [N,D]
        logits_M = agent.meta_actor.apply_fn({"params": agent.meta_actor.params}, h, train=False)  # [N,A+O]
        v = agent.meta_critic.apply_fn({"params": agent.meta_critic.params}, h, train=False).squeeze(-1)  # [N]

        # option Qs for flat π(a|s)
        q_opts = option_qs(agent, obsN, train=False)  # [O,N,A]

        logpi = flat_action_logprob_from_logits_and_q(
            meta_logits=logits_M,
            q_opts=q_opts,
            a=aN,
            A=A,
            meta_tau=meta_tau,
            opt_tau=opt_tau,
            eps=eps,
        )  # [N]

        # bootstrap V(s_{t+1})
        h1 = agent.enc.apply_fn({"params": agent.enc.params}, chw_to_nhwc(obs1N), train=False)
        v1 = agent.meta_critic.apply_fn({"params": agent.meta_critic.params}, h1, train=False).squeeze(-1)

        return _carry, (v, v1, logpi)

    _, (v_TN, v1_TN, logpi_TN) = jax.lax.scan(
        per_t_step, None, (obs_t, obs_tp1, a_TN), length=T
    )

    # IS ratios
    rho_TN = jnp.exp(logpi_TN - logmu_TN)              # [T,N]
    rho_bar_TN = jnp.clip(rho_TN, 0.0, rho_clip)
    c_TN       = jnp.clip(rho_TN, 0.0, c_clip)

    done_last = d_TN[-1]  # [N] float32
    vs_last_N = v1_TN[-1] * (1.0 - done_last)  # zero bootstrap on terminal

    scan_inp = (r_TN, d_TN, v_TN, v1_TN, rho_bar_TN, c_TN)
    _, (vs_TN, adv_TN) = jax.lax.scan(
        lambda carry, x: _vtrace_scan(carry, x, gamma=gamma),
        vs_last_N,
        scan_inp,
        reverse=True,
    )

    batch = {
        "obs": obs_t.reshape((B,) + obs_t.shape[2:]),
        "act_prim": a_TN.reshape((B,)),
        "logmu": logmu_TN.reshape((B,)),
        "v_old": v_TN.reshape((B,)),
        "vs_target": vs_TN.reshape((B,)),
        "adv_target": adv_TN.reshape((B,)),
    }
    stats = (jnp.mean(rho_bar_TN), jnp.mean(c_TN))
    return batch, stats


@partial(jax.jit, static_argnums=range(2, 8))
def update_meta_primitives_offpolicy_minibatch(
    agent: AgentState,
    mb,
    n_actions: int,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    norm_adv: bool,
    use_clip_vloss: bool,
):
    B = mb["act_prim"].shape[0]
    A = n_actions

    adv = mb["adv_target"].astype(jnp.float32)
    if norm_adv:
        adv = (adv - jnp.mean(adv)) / jnp.sqrt(jnp.var(adv) + 1e-8)

    act   = mb["act_prim"].astype(jnp.int32)
    logmu = mb["logmu"].astype(jnp.float32)
    v_targ = mb["vs_target"].astype(jnp.float32)

    meta_tau = getattr(args.algo, "meta_tau", 1.0)
    opt_tau  = getattr(args.algo, "opt_tau", 1.0)

    def loss_fn(enc_p, act_p, crt_p):
        # trunk
        h = agent.enc.apply_fn({"params": enc_p}, chw_to_nhwc(mb["obs"]), train=True)    # [B,D]
        logits_M = agent.meta_actor.apply_fn({"params": act_p}, h, train=True)          # [B,A+O]
        v = agent.meta_critic.apply_fn({"params": crt_p}, h, train=True).squeeze(-1)   # [B]

        # option q for flat π(a|s)
        q_opts = option_qs(agent, mb["obs"], train=True)  # [O,B,A]

        logpi = flat_action_logprob_from_logits_and_q(
            meta_logits=logits_M,
            q_opts=q_opts,
            a=act,
            A=A,
            meta_tau=meta_tau,
            opt_tau=opt_tau,
            eps=1e-12,
        )  # [B]

        # ratio = π/μ (numerically safe)
        logratio = logpi - logmu
        logratio = jnp.clip(logratio, -30.0, 30.0)  # prevents exp overflow / insane ratios
        ratio = jnp.exp(logratio)

        # clipped PG loss
        pg1 = -adv * ratio
        pg2 = -adv * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        pg_loss = jnp.mean(jnp.maximum(pg1, pg2))

        # value loss to V-trace target
        v_old = mb["v_old"]
        v_targ = jax.lax.stop_gradient(mb["vs_target"])
        v_loss_un = 0.5 * (v - v_targ) ** 2
        if use_clip_vloss:
            v_clip = v_old + jnp.clip(v - v_old, -clip_coef, clip_coef)
            v_loss_cl = 0.5 * (v_clip - v_targ) ** 2
            v_loss = jnp.mean(jnp.maximum(v_loss_un, v_loss_cl))
        else:
            v_loss = jnp.mean(v_loss_un)

        # entropy of meta (optional; keep as before)
        logp_all = jax.nn.log_softmax(logits_M / jnp.float32(meta_tau), axis=-1)
        p_all = jnp.exp(logp_all)
        ent_meta = jnp.mean(-(p_all * logp_all).sum(axis=-1))

        # total = pg_loss + vf_coef * v_loss - ent_coef * ent_meta
        total = v_loss
        aux = (pg_loss, v_loss, ent_meta, jnp.mean(ratio))
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
@partial(jax.jit, static_argnums=(4, 5))
def compute_td_lambda_targets(
    next_q: jnp.ndarray,      # [E]
    q_vals: jnp.ndarray,      # [T, E, A]
    rewards: jnp.ndarray,     # [T, E]
    dones: jnp.ndarray,       # [T, E]
    gammas: jnp.ndarray,      # [T, E] or [T]  per-step discounts
    td_lambda: float
) -> jnp.ndarray:            # [T, E]
    # G_{t}^lambda = r_t  --> immediate reward + gamma*[
    #                (1-lambda)V(s_{t+1})  --> one-step bootstrapping with gamma*(1-lambda) weight
    #                + lambda*G_{t+1}^lambda]  --> continuing lambda-return one-step further with gamma*lamda weight

    T, E, A = q_vals.shape
    gammas = jnp.broadcast_to(gammas, (T, E))

    # at t = T - 1
    # last-step return: r_T-1 + γ_T-1 * (1 - done_T-1)* next_q
    g_last = gammas[-1]  # [E] (if gammas is [T], broadcasts)
    # print(rewards.shape, g_last.shape, dones.shape, next_q.shape)
    G_last = rewards[-1] + g_last * (1 - dones[-1]) * next_q
    q_last = jnp.max(q_vals[-1], axis=-1)

    def _step(carry, data):
        # G_{t}^lambda = r_t + gamma*V(s_{t+1})  --> target bootstrap
        #                + gamma*lambda[G_{t+1}^lambda - V(s_{t+1})]  --> gamma*lambda*delta
        G_next, q_next = carry
        r_t, q_t, done_t, g_t = data

        B_t = r_t + g_t * (1 - done_t) * q_next  # (a) one-step bootstrap
        delta_t = G_next - q_next  # (b) “error” between λ-return and 1-step q
        G_t = B_t + g_t * td_lambda * delta_t  # (c) combine
        G_t = (1 - done_t) * G_t + done_t * r_t  # (d) reset when done
        q_prev = jnp.max(q_t, axis=-1)  # (e) update next_q for the next iteration
        return (G_t, q_prev), G_t

    elems = (rewards[:-1], q_vals[:-1], dones[:-1], gammas[:-1])
    (_, _), Gs = jax.lax.scan(_step, (G_last, q_last), elems, reverse=True)

    targets = jnp.concatenate([Gs, G_last[np.newaxis]])
    return targets


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def compute_options_td_lambda_targets(
    agent: AgentState,
    prim: PrimitiveStorage,
    r_int_TNO: jnp.ndarray,   # [T,N,O]
    gamma: float,
    td_lambda: float,
    n_actions: int,
    n_options: int,
):
    """
    Returns:
      targets_TNO: [T,N,O] intrinsic TD(lambda) targets for Q(s,a) per option
    """
    T, N = prim.actions.shape
    O = n_options
    A = n_actions

    # Q(s_t) for all options/actions
    obs_flat = prim.obs.reshape((T * N,) + prim.obs.shape[2:])                  # [B,C,H,W]
    q_flat = option_qs(agent, obs_flat, train=False)                            # [O,B,A]
    q_TNOA = jnp.transpose(q_flat, (1, 0, 2)).reshape((T, N, O, A))             # [T,N,O,A]

    # bootstrap next_q from final next_obs
    last_obs = prim.obs[-1]                                                    # [N,C,H,W]
    q_last = option_qs(agent, last_obs, train=False)                            # [O,N,A]
    next_q_last = jnp.max(q_last, axis=-1)                                      # [O,N]

    dones = prim.dones.astype(jnp.float32)                                      # [T,N]
    rewards = r_int_TNO.astype(jnp.float32)                                     # [T,N,O]

    # Compute per-option TD(λ) targets: vmap over O
    # q_vals_o: [T,N,A], rewards_o: [T,N], next_q_o: [N]
    def one_option(q_vals_o, r_o, next_q_o):
        gammas = jnp.full((T, N), jnp.float32(gamma))
        return compute_td_lambda_targets(
            next_q=next_q_o,
            q_vals=q_vals_o,
            rewards=r_o,
            dones=dones,
            gammas=gammas,
            td_lambda=td_lambda,
        )  # [T,N]

    q_o = jnp.transpose(q_TNOA, (2, 0, 1, 3))        # [O,T,N,A]
    r_o = jnp.transpose(rewards, (2, 0, 1))          # [O,T,N]
    targets_o = jax.vmap(one_option, in_axes=(0,0,0))(q_o, r_o, next_q_last)  # [O,T,N]
    targets_TNO = jnp.transpose(targets_o, (1, 2, 0)) # [T,N,O]
    return targets_TNO

@partial(jax.jit, static_argnums=(2,3,4))
def update_options_pqn_minibatch(
    agent: AgentState,
    mb,                     # dict minibatch
    n_actions: int,
    n_options: int,
    huber_delta: float = 0.0,   # 0 => MSE; set >0 to use Huber if needed
):
    """
    mb:
      obs:     [B,C,H,W]
      act:     [B]
      target:  [B,O]  TD(lambda) targets for executed action, per option
    """
    B = mb["act"].shape[0]
    A = n_actions
    O = n_options
    act = mb["act"].astype(jnp.int32)        # [B]
    targ_BO = mb["target"].astype(jnp.float32)  # [B,O]

    def loss_fn(enc_p, head_p):
        # forward
        obs = chw_to_nhwc(mb["obs"])
        h = agent.opt_q_enc.apply_fn({"params": enc_p}, obs, train=True)            # [B,D]
        q_OBA = agent.opt_q_head.apply_fn({"params": head_p}, h, train=True)        # [O,B,A]

        # gather Q(s,a) for executed primitive action a_t, for all options:
        q_OB = jnp.take_along_axis(q_OBA, act[None, :, None], axis=-1).squeeze(-1)  # [O,B]
        q_BO = jnp.transpose(q_OB, (1, 0))                                          # [B,O]

        err = q_BO - jax.lax.stop_gradient(targ_BO)

        if huber_delta > 0.0:
            abs_e = jnp.abs(err)
            quad = jnp.minimum(abs_e, huber_delta)
            lin = abs_e - quad
            loss = jnp.mean(0.5 * quad**2 + huber_delta * lin)
        else:
            loss = 0.5 * jnp.mean(err**2)

        q_mean = jnp.mean(q_BO)
        return loss, q_mean

    (loss, q_mean), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0,1))(
        agent.opt_q_enc.params, agent.opt_q_head.params
    )
    g_enc, g_head = grads

    agent = agent.replace(
        opt_q_enc=agent.opt_q_enc.apply_gradients(grads=g_enc),
        opt_q_head=agent.opt_q_head.apply_gradients(grads=g_head),
        grad_steps=agent.grad_steps + 1,
    )
    return agent, (loss, q_mean)

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

    key, k_meta = jax.random.split(key, 2)

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
    # (2) Low-level action execution + behavior probability μ_flat(a|s)
    # ------------------------------------------------------------
    is_prim = seg_meta < A
    sel_opt = (seg_meta - A).clip(0, O - 1)

    # Meta logits at current state
    meta_logits, _ = meta_logits_and_value(agent, obs, train=False)  # [N, A+O]

    # Option controller for execution (still greedy for behavior)
    q_opts = option_qs(agent, obs, train=False)                       # [O, N, A]
    q_sel  = q_opts[sel_opt, jnp.arange(N), :]                        # [N, A]
    a_opt  = jnp.argmax(q_sel, axis=-1).astype(jnp.int32)             # [N]

    # Executed primitive action
    action = jnp.where(is_prim, seg_meta, a_opt).astype(jnp.int32)    # [N]

    # Behavior prob of executed primitive action under *flat induced* μ(a|s)
    meta_tau = getattr(args.algo, "meta_tau", 1.0)
    opt_tau  = getattr(args.algo, "opt_tau", 1.0)
    logmu = flat_action_logprob_from_logits_and_q(
        meta_logits=meta_logits,   # [N,A+O]
        q_opts=q_opts,             # [O,N,A]
        a=action,                  # [N]
        A=A,
        meta_tau=meta_tau,
        opt_tau=opt_tau,
        eps=1e-12,
    )                               # [N]
    mu_prim = jnp.exp(logmu).astype(jnp.float32)

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
            “if option executed => mu=1 (greedy intra-option control; deterministic)”
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
    # (3) PQN option update from primitive rollout (intrinsic rewards, off-policy)
    # --------------------------------------------------------------------------
    # Pipeline:
    #   (a) Compute intrinsic rewards r_int[t,n,o] from Laplacian features φ(s)
    #       and normalize variance per option (RunningMeanStd).
    #   (b) Compute intrinsic TD(λ) targets G[t,n,o] using current option Q-values
    #       (no target network) and bootstrap from max_a Q_o(s_{t+1},a).
    #   (c) Flatten (t,n) -> B and do minibatched regression updates on:
    #         Q_o(s_t, a_t)  vs  G[t,n,o]
    #       where a_t is the executed primitive action (from either primitive meta
    #       action or option execution).
    # --------------------------------------------------------------------------
    r_int_TNO, done_TN, new_rms = compute_intrinsic_rewards_TNO(
        agent, prim, n_options=O, bi_dir_opt=bi_dir_opt, eps=1e-8
    )
    agent = agent.replace(int_rms=new_rms)

    # --- compute intrinsic TD(lambda) targets for options ---
    opt_td_lambda = getattr(args.algo, "opt_td_lambda", 0.9)
    targets_TNO = compute_options_td_lambda_targets(
        agent=agent,
        prim=prim,
        r_int_TNO=r_int_TNO,
        # gamma=args.env.gamma,
        gamma=0.99,
        td_lambda=opt_td_lambda,
        n_actions=A,
        n_options=O,
    )  # [T,N,O]

    # flatten dataset
    B = T * N
    opt_batch = {
        "obs": prim.obs.reshape((B,) + prim.obs.shape[2:]),  # [B,C,H,W]
        "act": prim.actions.reshape((B,)),  # [B]
        "target": targets_TNO.reshape((B, O)),  # [B,O]
    }

    # minibatches and epochs (use args.algo.n_opt_mini_batches if you have it, else n_mini_batches)
    n_opt_mini = getattr(args.algo, "n_opt_mini_batches", args.algo.n_mini_batches)

    keyF, sub = jax.random.split(keyF)
    opt_mbs = make_minibatches(sub, n_opt_mini, opt_batch)

    def opt_one_epoch(carry, _):
        ag = carry

        def opt_one_mb(ag2, mb):
            ag2, st = update_options_pqn_minibatch(
                ag2, mb,
                n_actions=A,
                n_options=O,
                huber_delta=getattr(args.algo, "opt_huber_delta", 0.0),
            )
            return ag2, st

        ag, st = jax.lax.scan(opt_one_mb, ag, opt_mbs)
        return ag, st

    opt_epochs = getattr(args.algo, "opt_upd_epochs", args.algo.upd_epochs)
    agent, opt_stats = jax.lax.scan(opt_one_epoch, agent, None, length=opt_epochs)
    opt_loss, opt_qmean = [x.mean() for x in opt_stats]


    # --------------------------------------------------------------------------
    # (4) Meta-primitives update (extrinsic) on ALL primitive steps (off-policy)
    #     - compute V-trace targets/advantages on ordered rollout
    #     - then minibatch SGD with PPO-style ratio clipping against stored logpi_old
    # --------------------------------------------------------------------------
    prim_targets, (prim_rho_mean, prim_c_mean) = compute_meta_prim_vtrace_targets(
        agent=agent,
        prim=prim,
        last_obs_chw=obsF,
        n_actions=A,
        gamma=args.env.gamma,
        rho_clip=1.0,
        c_clip=1.0,
    )

    keyF, sub = jax.random.split(keyF)
    prim_mbs = make_minibatches(sub, args.algo.n_mini_batches, prim_targets)

    def prim_one_epoch(carry, _):
        ag = carry
        def prim_one_mb(ag2, mb):
            ag2, st = update_meta_primitives_offpolicy_minibatch(
                ag2, mb,
                n_actions=A,
                clip_coef=args.algo.clip_coef,
                use_clip_vloss=args.algo.clip_vloss,
                ent_coef=args.algo.ent_coef,
                vf_coef=args.algo.vf_coef,
                norm_adv=args.algo.norm_adv,
            )
            return ag2, st
        ag, st = jax.lax.scan(prim_one_mb, ag, prim_mbs)
        return ag, st

    agent, prim_stats = jax.lax.scan(prim_one_epoch, agent, None, length=args.algo.upd_epochs)
    prim_loss, prim_pg, prim_v, prim_ent, prim_ratio = [x.mean() for x in prim_stats]
    # prim_loss = prim_pg = prim_v = prim_ent = prim_ratio = prim_rho_mean = prim_c_mean = jnp.nan

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
    lr = _current_lr(agent, args)
    mean_ptr = jnp.mean(meta.ptr.astype(jnp.float32))
    mean_dur = _wm(valid.reshape(-1).astype(jnp.float32), meta.durations.reshape(-1).astype(jnp.float32))
    opt_rate = _wm(valid.reshape(-1).astype(jnp.float32), (meta.actions.reshape(-1) >= A).astype(jnp.float32))

    metrics = {
        "env_step": agent.timesteps,
        "update_steps": agent.n_updates,
        "grad_steps": agent.grad_steps,
        "learning_rate": lr,

        "buff/mean_ptr": mean_ptr,
        "buff/mean_dur": mean_dur,
        "buff/opt_rate": opt_rate,

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
        "opt_pqn/td_loss": opt_loss,
        "opt_pqn/q_mean": opt_qmean,
        "opt_pqn/int_rms_var_mean": jnp.mean(agent.int_rms.var),

        # durations
        "opt/dur_min": jnp.min(opt_dur_table),
        "opt/dur_max": jnp.max(opt_dur_table),
        "opt/dur_mean": jnp.mean(opt_dur_table.astype(jnp.float32)),
    }

    # envpool infos
    metrics.update({k: v.mean() for k, v in infos.items()})
    metrics.update({"Max_Return": infos["returned_episode_returns"].max()})

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
    q_opts = option_qs(agent, obs_chw, train=False)   # [O,N,A]
    N = obs_chw.shape[0]
    q_sel = q_opts[opt_idx, jnp.arange(N), :]         # [N,A]
    return jnp.argmax(q_sel, axis=-1)


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
