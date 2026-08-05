import os
import time
import random
import traceback
import numpy as np

from functools import partial
from typing import Any, Union, List, NamedTuple

from src.config.procgen.pqn_laplacian import *
args = convert_args_to_dict(parse_args())

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import flax
import chex
import optax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from src.utils.procgen_wrappers import JaxLogProcgenEnvPoolWrapper
from src.mains.procgen.pqn_procgen_utils import (
    Enc_State,
    FFN_State,
    make_procgen_env,
    build_enc_state,
    build_ffn_state,
    build_lr_schedule,
    get_dual_q_values_procgen,
    procgen_rnd_frame,
    maybe_log_eval_metrics,
    make_wandb_log_callback,
    make_procgen_video_env,
    record_procgen_policy_and_option_videos,
)

from src.utils.jax_utils import qema_update
from src.networks.networks import Conv_Encoder
from src.mains.algorithms.RND import (
    RNDState, rnd_setup, rnd_intrinsic_reward,
    rnd_update_obs_rms, rnd_update_rew_rms, rnd_update_once_per_epoch
)
from src.mains.algorithms.PQN import (
    value_rescale, inv_value_rescale,
    compute_td_lambda_targets, compute_transformed_td_lambda_targets
)
from src.mains.algorithms.EO import (
    eigenvalues_to_option_durations_skip_first,
    update_option_networks_shared_dual,
)
from src.mains.algorithms.HEO import (
    MetaRingBuffer, unwrap_meta_ring, init_meta_buffer, get_meta_dual_q_seq,
    compute_meta_retrace_targets_dual, compute_meta_primitive_targets,
    smdp_learn_phase_dual, primitive_learn_phase_dual
)
from src.mains.algorithms.Laplacian import update_neo_laplacian_network
from src.mains.algorithms.NEO import (
    NEOScaleState, neo_scale_from_state_calibrated,
    compute_neo_options_lambda_targets_shared_dual,
)


@chex.dataclass(frozen=True)
class Prim_Transition:
    obs: chex.Array
    actions: chex.Array
    next_obs: chex.Array
    dones: chex.Array

    rewards: chex.Array
    int_rewards: chex.Array

    meta_action: chex.Array
    meta_q_ext_vals: chex.Array
    meta_q_int_vals: chex.Array

    option_q_ext_vals: chex.Array   # [E, O, A]
    option_q_int_vals: chex.Array   # [E, O, A]

    action_probs: chex.Array

    rnd_frame: chex.Array           # [E,H,W,3]
    rnd_bonus: chex.Array
    rnd_bonus_n: chex.Array
    inv_count: chex.Array


@chex.dataclass(frozen=True)
class Meta_Transition:
    obs: chex.Array
    actions: chex.Array
    ext_rewards: chex.Array
    int_rewards: chex.Array
    dones: chex.Array
    next_obs: chex.Array
    ext_discounts: chex.Array
    int_discounts: chex.Array
    behav_probs: chex.Array
    valid: chex.Array


class FlatStorage(struct.PyTreeNode):
    rnd_frame: chex.Array         # [B,H,W,3]


@struct.dataclass
class Lap_State(TrainState):
    betas: Any
    errors: Any
    quadratic_errors: Any
    barrier_coeffs: Any
    eigen_values: Any
    batch_stats: Union[FrozenDict, None]


class LapBatch(NamedTuple):
    obs: jnp.ndarray
    dones: jnp.ndarray
    actions: jnp.ndarray
    action_probs: jnp.ndarray
    counts: jnp.ndarray
    novelty: jnp.ndarray


@struct.dataclass
class RNDEpsState:
    # EMA percentiles used to map raw RND novelty -> base epsilon
    p5_ema: jnp.ndarray
    p95_ema: jnp.ndarray

    # Multiplicative controller on top of RND-based epsilon:
    #   eps_final = clip(lambda_scale * eps_rnd_base, eps_min, 1.0)
    lambda_scale: jnp.ndarray

    # Exponential moving average of realized exploration rate:
    #   eps_defacto_t = beta * eps_defacto_{t-1} + (1-beta) * realized_explore_t
    defacto_eps_ema: jnp.ndarray


@chex.dataclass(frozen=True)
class ExploreAux:
    # Actual epsilon used for action selection at each step/env
    eps_used: chex.Array          # [E]

    # Vanilla scheduled epsilon at this update
    eps_vanilla: chex.Array       # [E]

    # Base epsilon derived from RND novelty before controller scaling
    eps_rnd_base: chex.Array      # [E]

    # Indicator: did this env actually explore at this step?
    took_explore: chex.Array      # [E] bool


@struct.dataclass
class Train_State:
    enc_state: Enc_State
    q_ext_state: FFN_State
    q_int_state: FFN_State

    lap_state: Lap_State

    opt_enc_state: Enc_State
    opt_q_ext_states: List[FFN_State]
    opt_q_int_states: List[FFN_State]

    rnd_state: RNDState
    neo_state: NEOScaleState
    rnd_eps_state: RNDEpsState

    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def setup(key, args, obs, n_prim_actions):
    n_total_options = args.options.n_options
    assert not args.options.bidir_options
    n_meta_actions = n_prim_actions + n_total_options

    k_enc, k_qext, k_qint, k_opt_enc, k_lap, k_rnd, k_opt = jax.random.split(key, 7)

    # ------------------------------------------------------------------
    # main encoder / q heads
    # ------------------------------------------------------------------
    enc_state = build_enc_state(k_enc, args, obs, name="PQN:Enc")
    q_ext_state = build_ffn_state(k_qext, args, n_meta_actions, name="PQN:MetaQ_ext")
    q_int_state = build_ffn_state(k_qint, args, n_meta_actions, name="PQN:MetaQ_int")

    # ------------------------------------------------------------------
    # shared option encoder
    # ------------------------------------------------------------------
    opt_enc_state = build_enc_state(k_opt_enc, args, obs, name="Opt:Enc")

    # ------------------------------------------------------------------
    # per-option dual primitive q heads
    # ------------------------------------------------------------------
    opt_q_ext_states = []
    opt_q_int_states = []
    key_opt = k_opt
    for _ in range(n_total_options):
        key_opt, kqe, kqi = jax.random.split(key_opt, 3)
        opt_q_ext_states.append(build_ffn_state(kqe, args, n_prim_actions, name="Opt:Q_ext"))
        opt_q_int_states.append(build_ffn_state(kqi, args, n_prim_actions, name="Opt:Q_int"))

    # ------------------------------------------------------------------
    # laplacian encoder
    # ------------------------------------------------------------------
    obs0_hwc = np.transpose(obs[0], (1, 2, 0))[None]

    lap_network = Conv_Encoder(
        name="Lap:Enc",
        layer_dims=list(args.net.c_dims) + [args.laplacian.dims],
        activations=[nn.relu] * len(args.net.c_dims) + [None],
        min_scale=1,
        width_mul=1,
        impala_cnn=False,
        batch_norm=args.laplacian.batch_norm,
        layer_norm=args.laplacian.layer_norm,
    )

    lap_variables = lap_network.init(k_lap, obs0_hwc, train=False)
    lap_lr_schedule = build_lr_schedule(args)
    lap_optimizer = optax.chain(
        optax.clip_by_global_norm(args.laplacian.max_grad_norm),
        optax.adamw(learning_rate=lap_lr_schedule, eps=1.5e-4),
    )
    lap_batch_stats = lap_variables["batch_stats"] if args.laplacian.batch_norm else None
    lap_state = Lap_State.create(
        apply_fn=lap_network.apply,
        tx=lap_optimizer,
        params=lap_variables["params"],
        batch_stats=lap_batch_stats,
        barrier_coeffs=jnp.tril(jnp.ones((1, 1)), k=0) * args.laplacian.barrier.value,
        betas=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
        eigen_values=jnp.ones((args.laplacian.dims,), dtype=jnp.float32) * 1e-3,
        errors=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
        quadratic_errors=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
    )

    # ------------------------------------------------------------------
    # rnd
    # ------------------------------------------------------------------
    obs_shape = (obs.shape[2], obs.shape[3], obs.shape[1])
    rnd_state = rnd_setup(
        key=k_rnd,
        lr=args.net.lr,
        max_grad_norm=args.net.max_grad_norm,
        n_envs=args.env.n_envs,
        int_gamma=args.rnd.int_gamma,
        obs_shape=obs_shape,
    )

    neo_state = NEOScaleState(
        max_inv_ema=jnp.array(1.0, dtype=jnp.float32),
        mean_scale_ema=jnp.array(1.0, dtype=jnp.float32),
    )
    rnd_eps_state = RNDEpsState(
        p5_ema=jnp.array(0.0, dtype=jnp.float32),
        p95_ema=jnp.array(0.1, dtype=jnp.float32),

        # lambda starts neutral: eps_final ~= eps_rnd_base initially
        lambda_scale=jnp.array(1.0, dtype=jnp.float32),

        # start defacto EMA at vanilla eps_start so there is no artificial
        # initial mismatch at the beginning of training
        defacto_eps_ema=jnp.array(args.algo.eps.start, dtype=jnp.float32),
    )

    return Train_State(
        enc_state=enc_state,
        q_ext_state=q_ext_state,
        q_int_state=q_int_state,
        lap_state=lap_state,
        opt_enc_state=opt_enc_state,
        opt_q_ext_states=opt_q_ext_states,
        opt_q_int_states=opt_q_int_states,
        rnd_state=rnd_state,
        neo_state=neo_state,
        rnd_eps_state=rnd_eps_state,
    ), build_lr_schedule(args)


@partial(jax.jit, static_argnums=(2,))
def select_option_action(
    q_opt_sel,                 # [E, A]
    key,                       # PRNGKey
    option_policy_greedy: bool = True,
    option_policy_temp: float = 1.0,
):
    """
    Select primitive actions from an option policy.

    Args:
        q_opt_sel:
            Selected option Q-values for the current batch of environments.
            Shape [E, A].
        key:
            PRNG key.
        option_policy_greedy:
            If True, use argmax.
            If False, sample from softmax(q / temp).
        option_policy_temp:
            Softmax temperature when option_policy_greedy=False.

    Returns:
        act:  [E]      selected primitive action
        prob: [E]      probability of the selected action under the option policy
        dist: [E, A]   full option action distribution used for selection
    """
    if option_policy_greedy:
        act = jnp.argmax(q_opt_sel, axis=-1)
        dist = jax.nn.one_hot(act, q_opt_sel.shape[-1], dtype=jnp.float32)
        prob = jnp.ones((q_opt_sel.shape[0],), dtype=jnp.float32)
    else:
        temp = jnp.maximum(jnp.float32(option_policy_temp), 1e-6)
        logits = q_opt_sel / temp
        dist = jax.nn.softmax(logits, axis=-1)
        act = jax.random.categorical(key, logits, axis=-1)
        prob = jnp.take_along_axis(dist, act[:, None], axis=-1).squeeze(-1)
        prob = jnp.clip(prob, 1e-8, 1.0)

    return act, prob, dist


@partial(jax.jit, static_argnums=(3,))
def select_option_action_mixed(
    q_opt_sel,           # [E, A]
    key,                 # PRNGKey
    stochastic_mask,     # [E] bool ; True => sample softmax, False => argmax
    option_policy_temp: float = 1.0,
):
    """
    Mixed low-level option execution:
      - greedy for envs where stochastic_mask=False
      - softmax-sampled for envs where stochastic_mask=True
    """
    temp = jnp.maximum(jnp.float32(option_policy_temp), 1e-6)
    logits = q_opt_sel / temp
    dist = jax.nn.softmax(logits, axis=-1)                     # [E,A]

    greedy_act = jnp.argmax(q_opt_sel, axis=-1)                # [E]
    samp_act = jax.random.categorical(key, logits, axis=-1)    # [E]

    act = jnp.where(stochastic_mask, samp_act, greedy_act)     # [E]

    prob = jnp.take_along_axis(dist, act[:, None], axis=-1).squeeze(-1)
    prob = jnp.where(
        stochastic_mask,
        jnp.clip(prob, 1e-8, 1.0),
        jnp.ones_like(prob, dtype=jnp.float32),
    )

    return act, prob, dist


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def compute_option_dual_qs(
    obs,
    opt_enc_state,
    opt_q_ext_states,
    opt_q_int_states,
    use_enc_bn,
    use_q_bn,
    opt_int_coef,
    use_value_rescale,
):
    """
    obs: [E,C,H,W]
    returns:
        q_ext: [O,E,A]
        q_int: [O,E,A]
        q_sel: [O,E,A]
    """
    obs = jnp.transpose(obs, (0, 2, 3, 1))  # [E,H,W,C]

    enc_vars = {"params": opt_enc_state.params}
    if use_enc_bn:
        enc_vars["batch_stats"] = opt_enc_state.batch_stats
    emb = opt_enc_state.apply_fn(enc_vars, obs, train=False)  # [E,D]

    qext_params = jax.tree_util.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in opt_q_ext_states])
    qint_params = jax.tree_util.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in opt_q_int_states])

    qext_bs = (
        jax.tree_util.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in opt_q_ext_states])
        if use_q_bn else None
    )
    qint_bs = (
        jax.tree_util.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in opt_q_int_states])
        if use_q_bn else None
    )

    qext_apply = opt_q_ext_states[0].apply_fn
    qint_apply = opt_q_int_states[0].apply_fn

    def qext_one(p_q, bs_q):
        vars_q = {"params": p_q}
        if use_q_bn:
            vars_q["batch_stats"] = bs_q
        return qext_apply(vars_q, emb, train=False)  # [E,A]

    def qint_one(p_q, bs_q):
        vars_q = {"params": p_q}
        if use_q_bn:
            vars_q["batch_stats"] = bs_q
        return qint_apply(vars_q, emb, train=False)  # [E,A]

    q_ext = jax.vmap(qext_one, in_axes=(0, 0 if use_q_bn else None))(qext_params, qext_bs)
    q_int = jax.vmap(qint_one, in_axes=(0, 0 if use_q_bn else None))(qint_params, qint_bs)

    if use_value_rescale:
        q_sel_raw = inv_value_rescale(q_ext) + opt_int_coef * inv_value_rescale(q_int)
        q_sel = value_rescale(q_sel_raw)
    else:
        q_sel = q_ext + opt_int_coef * q_int

    return q_ext, q_int, q_sel


@jax.jit
def update_rnd_eps_percentile_ema(
    rnd_eps_state,
    novelty_vals,          # arbitrary shape, flattened internally
    tau=1e-3,
):
    x = novelty_vals.reshape(-1).astype(jnp.float32)

    p5 = jnp.percentile(x, 5.0)
    p95 = jnp.percentile(x, 95.0)

    new_p5 = (1.0 - tau) * rnd_eps_state.p5_ema + tau * p5
    new_p95 = (1.0 - tau) * rnd_eps_state.p95_ema + tau * p95

    # keep ordering valid
    new_p95 = jnp.maximum(new_p95, new_p5 + 1e-6)

    return rnd_eps_state.replace(
        p5_ema=new_p5,
        p95_ema=new_p95,
    )


@jax.jit
def novelty_to_linear_epsilon(
    novelty,               # [E]
    n_min,                 # scalar
    n_max,                 # scalar
    eps_min=0.01,
    eps_max=0.2,
):
    """
    x = clip((n - n_min) / (n_max - n_min), 0, 1)
    eps = eps_min + (eps_max - eps_min) * x
    """
    novelty = novelty.astype(jnp.float32)
    n_min = jnp.float32(n_min)
    n_max = jnp.float32(n_max)

    x = (novelty - n_min) / (n_max - n_min + 1e-8)
    x = jnp.clip(x, 0.0, 1.0)

    eps = eps_min + (eps_max - eps_min) * x
    return jnp.clip(eps, eps_min, eps_max)


@jax.jit
def apply_rnd_ema_controller(
    eps_rnd_base,        # [E]
    lambda_scale,        # scalar
    eps_min=0.01,
):
    """
    Controlled RND epsilon:
        eps_final(s) = clip(lambda_scale * eps_rnd_base(s), eps_min, 1.0)

    - eps_rnd_base(s) is the novelty-derived epsilon
    - lambda_scale is adjusted online so that the realized exploration rate
      matches the vanilla epsilon schedule on average
    """
    eps_final = lambda_scale * eps_rnd_base
    eps_final = jnp.clip(eps_final, eps_min, 1.0)
    return eps_final


@jax.jit
def update_rnd_eps_ema_controller(
    rnd_eps_state,
    eps_vanilla,            # scalar
    took_explore,           # arbitrary bool array, flattened internally
    ema_beta=0.99,
    controller_lr=5.0,
    lambda_min=0.25,
    lambda_max=4.0,
):
    """
    EMA control law.

    Realized exploration rate over the rollout:
        eps_realized = mean(1{explore})

    EMA of realized exploration:
        eps_defacto_t = beta * eps_defacto_{t-1} + (1-beta) * eps_realized

    Control error:
        error_t = eps_vanilla_t - eps_defacto_t

    Multiplicative controller:
        lambda_{t+1} = clip(
            lambda_t * exp(controller_lr * error_t),
            lambda_min, lambda_max
        )

    Interpretation:
      - if defacto < vanilla: error > 0 -> increase lambda -> more exploration
      - if defacto > vanilla: error < 0 -> decrease lambda -> less exploration
    """
    eps_realized = jnp.mean(took_explore.astype(jnp.float32))
    eps_defacto = (
        ema_beta * rnd_eps_state.defacto_eps_ema
        + (1.0 - ema_beta) * eps_realized
    )

    error = jnp.float32(eps_vanilla) - eps_defacto

    lambda_new = rnd_eps_state.lambda_scale * jnp.exp(controller_lr * error)
    lambda_new = jnp.clip(lambda_new, lambda_min, lambda_max)

    return rnd_eps_state.replace(
        lambda_scale=lambda_new,
        defacto_eps_ema=eps_defacto,
    )


@jax.jit
def combine_sched_and_rnd_epsilon(
    eps_sched,            # scalar
    eps_rnd_vec,          # [E]
):
    """
    Final epsilon is the larger of:
      - scheduled epsilon baseline
      - state-dependent RND epsilon
    """
    eps_sched_vec = jnp.full_like(eps_rnd_vec, jnp.float32(eps_sched))
    return jnp.maximum(eps_sched_vec, eps_rnd_vec)


@partial(jax.jit, static_argnums=range(12, 29))
def env_take_steps(
    carry, _unused,
    enc_state, q_ext_state, q_int_state,
    opt_enc_state, opt_q_ext_states, opt_q_int_states,
    rnd_state, rnd_eps_state, n_updates, opt_dur_table,
    env, eps_scheduler, mu_scheduler, gamma_ext, gamma_int, n_envs,
    use_main_bn, use_q_bn, n_total_options, int_coef, opt_int_coef, use_value_rescale,
    option_policy_greedy=True,   # unused for training-time option execution in this variant
    option_policy_temp=1.0,
    use_rnd_eps=False,
    rnd_eps_min=0.01,
    rnd_eps_max=0.2,
):
    (
        start_obs, start_meta, start_q_ext, start_q_int, start_ap,
        last_obs, env_state, last_meta, last_term, opt_count, opt_stoch_flag,
        cum_ext_r, cum_int_r, cum_gamma_ext, cum_gamma_int,
        meta_buf, key
    ) = carry

    E = n_envs
    key, k_u1, k_u2, k_opt, k_prim = jax.random.split(key, 5)

    sample_new = last_term

    q_ext, q_int = get_dual_q_values_procgen(
        last_obs, enc_state, q_ext_state, q_int_state,
        use_main_bn, use_q_bn, use_q_bn
    )

    A_meta = q_ext.shape[-1]
    A = A_meta - n_total_options
    O = n_total_options

    if use_value_rescale:
        q_sel_raw = inv_value_rescale(q_ext) + int_coef * inv_value_rescale(q_int)
        q_sel = value_rescale(q_sel_raw)
    else:
        q_sel = q_ext + int_coef * q_int

    greedy_meta = jnp.argmax(q_sel, axis=-1)

    mu_val = mu_scheduler(n_updates)
    eps_scalar = jnp.float32(eps_scheduler(n_updates))
    eps_vanilla_vec = jnp.full((E,), eps_scalar, dtype=jnp.float32)

    if use_rnd_eps:
        curr_rnd_frame = procgen_rnd_frame(last_obs)
        curr_novelty = rnd_intrinsic_reward(rnd_state, curr_rnd_frame)

        # Map current novelty to a state-dependent base epsilon in
        # [rnd_eps_min, rnd_eps_max].
        eps_rnd_base = novelty_to_linear_epsilon(
            novelty=curr_novelty,
            n_min=rnd_eps_state.p5_ema,
            n_max=rnd_eps_state.p95_ema,
            eps_min=rnd_eps_min,
            eps_max=rnd_eps_max,
        )

        # RND-controlled proposal:
        #   eps_rnd_ctrl(s) = clip(lambda * eps_rnd_base(s), rnd_eps_min, 1)
        eps_rnd_ctrl = apply_rnd_ema_controller(
            eps_rnd_base=eps_rnd_base,
            lambda_scale=rnd_eps_state.lambda_scale,
            eps_min=rnd_eps_min,
        )

        # Actual epsilon used by the policy:
        #   eps_used(s) = max(eps_vanilla, eps_rnd_ctrl(s))
        #
        # This preserves vanilla high exploration early in training, while
        # allowing novelty to selectively increase exploration later on.
        # eps_vec = jnp.maximum(eps_vanilla_vec, eps_rnd_ctrl)
        eps_vec = eps_rnd_ctrl
    else:
        eps_rnd_base = eps_vanilla_vec
        eps_vec = eps_vanilla_vec

    u1 = jax.random.uniform(k_u1, (E,))
    u2 = jax.random.uniform(k_u2, (E,))

    took_explore = (u1 < eps_vec)
    took_opt_branch = (u2 < mu_val)

    rand_opt = jax.random.randint(k_opt, (E,), A, A + O)
    rand_prim = jax.random.randint(k_prim, (E,), 0, A)

    choice = jnp.where(
        took_explore,
        jnp.where(took_opt_branch, rand_opt, rand_prim),
        greedy_meta,
    )

    new_opt_stoch_flag = took_explore & took_opt_branch & (choice >= A)
    opt_stoch_flag = jnp.where(sample_new, new_opt_stoch_flag, opt_stoch_flag)

    # exact old meta behavior probability computation, but with eps_vec
    p_opt_pick = eps_vec * mu_val
    p_prim_pick = eps_vec * (1.0 - mu_val)

    base_prim = p_prim_pick / A
    base_opt = p_opt_pick / O

    base = jnp.concatenate([
        jnp.full((E, A), 0.0, dtype=jnp.float32) + base_prim[:, None],
        jnp.full((E, O), 0.0, dtype=jnp.float32) + base_opt[:, None],
    ], axis=-1)

    soft = jax.nn.softmax(q_sel, axis=-1)
    best = (choice[:, None] == jnp.arange(A_meta)[None, :]).astype(jnp.float32)
    action_probs = base + (1.0 - eps_vec)[:, None] * soft * best
    behav_probs = action_probs[jnp.arange(E), choice]

    start_ap = jnp.where(sample_new, behav_probs, start_ap)

    curr_meta = jnp.where(sample_new, choice, last_meta)
    start_obs = jnp.where(sample_new[:, None, None, None], last_obs, start_obs)
    start_meta = jnp.where(sample_new, choice, start_meta)
    start_q_ext = jnp.where(sample_new[:, None], q_ext, start_q_ext)
    start_q_int = jnp.where(sample_new[:, None], q_int, start_q_int)

    q_opts_ext, q_opts_int, q_opts_sel = compute_option_dual_qs(
        last_obs,
        opt_enc_state,
        opt_q_ext_states,
        opt_q_int_states,
        use_main_bn,
        use_q_bn,
        opt_int_coef,
        use_value_rescale,
    )

    is_prim = curr_meta < A
    sel_opt = (curr_meta - A).clip(0, O - 1)

    q_opt_curr = q_opts_sel[sel_opt, jnp.arange(E), :]

    key, k_opt_act = jax.random.split(key)

    if option_policy_greedy:
        opt_act, opt_act_prob, _opt_dist = select_option_action(
            q_opt_curr,
            k_opt_act,
            option_policy_greedy=True,
            option_policy_temp=option_policy_temp,
        )
    else:
        option_exec_stochastic = (~is_prim) & opt_stoch_flag
        opt_act, opt_act_prob, _opt_dist = select_option_action_mixed(
            q_opt_curr,
            k_opt_act,
            stochastic_mask=option_exec_stochastic,
            option_policy_temp=option_policy_temp,
        )

    final_act = jnp.where(is_prim, curr_meta, opt_act)

    new_obs, new_env_state, ext_reward, new_done, info = env.step(env_state, final_act)

    rnd_frame = procgen_rnd_frame(new_obs)
    raw_rnd_bonus = rnd_intrinsic_reward(rnd_state, rnd_frame)

    opt_count = jnp.where(last_term, 1, opt_count + 1)
    safe_opt_idx = jnp.clip(sel_opt, 0, O - 1)
    option_dur = opt_dur_table[safe_opt_idx]

    curr_term = is_prim | (opt_count >= option_dur) | new_done
    next_opt_stoch_flag = jnp.where(curr_term, False, opt_stoch_flag)

    cum_ext_r = cum_ext_r + cum_gamma_ext * ext_reward
    cum_gamma_ext = cum_gamma_ext * gamma_ext

    cum_int_r = cum_int_r + cum_gamma_int * raw_rnd_bonus
    cum_gamma_int = cum_gamma_int * gamma_int

    meta_ext_reward = cum_ext_r
    meta_int_reward = cum_int_r
    meta_ext_discount = cum_gamma_ext
    meta_int_discount = cum_gamma_int

    cum_ext_r = jnp.where(curr_term, 0.0, cum_ext_r)
    cum_int_r = jnp.where(curr_term, 0.0, cum_int_r)
    cum_gamma_ext = jnp.where(curr_term, 1.0, cum_gamma_ext)
    cum_gamma_int = jnp.where(curr_term, 1.0, cum_gamma_int)

    ptr = meta_buf.ptr
    env_idx = jnp.arange(E)
    mask_i = curr_term.astype(jnp.int32)

    ext_rewards_b = meta_buf.ext_rewards
    int_rewards_b = meta_buf.int_rewards
    ext_discounts_b = meta_buf.ext_discounts
    int_discounts_b = meta_buf.int_discounts
    dones_b = meta_buf.dones
    start_obs_b = meta_buf.start_obs
    term_obs_b = meta_buf.term_obs
    actions_b = meta_buf.actions
    durations_b = meta_buf.durations
    behav_probs_b = meta_buf.behav_probs

    ext_rewards_b = ext_rewards_b.at[ptr, env_idx].set(jnp.where(mask_i, meta_ext_reward, ext_rewards_b[ptr, env_idx]))
    int_rewards_b = int_rewards_b.at[ptr, env_idx].set(jnp.where(mask_i, meta_int_reward, int_rewards_b[ptr, env_idx]))
    ext_discounts_b = ext_discounts_b.at[ptr, env_idx].set(jnp.where(mask_i, meta_ext_discount, ext_discounts_b[ptr, env_idx]))
    int_discounts_b = int_discounts_b.at[ptr, env_idx].set(jnp.where(mask_i, meta_int_discount, int_discounts_b[ptr, env_idx]))
    dones_b = dones_b.at[ptr, env_idx].set(jnp.where(mask_i, new_done.astype(jnp.int32), dones_b[ptr, env_idx]))
    start_obs_b = start_obs_b.at[ptr, env_idx].set(
        jnp.where(mask_i[:, None, None, None], start_obs, start_obs_b[ptr, env_idx])
    )
    term_obs_b = term_obs_b.at[ptr, env_idx].set(
        jnp.where(mask_i[:, None, None, None], new_obs, term_obs_b[ptr, env_idx])
    )
    actions_b = actions_b.at[ptr, env_idx].set(jnp.where(mask_i, start_meta, actions_b[ptr, env_idx]))
    durations_b = durations_b.at[ptr, env_idx].set(jnp.where(mask_i, opt_count, durations_b[ptr, env_idx]))
    behav_probs_b = behav_probs_b.at[ptr, env_idx].set(jnp.where(mask_i, start_ap, behav_probs_b[ptr, env_idx]))

    T = meta_buf.ext_rewards.shape[0]
    new_ptr = (ptr + mask_i) % T
    new_len = jnp.minimum(meta_buf.length + mask_i, T)
    overflow = (meta_buf.length == T) & (mask_i == 1)
    new_head = jnp.where(overflow, (meta_buf.head + 1) % T, meta_buf.head)

    meta_buf = MetaRingBuffer(
        ptr=new_ptr,
        head=new_head,
        length=new_len,
        start_obs=start_obs_b,
        term_obs=term_obs_b,
        actions=actions_b,
        ext_rewards=ext_rewards_b,
        int_rewards=int_rewards_b,
        ext_discounts=ext_discounts_b,
        int_discounts=int_discounts_b,
        dones=dones_b,
        durations=durations_b,
        behav_probs=behav_probs_b,
    )

    new_carry = (
        start_obs, start_meta, start_q_ext, start_q_int, start_ap,
        new_obs, new_env_state, curr_meta, curr_term, opt_count, next_opt_stoch_flag,
        cum_ext_r, cum_int_r, cum_gamma_ext, cum_gamma_int,
        meta_buf, key
    )

    prim_trans = Prim_Transition(
        obs=last_obs,
        actions=final_act,
        next_obs=new_obs,
        dones=new_done,
        rewards=ext_reward,
        int_rewards=jnp.zeros_like(raw_rnd_bonus),
        meta_action=curr_meta,
        meta_q_ext_vals=q_ext,
        meta_q_int_vals=q_int,
        option_q_ext_vals=jnp.transpose(q_opts_ext, (1, 0, 2)),
        option_q_int_vals=jnp.transpose(q_opts_int, (1, 0, 2)),
        action_probs=action_probs[jnp.arange(E), choice],
        rnd_frame=rnd_frame,
        rnd_bonus=raw_rnd_bonus,
        rnd_bonus_n=jnp.zeros_like(raw_rnd_bonus),
        inv_count=jnp.zeros_like(raw_rnd_bonus),
    )
    explore_aux = ExploreAux(
        eps_used=eps_vec,
        eps_vanilla=eps_vanilla_vec,
        eps_rnd_base=eps_rnd_base,
        took_explore=took_explore,
    )

    return new_carry, (prim_trans, info, explore_aux)


@partial(jax.jit, static_argnums=range(2, 7))
def update_step(runner_state, _unused, args, env, eps_scheduler, mu_scheduler, start_time):
    train_state, expl_state, key, opt_dur_table = runner_state

    n_total_options = n_base_options = args.options.n_options
    n_prim_actions = env.single_action_space.n
    n_meta_actions = n_prim_actions + n_total_options
    n_envs = args.env.n_envs
    opt_int_coef = args.neo.rnd_coef

    key, b_key = jax.random.split(key)

    step_fn_env = partial(
        env_take_steps,
        enc_state=train_state.enc_state,
        q_ext_state=train_state.q_ext_state,
        q_int_state=train_state.q_int_state,
        opt_enc_state=train_state.opt_enc_state,
        opt_q_ext_states=train_state.opt_q_ext_states,
        opt_q_int_states=train_state.opt_q_int_states,
        rnd_state=train_state.rnd_state,
        rnd_eps_state=train_state.rnd_eps_state,
        n_updates=train_state.n_updates,
        opt_dur_table=opt_dur_table,
        env=env,
        eps_scheduler=eps_scheduler,
        mu_scheduler=mu_scheduler,
        gamma_ext=args.env.gamma,
        gamma_int=args.rnd.int_gamma,
        n_envs=n_envs,
        use_main_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        n_total_options=n_total_options,
        int_coef=args.algo.int_coef,
        opt_int_coef=args.neo.rnd_coef,
        use_value_rescale=not args.algo.rew_scale,
        option_policy_greedy=args.options.policy_greedy,
        option_policy_temp=args.options.policy_temp,
        use_rnd_eps=args.algo.use_rnd_eps,
        # rnd_eps_min=args.algo.eps.end,
        rnd_eps_min=0.01,
        rnd_eps_max=args.algo.rnd_eps_max,
    )

    carryT, (prim_transitions, infos, explore_aux) = jax.lax.scan(step_fn_env, (*expl_state, b_key),
                                                                  None, args.env.n_steps)
    (*expl_state, _) = carryT
    expl_state = tuple(expl_state)

    meta_buf = expl_state[15]

    train_state = train_state.replace(
        timesteps=train_state.timesteps + args.env.n_steps * args.env.n_envs
    )

    # ------------------------------------------------------------
    # RND RMS + NEO counts
    # ------------------------------------------------------------
    burnin = train_state.timesteps < jnp.int32(args.rnd.obs_burnin)

    flat_rnd_frames = prim_transitions.rnd_frame.reshape((-1, *prim_transitions.rnd_frame.shape[2:]))

    rnd_state = rnd_update_obs_rms(train_state.rnd_state, flat_rnd_frames)

    raw_bonus = jax.lax.cond(
        burnin,
        lambda _: jnp.zeros_like(prim_transitions.rnd_bonus),
        lambda _: prim_transitions.rnd_bonus,
        operand=None,
    )

    rnd_state, rnd_bonus = jax.lax.cond(
        burnin,
        lambda r: (r, raw_bonus),
        lambda r: rnd_update_rew_rms(r, raw_bonus),
        rnd_state,
    )
    rnd_eps_state = jax.lax.cond(
        burnin,
        lambda s: s,
        lambda s: update_rnd_eps_percentile_ema(
            s,
            prim_transitions.rnd_bonus,  # raw rollout novelty before normalization replacement
            tau=getattr(args.algo, "rnd_eps_tau", 1e-3),
        ),
        train_state.rnd_eps_state,
    )
    # ------------------------------------------------------------
    # EMA budget controller for RND-based epsilon
    #
    # We compare:
    #   eps_vanilla  = scheduled epsilon target
    #   eps_defacto  = EMA of realized exploration decisions
    #
    # If defacto < vanilla -> increase lambda_scale
    # If defacto > vanilla -> decrease lambda_scale
    # ------------------------------------------------------------
    rnd_eps_state = jax.lax.cond(jnp.bool_(args.algo.use_rnd_eps),
                                 lambda s: update_rnd_eps_ema_controller(
                                     s,
                                     eps_vanilla=jnp.mean(explore_aux.eps_vanilla),
                                     took_explore=explore_aux.took_explore,
                                     ema_beta=getattr(args.algo, "rnd_eps_ema_beta", 0.99),
                                     controller_lr=getattr(args.algo, "rnd_eps_controller_lr", 5.0),
                                     lambda_min=getattr(args.algo, "rnd_eps_lambda_min", 0.5),
                                     lambda_max=getattr(args.algo, "rnd_eps_lambda_max", 4.0),
                                     # lambda_max=1/args.algo.rnd_eps_max,
                                 ),
                                 lambda s: s, rnd_eps_state)

    bonus_max_ema = qema_update(
        prev=rnd_state.rnd_bonus_max_ema,
        x=rnd_bonus,
        tau=getattr(args.neo, "tau_bonus_max", 1e-3),
        q=getattr(args.neo, "bonus_q", 0.99),
        monotone=True,
    )
    rnd_state = rnd_state.replace(rnd_bonus_max_ema=bonus_max_ema)

    rnd_bonus_n = rnd_bonus / (bonus_max_ema + 1e-8)
    inv_count = 1.0 / (rnd_bonus_n + 1e-6)

    inv_cap = getattr(args.neo, "inv_cap", 1e4)
    inv_used = jnp.minimum(inv_count, inv_cap)

    neo_max = qema_update(
        prev=train_state.neo_state.max_inv_ema,
        x=inv_used,
        tau=getattr(args.neo, "tau_max", 1e-3),
        q=getattr(args.neo, "q", 0.99),
        monotone=True,
    )
    neo_state = train_state.neo_state.replace(max_inv_ema=neo_max)

    zeta_rollout, neo_state = neo_scale_from_state_calibrated(
        inv_count=inv_used,
        neo_state=neo_state,
        delta=args.neo.delta,
        k=args.neo.k,
        target_mean=args.neo.target_mean,
        tau_mean=getattr(args.neo, "tau_mean", 1e-3),
    )

    prim_transitions = prim_transitions.replace(
        int_rewards=rnd_bonus,
        rnd_bonus=rnd_bonus,
        rnd_bonus_n=rnd_bonus_n,
        inv_count=inv_count,
    )
    train_state = train_state.replace(
        rnd_state=rnd_state,
        neo_state=neo_state,
        rnd_eps_state=rnd_eps_state,
    )

    # ------------------------------------------------------------
    # primitive targets for meta primitive rows
    # ------------------------------------------------------------
    next_q_ext_all, next_q_int_all = get_dual_q_values_procgen(
        prim_transitions.next_obs[-1],
        train_state.enc_state,
        train_state.q_ext_state,
        train_state.q_int_state,
        args.net.batch_norm,
        args.net.batch_norm,
        args.net.batch_norm,
    )

    next_prim_q_ext = jnp.max(next_q_ext_all[:, :n_prim_actions], axis=-1)
    next_prim_q_int = jnp.max(next_q_int_all[:, :n_prim_actions], axis=-1)

    prim_ext_targets = compute_meta_primitive_targets(
        next_prim_q=next_prim_q_ext,
        meta_q_vals=prim_transitions.meta_q_ext_vals[:, :, :n_prim_actions],
        rewards=prim_transitions.rewards,
        dones=prim_transitions.dones.astype(jnp.float32),
        gamma=args.env.gamma,
        td_lambda=args.algo.td_lambda,
    )

    prim_int_targets = compute_meta_primitive_targets(
        next_prim_q=next_prim_q_int,
        meta_q_vals=prim_transitions.meta_q_int_vals[:, :, :n_prim_actions],
        rewards=prim_transitions.int_rewards,
        dones=jnp.zeros_like(prim_transitions.dones, dtype=jnp.float32),
        gamma=args.rnd.int_gamma,
        td_lambda=args.algo.td_lambda,
    )

    # ------------------------------------------------------------
    # NEO option targets
    # ------------------------------------------------------------
    opt_ext_lambda_targets, opt_int_lambda_targets, eig_rewards, stall_mean = compute_neo_options_lambda_targets_shared_dual(
        train_state=train_state,
        transitions=prim_transitions,
        gamma=args.options.gamma,
        td_lambda=args.options.td_lambda,
        use_opt_bn=args.net.batch_norm,
        n_options=n_total_options,
        use_value_rescale=not args.algo.rew_scale,
        stall_coef=args.options.stall_coef,
    )

    # ------------------------------------------------------------
    # meta SMDP targets
    # ------------------------------------------------------------
    meta_seq = unwrap_meta_ring(meta_buf, args.options.m_steps)

    meta_q_ext_seq, meta_q_int_seq = get_meta_dual_q_seq(
        meta_seq["start_obs"],
        train_state.enc_state,
        train_state.q_ext_state,
        train_state.q_int_state,
        args.net.batch_norm,
        args.net.batch_norm,
        args.net.batch_norm,
    )

    start_obs_live = expl_state[0]
    last_obs_live = expl_state[5]
    last_term_live = expl_state[8]

    meta_boot_obs = jnp.where(
        last_term_live[:, None, None, None],
        last_obs_live,
        start_obs_live,
    )

    next_meta_q_ext, next_meta_q_int = get_dual_q_values_procgen(
        meta_boot_obs,
        train_state.enc_state,
        train_state.q_ext_state,
        train_state.q_int_state,
        args.net.batch_norm,
        args.net.batch_norm,
        args.net.batch_norm,
    )

    next_meta_q_ext = jnp.max(next_meta_q_ext, axis=-1)
    next_meta_q_int = jnp.max(next_meta_q_int, axis=-1)

    meta_ext_targets, meta_int_targets, meta_rho = compute_meta_retrace_targets_dual(
        actions=meta_seq["actions"],
        behav_probs=meta_seq["behav_probs"],
        next_q_ext=next_meta_q_ext,
        next_q_int=next_meta_q_int,
        ext_rewards=meta_seq["ext_rewards"],
        int_rewards=meta_seq["int_rewards"],
        dones=meta_seq["dones"].astype(jnp.float32),
        ext_discounts=meta_seq["ext_discounts"],
        int_discounts=meta_seq["int_discounts"],
        q_ext_vals=meta_q_ext_seq,
        q_int_vals=meta_q_int_seq,
        td_lambda=args.options.meta_td_lambda,
        use_safe_rho=True,
        retrace_clip_rho=1.0,
        retrace_clip_c=1.0,
        int_coef=args.algo.int_coef,
        use_value_rescale=not args.algo.rew_scale,
    )

    _smdp_learn_phase = partial(
        smdp_learn_phase_dual,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        int_coef=args.algo.int_coef,
        n_meta_actions=n_meta_actions,
        n_prim_actions=n_prim_actions,
    )

    _prim_learn_phase = partial(
        primitive_learn_phase_dual,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        int_coef=args.algo.int_coef,
        n_primitives=n_prim_actions,
    )

    _opt_policy_learn_phase = partial(
        update_option_networks_shared_dual,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        int_coef=args.neo.rnd_coef,
    )

    _update_lap_network = partial(
        update_neo_laplacian_network,
        args=args,
        n_actions=n_prim_actions,
    )

    def _learn_epoch(carry, _):
        train_state, key = carry
        key, shard_key, lap_key = jax.random.split(key, 3)

        def preprocess_pytree(x, key, n_shards):
            x = x.reshape(-1, *x.shape[2:])
            x = jax.random.permutation(key, x)
            x = x.reshape(n_shards, -1, *x.shape[1:])
            return x

        def preprocess_array(x, key, n_shards):
            x = x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)
            x = jax.random.permutation(key, x)
            x = x.reshape(n_shards, -1, *x.shape[1:])
            return x

        # Laplacian
        lap_batch = LapBatch(
            obs=prim_transitions.obs.transpose(1, 0, 2, 3, 4),
            dones=prim_transitions.dones.T,
            actions=prim_transitions.actions.T,
            action_probs=None,
            counts=zeta_rollout.T,
            novelty=prim_transitions.rnd_bonus.T,
        )

        train_state, (eig_loss, pos_loss, ortho_loss, dual_loss) = _update_lap_network(
            lap_key, train_state, lap_batch
        )

        # RND update once per epoch
        flat_storage = FlatStorage(rnd_frame=flat_rnd_frames)
        rnd_state, rnd_fwd_loss = rnd_update_once_per_epoch(
            train_state.rnd_state,
            flat_storage,
            args.rnd.upd_prop,
            burnin,
        )
        train_state = train_state.replace(rnd_state=rnd_state)

        rnd_fwd_l2 = prim_transitions.rnd_bonus.mean()

        # Meta SMDP
        meta_transitions = Meta_Transition(
            obs=meta_seq["start_obs"],
            actions=meta_seq["actions"],
            ext_rewards=meta_seq["ext_rewards"],
            int_rewards=meta_seq["int_rewards"],
            dones=meta_seq["dones"],
            next_obs=meta_seq["term_obs"],
            ext_discounts=meta_seq["ext_discounts"],
            int_discounts=meta_seq["int_discounts"],
            behav_probs=meta_seq["behav_probs"],
            valid=meta_seq["valid"],
        )

        meta_batches = jax.tree_util.tree_map(
            lambda x: preprocess_pytree(x, shard_key, args.algo.n_mini_batches),
            meta_transitions,
        )
        meta_ext_targets_mb = preprocess_array(meta_ext_targets, shard_key, args.algo.n_mini_batches)
        meta_int_targets_mb = preprocess_array(meta_int_targets, shard_key, args.algo.n_mini_batches)

        (train_state, _), (
            meta_losses, meta_qstats,
            meta_option_taken_count_mb, meta_primitive_taken_count_mb,
            meta_option_greedy_count_mb, meta_primitive_greedy_count_mb,
            meta_valid_count_mb
        ) = jax.lax.scan(
            _smdp_learn_phase,
            (train_state, shard_key),
            (meta_batches, meta_ext_targets_mb, meta_int_targets_mb),
        )

        # Primitive-row meta update
        prim_batches = jax.tree_util.tree_map(
            lambda x: preprocess_pytree(x, shard_key, args.algo.n_mini_batches),
            prim_transitions,
        )
        prim_ext_targets_mb = preprocess_array(prim_ext_targets, shard_key, args.algo.n_mini_batches)
        prim_int_targets_mb = preprocess_array(prim_int_targets, shard_key, args.algo.n_mini_batches)

        (train_state, _), (prim_losses, prim_qstats) = jax.lax.scan(
            _prim_learn_phase,
            (train_state, shard_key),
            (prim_batches, prim_ext_targets_mb, prim_int_targets_mb),
        )

        # Option policy update
        opt_batches = jax.tree_util.tree_map(
            lambda x: preprocess_pytree(x, shard_key, args.options.n_mini_batches),
            prim_transitions,
        )

        opt_ext_targets = jax.vmap(
            lambda targ_o: preprocess_array(targ_o, shard_key, args.options.n_mini_batches),
            in_axes=0,
        )(opt_ext_lambda_targets)
        opt_ext_targets = jnp.transpose(opt_ext_targets, (1, 0, 2))

        opt_int_targets = jax.vmap(
            lambda targ_o: preprocess_array(targ_o, shard_key, args.options.n_mini_batches),
            in_axes=0,
        )(opt_int_lambda_targets)
        opt_int_targets = jnp.transpose(opt_int_targets, (1, 0, 2))

        def _opt_mb_step(ts, shard):
            mb_data, mb_opt_ext_tgt, mb_opt_int_tgt = shard
            ts, (opt_loss, opt_q_t, opt_q_raw) = _opt_policy_learn_phase(
                ts, mb_data, mb_opt_ext_tgt, mb_opt_int_tgt
            )
            return ts, (opt_loss, opt_q_t, opt_q_raw)

        train_state, (optpol_losses, optpol_q_t, optpol_q_raw) = jax.lax.scan(
            _opt_mb_step,
            train_state,
            (opt_batches, opt_ext_targets, opt_int_targets),
        )

        return (train_state, key), (
            meta_losses, meta_qstats,
            meta_option_taken_count_mb, meta_primitive_taken_count_mb,
            meta_option_greedy_count_mb, meta_primitive_greedy_count_mb,
            meta_valid_count_mb,
            prim_losses, prim_qstats,
            optpol_losses, optpol_q_t, optpol_q_raw,
            eig_loss, pos_loss, ortho_loss, dual_loss,
            rnd_fwd_loss, rnd_fwd_l2,
        )

    key, epoch_key = jax.random.split(key)
    (train_state, _), epoch_stats = jax.lax.scan(
        _learn_epoch, (train_state, epoch_key), None, args.algo.n_epochs
    )
    train_state = train_state.replace(n_updates=train_state.n_updates + 1)

    meta_losses = epoch_stats[0]
    meta_qstats = epoch_stats[1]
    meta_option_taken_count_mb = epoch_stats[2]
    meta_primitive_taken_count_mb = epoch_stats[3]
    meta_option_greedy_count_mb = epoch_stats[4]
    meta_primitive_greedy_count_mb = epoch_stats[5]
    meta_valid_count_mb = epoch_stats[6]

    prim_losses = epoch_stats[7]
    prim_qstats = epoch_stats[8]

    optpol_losses = epoch_stats[9]
    optpol_q_t = epoch_stats[10]
    optpol_q_raw = epoch_stats[11]

    eig_loss = epoch_stats[12]
    pos_loss = epoch_stats[13]
    ortho_loss = epoch_stats[14]
    dual_loss = epoch_stats[15]

    rnd_fwd_losses = epoch_stats[16]
    rnd_fwd_l2s = epoch_stats[17]

    meta_ext_loss = meta_qstats[0].mean()
    meta_int_loss = meta_qstats[1].mean()
    meta_q_ext_mean_t = meta_qstats[2].mean()
    meta_q_ext_mean_raw = meta_qstats[3].mean()
    meta_q_int_mean_t = meta_qstats[4].mean()
    meta_q_int_mean_raw = meta_qstats[5].mean()
    meta_q_sel_mean_t = meta_qstats[6].mean()
    meta_q_sel_mean_raw = meta_qstats[7].mean()

    prim_ext_loss = prim_qstats[0].mean()
    prim_int_loss = prim_qstats[1].mean()
    prim_q_ext_mean_t = prim_qstats[2].mean()
    prim_q_ext_mean_raw = prim_qstats[3].mean()
    prim_q_int_mean_t = prim_qstats[4].mean()
    prim_q_int_mean_raw = prim_qstats[5].mean()
    prim_q_sel_mean_t = prim_qstats[6].mean()
    prim_q_sel_mean_raw = prim_qstats[7].mean()

    meta_option_taken_count = meta_option_taken_count_mb.sum()
    meta_primitive_taken_count = meta_primitive_taken_count_mb.sum()
    meta_option_greedy_count = meta_option_greedy_count_mb.sum()
    meta_primitive_greedy_count = meta_primitive_greedy_count_mb.sum()
    meta_valid_count = meta_valid_count_mb.sum()

    optpol_td_loss = optpol_losses.mean(axis=(0, 1))
    optpol_q_mean_t = optpol_q_t.mean(axis=(0, 1))
    optpol_q_mean_raw = optpol_q_raw.mean(axis=(0, 1))

    rnd_fwd_loss = rnd_fwd_losses.mean()
    rnd_fwd_l2 = rnd_fwd_l2s.mean()

    if args.options.eigdur_options:
        opt_dur_table = eigenvalues_to_option_durations_skip_first(
            train_state.lap_state.eigen_values,
            n_base_options=n_base_options,
            bidir_options=False,
            min_dur=args.options.opt_dur_min,
            max_dur=args.options.opt_dur_max,
            eps=1e-6,
        )
    else:
        opt_dur_table = jnp.full(
            (n_total_options,),
            jnp.int32(args.options.opt_dur),
            dtype=jnp.int32,
        )

    metrics = {
        "env_step": train_state.timesteps,
        "update_steps": train_state.n_updates,
        "env_frame": train_state.timesteps,
        "grad_steps": train_state.grad_steps,

        "Loss/SMDP_total": meta_losses.mean(),
        "Loss/SMDP_ext": meta_ext_loss,
        "Loss/SMDP_int": meta_int_loss,

        "Loss/Primitive_total": prim_losses.mean(),
        "Loss/Primitive_ext": prim_ext_loss,
        "Loss/Primitive_int": prim_int_loss,

        "Loss/OptionPolicy": optpol_losses.mean(),

        "Q/SMDP_ext_mean_t": meta_q_ext_mean_t,
        "Q/SMDP_ext_mean_raw": meta_q_ext_mean_raw,
        "Q/SMDP_int_mean_t": meta_q_int_mean_t,
        "Q/SMDP_int_mean_raw": meta_q_int_mean_raw,
        "Q/SMDP_sel_mean_t": meta_q_sel_mean_t,
        "Q/SMDP_sel_mean_raw": meta_q_sel_mean_raw,

        "Q/Primitive_ext_mean_t": prim_q_ext_mean_t,
        "Q/Primitive_ext_mean_raw": prim_q_ext_mean_raw,
        "Q/Primitive_int_mean_t": prim_q_int_mean_t,
        "Q/Primitive_int_mean_raw": prim_q_int_mean_raw,
        "Q/Primitive_sel_mean_t": prim_q_sel_mean_t,
        "Q/Primitive_sel_mean_raw": prim_q_sel_mean_raw,

        "Eigen/Total_loss": eig_loss.mean(),
        "Eigen/Pos_loss": pos_loss.mean(),
        "Eigen/Ortho_loss": ortho_loss.mean(),
        "Eigen/Dual_loss": dual_loss.mean(),
        "Eigen/Barrier": train_state.lap_state.barrier_coeffs[0, 0],
        "Eigen/Max_EigenValue": jnp.max(train_state.lap_state.eigen_values),

        "Options/dur_min": opt_dur_table.min(),
        "Options/dur_max": opt_dur_table.max(),
        "Options/dur_mean": opt_dur_table.astype(jnp.float32).mean(),
        "Options/stall_mean": stall_mean,

        "Meta/rho_mean": meta_rho.mean(),
        "Meta/rho_max": meta_rho.max(),

        "Meta/option_taken_count": meta_option_taken_count,
        "Meta/primitive_taken_count": meta_primitive_taken_count,
        "Meta/option_greedy_count": meta_option_greedy_count,
        "Meta/primitive_greedy_count": meta_primitive_greedy_count,
        "Meta/valid_count": meta_valid_count,

        "Meta/option_taken_frac": meta_option_taken_count / (meta_valid_count + 1e-8),
        "Meta/primitive_taken_frac": meta_primitive_taken_count / (meta_valid_count + 1e-8),
        "Meta/option_greedy_frac": meta_option_greedy_count / (meta_valid_count + 1e-8),
        "Meta/primitive_greedy_frac": meta_primitive_greedy_count / (meta_valid_count + 1e-8),
    }

    for i in range(n_total_options):
        metrics[f"Options/TD_loss_{i}"] = optpol_td_loss[i]
        metrics[f"Options/Q_mean_t_{i}"] = optpol_q_mean_t[i]
        metrics[f"Options/Q_mean_raw_{i}"] = optpol_q_mean_raw[i]

    nov = prim_transitions.rnd_bonus
    nov_n = prim_transitions.rnd_bonus_n
    inv = prim_transitions.inv_count
    bonus_max = nov.max()

    eps_sched_scalar = jnp.mean(explore_aux.eps_vanilla)
    eps_rnd_base_rollout = explore_aux.eps_rnd_base.reshape(-1)
    eps_final_rollout = explore_aux.eps_used.reshape(-1)
    eps_realized_rollout = explore_aux.took_explore.astype(jnp.float32).reshape(-1)

    metrics.update({
        "Explore/Min_Novelty": nov.min(),
        "Explore/Mean_Novelty": nov.mean(),
        "Explore/Max_Novelty": nov.max(),

        "Explore/Min_Novelty_Norm": nov_n.min(),
        "Explore/Mean_Novelty_Norm": nov_n.mean(),
        "Explore/Max_Novelty_Norm": nov_n.max(),

        "Counts/MinInv": inv.min(),
        "Counts/MeanInv": inv.mean(),
        "Counts/MaxInv": inv.max(),

        "RND/int_reward_mean": prim_transitions.int_rewards.mean(),
        "RND/int_reward_max": prim_transitions.int_rewards.max(),
        "RND/int_reward_sum_per_env_mean": prim_transitions.int_rewards.sum(axis=0).mean(),

        "RND/fwd_loss": rnd_fwd_loss,
        "RND/fwd_l2": rnd_fwd_l2,
        "RND/obs_std": jnp.sqrt(train_state.rnd_state.obs_rms.var).mean(),
        "RND/rew_std": jnp.sqrt(train_state.rnd_state.rew_rms.var),
        "RND/burnin": burnin.astype(jnp.float32),

        "Explore/bonus_p50": jnp.percentile(nov, 50.0),
        "Explore/bonus_p90": jnp.percentile(nov, 90.0),
        "Explore/bonus_p99": jnp.percentile(nov, 99.0),

        "NEO/max_inv_ema": train_state.neo_state.max_inv_ema,
        "NEO/zeta_min": zeta_rollout.min(),
        "NEO/zeta_mean": zeta_rollout.mean(),
        "NEO/zeta_max": zeta_rollout.max(),

        "EigenRew/max": eig_rewards.max(),
        "EigenRew/mean": eig_rewards.mean(),
        "EigenRew/min": eig_rewards.min(),

        "RND/bonus_max_rollout": bonus_max,
        "RND/bonus_max_ema": train_state.rnd_state.rnd_bonus_max_ema,

        "RND_Eps/p5_ema": rnd_eps_state.p5_ema,
        "RND_Eps/p95_ema": rnd_eps_state.p95_ema,

        # Vanilla scheduled epsilon target:
        #   epsilon_vanilla_t
        "RND_Eps/vanilla_eps": eps_sched_scalar,

        # EMA of realized exploration:
        #   epsilon_defacto_t = beta * epsilon_defacto_{t-1}
        #                     + (1-beta) * realized_explore_t
        "RND_Eps/defacto_eps_ema": rnd_eps_state.defacto_eps_ema,

        # Controller gain:
        #   epsilon_final = lambda_scale * epsilon_rnd_base
        "RND_Eps/lambda_scale": rnd_eps_state.lambda_scale,

        # Rollout diagnostics
        "RND_Eps/realized_explore_mean": eps_realized_rollout.mean(),
        "RND_Eps/rnd_base_eps_mean": eps_rnd_base_rollout.mean(),
        "RND_Eps/rnd_base_eps_min": eps_rnd_base_rollout.min(),
        "RND_Eps/rnd_base_eps_max": eps_rnd_base_rollout.max(),

        "RND_Eps/final_eps_mean": eps_final_rollout.mean(),
        "RND_Eps/final_eps_min": eps_final_rollout.min(),
        "RND_Eps/final_eps_max": eps_final_rollout.max(),

        # Useful to directly see the controller mismatch
        "RND_Eps/vanilla_minus_defacto": eps_sched_scalar - rnd_eps_state.defacto_eps_ema,

        "RND_Eps/enabled": jnp.float32(getattr(args.algo, "use_rnd_eps", False)),
    })

    metrics.update({
        "Max_Return": infos["returned_episode_returns"].max(),
        "returned_episode_returns": infos["returned_episode_returns"].mean(),
        "returned_episode_lengths": infos["returned_episode_lengths"].mean(),
        "returned_episode": infos["returned_episode"].mean(),
        "elapsed_step": infos["elapsed_step"].mean(),
        "returned_episode_normalized_returns": infos["returned_episode_normalized_returns"].mean(),
        "Max_Normalized_Return": infos["returned_episode_normalized_returns"].max(),
    })

    pass_and_log = make_wandb_log_callback(args, start_time)
    if args.wb.on:
        do_log = (
            (train_state.n_updates % 10 == 0)
            | (train_state.n_updates == (args.algo.n_upds - 1))
        )
        _ = jax.lax.cond(do_log, lambda m: pass_and_log(m), lambda m: None, metrics)

    runner_state = (train_state, tuple(expl_state), key, opt_dur_table)
    return runner_state, metrics


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 6, 7, 8))
def evaluate_greedy_hneo(
    train_state,
    eval_env,
    n_eval_steps,
    use_main_bn,
    use_q_bn,
    opt_dur_table,
    n_total_options,
    option_policy_greedy=True,
    option_policy_temp=1.0,
):
    """
    HNEO evaluation.

    Meta policy:
      - always greedy from meta Q_ext
      - primitive meta-actions execute immediately
      - option meta-actions invoke the corresponding option policy

    Option low-level execution:
      - greedy if option_policy_greedy=True
      - softmax-sampled if option_policy_greedy=False

    Uses ONLY extrinsic heads for evaluation.
    """
    init_obs, env_state = eval_env.reset()

    E = init_obs.shape[0]
    last_meta = jnp.full((E,), -1, dtype=jnp.int32)
    last_term = jnp.ones((E,), dtype=bool)
    opt_count = jnp.zeros((E,), dtype=jnp.int32)

    carry = (init_obs, env_state, last_meta, last_term, opt_count, jax.random.PRNGKey(0))

    def _step(carry, _):
        obs, env_state, last_meta, last_term, opt_count, key = carry

        q_ext, _ = get_dual_q_values_procgen(
            obs,
            train_state.enc_state,
            train_state.q_ext_state,
            train_state.q_int_state,
            use_main_bn,
            use_q_bn,
            use_q_bn,
        )

        A_meta = q_ext.shape[-1]
        A = A_meta - n_total_options
        O = n_total_options

        greedy_meta = jnp.argmax(q_ext, axis=-1)
        curr_meta = jnp.where(last_term, greedy_meta, last_meta)

        q_opts_ext, _, _ = compute_option_dual_qs(
            obs,
            train_state.opt_enc_state,
            train_state.opt_q_ext_states,
            train_state.opt_q_int_states,
            use_main_bn,
            use_q_bn,
            opt_int_coef=0.0,
            use_value_rescale=False,
        )

        is_prim = curr_meta < A
        sel_opt = jnp.clip(curr_meta - A, 0, O - 1)

        q_opt_curr = q_opts_ext[sel_opt, jnp.arange(E), :]

        key, k_opt = jax.random.split(key)
        opt_act, _, _ = select_option_action(
            q_opt_curr,
            k_opt,
            option_policy_greedy=option_policy_greedy,
            option_policy_temp=option_policy_temp,
        )

        final_act = jnp.where(is_prim, curr_meta, opt_act)

        next_obs, next_env_state, reward, done, info = eval_env.step(env_state, final_act)

        opt_count_new = jnp.where(last_term, 1, opt_count + 1)
        safe_opt_idx = jnp.clip(sel_opt, 0, O - 1)
        option_dur = opt_dur_table[safe_opt_idx]

        curr_term = is_prim | (opt_count_new >= option_dur) | done
        next_opt_count = jnp.where(curr_term, 0, opt_count_new)

        next_carry = (
            next_obs,
            next_env_state,
            curr_meta,
            curr_term,
            next_opt_count,
            key,
        )
        return next_carry, info

    (_, _, _, _, _, _), infos = jax.lax.scan(_step, carry, None, length=n_eval_steps)

    metrics = {
        "eval/returned_episode_returns": infos["returned_episode_returns"].mean(),
        "eval/returned_episode_normalized_returns": infos["returned_episode_normalized_returns"].mean(),
        "eval/returned_episode_lengths": infos["returned_episode_lengths"].mean(),
        "eval/returned_episode": infos["returned_episode"].mean(),
        "eval/Max_Return": infos["returned_episode_returns"].max(),
        "eval/Max_Normalized_Return": infos["returned_episode_normalized_returns"].max(),
    }
    return metrics


def p_q_learning(key, env, train_state, eval_env=None, video_env=None):
    init_obs, env_state = env.reset()

    n_envs = args.env.n_envs
    n_total_options = args.options.n_options
    T_max = args.options.m_steps

    if args.options.eigdur_options:
        opt_dur_table = eigenvalues_to_option_durations_skip_first(
            train_state.lap_state.eigen_values,
            n_base_options=args.options.n_options,
            bidir_options=False,
            min_dur=args.options.opt_dur_min,
            max_dur=args.options.opt_dur_max,
            eps=1e-6,
        )
    else:
        opt_dur_table = jnp.full(
            (n_total_options,),
            jnp.int32(args.options.opt_dur),
            dtype=jnp.int32,
        )

    meta_buf = init_meta_buffer(init_obs, n_envs, T_max)

    init_q_ext, init_q_int = get_dual_q_values_procgen(
        init_obs,
        train_state.enc_state,
        train_state.q_ext_state,
        train_state.q_int_state,
        args.net.batch_norm,
        args.net.batch_norm,
        args.net.batch_norm,
    )

    start_ap = jnp.zeros((n_envs,), dtype=jnp.float32)

    curr_meta = jnp.full((n_envs,), -1, dtype=jnp.int32)
    last_term = jnp.ones((n_envs,), dtype=bool)
    opt_count = jnp.zeros((n_envs,), dtype=jnp.int32)

    cum_ext_r = jnp.zeros((n_envs,), dtype=jnp.float32)
    cum_int_r = jnp.zeros((n_envs,), dtype=jnp.float32)
    cum_gamma_ext = jnp.ones((n_envs,), dtype=jnp.float32)
    cum_gamma_int = jnp.ones((n_envs,), dtype=jnp.float32)
    opt_stoch_flag = jnp.zeros((n_envs,), dtype=bool)

    expl_state = (
        init_obs, curr_meta, init_q_ext, init_q_int, start_ap,
        init_obs, env_state, curr_meta, last_term, opt_count, opt_stoch_flag,
        cum_ext_r, cum_int_r, cum_gamma_ext, cum_gamma_int,
        meta_buf,
    )

    eps_scheduler = optax.linear_schedule(
        args.algo.eps.start,
        args.algo.eps.end,
        int(args.algo.eps.frac * args.algo.n_upds),
    )

    mu_scheduler = optax.linear_schedule(
        args.options.mu_prob,
        args.options.mu_prob,
        int(args.algo.n_upds),
    )

    runner_state = (train_state, expl_state, key, opt_dur_table)

    start_time = time.time()
    step_fn_update = partial(
        update_step,
        args=args,
        env=env,
        eps_scheduler=eps_scheduler,
        mu_scheduler=mu_scheduler,
        start_time=start_time,
    )

    for upd in range(int(args.algo.n_upds)):
        runner_state, metrics = step_fn_update(runner_state, None)

        train_state_now, _, key_now, opt_dur_table = runner_state

        # -------------------------------
        # Evaluation
        # -------------------------------
        if eval_env is not None:
            do_eval = (upd % args.env.eval_interval == 0) or (upd == args.algo.n_upds - 1)

            if do_eval:
                max_ep_steps = getattr(args.env, "max_episode_steps", None)
                eval_steps = int(max_ep_steps) if max_ep_steps is not None else 1000

                eval_metrics = evaluate_greedy_hneo(
                    train_state_now,
                    eval_env,
                    eval_steps,
                    args.net.batch_norm,
                    args.net.batch_norm,
                    opt_dur_table,
                    args.options.n_options,
                    option_policy_greedy=True,
                    option_policy_temp=1,
                )
                maybe_log_eval_metrics(args, eval_metrics, train_state_now.timesteps)

        # -------------------------------
        # Video logging
        # -------------------------------
        if args.wb.on and args.algo.save_video and ((upd % int(args.algo.n_upds/3) == 0) or (upd == args.algo.n_upds - 1)):
            # print(f"[VIDEO] trigger hit at update {int(train_state_now.n_updates)}")
            log_hneo_videos_to_wandb(
                args=args,
                train_state=train_state_now,
                video_env=video_env,
                opt_dur_table=opt_dur_table,
                step=train_state_now.n_updates,
            )


def log_hneo_videos_to_wandb(args, train_state, video_env, opt_dur_table, step):
    import os
    import time
    import wandb

    run_id = wandb.run.id if wandb.run is not None else f"seed_{args.exp.seed}"
    run_name = wandb.run.name if wandb.run is not None else "offline"
    safe_run_name = run_name.replace("/", "_").replace(" ", "_")

    video_dir = os.path.join("videos", f"{safe_run_name}_{run_id}")
    prefix = f"run_{run_id}_seed_{int(args.exp.seed)}_upd_{int(step)}"

    outputs = record_procgen_policy_and_option_videos(
        out_dir=video_dir,
        env=video_env,
        enc_state=train_state.enc_state,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        n_steps_policy=200,
        n_steps_option=50,
        fps=10,
        prefix=prefix,
        has_options=True,
        n_total_options=args.options.n_options,
        opt_dur_table=opt_dur_table,
        meta_q_ext_state=train_state.q_ext_state,
        meta_q_int_state=train_state.q_int_state,
        meta_int_coef=args.algo.int_coef,
        option_enc_state=train_state.opt_enc_state,
        opt_q_ext_states=train_state.opt_q_ext_states,
        opt_q_int_states=train_state.opt_q_int_states,
        option_int_coef=args.neo.rnd_coef,
        use_value_rescale=not args.algo.rew_scale,
        value_rescale_fn=value_rescale,
        inv_value_rescale_fn=inv_value_rescale,
        meta_eval_mode="ext",
        option_eval_mode="ext",
        option_policy_greedy=args.options.policy_greedy,
        option_policy_temp=args.options.policy_temp,
    )

    log_dict = {
        "video/greedy_policy": wandb.Video(outputs["greedy"], fps=10, format="mp4"),
    }

    for oid, path in outputs["options"].items():
        log_dict[f"video/option_{oid}"] = wandb.Video(path, fps=10, format="mp4")

    wandb.log(log_dict, step=int(step))


if __name__ == "__main__":
    TrainEnv = make_procgen_env(args, eval_env=False)
    EvalEnv = None
    VideoEnv = None
    if getattr(args.env, "n_eval_envs", 0) > 0:
        EvalEnv = make_procgen_env(args, eval_env=True)

    n_actions = TrainEnv.single_action_space.n

    if args.wb.on:
        create_wb_logger(args)
        VideoEnv = make_procgen_video_env(args, wrapper_cls=JaxLogProcgenEnvPoolWrapper)
        wandb.save('src/networks/networks.py', policy='now')
        wandb.save('src/losses/laplacian.py', policy='now')
        wandb.save('src/mains/algorithms/PQN.py', policy='now')
        wandb.save('src/mains/algorithms/RND.py', policy='now')
        wandb.save('src/mains/algorithms/NEO.py', policy='now')
        wandb.save('src/mains/algorithms/EO.py', policy='now')
        wandb.save('src/mains/algorithms/HEO.py', policy='now')
        wandb.save('src/mains/algorithms/Laplacian.py', policy='now')

    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    key, setup_key, q_key = jax.random.split(jax.random.PRNGKey(args.exp.seed), 3)

    with jax.default_device(jax.devices("gpu")[0]):
        try:
            TrainEnv = JaxLogProcgenEnvPoolWrapper(TrainEnv)
            if EvalEnv is not None:
                EvalEnv = JaxLogProcgenEnvPoolWrapper(EvalEnv)

            init_obs, _ = TrainEnv.reset()
            train_state, lr_schedule = setup(setup_key, args, init_obs, n_actions)
            p_q_learning(q_key, TrainEnv, train_state, eval_env=EvalEnv, video_env=VideoEnv)

        except Exception:
            traceback.print_exc()
            if args.wb.on:
                wandb.finish(exit_code=1)
        finally:
            TrainEnv.close()
            if EvalEnv is not None:
                EvalEnv.close()
            if VideoEnv is not None:
                VideoEnv.close()
            if args.wb.on:
                close_logger()
            jax.clear_backends()
