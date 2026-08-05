import math
import random
import traceback
from copy import deepcopy
from typing import Any, Union, List, NamedTuple

import envpool
import numpy as np
from etils import epath
from functools import partial

import os, sys
sys.path.append(os.getcwd())

from src.config.atari.pqn_laplacian import *
args = convert_args_to_dict(parse_args())
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.gpu.perc)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Helps with fragmentation

import jax
import flax
import chex
import optax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
import orbax.checkpoint as ocp
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from src.utils.jax_utils import learning_rate_fn, qema_update
from src.utils.purejax_atari_wrappers import JaxLogEnvPoolWrapper
from src.networks.networks import FFNetwork, Conv_Encoder, SA_LapNetwork
from src.mains.algorithms.RND import (RNDState, rnd_setup, rnd_intrinsic_reward,
                                      rnd_update_obs_rms, rnd_update_rew_rms, rnd_update_once_per_epoch)
from src.mains.algorithms.PQN import (value_rescale, inv_value_rescale,
                                      get_dual_q_values, learn_phase_dual,
                                      compute_td_lambda_targets, compute_transformed_td_lambda_targets)
from src.mains.algorithms.EO import (eigenvalues_to_option_durations_skip_first, update_option_networks_shared)
from src.mains.algorithms.HEO import (MetaRingBuffer, unwrap_meta_ring, init_meta_buffer, get_meta_q_seq,
                                      compute_meta_retrace_targets, compute_meta_primitive_targets, smdp_learn_phase,
                                      primitive_learn_phase)
from src.mains.algorithms.Laplacian import update_neo_laplacian_network
from src.mains.algorithms.NEO import (NEOScaleState, neo_scale_from_state_calibrated,
                                      compute_neo_options_lambda_targets_shared)


@chex.dataclass(frozen=True)
class Prim_Transition:
    obs: chex.Array                 # [E,C,H,W]
    actions: chex.Array             # executed primitive action [E]
    next_obs: chex.Array            # [E,C,H,W]
    dones: chex.Array               # [E]
    rewards: chex.Array             # extrinsic reward [E]

    meta_action: chex.Array         # [E] current meta action in [0..A+O-1]
    meta_q_vals: chex.Array         # [E, A+O]
    option_q_vals: chex.Array       # [E, O, A]
    action_probs: chex.Array        # [E] primitive behavior prob if needed for lap IS

    rnd_frame: chex.Array           # [E,84,84,1]
    rnd_bonus: chex.Array           # [E]
    rnd_bonus_n: chex.Array         # [E]
    inv_count: chex.Array           # [E]


@chex.dataclass(frozen=True)
class Meta_Transition:
    obs: chex.Array           # [B,C,H,W]
    actions: chex.Array       # [B] in [0..A+O-1]
    rewards: chex.Array       # [B]
    dones: chex.Array         # [B]
    next_obs: chex.Array      # [B,C,H,W]
    discounts: chex.Array     # [B]
    behav_probs: chex.Array   # [B]
    valid: chex.Array         # [B]


class Enc_State(TrainState):
    batch_stats: Union[None, flax.core.FrozenDict] = None


class FFN_State(TrainState):
    batch_stats: Union[None, flax.core.FrozenDict] = None


class FlatStorage(struct.PyTreeNode):
    rnd_frame: chex.Array   # [B,H,W,1]


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
class Train_State:
    enc_state: Enc_State
    q_state: FFN_State              # meta-Q over A+O

    lap_state: Lap_State

    opt_enc_state: Enc_State
    opt_q_states: List[FFN_State]

    rnd_state: RNDState
    neo_state: NEOScaleState

    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_env(args):
    env_kwargs = {'episodic_life': True, 'reward_clip': args.algo.rew_scale, 'repeat_action_probability': 0.,
                  'frame_skip': 4, 'noop_max': 30, 'full_action_space': False}
    env = envpool.make(args.env.id+'-v5', env_type="gym", num_envs=args.env.n_total_envs, seed=args.exp.seed, **env_kwargs)
    env.num_envs = args.env.n_total_envs
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    env.name = args.env.id+'-v5'
    return env


def setup(key, args, obs, n_prim_actions):
    n_total_options = args.options.n_options
    assert not args.options.bidir_options
    n_meta_actions = n_prim_actions + n_total_options

    k_enc, k_q, k_opt_enc, k_lap, k_rnd = jax.random.split(key, 5)

    common_encoder_kwargs = dict(
        min_scale=1,
        width_mul=1,
        impala_cnn=False,
        batch_norm=args.net.batch_norm,
        layer_norm=args.net.layer_norm,
    )

    assert len(args.net.c_dims) > 0

    # -------------------------
    # Main encoder + Q
    # -------------------------
    enc_network = Conv_Encoder(
        name='PQN:Enc',
        layer_dims=list(args.net.c_dims),
        activations=[nn.relu] * len(args.net.c_dims),
        **common_encoder_kwargs,
    )

    q_network = FFNetwork(
        name='PQN:MetaQ',
        stop_grad=False,
        ip_batch_norm=False,
        batch_norm=args.net.batch_norm,
        layer_norm=args.net.layer_norm,
        layer_dims=list(args.net.q_dims) + [n_meta_actions],
        activations=[nn.relu] * len(args.net.q_dims) + [None],
    )

    # -------------------------
    # Shared option encoder
    # -------------------------
    opt_enc_network = Conv_Encoder(
        name='Opt:Enc',
        layer_dims=list(args.net.c_dims),
        activations=[nn.relu] * len(args.net.c_dims),
        **common_encoder_kwargs,
    )

    # -------------------------
    # Per-option Q-head template
    # -------------------------
    opt_q_network = FFNetwork(
        name='Opt:Q',
        stop_grad=False,
        ip_batch_norm=False,
        batch_norm=args.net.batch_norm,
        layer_norm=args.net.layer_norm,
        layer_dims=list(args.net.q_dims) + [n_prim_actions],
        activations=[nn.relu] * len(args.net.q_dims) + [None],
    )

    # -------------------------
    # Laplacian encoder
    # -------------------------
    lap_network = Conv_Encoder(
        name='Lap:Enc',
        layer_dims=list(args.net.c_dims) + [args.laplacian.dims],
        activations=[nn.relu] * len(args.net.c_dims) + [None],
        min_scale=1,
        width_mul=1,
        impala_cnn=False,
        batch_norm=args.laplacian.batch_norm,
        layer_norm=args.laplacian.layer_norm,
    )

    # -------------------------
    # Dummy inputs
    # -------------------------
    obs0 = np.transpose(obs[0], (1, 2, 0))[None]
    dummy_embedding = jnp.ones((1, args.net.c_dims[-1]), dtype=jnp.float32)

    print(enc_network.tabulate(k_enc, obs0))
    print(q_network.tabulate(k_q, dummy_embedding))
    print(opt_enc_network.tabulate(k_opt_enc, obs0))
    print(opt_q_network.tabulate(k_q, dummy_embedding))

    lr_schedule = learning_rate_fn(
        init_lr=args.net.lr,
        alpha=1.0,
        num_epochs=args.algo.n_upds,
        warmup_epochs=args.algo.n_upds // 4,
        steps_per_epoch=args.algo.n_mini_batches * args.algo.n_epochs,
    )

    # -------------------------
    # Main states
    # -------------------------
    enc_variables = enc_network.init(k_enc, obs0, train=False)
    enc_optimizer = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adamw(learning_rate=lr_schedule, eps=1.5e-4),
    )
    enc_batch_stats = enc_variables['batch_stats'] if args.net.batch_norm else None
    enc_state = Enc_State.create(
        apply_fn=enc_network.apply,
        tx=enc_optimizer,
        params=enc_variables['params'],
        batch_stats=enc_batch_stats,
    )

    q_variables = q_network.init(k_q, dummy_embedding, train=False)
    q_optimizer = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adamw(learning_rate=lr_schedule, eps=1.5e-4),
    )
    q_batch_stats = q_variables['batch_stats'] if args.net.batch_norm else None
    q_state = FFN_State.create(
        apply_fn=q_network.apply,
        tx=q_optimizer,
        params=q_variables['params'],
        batch_stats=q_batch_stats,
    )

    # -------------------------
    # Shared option encoder state
    # -------------------------
    opt_enc_variables = opt_enc_network.init(k_opt_enc, obs0, train=False)
    opt_enc_optimizer = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adamw(learning_rate=lr_schedule, eps=1.5e-4),
    )
    opt_enc_batch_stats = opt_enc_variables['batch_stats'] if args.net.batch_norm else None
    opt_enc_state = Enc_State.create(
        apply_fn=opt_enc_network.apply,
        tx=opt_enc_optimizer,
        params=opt_enc_variables['params'],
        batch_stats=opt_enc_batch_stats,
    )

    # -------------------------
    # Separate option Q-head states
    # -------------------------
    opt_q_states = []
    key_opt = key
    for _ in range(n_total_options):
        key_opt, kq = jax.random.split(key_opt)
        opt_q_variables = opt_q_network.init(kq, dummy_embedding, train=False)
        opt_q_optimizer = optax.chain(
            optax.clip_by_global_norm(args.net.max_grad_norm),
            optax.adamw(learning_rate=lr_schedule, eps=1.5e-4),
        )
        opt_q_batch_stats = opt_q_variables['batch_stats'] if args.net.batch_norm else None
        opt_q_state = FFN_State.create(
            apply_fn=opt_q_network.apply,
            tx=opt_q_optimizer,
            params=opt_q_variables['params'],
            batch_stats=opt_q_batch_stats,
        )
        opt_q_states.append(opt_q_state)

    # -------------------------
    # Laplacian state
    # -------------------------
    lap_variables = lap_network.init(k_lap, obs0, train=False)
    lap_lr_schedule = learning_rate_fn(
        init_lr=args.laplacian.lr,
        alpha=1.0,
        num_epochs=args.algo.n_upds,
        warmup_epochs=args.algo.n_upds // 4,
        steps_per_epoch=max(1, args.algo.n_epochs),
    )
    lap_optimizer = optax.chain(
        optax.clip_by_global_norm(args.laplacian.max_grad_norm),
        optax.adamw(learning_rate=lap_lr_schedule, eps=1.5e-4),
    )
    lap_batch_stats = lap_variables['batch_stats'] if args.laplacian.batch_norm else None
    lap_state = Lap_State.create(
        apply_fn=lap_network.apply,
        tx=lap_optimizer,
        params=lap_variables['params'],
        batch_stats=lap_batch_stats,
        barrier_coeffs=jnp.tril(jnp.ones((1, 1)), k=0) * args.laplacian.barrier.value,
        betas=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
        eigen_values=jnp.ones((args.laplacian.dims,), dtype=jnp.float32) * 1e-3,
        errors=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
        quadratic_errors=jnp.zeros((args.laplacian.dims, args.laplacian.dims)),
    )

    rnd_state = rnd_setup(
        key=k_rnd,
        lr=args.net.lr,
        max_grad_norm=args.net.max_grad_norm,
        n_envs=args.env.n_envs,
        int_gamma=args.rnd.int_gamma,
        update_proportion=args.rnd.upd_prop,
    )

    neo_state = NEOScaleState(
        max_inv_ema=jnp.array(1.0, dtype=jnp.float32),
        mean_scale_ema=jnp.array(1.0, dtype=jnp.float32),
    )

    train_state = Train_State(
        enc_state=enc_state,
        q_state=q_state,
        lap_state=lap_state,
        opt_enc_state=opt_enc_state,
        opt_q_states=opt_q_states,
        rnd_state=rnd_state,
        neo_state=neo_state,
    )
    return train_state, lr_schedule


@jax.jit
def eps_greedy_exploration(rng, q_vals, eps):
    rng_a, rng_e = jax.random.split(rng)  # a key for sampling random actions and one for picking
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    chosen_actions = jnp.where(jax.random.uniform(rng_e, greedy_actions.shape) < eps,
                               jax.random.randint(rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]),
                               greedy_actions)
    return chosen_actions


@partial(jax.jit, static_argnums=(3, 4))
def get_q_values(obs, enc_state, q_state, use_enc_bn, use_q_bn):
    obs = jnp.transpose(obs, (0, 2, 3, 1))
    enc_vars = {'params': enc_state.params, 'batch_stats': enc_state.batch_stats} if use_enc_bn else {'params': enc_state.params}
    embeddings = enc_state.apply_fn(enc_vars, obs)

    q_vars = {'params': q_state.params, 'batch_stats': q_state.batch_stats} if use_q_bn else {'params': q_state.params}
    q_values = q_state.apply_fn(q_vars, embeddings)
    return q_values


@partial(jax.jit, static_argnums=(3, 4))
def compute_option_qs(obs, opt_enc_state, opt_q_states, use_enc_bn, use_q_bn):
    """
    obs: [E,C,H,W]
    returns: [O,E,A]
    """
    obs = jnp.transpose(obs, (0, 2, 3, 1))  # [E,H,W,C]

    # shared encoder
    enc_vars = {'params': opt_enc_state.params}
    if use_enc_bn:
        enc_vars['batch_stats'] = opt_enc_state.batch_stats
    emb = opt_enc_state.apply_fn(enc_vars, obs, train=False)   # [E,D]

    # per-option heads
    q_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in opt_q_states])
    q_bs = jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in opt_q_states]) if use_q_bn else None
    q_apply = opt_q_states[0].apply_fn

    def q_one(p_q, bs_q):
        vars_q = {'params': p_q}
        if use_q_bn:
            vars_q['batch_stats'] = bs_q
        return q_apply(vars_q, emb, train=False)  # [E,A]

    q_opts = jax.vmap(q_one, in_axes=(0, 0 if use_q_bn else None))(q_params, q_bs)  # [O,E,A]
    return q_opts


@partial(jax.jit, static_argnums=range(9, 19))
def env_take_steps(
    carry, _unused,
    enc_state, meta_q_state, opt_enc_state, opt_q_states, rnd_state, n_updates, opt_dur_table,
    env, eps_scheduler, mu_scheduler, gamma, n_tr_envs, n_te_envs,
    use_main_bn, use_q_bn, save_act_probs,
    n_total_options,
):
    (
        start_obs, start_meta, start_q, start_ap,
        last_obs, env_state, last_meta, last_term, opt_count,
        cum_r, cum_gamma,
        meta_buf, key
    ) = carry

    E = n_tr_envs + n_te_envs
    key, k_u1, k_u2, k_opt, k_prim, _ = jax.random.split(key, 6)

    # ------------------------------------------------------------
    # Meta decision
    # ------------------------------------------------------------
    sample_new = last_term

    q_meta = get_q_values(last_obs, enc_state, meta_q_state, use_main_bn, use_q_bn)  # [E, A+O]
    A_meta = q_meta.shape[-1]
    A = A_meta - n_total_options
    O = n_total_options

    greedy_meta = jnp.argmax(q_meta, axis=-1)

    eps_val = eps_scheduler(n_updates)
    mu_val = mu_scheduler(n_updates)

    eps = jnp.concatenate([
        jnp.full((n_tr_envs,), eps_val, dtype=jnp.float32),
        jnp.zeros((n_te_envs,), dtype=jnp.float32),
    ], axis=0)

    u1 = jax.random.uniform(k_u1, (E,))
    u2 = jax.random.uniform(k_u2, (E,))

    rand_opt = jax.random.randint(k_opt, (E,), A, A + O)
    rand_prim = jax.random.randint(k_prim, (E,), 0, A)

    choice = jnp.where(
        u1 < eps,
        jnp.where(u2 < mu_val, rand_opt, rand_prim),
        greedy_meta,
    )

    # ------------------------------------------------------------
    # Behavior probabilities for chosen meta action
    # ------------------------------------------------------------
    p_opt_pick = eps_val * mu_val
    p_prim_pick = eps_val * (1.0 - mu_val)

    base_prim = p_prim_pick / A
    base_opt = p_opt_pick / O

    base = jnp.concatenate([
        jnp.full((E, A), base_prim, dtype=jnp.float32),
        jnp.full((E, O), base_opt, dtype=jnp.float32),
    ], axis=-1)

    soft = jax.nn.softmax(q_meta, axis=-1)
    best = (choice[:, None] == jnp.arange(A_meta)[None, :]).astype(jnp.float32)
    action_probs = base + (1.0 - eps)[:, None] * soft * best
    behav_probs = action_probs[jnp.arange(E), choice]

    start_ap = jnp.where(sample_new, behav_probs, start_ap)

    # ------------------------------------------------------------
    # Install or continue current meta action
    # ------------------------------------------------------------
    curr_meta = jnp.where(sample_new, choice, last_meta)
    start_obs = jnp.where(sample_new[:, None, None, None], last_obs, start_obs)
    start_meta = jnp.where(sample_new, choice, start_meta)
    start_q = jnp.where(sample_new[:, None], q_meta, start_q)

    # ------------------------------------------------------------
    # Low-level action
    # ------------------------------------------------------------
    q_opts = compute_option_qs(last_obs, opt_enc_state, opt_q_states, use_main_bn, use_q_bn)  # [O,E,A]

    is_prim = curr_meta < A
    sel_opt = (curr_meta - A).clip(0, O - 1)
    greedy_op = q_opts[sel_opt, jnp.arange(E), :].argmax(axis=-1)
    final_act = jnp.where(is_prim, curr_meta, greedy_op)

    prim_action_probs = None
    if save_act_probs:
        prim_prob_from_meta = action_probs[jnp.arange(E), final_act]
        pi_opt = jax.nn.softmax(q_opts[sel_opt, jnp.arange(E), :], axis=-1)
        opt_low_prob = pi_opt[jnp.arange(E), final_act]
        prim_action_probs = jnp.where(
            is_prim,
            prim_prob_from_meta,
            start_ap * opt_low_prob,
        )

    # ------------------------------------------------------------
    # Step env
    # ------------------------------------------------------------
    new_obs, new_env_state, reward, new_done, info = env.step(env_state, final_act)

    # ------------------------------------------------------------
    # Raw RND bonus (match PQN_RND / PQN_NEO semantics)
    # ------------------------------------------------------------
    rnd_frame = new_obs[:, 3, :, :].astype(jnp.float32)[..., None]   # [E,84,84,1]
    rnd_bonus = rnd_intrinsic_reward(rnd_state, rnd_frame)           # raw predictor error

    # ------------------------------------------------------------
    # Fixed-duration termination
    # ------------------------------------------------------------
    opt_count = jnp.where(last_term, 1, opt_count + 1)

    safe_opt_idx = jnp.clip(sel_opt, 0, O - 1)
    option_dur = opt_dur_table[safe_opt_idx]

    curr_term = is_prim | (opt_count >= option_dur) | new_done

    # ------------------------------------------------------------
    # Segment accounting
    # ------------------------------------------------------------
    cum_r = cum_r + cum_gamma * reward
    cum_gamma = cum_gamma * gamma

    meta_reward = cum_r
    meta_discount = cum_gamma

    cum_r = jnp.where(curr_term, 0.0, cum_r)
    cum_gamma = jnp.where(curr_term, 1.0, cum_gamma)

    # ------------------------------------------------------------
    # Write meta ring buffer only where current segment terminates
    # ------------------------------------------------------------
    ptr = meta_buf.ptr
    env_idx = jnp.arange(E)
    mask_i = curr_term.astype(jnp.int32)

    rewards_b = meta_buf.rewards
    discounts_b = meta_buf.discounts
    dones_b = meta_buf.dones
    start_obs_b = meta_buf.start_obs
    term_obs_b = meta_buf.term_obs
    actions_b = meta_buf.actions
    durations_b = meta_buf.durations
    behav_probs_b = meta_buf.behav_probs

    rewards_b = rewards_b.at[ptr, env_idx].set(jnp.where(mask_i, meta_reward, rewards_b[ptr, env_idx]))
    discounts_b = discounts_b.at[ptr, env_idx].set(jnp.where(mask_i, meta_discount, discounts_b[ptr, env_idx]))
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

    T = meta_buf.rewards.shape[0]
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
        rewards=rewards_b,
        dones=dones_b,
        discounts=discounts_b,
        durations=durations_b,
        behav_probs=behav_probs_b,
    )

    new_carry = (
        start_obs, start_meta, start_q, start_ap,
        new_obs, new_env_state, curr_meta, curr_term, opt_count,
        cum_r, cum_gamma,
        meta_buf, key
    )

    prim_trans = Prim_Transition(
        obs=last_obs,
        actions=final_act,
        next_obs=new_obs,
        dones=new_done,
        rewards=reward,
        meta_action=curr_meta,
        meta_q_vals=q_meta,
        option_q_vals=jnp.transpose(q_opts, (1, 0, 2)),
        action_probs=prim_action_probs if prim_action_probs is not None else action_probs[jnp.arange(E), choice],
        rnd_frame=rnd_frame,
        rnd_bonus=rnd_bonus,
        rnd_bonus_n=jnp.zeros_like(rnd_bonus),
        inv_count=jnp.zeros_like(rnd_bonus),
    )

    return new_carry, (prim_trans, info)


@partial(jax.jit, static_argnums=range(2, 7))
def update_step(runner_state, _unused, args, env, eps_scheduler, mu_scheduler, start_time):
    (train_state, expl_state, key, opt_dur_table) = runner_state

    n_total_options = n_base_options = args.options.n_options
    n_prim_actions = env.single_action_space.n
    n_meta_actions = n_prim_actions + n_total_options
    n_train_envs = args.env.n_envs

    # ------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------
    key, b_key = jax.random.split(key)

    step_fn_env = partial(
        env_take_steps,
        enc_state=train_state.enc_state,
        meta_q_state=train_state.q_state,
        opt_enc_state=train_state.opt_enc_state,
        opt_q_states=train_state.opt_q_states,
        rnd_state=train_state.rnd_state,
        n_updates=train_state.n_updates,
        opt_dur_table=opt_dur_table,
        env=env,
        eps_scheduler=eps_scheduler,
        mu_scheduler=mu_scheduler,
        gamma=args.env.gamma,
        n_tr_envs=args.env.n_envs,
        n_te_envs=args.env.n_test_envs,
        use_main_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        save_act_probs=False,
        n_total_options=n_total_options,
    )

    carryT, (prim_transitions_all, infos) = jax.lax.scan(
        step_fn_env,
        (*expl_state, b_key),
        None,
        args.env.n_steps,
    )
    (*expl_state, _) = carryT
    expl_state = tuple(expl_state)

    meta_buf_all = expl_state[11]

    # ------------------------------------------------------------
    # Slice train envs only
    # ------------------------------------------------------------
    if args.env.n_test_envs > 0:
        prim_transitions = jax.tree_util.tree_map(
            lambda x: x[:, :n_train_envs], prim_transitions_all
        )

        meta_buf = MetaRingBuffer(
            ptr=meta_buf_all.ptr[:n_train_envs],
            head=meta_buf_all.head[:n_train_envs],
            length=meta_buf_all.length[:n_train_envs],
            start_obs=meta_buf_all.start_obs[:, :n_train_envs],
            term_obs=meta_buf_all.term_obs[:, :n_train_envs],
            actions=meta_buf_all.actions[:, :n_train_envs],
            rewards=meta_buf_all.rewards[:, :n_train_envs],
            dones=meta_buf_all.dones[:, :n_train_envs],
            discounts=meta_buf_all.discounts[:, :n_train_envs],
            durations=meta_buf_all.durations[:, :n_train_envs],
            behav_probs=meta_buf_all.behav_probs[:, :n_train_envs],
        )
    else:
        prim_transitions = prim_transitions_all
        meta_buf = meta_buf_all

    train_state = train_state.replace(
        timesteps=train_state.timesteps + args.env.n_steps * args.env.n_envs
    )

    # ------------------------------------------------------------
    # RND RMS updates + NEO counts
    # Base semantics match PQN_RND / PQN_NEO
    # ------------------------------------------------------------
    burnin = train_state.timesteps < jnp.int32(args.rnd.obs_burnin)

    flat_rnd_frames = prim_transitions.rnd_frame.reshape((-1, 84, 84, 1))

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
        rnd_bonus=rnd_bonus,
        rnd_bonus_n=rnd_bonus_n,
        inv_count=inv_count,
    )

    train_state = train_state.replace(
        rnd_state=rnd_state,
        neo_state=neo_state,
    )

    # ------------------------------------------------------------
    # Build primitive targets for meta primitive rows
    # ------------------------------------------------------------
    next_q_all = get_q_values(
        prim_transitions.next_obs[-1],
        train_state.enc_state,
        train_state.q_state,
        args.net.batch_norm,
        args.net.batch_norm,
    )

    next_prim_q = jnp.max(next_q_all[:, :n_prim_actions], axis=-1)

    prim_lambda_targets = compute_meta_primitive_targets(
        next_prim_q=next_prim_q,
        meta_q_vals=prim_transitions.meta_q_vals[:, :, :n_prim_actions],
        rewards=prim_transitions.rewards,
        dones=prim_transitions.dones.astype(jnp.float32),
        gamma=args.env.gamma,
        td_lambda=args.algo.td_lambda,
    )

    # ------------------------------------------------------------
    # Build NEO option intrinsic targets
    # ------------------------------------------------------------
    opt_lambda_targets, eig_rewards = compute_neo_options_lambda_targets_shared(
        train_state=train_state,
        transitions=prim_transitions,
        gamma=args.env.gamma,
        td_lambda=args.algo.td_lambda,
        use_opt_bn=args.net.batch_norm,
        n_options=n_total_options,
        use_value_rescale=not args.algo.rew_scale,
        rnd_coef=args.neo.rnd_coef,
    )

    # ------------------------------------------------------------
    # Build meta SMDP targets from ring buffer
    # ------------------------------------------------------------
    meta_seq = unwrap_meta_ring(meta_buf, args.options.m_steps)

    meta_q_seq = get_meta_q_seq(
        meta_seq["start_obs"],
        train_state.enc_state,
        train_state.q_state,
        args.net.batch_norm,
        args.net.batch_norm,
    )

    start_obs_live = expl_state[0][:n_train_envs]
    last_obs_live = expl_state[4][:n_train_envs]
    last_term_live = expl_state[7][:n_train_envs]

    meta_boot_obs = jnp.where(
        last_term_live[:, None, None, None],
        last_obs_live,
        start_obs_live,
    )

    meta_boot_q = get_q_values(
        meta_boot_obs,
        train_state.enc_state,
        train_state.q_state,
        args.net.batch_norm,
        args.net.batch_norm,
    )
    next_meta_q = jnp.max(meta_boot_q, axis=-1)

    meta_lambda_targets, meta_rho = compute_meta_retrace_targets(
        actions=meta_seq["actions"],
        behav_probs=meta_seq["behav_probs"],
        next_q=next_meta_q,
        rewards=meta_seq["rewards"],
        dones=meta_seq["dones"].astype(jnp.float32),
        discounts=meta_seq["discounts"],
        q_vals=meta_q_seq,
        td_lambda=args.options.meta_td_lambda,
        use_safe_rho=True,
        retrace_clip_rho=1.0,
        retrace_clip_c=1.0,
    )

    # ------------------------------------------------------------
    # Learning phase callables
    # ------------------------------------------------------------
    _smdp_learn_phase = partial(
        smdp_learn_phase,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
        n_meta_actions=n_meta_actions,
        n_prim_actions=n_prim_actions,
    )

    _prim_learn_phase = partial(
        primitive_learn_phase,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
    )

    _opt_policy_learn_phase = partial(
        update_option_networks_shared,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
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

        # --------------------------------------------------------
        # (1) Laplacian update
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # (1b) RND update once per epoch (match PQN_RND / PQN_NEO)
        # --------------------------------------------------------
        flat_storage = FlatStorage(rnd_frame=flat_rnd_frames)
        rnd_state, rnd_fwd_loss = rnd_update_once_per_epoch(
            train_state.rnd_state,
            flat_storage,
            args.rnd.upd_prop,
            burnin,
        )
        train_state = train_state.replace(rnd_state=rnd_state)

        rnd_fwd_l2 = prim_transitions.rnd_bonus.mean()

        # --------------------------------------------------------
        # (2) SMDP meta update
        # --------------------------------------------------------
        meta_transitions = Meta_Transition(
            obs=meta_seq["start_obs"],
            actions=meta_seq["actions"],
            rewards=meta_seq["rewards"],
            dones=meta_seq["dones"],
            next_obs=meta_seq["term_obs"],
            discounts=meta_seq["discounts"],
            behav_probs=meta_seq["behav_probs"],
            valid=meta_seq["valid"],
        )

        meta_batches = jax.tree_util.tree_map(
            lambda x: preprocess_pytree(x, shard_key, args.algo.n_mini_batches),
            meta_transitions,
        )
        meta_targets_mb = preprocess_array(
            meta_lambda_targets, shard_key, args.algo.n_mini_batches
        )

        (train_state, _), (
            meta_losses, meta_qstats,
            meta_option_taken_count_mb, meta_primitive_taken_count_mb,
            meta_option_greedy_count_mb, meta_primitive_greedy_count_mb,
            meta_valid_count_mb
        ) = jax.lax.scan(
            _smdp_learn_phase, (train_state, shard_key), (meta_batches, meta_targets_mb)
        )

        # --------------------------------------------------------
        # (3) Primitive-row meta update
        # --------------------------------------------------------
        prim_batches = jax.tree_util.tree_map(
            lambda x: preprocess_pytree(x, shard_key, args.algo.n_mini_batches),
            prim_transitions,
        )
        prim_targets_mb = preprocess_array(
            prim_lambda_targets, shard_key, args.algo.n_mini_batches
        )

        (train_state, _), (prim_losses, prim_qstats) = jax.lax.scan(
            _prim_learn_phase,
            (train_state, shard_key),
            (prim_batches, prim_targets_mb),
        )

        # --------------------------------------------------------
        # (4) Option-policy update
        # --------------------------------------------------------
        opt_batches = jax.tree_util.tree_map(
            lambda x: preprocess_pytree(x, shard_key, args.options.n_mini_batches),
            prim_transitions,
        )

        opt_targets = jax.vmap(
            lambda targ_o: preprocess_array(targ_o, shard_key, args.options.n_mini_batches),
            in_axes=0,
        )(opt_lambda_targets)
        opt_targets = jnp.transpose(opt_targets, (1, 0, 2))

        def _opt_mb_step(ts, shard):
            mb_data, mb_opt_tgt = shard
            ts, (opt_loss, opt_q_t, opt_q_raw) = _opt_policy_learn_phase(ts, mb_data, mb_opt_tgt)
            return ts, (opt_loss, opt_q_t, opt_q_raw)

        train_state, (optpol_losses, optpol_q_t, optpol_q_raw) = jax.lax.scan(
            _opt_mb_step,
            train_state,
            (opt_batches, opt_targets),
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

    meta_q_mean_t = meta_qstats[0].mean()
    meta_q_mean_raw = meta_qstats[1].mean()

    meta_option_taken_count = meta_option_taken_count_mb.sum()
    meta_primitive_taken_count = meta_primitive_taken_count_mb.sum()
    meta_option_greedy_count = meta_option_greedy_count_mb.sum()
    meta_primitive_greedy_count = meta_primitive_greedy_count_mb.sum()
    meta_valid_count = meta_valid_count_mb.sum()

    prim_q_mean_t = prim_qstats[0].mean()
    prim_q_mean_raw = prim_qstats[1].mean()

    optpol_td_loss = optpol_losses.mean(axis=(0, 1))
    optpol_q_mean_t = optpol_q_t.mean(axis=(0, 1))
    optpol_q_mean_raw = optpol_q_raw.mean(axis=(0, 1))

    rnd_fwd_loss = rnd_fwd_losses.mean()
    rnd_fwd_l2 = rnd_fwd_l2s.mean()

    # ------------------------------------------------------------
    # Update option durations
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    metrics = {
        "env_step": train_state.timesteps,
        "update_steps": train_state.n_updates,
        "env_frame": train_state.timesteps * env.observation_space.shape[0],
        "grad_steps": train_state.grad_steps,

        "Loss/SMDP": meta_losses.mean(),
        "Loss/Primitive": prim_losses.mean(),
        "Loss/OptionPolicy": optpol_losses.mean(),

        "Q/SMDP_mean_t": meta_q_mean_t,
        "Q/SMDP_mean_raw": meta_q_mean_raw,
        "Q/Primitive_mean_t": prim_q_mean_t,
        "Q/Primitive_mean_raw": prim_q_mean_raw,

        "Eigen/Total_loss": eig_loss.mean(),
        "Eigen/Pos_loss": pos_loss.mean(),
        "Eigen/Ortho_loss": ortho_loss.mean(),
        "Eigen/Dual_loss": dual_loss.mean(),
        "Eigen/Barrier": train_state.lap_state.barrier_coeffs[0, 0],
        "Eigen/Max_EigenValue": jnp.max(train_state.lap_state.eigen_values),

        "Options/dur_min": opt_dur_table.min(),
        "Options/dur_max": opt_dur_table.max(),
        "Options/dur_mean": opt_dur_table.astype(jnp.float32).mean(),

        "Meta/rho_mean": meta_rho.mean(),
        "Meta/rho_max": meta_rho.max(),

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

        "RND/fwd_loss": rnd_fwd_loss,
        "RND/fwd_l2": rnd_fwd_l2,
        "RND/obs_std": jnp.sqrt(train_state.rnd_state.obs_rms.var).mean(),
        "RND/rew_std": jnp.sqrt(train_state.rnd_state.rew_rms.var),

        "Explore/bonus_p50": jnp.percentile(nov, 50.0),
        "Explore/bonus_p90": jnp.percentile(nov, 90.0),
        "Explore/bonus_p99": jnp.percentile(nov, 99.0),

        "NEO/max_inv_ema": train_state.neo_state.max_inv_ema,
        "NEO/zeta_min": zeta_rollout.min(),
        "NEO/zeta_mean": zeta_rollout.mean(),
        "NEO/zeta_max": zeta_rollout.max(),

        "RND/bonus_max_rollout": bonus_max,
        "RND/bonus_max_ema": train_state.rnd_state.rnd_bonus_max_ema,
    })

    if args.env.n_test_envs > 0:
        train_infos = jax.tree_util.tree_map(lambda x: x[:, :-args.env.n_test_envs], infos)
        test_infos = jax.tree_util.tree_map(lambda x: x[:, -args.env.n_test_envs:], infos)

        metrics.update({"Max_Return": train_infos["returned_episode_returns"].max()})
        metrics.update({k: v.mean() for k, v in train_infos.items()})
        metrics.update({f"test/{k}": v.mean() for k, v in test_infos.items()})
        metrics.update({"test/Max_Return": test_infos["returned_episode_returns"].max()})
    else:
        metrics.update({"Max_Return": infos["returned_episode_returns"].max()})
        metrics.update({k: v.mean() for k, v in infos.items()})

    def _log_to_wandb(m):
        elapsed = time.time() - start_time
        total_env = m["update_steps"] * args.env.n_steps * args.env.n_envs
        m["SPS"] = total_env / elapsed
        wandb.log(m, step=int(m["update_steps"]))

    @jax.jit
    def _pass_and_log(m):
        return jax.debug.callback(_log_to_wandb, m)

    if args.wb.on:
        do_log = (
            (train_state.n_updates % 10 == 0)
            | (train_state.n_updates == (args.algo.n_upds - 1))
        )
        _ = jax.lax.cond(do_log, lambda m: _pass_and_log(m), lambda m: None, metrics)

    runner_state = (train_state, tuple(expl_state), key, opt_dur_table)
    return runner_state, metrics


def p_q_learning(key, env, train_state):
    init_obs, env_state = env.reset()

    n_total_envs = args.env.n_total_envs
    n_prim_actions = env.single_action_space.n
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

    meta_buf = init_meta_buffer(init_obs, n_total_envs, T_max)

    curr_meta = jnp.full((n_total_envs,), -1, dtype=jnp.int32)
    last_term = jnp.ones((n_total_envs,), dtype=bool)
    opt_count = jnp.zeros((n_total_envs,), dtype=jnp.int32)

    cum_r = jnp.zeros((n_total_envs,), dtype=jnp.float32)
    cum_gamma = jnp.ones((n_total_envs,), dtype=jnp.float32)

    start_obs = init_obs
    start_meta = curr_meta

    init_q = get_q_values(
        init_obs,
        train_state.enc_state,
        train_state.q_state,
        args.net.batch_norm,
        args.net.batch_norm,
    )
    start_ap = jnp.zeros((n_total_envs,), dtype=jnp.float32)

    expl_state = (
        start_obs, start_meta, init_q, start_ap,
        init_obs, env_state, curr_meta, last_term, opt_count,
        cum_r, cum_gamma,
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
        int(1 * args.algo.n_upds),
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
    runner_state, metrics = jax.lax.scan(step_fn_update, runner_state, None, args.algo.n_upds)
    return runner_state, metrics


# ------------------------------------------------------------------------------
# Main Entry Point: Environment and Logger Initialization
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    Env = make_env(args)
    n_actions = Env.single_action_space.n
    if args.wb.on:
        create_wb_logger(args)
        wandb.save('src/networks/networks.py', policy='now')
        wandb.save('src/losses/laplacian.py', policy='now')
        wandb.save('src/mains/algorithms/PQN.py', policy='now')
        wandb.save('src/mains/algorithms/RND.py', policy='now')
        wandb.save('src/mains/algorithms/NEO.py', policy='now')
        wandb.save('src/mains/algorithms/EO.py', policy='now')
        wandb.save('src/mains/algorithms/HEO.py', policy='now')
        wandb.save('src/mains/algorithms/Laplacian.py', policy='now')

    # Set seeds for reproducibility.
    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    Key, setup_key, q_key = jax.random.split(jax.random.PRNGKey(args.exp.seed), 3)

    with jax.default_device(jax.devices('gpu')[0]):
        try:
            Env = JaxLogEnvPoolWrapper(Env)
            init_obs, _ = Env.reset()
            (Train_State, Lr_Schedule) = setup(setup_key, args, init_obs, n_actions)
            p_q_learning(q_key, Env, Train_State)

        except Exception as e:
            traceback.print_exc()  # This prints the full traceback with line numbers.
            wandb.finish(exit_code=1)
        finally:
            Env.close()
            close_logger() if args.wb.on else None
            jax.clear_backends()
