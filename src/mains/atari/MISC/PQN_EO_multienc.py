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

from src.utils.jax_utils import *
from src.losses.laplacian import *
from src.utils.rl_jax import (
    learn_phase,
    value_rescale, inv_value_rescale,
    compute_td_lambda_targets, compute_transformed_td_lambda_targets,
    compute_options_lambda_targets, update_option_networks,
    update_laplacian_network,
)
from src.networks.networks import FFNetwork, Conv_Encoder, SA_LapNetwork
from src.utils.purejax_atari_wrappers import JaxLogEnvPoolWrapper  # Wrapper to log envpool outputs


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array              # [E,C,H,W]
    actions: chex.Array          # executed primitive action [E]
    rewards: chex.Array          # extrinsic reward [E]
    dones: chex.Array            # [E]
    next_obs: chex.Array         # [E,C,H,W]

    q_vals: chex.Array           # main primitive-action Q-values [E, A_prim]
    option_idx: chex.Array       # active option index, -1 means no option [E]
    option_q_vals: chex.Array    # [E, O, A_prim]
    action_probs: chex.Array     # optional for lap IS, else None


class Enc_State(TrainState):
    batch_stats: Union[None, flax.core.FrozenDict] = None


class FFN_State(TrainState):
    batch_stats: Union[None, flax.core.FrozenDict] = None


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


@struct.dataclass
class Train_State:
    enc_state: Enc_State
    q_state: FFN_State
    lap_state: Lap_State

    opt_enc_states: List[Enc_State]
    opt_q_states: List[FFN_State]

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


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def eigenvalues_to_option_durations_skip_first(
    eigs: jnp.ndarray,          # [D] or [1,D]
    n_base_options: int,
    bidir_options: bool,
    min_dur: int,
    max_dur: int,
    eps: float = 1e-6,
):
    eigs = jnp.reshape(eigs, (-1,))
    eigs = jnp.nan_to_num(eigs, nan=0.0, posinf=0.0, neginf=0.0)
    eigs = jnp.clip(eigs, 0.0, jnp.inf)

    # skip first eigendimension
    usable = eigs[1:1 + n_base_options]
    lam = jnp.maximum(usable, eps)
    dur = jnp.ceil(1.0 / lam).astype(jnp.int32)
    dur = jnp.clip(dur, min_dur, max_dur)

    if bidir_options:
        dur = jnp.concatenate([dur, dur], axis=0)
    return dur


def setup(key, args, obs, n_prim_actions):
    n_base_options = args.options.n_options
    n_total_options = 2 * n_base_options if args.options.bidir_options else n_base_options

    k_enc, k_q, k_lap = jax.random.split(key, 3)

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
        name="PQN:Enc",
        layer_dims=list(args.net.c_dims),
        activations=[nn.relu] * len(args.net.c_dims),
        **common_encoder_kwargs,
    )

    q_network = FFNetwork(
        name="PQN:Q",
        stop_grad=False,
        ip_batch_norm=False,
        batch_norm=args.net.batch_norm,
        layer_norm=args.net.layer_norm,
        layer_dims=list(args.net.q_dims) + [n_prim_actions],
        activations=[nn.relu] * len(args.net.q_dims) + [None],
    )

    # -------------------------
    # Option encoder + head templates
    # -------------------------
    opt_enc_network = Conv_Encoder(
        name="Opt:Enc",
        layer_dims=list(args.net.c_dims),
        activations=[nn.relu] * len(args.net.c_dims),
        **common_encoder_kwargs,
    )

    opt_q_network = FFNetwork(
        name="Opt:Q",
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
        name="Lap:Enc",
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
    obs0 = np.transpose(obs[0], (1, 2, 0))[None]  # [1,H,W,C]
    dummy_embedding = jnp.ones((1, args.net.c_dims[-1]), dtype=jnp.float32)

    print(enc_network.tabulate(k_enc, obs0))
    print(q_network.tabulate(k_q, dummy_embedding))
    print(opt_enc_network.tabulate(k_q, obs0))
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
    enc_batch_stats = enc_variables["batch_stats"] if args.net.batch_norm else None
    enc_state = Enc_State.create(
        apply_fn=enc_network.apply,
        tx=enc_optimizer,
        params=enc_variables["params"],
        batch_stats=enc_batch_stats,
    )

    q_variables = q_network.init(k_q, dummy_embedding, train=False)
    q_optimizer = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adamw(learning_rate=lr_schedule, eps=1.5e-4),
    )
    q_batch_stats = q_variables["batch_stats"] if args.net.batch_norm else None
    q_state = FFN_State.create(
        apply_fn=q_network.apply,
        tx=q_optimizer,
        params=q_variables["params"],
        batch_stats=q_batch_stats,
    )

    # -------------------------
    # Per-option states
    # -------------------------
    opt_enc_states = []
    opt_q_states = []
    key_opt = key

    for _ in range(n_total_options):
        key_opt, k1, k2 = jax.random.split(key_opt, 3)

        opt_enc_variables = opt_enc_network.init(k1, obs0, train=False)
        opt_enc_optimizer = optax.chain(
            optax.clip_by_global_norm(args.net.max_grad_norm),
            optax.adamw(learning_rate=lr_schedule, eps=1.5e-4),
        )
        opt_enc_batch_stats = opt_enc_variables["batch_stats"] if args.net.batch_norm else None
        opt_enc_state = Enc_State.create(
            apply_fn=opt_enc_network.apply,
            tx=opt_enc_optimizer,
            params=opt_enc_variables["params"],
            batch_stats=opt_enc_batch_stats,
        )
        opt_enc_states.append(opt_enc_state)

        opt_q_variables = opt_q_network.init(k2, dummy_embedding, train=False)
        opt_q_optimizer = optax.chain(
            optax.clip_by_global_norm(args.net.max_grad_norm),
            optax.adamw(learning_rate=lr_schedule, eps=1.5e-4),
        )
        opt_q_batch_stats = opt_q_variables["batch_stats"] if args.net.batch_norm else None
        opt_q_state = FFN_State.create(
            apply_fn=opt_q_network.apply,
            tx=opt_q_optimizer,
            params=opt_q_variables["params"],
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

    train_state = Train_State(
        enc_state=enc_state,
        q_state=q_state,
        lap_state=lap_state,
        opt_enc_states=opt_enc_states,
        opt_q_states=opt_q_states,
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
def compute_option_qs(obs, opt_enc_states, opt_q_states, use_enc_bn, use_q_bn):
    """
    obs: [E,C,H,W]
    returns: [O,E,A]
    """
    O = len(opt_enc_states)
    obs = jnp.transpose(obs, (0, 2, 3, 1))  # [E,H,W,C]

    enc_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in opt_enc_states])
    q_params = jax.tree_map(lambda *ps: jnp.stack(ps), *[s.params for s in opt_q_states])

    enc_bs = jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in opt_enc_states]) if use_enc_bn else None
    q_bs = jax.tree_map(lambda *bs: jnp.stack(bs), *[s.batch_stats for s in opt_q_states]) if use_q_bn else None

    enc_apply = opt_enc_states[0].apply_fn
    q_apply = opt_q_states[0].apply_fn

    def encode_one(p_enc, bs_enc):
        vars_e = {"params": p_enc}
        if use_enc_bn:
            vars_e["batch_stats"] = bs_enc
        return enc_apply(vars_e, obs, train=False)

    embs = jax.vmap(encode_one, in_axes=(0, 0 if use_enc_bn else None))(enc_params, enc_bs)  # [O,E,D]

    def q_one(p_q, bs_q, emb):
        vars_q = {"params": p_q}
        if use_q_bn:
            vars_q["batch_stats"] = bs_q
        return q_apply(vars_q, emb, train=False)

    q_opts = jax.vmap(q_one, in_axes=(0, 0 if use_q_bn else None, 0))(q_params, q_bs, embs)  # [O,E,A]
    return q_opts


@partial(jax.jit, static_argnums=range(8, 17))
def env_take_steps(
    carry, _unused,
    enc_state, q_state, opt_enc_states, opt_q_states, n_updates, option_dur_table,
    env, eps_scheduler, mu_prob, n_tr_envs, n_te_envs,
    use_main_bn, use_q_bn, save_act_probs,
    n_options,
):
    last_obs, env_state, curr_opts, opt_left, key = carry
    key, k_eps, k_mu, k_idx, k_act = jax.random.split(key, 5)

    E = n_tr_envs + n_te_envs
    n_prim_actions = env.single_action_space.n

    q_main = get_q_values(last_obs, enc_state, q_state, use_main_bn, use_q_bn)   # [E,A]
    q_opts = compute_option_qs(last_obs, opt_enc_states, opt_q_states, use_main_bn, use_q_bn)  # [O,E,A]

    need_decision = (opt_left <= 0)

    eps_val = eps_scheduler(n_updates)
    eps = jnp.concatenate([
        jnp.full((n_tr_envs,), eps_val, dtype=jnp.float32),
        jnp.zeros((n_te_envs,), dtype=jnp.float32),
    ])

    u_eps = jax.random.uniform(k_eps, shape=(E,))
    u_mu = jax.random.uniform(k_mu, shape=(E,))

    do_explore = need_decision & (u_eps < eps)
    pick_opt = do_explore & (u_mu < mu_prob)
    pick_prim = do_explore & (u_mu >= mu_prob)

    idx_keys = jax.random.split(k_idx, E)
    sampled_opts = jax.vmap(lambda k: jax.random.randint(k, (), 0, n_options))(idx_keys)

    greedy_main = jnp.argmax(q_main, axis=-1)

    safe_curr_opts = jnp.clip(curr_opts, 0, n_options - 1)
    q_opt_cont = q_opts[safe_curr_opts, jnp.arange(E), :]   # [E,A]
    greedy_opt_cont = jnp.argmax(q_opt_cont, axis=-1)

    q_opt_new = q_opts[sampled_opts, jnp.arange(E), :]      # [E,A]
    greedy_opt_new = jnp.argmax(q_opt_new, axis=-1)

    rand_prims = jax.random.randint(k_act, shape=(E,), minval=0, maxval=n_prim_actions)

    continuing_opt = ~need_decision
    new_actions = jnp.where(
        pick_opt,
        greedy_opt_new,
        jnp.where(
            pick_prim,
            rand_prims,
            jnp.where(continuing_opt, greedy_opt_cont, greedy_main),
        ),
    )

    chosen_option_idx = jnp.where(
        pick_opt,
        sampled_opts,
        jnp.where(need_decision, -1, curr_opts),
    )

    chosen_dur = option_dur_table[sampled_opts]
    new_opt_left = jnp.where(
        pick_opt,
        chosen_dur - 1,
        jnp.where(continuing_opt, opt_left - 1, 0),
    )
    new_opt_left = jnp.maximum(new_opt_left, 0)

    new_obs, new_env_state, reward, new_done, info = env.step(env_state, new_actions)

    next_curr_opts = jnp.where(new_done, -1, chosen_option_idx)
    next_opt_left = jnp.where(new_done, 0, new_opt_left)

    action_probs = None
    if save_act_probs:
        behav_q = jnp.where(
            pick_opt[:, None],
            q_opt_new,
            jnp.where(
                continuing_opt[:, None],
                q_opt_cont,
                q_main,
            ),
        )
        q_probs = jax.nn.softmax(behav_q, axis=-1)
        base_prob = eps[:, None] / behav_q.shape[-1]
        greedy_i = jnp.argmax(behav_q, axis=-1)
        action_probs = base_prob + (1.0 - eps)[:, None] * (
            (new_actions[:, None] == greedy_i[:, None]).astype(jnp.float32)
        ) * q_probs

    transition = Transition(
        obs=last_obs,
        actions=new_actions,
        rewards=reward,
        dones=new_done,
        next_obs=new_obs,
        q_vals=q_main,
        option_idx=chosen_option_idx,   # important: store behavior option, not post-done reset
        option_q_vals=jnp.transpose(q_opts, (1, 0, 2)),
        action_probs=action_probs,
    )

    return (new_obs, new_env_state, next_curr_opts, next_opt_left, key), (transition, info)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def update_step(runner_state, _unused, args, env, eps_scheduler, start_time):
    (train_state, expl_state, key, opt_dur_table) = runner_state

    n_base_options = args.options.n_options
    n_total_options = 2 * n_base_options if args.options.bidir_options else n_base_options
    n_prim_actions = env.single_action_space.n

    # ------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------
    key, b_key = jax.random.split(key)

    step_fn_env = partial(
        env_take_steps,
        enc_state=train_state.enc_state,
        q_state=train_state.q_state,
        opt_enc_states=train_state.opt_enc_states,
        opt_q_states=train_state.opt_q_states,
        n_updates=train_state.n_updates,
        option_dur_table=opt_dur_table,
        env=env,
        eps_scheduler=eps_scheduler,
        mu_prob=args.options.mu_prob,
        n_tr_envs=args.env.n_envs,
        n_te_envs=args.env.n_test_envs,
        use_main_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        save_act_probs=False,
        n_options=n_total_options,
    )

    (*expl_state, _), (transitions, infos) = jax.lax.scan(step_fn_env, (*expl_state, b_key), None, args.env.n_steps)
    expl_state = tuple(expl_state)

    # Keep only train envs for learning
    if args.env.n_test_envs > 0:
        transitions = jax.tree_util.tree_map(lambda x: x[:, :-args.env.n_test_envs], transitions)

    train_state = train_state.replace(timesteps=train_state.timesteps + args.env.n_steps * args.env.n_envs)

    # ------------------------------------------------------------
    # Main primitive PQN targets
    # ------------------------------------------------------------
    next_q_vals = get_q_values(
        transitions.next_obs[-1],
        train_state.enc_state,
        train_state.q_state,
        args.net.batch_norm,
        args.net.batch_norm,
    )  # [E, A_prim]
    next_q = jnp.max(next_q_vals, axis=-1)

    if args.algo.rew_scale:
        compute_main_targets = partial(
            compute_td_lambda_targets,
            gamma=args.env.gamma,
            td_lambda=args.algo.td_lambda,
        )
    else:
        compute_main_targets = partial(
            compute_transformed_td_lambda_targets,
            gamma=args.env.gamma,
            td_lambda=args.algo.td_lambda,
        )

    lambda_targets = compute_main_targets(
        next_q,
        transitions.q_vals,      # primitive Q-values from rollout
        transitions.rewards,     # extrinsic rewards
        transitions.dones,
    )

    _main_learn_phase = partial(
        learn_phase,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
    )

    # ------------------------------------------------------------
    # Option intrinsic targets from Laplacian, skipping first eigvec
    # ------------------------------------------------------------
    opt_lambda_targets = compute_options_lambda_targets(
        train_state=train_state,
        transitions=transitions,
        gamma=args.env.gamma,
        td_lambda=args.algo.td_lambda,
        use_opt_bn=args.net.batch_norm,
        use_lap_bn=args.laplacian.batch_norm,
        n_base_options=n_base_options,
        bidir_options=args.options.bidir_options,
        use_value_rescale=not args.algo.rew_scale,
    )  # [O,T,E]

    _opt_learn_phase = partial(
        update_option_networks,
        use_enc_bn=args.net.batch_norm,
        use_q_bn=args.net.batch_norm,
        use_value_rescale=not args.algo.rew_scale,
    )

    # ------------------------------------------------------------
    # Laplacian update hook
    # ------------------------------------------------------------
    _update_lap_network = partial(
        update_laplacian_network,
        args=args,
        n_actions=n_prim_actions,
    )

    def _learn_epoch(carry, _):
        train_state, key = carry
        key, _key, lap_key = jax.random.split(key, 3)

        def preprocess_transition(x, key, n_shards):
            x = x.reshape(-1, *x.shape[2:])            # [T*E, ...]
            x = jax.random.permutation(key, x)
            x = x.reshape(n_shards, -1, *x.shape[1:]) # [M,B,...]
            return x

        # --------------------------------------------------------
        # (1) Laplacian update
        # --------------------------------------------------------
        lap_batches = LapBatch(
            obs=transitions.obs.transpose(1, 0, 2, 3, 4),  # [E,T,C,H,W]
            dones=transitions.dones.T,  # [E,T]
            actions=transitions.actions.T,  # [E,T]
            action_probs=None,
        )

        train_state, (eig_loss, pos_loss, ortho_loss, dual_loss) = _update_lap_network(lap_key, train_state, lap_batches)

        # --------------------------------------------------------
        # (2) Main primitive PQN update
        # --------------------------------------------------------
        main_batches = jax.tree_util.tree_map(
            lambda x: preprocess_transition(x, _key, args.algo.n_mini_batches),
            transitions,
        )
        main_targets_mb = preprocess_transition(
            lambda_targets, _key, args.algo.n_mini_batches
        )

        (train_state, _), (main_losses, main_qstats) = jax.lax.scan(
            _main_learn_phase,
            (train_state, _key),
            (main_batches, main_targets_mb),
        )

        # --------------------------------------------------------
        # (3) Option controller update
        # --------------------------------------------------------
        opt_batches = jax.tree_util.tree_map(
            lambda x: preprocess_transition(x, _key, args.options.n_mini_batches),
            transitions,
        )

        opt_targets = jax.vmap(
            lambda targ_o: preprocess_transition(targ_o, _key, args.options.n_mini_batches),
            in_axes=0
        )(opt_lambda_targets)                                # [O,M,B]
        opt_targets = jnp.transpose(opt_targets, (1, 0, 2))  # [M,O,B]

        def _opt_mb_step(ts, shards):
            mb_data, mb_opt_tgt = shards
            ts, (opt_loss, opt_q_t, opt_q_raw) = _opt_learn_phase(ts, mb_data, mb_opt_tgt)
            return ts, (opt_loss, opt_q_t, opt_q_raw)

        train_state, (opt_losses, opt_q_t, opt_q_raw) = jax.lax.scan(
            _opt_mb_step,
            train_state,
            (opt_batches, opt_targets),
        )

        return (train_state, key), (
            main_losses,
            main_qstats,
            opt_losses,
            opt_q_t,
            opt_q_raw,
            eig_loss,
            pos_loss,
            ortho_loss,
            dual_loss,
        )

    key, _key = jax.random.split(key)
    (train_state, _), epoch_stats = jax.lax.scan(
        _learn_epoch, (train_state, _key), None, args.algo.n_epochs
    )
    train_state = train_state.replace(n_updates=train_state.n_updates + 1)

    main_losses = epoch_stats[0]
    main_qstats = epoch_stats[1]
    opt_losses = epoch_stats[2]
    opt_q_t = epoch_stats[3]
    opt_q_raw = epoch_stats[4]
    eig_loss = epoch_stats[5]
    pos_loss = epoch_stats[6]
    ortho_loss = epoch_stats[7]
    dual_loss = epoch_stats[8]

    q_mean_t = main_qstats[0].mean()
    q_mean_raw = main_qstats[1].mean()

    opt_td_loss = opt_losses.mean(axis=(0, 1))
    opt_q_mean_t = opt_q_t.mean(axis=(0, 1))
    opt_q_mean_raw = opt_q_raw.mean(axis=(0, 1))

    # ------------------------------------------------------------
    # Update option durations from eigenvalues if requested
    # ------------------------------------------------------------
    if args.options.eigdur_options:
        opt_dur_table = eigenvalues_to_option_durations_skip_first(
            train_state.lap_state.eigen_values,
            n_base_options=n_base_options,
            bidir_options=args.options.bidir_options,
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

        "TD_loss": main_losses.mean(),
        "Q_mean_transformed": q_mean_t,
        "Q_mean_raw": q_mean_raw,

        "Eigen/Total_loss": eig_loss.mean(),
        "Eigen/Pos_loss": pos_loss.mean(),
        "Eigen/Ortho_loss": ortho_loss.mean(),
        "Eigen/Dual_loss": dual_loss.mean(),
        "Eigen/Barrier": train_state.lap_state.barrier_coeffs[0, 0],
        "Eigen/Max_EigenValue": jnp.max(train_state.lap_state.eigen_values),

        "Options/dur_min": opt_dur_table.min(),
        "Options/dur_max": opt_dur_table.max(),
        "Options/dur_mean": opt_dur_table.astype(jnp.float32).mean(),
    }

    for i in range(n_total_options):
        metrics[f"Options/TD_loss_{i}"] = opt_td_loss[i]
        metrics[f"Options/Q_mean_t_{i}"] = opt_q_mean_t[i]
        metrics[f"Options/Q_mean_raw_{i}"] = opt_q_mean_raw[i]

    if args.env.n_test_envs > 0:
        train_infos = jax.tree_util.tree_map(lambda x: x[:, :-args.env.n_test_envs], infos)
        test_infos = jax.tree_util.tree_map(lambda x: x[:, -args.env.n_test_envs:], infos)

        metrics.update({'Max_Return': train_infos['returned_episode_returns'].max()})
        metrics.update({k: v.mean() for k, v in train_infos.items()})
        metrics.update({f"test/{k}": v.mean() for k, v in test_infos.items()})
        metrics.update({'test/Max_Return': test_infos['returned_episode_returns'].max()})
    else:
        metrics.update({'Max_Return': infos['returned_episode_returns'].max()})
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
        do_log = (train_state.n_updates % 10 == 0) | (train_state.n_updates == (args.algo.n_upds - 1))
        _ = jax.lax.cond(do_log, lambda m: _pass_and_log(m), lambda m: None, metrics)

    runner_state = (train_state, tuple(expl_state), key, opt_dur_table)
    return runner_state, metrics


def p_q_learning(key, env, train_state):
    init_obs, env_state = env.reset()

    n_base_options = args.options.n_options
    n_total_options = 2 * n_base_options if args.options.bidir_options else n_base_options

    if args.options.eigdur_options:
        opt_dur_table = eigenvalues_to_option_durations_skip_first(
            train_state.lap_state.eigen_values,
            n_base_options=n_base_options,
            bidir_options=args.options.bidir_options,
            min_dur=args.options.opt_dur_min,
            max_dur=args.options.opt_dur_max,
            eps=1e-6,
        )
    else:
        opt_dur_table = jnp.full((n_total_options,), jnp.int32(args.options.opt_dur), dtype=jnp.int32)

    curr_opts = jnp.full((args.env.n_total_envs,), -1, dtype=jnp.int32)
    opt_left = jnp.zeros((args.env.n_total_envs,), dtype=jnp.int32)

    expl_state = (init_obs, env_state, curr_opts, opt_left)

    eps_scheduler = optax.linear_schedule(
        args.algo.eps.start,
        args.algo.eps.end,
        int(args.algo.eps.frac * args.algo.n_upds),
    )

    runner_state = (train_state, expl_state, key, opt_dur_table)

    start_time = time.time()
    step_fn_update = partial(
        update_step,
        args=args,
        env=env,
        eps_scheduler=eps_scheduler,
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
