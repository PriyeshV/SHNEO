import os
import time
import wandb
import random
import traceback
import numpy as np

from functools import partial
from typing import Any, Union, List

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
    make_procgen_video_env,
    build_enc_state,
    build_ffn_state,
    build_lr_schedule,
    get_dual_q_values_procgen,
    procgen_rnd_frame,
    make_wandb_log_callback,
    record_procgen_policy_and_option_videos,
)
from src.networks.networks import Conv_Encoder
from src.mains.algorithms.RND import RNDState, rnd_setup
from src.mains.algorithms.NEO import NEOScaleState
from src.mains.algorithms.HEO import init_meta_buffer
from src.mains.algorithms.EO import eigenvalues_to_option_durations_skip_first
from src.mains.algorithms.PQN import value_rescale, inv_value_rescale

from src.mains.agents.pqn_hneo_core import (
    Lap_State,
    env_take_steps_common,
    update_step_common,
    select_option_action,
)


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

    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def setup(key, args, obs, n_prim_actions):
    n_total_options = args.options.n_options
    assert not args.options.bidir_options
    n_meta_actions = n_prim_actions + n_total_options

    k_enc, k_qext, k_qint, k_opt_enc, k_lap, k_rnd, k_opt = jax.random.split(key, 7)

    enc_state = build_enc_state(k_enc, args, obs, name="PQN:Enc")
    q_ext_state = build_ffn_state(k_qext, args, n_meta_actions, name="PQN:MetaQ_ext")
    q_int_state = build_ffn_state(k_qint, args, n_meta_actions, name="PQN:MetaQ_int")

    opt_enc_state = build_enc_state(k_opt_enc, args, obs, name="Opt:Enc")

    opt_q_ext_states = []
    opt_q_int_states = []
    key_opt = k_opt
    for _ in range(n_total_options):
        key_opt, kqe, kqi = jax.random.split(key_opt, 3)
        opt_q_ext_states.append(build_ffn_state(kqe, args, n_prim_actions, name="Opt:Q_ext"))
        opt_q_int_states.append(build_ffn_state(kqi, args, n_prim_actions, name="Opt:Q_int"))

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
    ), build_lr_schedule(args)


def p_q_learning(key, env, train_state, video_env=None):
    init_obs, env_state = env.reset()

    n_total_envs = args.env.n_total_envs
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
        opt_dur_table = jnp.full((n_total_options,), jnp.int32(args.options.opt_dur), dtype=jnp.int32)

    meta_buf = init_meta_buffer(init_obs, n_total_envs, T_max)

    # Unified rollout state.
    # This must match pqn_hneo_core.env_take_steps_common carry layout.
    seg_start_obs = init_obs
    seg_start_meta = jnp.full((n_total_envs,), -1, dtype=jnp.int32)
    seg_start_behav_prob = jnp.ones((n_total_envs,), dtype=jnp.float32)

    obs_t = init_obs

    active_meta = jnp.full((n_total_envs,), -1, dtype=jnp.int32)
    opt_steps_left = jnp.zeros((n_total_envs,), dtype=jnp.int32)

    option_stoch = jnp.zeros((n_total_envs,), dtype=bool)

    seg_ext_return = jnp.zeros((n_total_envs,), dtype=jnp.float32)
    seg_int_return = jnp.zeros((n_total_envs,), dtype=jnp.float32)
    seg_ext_discount = jnp.ones((n_total_envs,), dtype=jnp.float32)
    seg_int_discount = jnp.ones((n_total_envs,), dtype=jnp.float32)
    seg_len = jnp.zeros((n_total_envs,), dtype=jnp.int32)

    expl_state = (
        seg_start_obs,  # 0
        seg_start_meta,  # 1
        seg_start_behav_prob,  # 2

        obs_t,  # 3
        env_state,  # 4

        active_meta,  # 5
        opt_steps_left,  # 6

        option_stoch,  # 7

        seg_ext_return,  # 8
        seg_int_return,  # 9
        seg_ext_discount,  # 10
        seg_int_discount,  # 11
        seg_len,  # 12

        meta_buf,  # 13
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
        update_step_common,
        args=args,
        env=env,
        eps_scheduler=eps_scheduler,
        mu_scheduler=mu_scheduler,
        start_time=start_time,
        get_dual_q_values_fn=get_dual_q_values_procgen,
        rnd_frame_fn=procgen_rnd_frame,
    )

    for upd in range(int(args.algo.n_upds)):
        runner_state, metrics = step_fn_update(runner_state, None)

        train_state_now, _, _, opt_dur_table = runner_state

        if args.wb.on and args.algo.save_video and (
            (upd % max(1, int(args.algo.n_upds / 3)) == 0) or (upd == args.algo.n_upds - 1)
        ):
            log_hneo_videos_to_wandb(
                args=args,
                train_state=train_state_now,
                video_env=video_env,
                opt_dur_table=opt_dur_table,
                step=train_state_now.n_updates,
            )


def log_hneo_videos_to_wandb(args, train_state, video_env, opt_dur_table, step):

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
        option_policy_greedy=False,
        option_policy_temp=0.,
    )

    log_dict = {
        "video/greedy_policy": wandb.Video(outputs["greedy"], fps=10, format="mp4"),
    }
    for oid, path in outputs["options"].items():
        log_dict[f"video/option_{oid}"] = wandb.Video(path, fps=10, format="mp4")

    wandb.log(log_dict, step=int(step))


if __name__ == "__main__":
    TrainEnv = make_procgen_env(args)
    VideoEnv = None
    n_actions = TrainEnv.single_action_space.n

    if args.wb.on:
        create_wb_logger(args)
        VideoEnv = make_procgen_video_env(args, wrapper_cls=JaxLogProcgenEnvPoolWrapper)

    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    key, setup_key, q_key = jax.random.split(jax.random.PRNGKey(args.exp.seed), 3)

    with jax.default_device(jax.devices("gpu")[0]):
        try:
            TrainEnv = JaxLogProcgenEnvPoolWrapper(TrainEnv)

            init_obs, _ = TrainEnv.reset()
            train_state, _ = setup(setup_key, args, init_obs, n_actions)
            p_q_learning(q_key, TrainEnv, train_state, video_env=VideoEnv)

        except Exception:
            traceback.print_exc()
            if args.wb.on:
                wandb.finish(exit_code=1)
        finally:
            TrainEnv.close()
            if VideoEnv is not None:
                VideoEnv.close()
            if args.wb.on:
                close_logger()
            jax.clear_backends()