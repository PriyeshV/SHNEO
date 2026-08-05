# procgen/PQN_NEO.py
import os, sys
sys.path.append(os.getcwd())

import time
import random
import traceback
import numpy as np

from typing import List
from functools import partial

from src.config.procgen.pqn_laplacian import *
args = convert_args_to_dict(parse_args())

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import optax
import jax.numpy as jnp
from flax import struct

from src.utils.procgen_wrappers import JaxLogProcgenEnvPoolWrapper
from src.mains.procgen.pqn_procgen_utils import (
    Enc_State,
    FFN_State,
    make_procgen_env,
    build_enc_state,
    build_ffn_state,
    procgen_rnd_frame,
)
from src.mains.algorithms.RND import RNDState, rnd_setup
from src.mains.algorithms.NEO import NEOScaleState
from src.mains.procgen.pqn_procgen_utils import get_dual_q_values_procgen
from src.mains.agents.pqn_neo_core import (
    Lap_State,
    build_lap_state_common,
    eigenvalues_to_option_durations_skip_first,
    update_step_common,
    make_wandb_log_callback,
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

    k_enc, k_qext, k_qint, k_opt_enc, k_lap, k_rnd, k_opt = jax.random.split(key, 7)

    enc_state = build_enc_state(k_enc, args, obs, name="PQN:Enc")
    q_ext_state = build_ffn_state(k_qext, args, n_prim_actions, name="PQN:Q_ext")
    q_int_state = build_ffn_state(k_qint, args, n_prim_actions, name="PQN:Q_int")

    opt_enc_state = build_enc_state(k_opt_enc, args, obs, name="Opt:Enc")

    opt_q_ext_states = []
    opt_q_int_states = []
    key_opt = k_opt
    for _ in range(n_total_options):
        key_opt, kqe, kqi = jax.random.split(key_opt, 3)
        opt_q_ext_states.append(build_ffn_state(kqe, args, n_prim_actions, name="Opt:Q_ext"))
        opt_q_int_states.append(build_ffn_state(kqi, args, n_prim_actions, name="Opt:Q_int"))

    lap_state = build_lap_state_common(k_lap, args, obs)

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
    )


def p_q_learning(key, env, train_state):
    init_obs, env_state = env.reset()

    n_base_options = args.options.n_options
    n_total_options = args.options.n_options

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

    E = args.env.n_total_envs
    expl_state = (
        init_obs,  # current obs
        env_state,
        jnp.full((E,), -1, dtype=jnp.int32),  # active_opt
        jnp.zeros((E,), dtype=jnp.int32),  # opt_steps_left
    )
    runner_state = (train_state, expl_state, key, opt_dur_table)

    eps_scheduler = optax.linear_schedule(
        args.algo.eps.start,
        args.algo.eps.end,
        int(args.algo.eps.frac * args.algo.n_upds),
    )

    start_time = time.time()
    step_fn_update = partial(
        update_step_common,
        args=args,
        env=env,
        eps_scheduler=eps_scheduler,
        start_time=start_time,
        get_dual_q_values_fn=get_dual_q_values_procgen,
        rnd_frame_fn=procgen_rnd_frame,
        make_wandb_log_callback_fn=make_wandb_log_callback,
    )

    for _ in range(int(args.algo.n_upds)):
        runner_state, metrics = step_fn_update(runner_state, None)

    return runner_state


if __name__ == "__main__":
    TrainEnv = make_procgen_env(args)
    n_actions = TrainEnv.single_action_space.n

    if args.wb.on:
        create_wb_logger(args)

    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    key, setup_key, q_key = jax.random.split(jax.random.PRNGKey(args.exp.seed), 3)

    with jax.default_device(jax.devices("gpu")[0]):
        try:
            TrainEnv = JaxLogProcgenEnvPoolWrapper(TrainEnv)
            init_obs, _ = TrainEnv.reset()
            Train_State_ = setup(setup_key, args, init_obs, n_actions)
            p_q_learning(q_key, TrainEnv, Train_State_)

        except Exception:
            traceback.print_exc()
            if args.wb.on:
                import wandb
                wandb.finish(exit_code=1)
        finally:
            TrainEnv.close()
            if args.wb.on:
                close_logger()
            jax.clear_backends()