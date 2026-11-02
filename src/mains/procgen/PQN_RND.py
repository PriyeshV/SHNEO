import os, sys
sys.path.append(os.getcwd())

import random
import time
import traceback
import numpy as np
from functools import partial

from src.config.procgen.pqn_laplacian import *
args = convert_args_to_dict(parse_args())

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import optax
from flax import struct

from src.mains.algorithms.RND import RNDState, rnd_setup
from src.utils.procgen_wrappers import JaxLogProcgenEnvPoolWrapper
from src.mains.procgen.pqn_procgen_utils import (
    Enc_State,
    FFN_State,
    make_procgen_env,
    build_enc_state,
    build_ffn_state,
    build_lr_schedule,
    eps_greedy_exploration,
    get_dual_q_values_procgen,
    procgen_rnd_frame,
    make_wandb_log_callback,
)
from src.mains.agents.pqn_rnd_core import update_step_common


@struct.dataclass
class Train_State:
    enc_state: Enc_State
    q_ext_state: FFN_State
    q_int_state: FFN_State
    rnd_state: RNDState
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def setup(key, args, obs, n_actions):
    enc_key, qext_key, qint_key, rnd_key = jax.random.split(key, 4)

    enc_state = build_enc_state(enc_key, args, obs, name="PQN:Enc")
    q_ext_state = build_ffn_state(qext_key, args, n_actions, name="PQN:Q_ext")
    q_int_state = build_ffn_state(qint_key, args, n_actions, name="PQN:Q_int")

    obs_shape = (obs.shape[2], obs.shape[3], obs.shape[1])
    rnd_state = rnd_setup(
        key=rnd_key,
        lr=args.net.lr,
        max_grad_norm=args.net.max_grad_norm,
        n_envs=args.env.n_envs,
        int_gamma=args.rnd.int_gamma,
        obs_shape=obs_shape,
    )

    return Train_State(
        enc_state=enc_state,
        q_ext_state=q_ext_state,
        q_int_state=q_int_state,
        rnd_state=rnd_state,
    ), build_lr_schedule(args)


def p_q_learning(key, env, train_state):
    init_obs, env_state = env.reset()
    expl_state = (init_obs, env_state)

    eps_scheduler = optax.linear_schedule(
        args.algo.eps.start,
        args.algo.eps.end,
        int(args.algo.eps.frac * args.algo.n_upds),
    )

    runner_state = (train_state, expl_state, key)

    start_time = time.time()
    step_fn_update = partial(
        update_step_common,
        args=args,
        env=env,
        eps_scheduler=eps_scheduler,
        start_time=start_time,
        eps_greedy_exploration_fn=eps_greedy_exploration,
        get_dual_q_values_fn=get_dual_q_values_procgen,
        rnd_frame_fn=procgen_rnd_frame,
        make_wandb_log_callback_fn=make_wandb_log_callback,
    )

    for _ in range(int(args.algo.n_upds)):
        runner_state, metrics = step_fn_update(runner_state, None)


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
            train_state, _ = setup(setup_key, args, init_obs, n_actions)
            p_q_learning(q_key, TrainEnv, train_state)

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