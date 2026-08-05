import os, sys
sys.path.append(os.getcwd())

import random
import time
import traceback
from functools import partial

import numpy as np

from src.config.procgen.pqn_laplacian import *
args = convert_args_to_dict(parse_args())

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import optax
from flax import struct

from src.utils.procgen_wrappers import JaxLogProcgenEnvPoolWrapper
from src.mains.procgen.pqn_procgen_utils import (
    Enc_State,
    FFN_State,
    make_procgen_env,
    build_enc_state,
    build_ffn_state,
    build_lr_schedule,
    eps_greedy_exploration,
    get_q_values_procgen,
    make_wandb_log_callback,
)
from src.mains.agents.pqn_core import update_step_common


@struct.dataclass
class Train_State:
    enc_state: Enc_State
    q_state: FFN_State
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def setup(key, args, obs, n_actions):
    enc_key, q_key = jax.random.split(key, 2)
    enc_state = build_enc_state(enc_key, args, obs, name="PQN:Enc")
    q_state = build_ffn_state(q_key, args, n_actions, name="PQN:Q")
    return Train_State(enc_state=enc_state, q_state=q_state), build_lr_schedule(args)


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
        get_q_values_fn=get_q_values_procgen,
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