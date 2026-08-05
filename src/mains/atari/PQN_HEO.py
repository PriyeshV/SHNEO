import os, sys
sys.path.append(os.getcwd())

import random
import traceback
import numpy as np
from functools import partial

from src.config.atari.pqn_laplacian import *
args = convert_args_to_dict(parse_args())

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import optax
from flax import struct

from src.utils.purejax_atari_wrappers import JaxLogEnvPoolWrapper
from src.mains.atari.pqn_atari_utils import (
    Enc_State,
    FFN_State,
    make_atari_env,
    build_enc_state,
    build_ffn_state,
    build_lr_schedule,
)
from src.mains.algorithms.HEO import init_meta_buffer
from src.mains.agents.pqn_heo_core import (
    Lap_State,
    build_lap_state_common,
    eigenvalues_to_option_durations_skip_first,
    update_step_common,
)


@struct.dataclass
class Train_State:
    enc_state: Enc_State
    q_state: FFN_State
    lap_state: Lap_State
    opt_enc_state: Enc_State
    opt_q_states: list
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def setup(key, args, obs, n_prim_actions):
    n_base_options = args.options.n_options
    n_total_options = 2 * n_base_options if args.options.bidir_options else n_base_options
    n_meta_actions = n_prim_actions + n_total_options

    k_enc, k_q, k_opt_enc, k_lap, k_opt = jax.random.split(key, 5)

    enc_state = build_enc_state(k_enc, args, obs, name="PQN:Enc")
    q_state = build_ffn_state(k_q, args, n_meta_actions, name="PQN:MetaQ")
    opt_enc_state = build_enc_state(k_opt_enc, args, obs, name="Opt:Enc")

    opt_q_states = []
    key_opt = k_opt
    for _ in range(n_total_options):
        key_opt, kq = jax.random.split(key_opt)
        opt_q_states.append(build_ffn_state(kq, args, n_prim_actions, name="Opt:Q"))

    lap_state = build_lap_state_common(k_lap, args, obs)

    train_state = Train_State(
        enc_state=enc_state,
        q_state=q_state,
        lap_state=lap_state,
        opt_enc_state=opt_enc_state,
        opt_q_states=opt_q_states,
    )
    return train_state, build_lr_schedule(args)


def p_q_learning(key, env, train_state):
    init_obs, env_state = env.reset()

    n_base_options = args.options.n_options
    n_total_options = 2 * n_base_options if args.options.bidir_options else n_base_options
    n_total_envs = args.env.n_total_envs

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
        opt_dur_table = jax.numpy.full(
            (n_total_options,),
            jax.numpy.int32(args.options.opt_dur),
            dtype=jax.numpy.int32,
        )

    meta_buf = init_meta_buffer(init_obs, n_total_envs, args.options.m_steps)

    seg_start_obs = init_obs
    seg_start_meta = jax.numpy.full((n_total_envs,), -1, dtype=jax.numpy.int32,)
    seg_start_behav_prob = jax.numpy.ones((n_total_envs,), dtype=jax.numpy.float32)
    active_meta = jax.numpy.full((n_total_envs,), -1, dtype=jax.numpy.int32)
    opt_steps_left = jax.numpy.zeros((n_total_envs,), dtype=jax.numpy.int32)

    seg_ext_return = jax.numpy.zeros((n_total_envs,), dtype=jax.numpy.float32)
    seg_ext_discount = jax.numpy.ones((n_total_envs,), dtype=jax.numpy.float32)
    seg_len = jax.numpy.zeros((n_total_envs,), dtype=jax.numpy.int32)

    expl_state = (
        seg_start_obs,
        seg_start_meta,
        seg_start_behav_prob,

        init_obs,
        env_state,

        active_meta,
        opt_steps_left,

        seg_ext_return,
        seg_ext_discount,
        seg_len,

        meta_buf,
    )

    eps_scheduler = optax.linear_schedule(
        args.algo.eps.start,
        args.algo.eps.end,
        int(args.algo.eps.frac * args.algo.n_upds),
    )

    runner_state = (train_state, expl_state, key, opt_dur_table)

    start_time = __import__("time").time()
    step_fn_update = partial(
        update_step_common,
        args=args,
        env=env,
        eps_scheduler=eps_scheduler,
        start_time=start_time,
    )
    runner_state, metrics = jax.lax.scan(step_fn_update, runner_state, None, args.algo.n_upds)
    return runner_state, metrics


if __name__ == "__main__":
    Env = make_atari_env(args)
    n_actions = Env.single_action_space.n

    if args.wb.on:
        create_wb_logger(args)
        import wandb
        wandb.save("src/networks/networks.py", policy="now")
        wandb.save("src/losses/laplacian.py", policy="now")
        wandb.save("src/mains/algorithms/PQN.py", policy="now")
        wandb.save("src/mains/algorithms/EO.py", policy="now")
        wandb.save("src/mains/algorithms/HEO.py", policy="now")
        wandb.save("src/mains/algorithms/Laplacian.py", policy="now")

    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    Key, setup_key, q_key = jax.random.split(jax.random.PRNGKey(args.exp.seed), 3)

    with jax.default_device(jax.devices("gpu")[0]):
        try:
            Env = JaxLogEnvPoolWrapper(Env)
            init_obs, _ = Env.reset()
            Train_State_, _ = setup(setup_key, args, init_obs, n_actions)
            p_q_learning(q_key, Env, Train_State_)

        except Exception:
            traceback.print_exc()
            if args.wb.on:
                import wandb
                wandb.finish(exit_code=1)
        finally:
            Env.close()
            if args.wb.on:
                close_logger()
            jax.clear_backends()