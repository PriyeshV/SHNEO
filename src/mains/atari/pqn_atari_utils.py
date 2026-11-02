from dataclasses import replace
from functools import partial
from typing import Union

import envpool
import jax
import jax.numpy as jnp

from src.mains.common.pqn_env_utils import (
    Enc_State,
    FFN_State,
    obs_chw_to_hwc,
    single_obs_chw_to_hwc,
    build_encoder_network,
    build_q_network,
    build_lr_schedule,
    build_enc_state,
    build_ffn_state,
    eps_greedy_exploration,
    get_q_values_generic,
    get_dual_q_values_generic,
    record_policy_and_option_videos_generic,
    make_wandb_log_callback,
)


def make_atari_env(args):
    env_kwargs = {
        "episodic_life": True,
        "reward_clip": args.algo.rew_scale,
        "repeat_action_probability": 0.0,
        "frame_skip": 4,
        "noop_max": 30,
        "full_action_space": False,
    }
    env = envpool.make(
        args.env.id + "-v5",
        env_type="gym",
        num_envs=int(args.env.n_total_envs),
        seed=int(args.exp.seed),
        **env_kwargs,
    )
    env.num_envs = int(args.env.n_total_envs)
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    env.name = args.env.id + "-v5"
    return env


def make_atari_video_env(args, wrapper_cls=None):
    video_env_cfg = replace(args.env, n_envs=1, n_test_envs=0, n_total_envs=1)
    video_args = replace(args, env=video_env_cfg)
    env = make_atari_env(video_args)
    if wrapper_cls is not None:
        env = wrapper_cls(env)
    return env


@jax.jit
def atari_rnd_frame(obs):
    # take last frame from stacked Atari observation [B,C,H,W]
    return obs[:, 3, :, :].astype(jnp.float32)[..., None]


@partial(jax.jit, static_argnums=(3, 4))
def get_q_values_atari(obs, enc_state, q_state, use_enc_bn, use_q_bn):
    return get_q_values_generic(obs, enc_state, q_state, use_enc_bn, use_q_bn)


@partial(jax.jit, static_argnums=(4, 5, 6))
def get_dual_q_values_atari(obs, enc_state, q_ext_state, q_int_state, use_enc_bn, use_qext_bn, use_qint_bn):
    return get_dual_q_values_generic(obs, enc_state, q_ext_state, q_int_state, use_enc_bn, use_qext_bn, use_qint_bn)


def record_atari_policy_and_option_videos(**kwargs):
    return record_policy_and_option_videos_generic(**kwargs)