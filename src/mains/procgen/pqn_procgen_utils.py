from dataclasses import replace
from functools import partial

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


# ---------------------------------------------------------------------
# Procgen task / env helpers
# ---------------------------------------------------------------------
_DISTRIBUTION_MODE_MAP = {
    "easy": 0,
    "hard": 1,
    "extreme": 2,
    "memory": 10,
}


def as_distribution_mode(x):
    if isinstance(x, str):
        x_low = x.lower()
        if x_low not in _DISTRIBUTION_MODE_MAP:
            raise ValueError(f"Unknown distribution_mode: {x}")
        return _DISTRIBUTION_MODE_MAP[x_low]
    return int(x)


def normalize_procgen_task_id(env_id: str, distribution_mode):
    if env_id.endswith("-v0"):
        return env_id

    mode_inv = {
        0: "Easy",
        1: "Hard",
        2: "Extreme",
        10: "Memory",
    }
    dist = as_distribution_mode(distribution_mode)
    if dist not in mode_inv:
        raise ValueError(f"Unsupported Procgen distribution_mode for task id construction: {dist}")

    name = env_id.strip()
    name = name[0].upper() + name[1:]
    return f"{name}{mode_inv[dist]}-v0"


def make_procgen_env(args):
    distribution_mode = as_distribution_mode(args.env.distribution_mode)
    task_id = normalize_procgen_task_id(args.env.id, distribution_mode)

    env_kwargs = dict(
        num_envs=int(args.env.n_total_envs),
        seed=int(args.exp.seed),
        channel_first=True,
        num_levels=1,
        start_level=args.env.start_level,
        distribution_mode=distribution_mode,
        center_agent=bool(getattr(args.env, "center_agent", True)),
        use_backgrounds=bool(getattr(args.env, "use_backgrounds", True)),
        use_monochrome_assets=bool(getattr(args.env, "use_monochrome_assets", False)),
        restrict_themes=bool(getattr(args.env, "restrict_themes", False)),
        use_generated_assets=bool(getattr(args.env, "use_generated_assets", False)),
        paint_vel_info=bool(getattr(args.env, "paint_vel_info", False)),
        use_easy_jump=bool(getattr(args.env, "use_easy_jump", False)),
    )

    if getattr(args.env, "max_episode_steps", None) is not None:
        env_kwargs["max_episode_steps"] = int(args.env.max_episode_steps)

    env = envpool.make(task_id, env_type="gym", **env_kwargs)
    env.num_envs = int(args.env.n_total_envs)
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    env.name = task_id
    return env


def make_procgen_video_env(args, wrapper_cls=None):
    video_env_cfg = replace(args.env, n_envs=1, n_test_envs=0, n_total_envs=1)
    video_args = replace(args, env=video_env_cfg)
    env = make_procgen_env(video_args)
    if wrapper_cls is not None:
        env = wrapper_cls(env)
    return env


@jax.jit
def procgen_rnd_frame(obs):
    return obs_chw_to_hwc(obs).astype(jnp.float32)


@partial(jax.jit, static_argnums=(3, 4))
def get_q_values_procgen(obs, enc_state, q_state, use_enc_bn, use_q_bn):
    return get_q_values_generic(obs, enc_state, q_state, use_enc_bn, use_q_bn)


@partial(jax.jit, static_argnums=(4, 5, 6))
def get_dual_q_values_procgen(obs, enc_state, q_ext_state, q_int_state, use_enc_bn, use_qext_bn, use_qint_bn):
    return get_dual_q_values_generic(obs, enc_state, q_ext_state, q_int_state, use_enc_bn, use_qext_bn, use_qint_bn)


def record_procgen_policy_and_option_videos(**kwargs):
    return record_policy_and_option_videos_generic(**kwargs)