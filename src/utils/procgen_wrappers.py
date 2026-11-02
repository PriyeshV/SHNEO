import jax
import jax.numpy as jnp
from flax import struct
from functools import partial

import gym
from packaging import version

is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")
assert is_legacy_gym, "Current version supports only gym<=0.23.1"


# ---------------------------------------------------------------------
# Procgen max-min score table
# Keys are canonical Procgen env names, lowercase.
# Each entry stores (hard_min, hard_max, easy_min, easy_max)
# ---------------------------------------------------------------------
PROCGEN_SCORE_BOUNDS = {
    "coinrun":   (5.0, 10.0, 5.0, 10.0),
    "starpilot": (1.5, 35.0, 2.5, 64.0),
    "caveflyer": (2.0, 13.4, 3.5, 12.0),
    "dodgeball": (1.5, 19.0, 1.5, 19.0),
    "fruitbot":  (-0.5, 27.2, -1.5, 32.4),
    "chaser":    (0.5, 14.2, 0.5, 13.0),
    "miner":     (1.5, 20.0, 1.5, 13.0),
    "jumper":    (1.0, 10.0, 3.0, 10.0),
    "leaper":    (1.5, 10.0, 3.0, 10.0),
    "maze":      (4.0, 10.0, 5.0, 10.0),
    "bigfish":   (0.0, 40.0, 1.0, 40.0),
    "heist":     (2.0, 10.0, 3.5, 10.0),
    "climber":   (1.0, 12.6, 2.0, 12.6),
    "plunder":   (3.0, 30.0, 4.5, 30.0),
    "ninja":     (2.0, 10.0, 3.5, 10.0),
    "bossfight": (0.5, 13.0, 0.5, 13.0),
}


def _canonical_procgen_name(env_name: str) -> str:
    """
    Converts names like:
      - 'ClimberHard-v0'
      - 'climber'
      - 'ClimberEasy-v0'
    to:
      - 'climber'
    """
    x = env_name.lower().strip()

    if x.endswith("-v0"):
        x = x[:-3]

    for suffix in ("easy", "hard", "extreme", "memory"):
        if x.endswith(suffix):
            x = x[: -len(suffix)]
            break

    return x


def _distribution_mode_str(distribution_mode):
    if isinstance(distribution_mode, str):
        return distribution_mode.lower()

    mode_map = {
        0: "easy",
        1: "hard",
        2: "extreme",
        10: "memory",
    }
    return mode_map.get(int(distribution_mode), "hard")


def _get_procgen_score_bounds(env_name: str, distribution_mode="hard"):
    """
    Returns (r_min, r_max) for max-min normalization.
    For 'extreme' and 'memory', we fall back to hard bounds.
    """
    canonical = _canonical_procgen_name(env_name)
    if canonical not in PROCGEN_SCORE_BOUNDS:
        return None

    hard_min, hard_max, easy_min, easy_max = PROCGEN_SCORE_BOUNDS[canonical]
    mode = _distribution_mode_str(distribution_mode)

    if mode == "easy":
        return easy_min, easy_max

    # hard / extreme / memory -> use hard bounds
    return hard_min, hard_max


@jax.jit
def _normalize_procgen_return(ret, r_min, r_max):
    denom = jnp.maximum(r_max - r_min, 1e-8)
    return (ret - r_min) / denom


@struct.dataclass
class LogProcgenEnvState:
    handle: jnp.ndarray
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    returned_episode_normalized_returns: jnp.ndarray


class JaxLogProcgenEnvPoolWrapper(gym.Wrapper):
    """
    EnvPool+XLA wrapper for Procgen.

    Logs:
      - raw episodic returns
      - episodic lengths
      - max-min normalized episodic returns

    Normalization:
      normalized = (R - R_min) / (R_max - R_min)
    where (R_min, R_max) are Procgen reference bounds for the current game
    and distribution mode (easy/hard). For extreme/memory, hard bounds are used.
    """

    def __init__(self, env, reset_info=True, distribution_mode=None):
        super(JaxLogProcgenEnvPoolWrapper, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.env_name = getattr(env, "name", "procgen")
        self.reset_info = reset_info

        handle, recv, send, step = env.xla()
        self.init_handle = handle
        self.send_f = send
        self.recv_f = recv
        self.step_f = step

        # infer distribution mode if not explicitly passed
        if distribution_mode is None:
            distribution_mode = getattr(env, "distribution_mode", "hard")
        self.distribution_mode = distribution_mode

        bounds = _get_procgen_score_bounds(self.env_name, self.distribution_mode)
        self.has_score_bounds = True
        rmin, rmax = bounds
        self.score_rmin = jnp.array(rmin, dtype=jnp.float32)
        self.score_rmax = jnp.array(rmax, dtype=jnp.float32)

    def reset(self, **kwargs):
        observations = super(JaxLogProcgenEnvPoolWrapper, self).reset(**kwargs)

        env_state = LogProcgenEnvState(
            handle=jnp.array(self.init_handle),
            episode_returns=jnp.zeros(self.num_envs, dtype=jnp.float32),
            episode_lengths=jnp.zeros(self.num_envs, dtype=jnp.float32),
            returned_episode_returns=jnp.zeros(self.num_envs, dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros(self.num_envs, dtype=jnp.float32),
            returned_episode_normalized_returns=jnp.zeros(self.num_envs, dtype=jnp.float32),
        )
        return observations, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        new_handle, (observations, rewards, dones, infos) = self.step_f(state.handle, action)

        done_f = dones.astype(jnp.float32)

        new_episode_return = state.episode_returns + rewards
        new_episode_length = state.episode_lengths + 1.0

        new_returned_episode_returns = jnp.where(
            dones, new_episode_return, state.returned_episode_returns
        )
        new_returned_episode_lengths = jnp.where(
            dones, new_episode_length, state.returned_episode_lengths
        )

        if self.has_score_bounds:
            normalized_returns = _normalize_procgen_return(
                new_episode_return,
                self.score_rmin,
                self.score_rmax,
            )
        else:
            normalized_returns = new_episode_return

        new_returned_episode_normalized_returns = jnp.where(
            dones, normalized_returns, state.returned_episode_normalized_returns
        )

        state = state.replace(
            handle=new_handle,
            episode_returns=new_episode_return * (1.0 - done_f),
            episode_lengths=new_episode_length * (1.0 - done_f),
            returned_episode_returns=new_returned_episode_returns,
            returned_episode_lengths=new_returned_episode_lengths,
            returned_episode_normalized_returns=new_returned_episode_normalized_returns,
        )

        if self.reset_info:
            infos = {}

        infos["returned_episode_returns"] = state.returned_episode_returns
        infos["returned_episode_lengths"] = state.returned_episode_lengths
        infos["returned_episode_normalized_returns"] = state.returned_episode_normalized_returns
        infos["returned_episode"] = dones.astype(jnp.float32)

        # current live episode length
        infos["elapsed_step"] = state.episode_lengths

        return observations, state, rewards, dones, infos