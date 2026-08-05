"""Shared environment and logging definitions for Brax SAC agents."""

import importlib.machinery
import inspect
import sys
import time
import types
from functools import partial
from typing import Any, Dict, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs
from brax.training.agents.sac import networks as sac_networks


# Stable names: SAC, SAC_RND, and SAC_NEO must keep these aliases identical.
SAC_METRIC_ALIASES = {
    "training/critic_loss": "losses/sac_critic_loss",
    "training/actor_loss": "losses/sac_actor_loss",
    "training/alpha_loss": "losses/sac_alpha_loss",
    "training/alpha": "values/alpha",
    "training/buffer_current_size": "replay/size",
    "training/sps": "values/SPS",
    "eval/episode_reward": "test/episodic_return",
    "eval/avg_episode_length": "test/episodic_length",
    "eval/episode_success": "test/success_rate",
}

# PQN_RND-compatible names emitted by SAC_RND's custom Brax training loop.
# The loop prefixes every learner diagnostic with ``training/``; these aliases
# remove only that transport prefix so dashboards can compare the algorithms.
SAC_RND_METRIC_ALIASES = {
    "training/loss/total_td": "loss/total_td",
    "training/loss/ext_td": "loss/ext_td",
    "training/loss/int_td": "loss/int_td",
    "training/Q_mean_raw": "Q_mean_raw",
    "training/Q_int_mean_raw": "Q_int_mean_raw",
    "training/RND/int_reward_mean": "RND/int_reward_mean",
    "training/RND/int_reward_max": "RND/int_reward_max",
    "training/RND/int_reward_sum_per_env_mean": "RND/int_reward_sum_per_env_mean",
    "training/RND/fwd_loss": "RND/fwd_loss",
    "training/RND/update": "RND/update",
    "training/RND/burnin": "RND/burnin",
}


def make_network_factory(args):
    """Builds a Brax SAC network factory with project-configured dimensions."""

    make_networks_signature = inspect.signature(sac_networks.make_sac_networks)
    supported = make_networks_signature.parameters

    network_kwargs: Dict[str, Any] = {
        "hidden_layer_sizes": args.net.c_dims,
    }
    layer_norm_names = ("policy_network_layer_norm", "q_network_layer_norm")
    if all(name in supported for name in layer_norm_names):
        network_kwargs.update(
            policy_network_layer_norm=args.net.layer_norm,
            q_network_layer_norm=args.net.layer_norm,
        )
    elif args.net.layer_norm:
        raise RuntimeError(
            "This installed Brax version does not expose SAC layer-normalization "
            "arguments. Upgrade Brax or run with --layer_norm False."
        )

    return partial(sac_networks.make_sac_networks, **network_kwargs)


def _scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Moves scalar JAX metrics to host and ignores non-scalar diagnostics."""

    result: Dict[str, float] = {}
    for name, value in metrics.items():
        host_value = np.asarray(jax.device_get(value))
        if host_value.size == 1:
            result[name] = float(host_value.reshape(()))
    return result


class SuccessRateAUC:
    """Accumulates evaluation success and computes trapezoidal AUC.

    The raw AUC has units of environment steps.  The normalized AUC divides
    by the evaluated step interval and is therefore the average success rate
    over training, which is the useful quantity when runs have equal horizons.
    """

    def __init__(self) -> None:
        self._success_by_step: Dict[int, float] = {}

    def add(self, step: int, success_rate: float) -> None:
        self._success_by_step[int(step)] = float(success_rate)

    def compute(self) -> Dict[str, float]:
        if len(self._success_by_step) < 2:
            return {}
        ordered = sorted(self._success_by_step.items())
        steps = np.asarray([item[0] for item in ordered], dtype=np.float64)
        success = np.asarray([item[1] for item in ordered], dtype=np.float64)
        widths = np.diff(steps)
        auc = float(np.sum(widths * (success[:-1] + success[1:]) * 0.5))
        span = float(steps[-1] - steps[0])
        if span <= 0.0:
            return {}
        return {
            "test/success_rate_auc_env_steps": auc,
            "test/success_rate_auc_normalized": auc / span,
            "test/success_rate_auc_start_step": float(steps[0]),
            "test/success_rate_auc_end_step": float(steps[-1]),
            "test/success_rate_auc_num_points": float(len(steps)),
        }

    def print_summary(self) -> Dict[str, float]:
        result = self.compute()
        if not result:
            print(
                "Final test/success_rate AUC unavailable: at least two "
                "evaluation points are required.",
                flush=True,
            )
            return result
        print(
            "Final test/success_rate AUC | "
            f"auc_env_steps={result['test/success_rate_auc_env_steps']:.6g} | "
            f"auc_normalized={result['test/success_rate_auc_normalized']:.6f} | "
            f"range=[{int(result['test/success_rate_auc_start_step'])}, "
            f"{int(result['test/success_rate_auc_end_step'])}] | "
            f"points={int(result['test/success_rate_auc_num_points'])}",
            flush=True,
        )
        return result


def make_progress_fn(args, extra_aliases=None, success_auc=None):
    """Creates the shared console/W&B callback used by all Brax SAC agents."""

    start_time = time.time()
    aliases = {**SAC_METRIC_ALIASES, **(extra_aliases or {})}

    def progress(num_steps: int, metrics: Mapping[str, Any]) -> None:
        step = int(num_steps)
        raw = _scalar_metrics(metrics)
        elapsed = max(time.time() - start_time, 1e-8)

        prefill_actor_steps = -(-args.algo.starts // args.env.n_envs)
        prefill_env_steps = (
            prefill_actor_steps * args.env.n_envs * args.env.action_repeat
        )
        actor_steps = max(step - prefill_env_steps, 0) // (
            args.env.n_envs * args.env.action_repeat
        )
        logged: Dict[str, float] = {
            **raw,
            "env_step": step,
            "update_steps": actor_steps,
            "grad_steps": actor_steps * args.algo.grad_updates_per_step,
            "values/SPS": raw.get("training/sps", step / elapsed),
        }
        for brax_name, stable_name in aliases.items():
            if brax_name in raw:
                logged[stable_name] = raw[brax_name]

        if success_auc is not None and "test/success_rate" in logged:
            success_auc.add(step, logged["test/success_rate"])

        message = [f"env_step={step}"]
        if "test/episodic_return" in logged:
            message.append(f"return={logged['test/episodic_return']:.3f}")
        if "test/success_rate" in logged:
            message.append(f"success={logged['test/success_rate']:.3f}")
        if "values/SPS" in logged:
            message.append(f"SPS={logged['values/SPS']:.0f}")
        print(" | ".join(message), flush=True)

        if args.wb.on:
            import wandb

            wandb.log(logged, step=step)

    return progress


class TerminateOnSuccessWrapper(envs.Wrapper):
    """Terminates a goal episode on its first successful transition."""

    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, state, action):
        next_state = self.env.step(state, action)
        if "success" not in next_state.metrics:
            raise KeyError(
                "e_terminate_on_success=True requires an environment metric "
                "named 'success'"
            )
        done = jnp.maximum(next_state.done, next_state.metrics["success"])
        return next_state.replace(done=done)


def ensure_pixel_ant_maze(environment, args):
    """Idempotently adds PixelAntMaze to a possibly pre-wrapped environment.

    Keeping this separate from ``make_environment`` matters for callers that
    construct an environment themselves and then pass ``environment=...`` to
    the SAC-RND trainer.
    """

    if not getattr(args.env, "pixel_obs", False):
        return environment
    layouts = {
        "ant_u_maze": "u_maze",
        "ant_big_maze": "big_maze",
        "ant_hardest_maze": "hardest_maze",
    }
    if args.env.source != "jaxgcrl" or args.env.id not in layouts:
        raise ValueError("e_pixel_obs requires a supported JaxGCRL AntMaze")

    # Environment wrappers expose their child as ``env``.  Look through the
    # full chain so calling this from both make_environment and train is safe.
    current = environment
    visited = set()
    while current is not None and id(current) not in visited:
        if getattr(current, "is_pixel_ant_maze_wrapper", False):
            return environment
        visited.add(id(current))
        current = getattr(current, "env", None)

    try:
        # Some experiments keep the complete pixel stack in
        # src.mains.pixel_brax instead of src.mains.ant_maze.
        from src.mains.pixel_brax.pixel_ant_maze import PixelAntMazeWrapper
    except ImportError:
        from src.mains.ant_maze.pixel_ant_maze import PixelAntMazeWrapper

    return PixelAntMazeWrapper(
        environment,
        maze_layout_name=layouts[args.env.id],
        height=args.env.pixel_size,
        observation_mode="state",
    )


def make_environment(args):
    """Creates either a JaxGCRL AntMaze or a stock Brax environment."""

    if args.env.source == "jaxgcrl":
        layouts = {
            "ant_u_maze": "u_maze",
            "ant_big_maze": "big_maze",
            "ant_hardest_maze": "hardest_maze",
        }
        if args.env.id not in layouts:
            raise ValueError(
                f"Unsupported JaxGCRL AntMaze id: {args.env.id!r}. Choose "
                f"one of {tuple(layouts)}."
            )

        # JaxGCRL's package initializer imports legacy agents which still
        # reference brax.v1.  The AntMaze environment itself uses current Brax,
        # so load that submodule without executing the unrelated initializer.
        package_spec = importlib.machinery.PathFinder.find_spec("jaxgcrl")
        if package_spec is None or not package_spec.submodule_search_locations:
            raise ImportError(
                "AntMaze is not included in stock Brax. Install jaxgcrl, or "
                "smoke-test with --e_env_source brax --e_id ant."
            )
        if "jaxgcrl" not in sys.modules:
            package = types.ModuleType("jaxgcrl")
            package.__file__ = package_spec.origin
            package.__package__ = "jaxgcrl"
            package.__path__ = list(package_spec.submodule_search_locations)
            synthetic_spec = importlib.machinery.ModuleSpec(
                "jaxgcrl", loader=None, is_package=True
            )
            synthetic_spec.submodule_search_locations = package.__path__
            package.__spec__ = synthetic_spec
            sys.modules["jaxgcrl"] = package

        from jaxgcrl.envs.ant_maze import AntMaze

        environment = AntMaze(
            backend=args.env.backend,
            maze_layout_name=layouts[args.env.id],
        )
        fixed_goal = getattr(args.env, "fixed_goal", None)
        if fixed_goal is not None:
            goal = np.asarray(fixed_goal, dtype=np.float32)
            valid_goals = np.asarray(environment.possible_goals)
            matches = np.all(
                np.isclose(valid_goals, goal[None, :], atol=1e-6), axis=-1
            )
            if not np.any(matches):
                choices = [tuple(map(float, item)) for item in valid_goals]
                raise ValueError(
                    f"e_fixed_goal={tuple(goal)} is not a designated goal "
                    f"for {args.env.id}. Valid coordinates: {choices}"
                )
            # AntMaze._random_target samples only from possible_goals.  A
            # singleton array therefore gives every train/eval reset the same
            # desired goal while preserving the environment implementation.
            environment.possible_goals = jnp.asarray(goal[None, :])
        environment = ensure_pixel_ant_maze(environment, args)
        if args.env.terminate_on_success:
            environment = TerminateOnSuccessWrapper(environment)
        return environment

    if "maze" in args.env.id.lower():
        raise ValueError(
            f"'{args.env.id}' is not in the stock Brax registry. Use "
            "--e_env_source jaxgcrl for an AntMaze task."
        )
    if getattr(args.env, "fixed_goal", None) is not None:
        raise ValueError("e_fixed_goal is supported only for JaxGCRL AntMaze")
    if getattr(args.env, "pixel_obs", False):
        raise ValueError("e_pixel_obs is supported only for JaxGCRL AntMaze")
    return envs.get_environment(env_name=args.env.id, backend=args.env.backend)


__all__ = [
    "SAC_METRIC_ALIASES",
    "SAC_RND_METRIC_ALIASES",
    "SuccessRateAUC",
    "TerminateOnSuccessWrapper",
    "ensure_pixel_ant_maze",
    "make_environment",
    "make_network_factory",
    "make_progress_fn",
]
