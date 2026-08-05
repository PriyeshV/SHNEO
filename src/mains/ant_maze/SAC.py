"""JAX SAC baseline for AntMaze using Brax's maintained SAC trainer.

The default environment is JaxGCRL's sparse ``ant_u_maze`` PipelineEnv,
because stock Brax only registers the locomotion task ``ant``.  The training
algorithm itself is Brax SAC: a tanh-Gaussian actor, twin Q-functions, target
Q-functions, automatic entropy tuning, and uniform replay.

The W&B aliases in ``SAC_METRIC_ALIASES`` are part of the experiment
interface.  Keep them unchanged in SAC_NEO so baseline and reliability-weighted
runs can be plotted together.
"""

import inspect
import os
import sys
from typing import Any, Dict

sys.path.append(os.getcwd())

from src.config.ant_maze.sac_laplacian import (  # noqa: E402
    close_logger,
    convert_args_to_dict,
    create_wb_logger,
    parse_args,
)


ARGS = convert_args_to_dict(parse_args("sac"))

# These must be set before importing JAX.  With preallocation disabled the
# memory fraction is only a ceiling hint, but retaining gpu_perc keeps the
# launcher interface consistent with the rest of this repository.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(ARGS.gpu.id))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(ARGS.gpu.perc))

import jax  # noqa: E402
from brax.training.agents.sac import train as sac_train  # noqa: E402
from src.mains.ant_maze.sac_utils import (  # noqa: E402
    SuccessRateAUC,
    make_environment,
    make_network_factory,
    make_progress_fn,
)


def train(args=ARGS, environment=None):
    """Runs SAC and returns Brax's (inference_fn, params, final_metrics)."""

    if args.algo.save_video:
        raise NotImplementedError(
            "Video export is not part of this first SAC baseline. Set "
            "--a_save_video False; checkpointing is available through "
            "--a_checkpoint_logdir."
        )

    if environment is None:
        environment = make_environment(args)
    network_factory = make_network_factory(args)
    success_auc = SuccessRateAUC()

    train_kwargs: Dict[str, Any] = {
        "environment": environment,
        "num_timesteps": args.algo.tot_steps,
        "episode_length": args.env.max_episode_steps,
        "action_repeat": args.env.action_repeat,
        "num_envs": args.env.n_envs,
        "num_eval_envs": args.env.n_test_envs,
        "learning_rate": args.net.lr,
        "discounting": args.env.gamma,
        "seed": args.exp.seed,
        "batch_size": args.algo.batch_size,
        "num_evals": args.algo.n_evals,
        "normalize_observations": args.env.normalize_obs,
        "max_devices_per_host": args.algo.max_devices_per_host,
        "reward_scaling": args.algo.reward_scale,
        "tau": args.algo.tau,
        "min_replay_size": args.algo.starts,
        "max_replay_size": args.algo.replay_size,
        "grad_updates_per_step": args.algo.grad_updates_per_step,
        "deterministic_eval": args.algo.deterministic_eval,
        "network_factory": network_factory,
        "progress_fn": make_progress_fn(args, success_auc=success_auc),
    }

    train_signature = inspect.signature(sac_train.train).parameters
    if "max_grad_norm" in train_signature:
        train_kwargs["max_grad_norm"] = args.net.max_grad_norm
    required = set(train_kwargs)
    unsupported = sorted(required.difference(train_signature))
    if unsupported:
        raise RuntimeError(
            "The installed Brax SAC API is incompatible with this entry point. "
            f"Unsupported arguments: {unsupported}. Install a current Brax "
            "release compatible with JaxGCRL."
        )
    if args.algo.checkpoint_logdir is not None:
        if "checkpoint_logdir" not in train_signature:
            raise RuntimeError(
                "The installed Brax version does not support checkpoint_logdir."
            )
        train_kwargs["checkpoint_logdir"] = args.algo.checkpoint_logdir

    inference_fn, params, final_metrics = sac_train.train(**train_kwargs)
    auc_metrics = success_auc.print_summary()
    final_metrics = {**final_metrics, **auc_metrics}
    if args.wb.on:
        import wandb

        for name, value in auc_metrics.items():
            wandb.run.summary[name] = value
    return inference_fn, params, final_metrics


def main() -> None:
    print("Available JAX devices:", jax.devices())
    print(
        f"Environment: {ARGS.env.source}:{ARGS.env.id} "
        f"(backend={ARGS.env.backend}, "
        f"{'sparse goal task' if ARGS.env.source == 'jaxgcrl' else 'stock task'}, "
        f"fixed_goal={ARGS.env.fixed_goal})"
    )

    failed = False
    logger_started = False
    try:
        # Validate all environment imports before creating a W&B run.  This
        # prevents dependency/setup failures from producing empty runs.
        environment = make_environment(ARGS)
        if ARGS.wb.on:
            create_wb_logger(ARGS)
            logger_started = True
        train(ARGS, environment=environment)
    except Exception:
        failed = True
        raise
    finally:
        if logger_started:
            if failed:
                import wandb

                wandb.finish(exit_code=1)
            else:
                close_logger()
        jax.clear_caches()


if __name__ == "__main__":
    main()
