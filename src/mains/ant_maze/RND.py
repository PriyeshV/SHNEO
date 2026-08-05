"""SAC with Random Network Distillation for Brax AntMaze.

This follows the repository's PQN_RND decomposition: separate extrinsic and
intrinsic twin critics are learned, while one stochastic actor maximizes
``min(Q_ext) + a_int_coef * min(Q_int)``.  By default the RND
target/predictor are MLPs over goal-free state.  With ``--e_pixel_obs True``
they use convolutional networks over PixelAntMaze frames instead, while the
SAC actor and critics keep the same vector observation as the baseline.

The surrounding environment, evaluation, replay, batching, and logging flow
tracks Brax SAC so SAC.py and SAC_RND.py remain directly comparable.
"""

import functools
import inspect
import os
import sys
import time
from typing import Any, Dict, Mapping, Tuple

sys.path.append(os.getcwd())

from src.config.ant_maze.sac_laplacian import (  # noqa: E402
    close_logger,
    convert_args_to_dict,
    create_wb_logger,
    parse_args,
)


# SAC_NEO reuses this trainer with an exploration module.  In library mode its
# wrapper has already parsed arguments and configured JAX before importing us.
_LIBRARY_MODE = os.environ.get("HEO_SAC_RND_LIBRARY_MODE", "0") == "1"
if _LIBRARY_MODE:
    ARGS = None
    RND = None
else:
    ARGS = convert_args_to_dict(parse_args("sac_rnd"))
    RND = ARGS.rnd
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(ARGS.gpu.id))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(ARGS.gpu.perc))

import flax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from brax import envs  # noqa: E402
from brax.io import model  # noqa: E402
from brax.training import acting, pmap, replay_buffers, types  # noqa: E402
from brax.training.acme import running_statistics, specs  # noqa: E402
from brax.training.agents.sac import networks as sac_networks  # noqa: E402
from flax import linen as nn  # noqa: E402
try:  # noqa: E402
    from src.mains.ant_maze.sac_utils import (
        SAC_RND_METRIC_ALIASES,
        SuccessRateAUC,
        ensure_pixel_ant_maze,
        make_environment,
        make_network_factory,
        make_progress_fn,
    )
except ImportError:  # Support the original sac_utils.py used by SAC.py.
    from src.mains.ant_maze.sac_utils import SAC_METRIC_ALIASES, make_environment

    SAC_RND_METRIC_ALIASES = {
        "training/loss/total_td": "loss/total_td",
        "training/loss/ext_td": "loss/ext_td",
        "training/loss/int_td": "loss/int_td",
        "training/Q_mean_raw": "Q_mean_raw",
        "training/Q_int_mean_raw": "Q_int_mean_raw",
        "training/RND/int_reward_mean": "RND/int_reward_mean",
        "training/RND/int_reward_max": "RND/int_reward_max",
        "training/RND/int_reward_sum_per_env_mean": (
            "RND/int_reward_sum_per_env_mean"
        ),
        "training/RND/fwd_loss": "RND/fwd_loss",
        "training/RND/update": "RND/update",
        "training/RND/burnin": "RND/burnin",
    }

    def make_network_factory(args):
        """Build a SAC network factory across supported Brax versions."""

        supported = inspect.signature(
            sac_networks.make_sac_networks
        ).parameters
        kwargs = {"hidden_layer_sizes": args.net.c_dims}
        layer_norm_names = (
            "policy_network_layer_norm",
            "q_network_layer_norm",
        )
        if all(name in supported for name in layer_norm_names):
            kwargs.update(
                policy_network_layer_norm=args.net.layer_norm,
                q_network_layer_norm=args.net.layer_norm,
            )
        elif args.net.layer_norm:
            raise RuntimeError(
                "This Brax version does not expose SAC layer-normalization "
                "arguments. Run with --layer_norm False."
            )
        return functools.partial(sac_networks.make_sac_networks, **kwargs)

    def _scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
        result = {}
        for name, value in metrics.items():
            host_value = np.asarray(jax.device_get(value))
            if host_value.size == 1:
                result[name] = float(host_value.reshape(()))
        return result

    def make_progress_fn(args, extra_aliases=None, success_auc=None):
        """Create the same stable console/W&B logging used by SAC.py."""

        start_time = time.time()
        aliases = {**SAC_METRIC_ALIASES, **(extra_aliases or {})}

        def progress(num_steps: int, metrics: Mapping[str, Any]) -> None:
            step = int(num_steps)
            raw = _scalar_metrics(metrics)
            elapsed = max(time.time() - start_time, 1e-8)
            prefill_actor_steps = -(-args.algo.starts // args.env.n_envs)
            prefill_env_steps = (
                prefill_actor_steps
                * args.env.n_envs
                * args.env.action_repeat
            )
            actor_steps = max(step - prefill_env_steps, 0) // (
                args.env.n_envs * args.env.action_repeat
            )
            logged = {
                **raw,
                "env_step": step,
                "update_steps": actor_steps,
                "grad_steps": (
                    actor_steps * args.algo.grad_updates_per_step
                ),
                "values/SPS": raw.get("training/sps", step / elapsed),
            }
            for source_name, stable_name in aliases.items():
                if source_name in raw:
                    logged[stable_name] = raw[source_name]

            if success_auc is not None and "test/success_rate" in logged:
                success_auc.add(step, logged["test/success_rate"])

            message = [f"env_step={step}"]
            if "test/episodic_return" in logged:
                message.append(
                    f"return={logged['test/episodic_return']:.3f}"
                )
            if "test/success_rate" in logged:
                message.append(
                    f"success={logged['test/success_rate']:.3f}"
                )
            message.append(f"SPS={logged['values/SPS']:.0f}")
            print(" | ".join(message), flush=True)

            if args.wb.on:
                import wandb

                wandb.log(logged, step=step)

        return progress

    class SuccessRateAUC:
        """Compatibility implementation for an older shared sac_utils."""

        def __init__(self):
            self._success_by_step = {}

        def add(self, step, success_rate):
            self._success_by_step[int(step)] = float(success_rate)

        def compute(self):
            if len(self._success_by_step) < 2:
                return {}
            ordered = sorted(self._success_by_step.items())
            steps = np.asarray([item[0] for item in ordered], dtype=np.float64)
            success = np.asarray([item[1] for item in ordered], dtype=np.float64)
            auc = float(
                np.sum(np.diff(steps) * (success[:-1] + success[1:]) * 0.5)
            )
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

        def print_summary(self):
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

    def ensure_pixel_ant_maze(environment, args):
        """Compatibility fallback for an older shared sac_utils module."""

        if not getattr(args.env, "pixel_obs", False):
            return environment
        layouts = {
            "ant_u_maze": "u_maze",
            "ant_big_maze": "big_maze",
            "ant_hardest_maze": "hardest_maze",
        }
        current = environment
        while current is not None:
            if getattr(current, "is_pixel_ant_maze_wrapper", False):
                return environment
            current = getattr(current, "env", None)
        try:
            from src.mains.pixel_brax.pixel_ant_maze import PixelAntMazeWrapper
        except ImportError:
            from src.mains.ant_maze.pixel_ant_maze import PixelAntMazeWrapper

        return PixelAntMazeWrapper(
            environment,
            maze_layout_name=layouts[args.env.id],
            height=args.env.pixel_size,
            observation_mode="state",
        )


Transition = types.Transition
PRNGKey = types.PRNGKey
Params = types.Params
ReplayBufferState = Any
_PMAP_AXIS_NAME = "i"
_ANTMAZE_GOAL_DIMS = 2


class RNDTarget(nn.Module):
    """Fixed RND target supporting either vector or HWC pixel input."""

    hidden_dims: Tuple[int, ...]
    feature_dim: int
    pixel_input: bool = False

    @nn.compact
    def __call__(self, obs):
        x = obs
        if self.pixel_input:
            for features, kernel, stride in (
                (32, (8, 8), (4, 4)),
                (64, (4, 4), (2, 2)),
                (64, (3, 3), (1, 1)),
            ):
                x = nn.leaky_relu(
                    nn.Conv(features, kernel, strides=stride, padding="VALID")(x),
                    negative_slope=0.2,
                )
            x = x.reshape((x.shape[0], -1))
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width)(x))
        return nn.Dense(self.feature_dim)(x)


class RNDPredictor(nn.Module):
    """Trainable RND predictor for vector or HWC pixel input."""

    hidden_dims: Tuple[int, ...]
    feature_dim: int
    pixel_input: bool = False

    @nn.compact
    def __call__(self, obs):
        x = obs
        if self.pixel_input:
            for features, kernel, stride in (
                (32, (8, 8), (4, 4)),
                (64, (4, 4), (2, 2)),
                (64, (3, 3), (1, 1)),
            ):
                x = nn.leaky_relu(
                    nn.Conv(features, kernel, strides=stride, padding="VALID")(x),
                    negative_slope=0.2,
                )
            x = x.reshape((x.shape[0], -1))
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width)(x))
        x = nn.relu(nn.Dense(self.feature_dim)(x))
        x = nn.relu(nn.Dense(self.feature_dim)(x))
        return nn.Dense(self.feature_dim)(x)


@flax.struct.dataclass
class TrainingState:
    """Learner state, including the decomposed critics and vector RND state."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    rnd_optimizer_state: optax.OptState
    rnd_predictor_params: Params
    rnd_target_params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    rnd_obs_rms: running_statistics.RunningStatisticsState
    rnd_rew_rms: running_statistics.RunningStatisticsState
    rnd_return: jax.Array
    gradient_steps: jax.Array
    env_steps: jax.Array
    exploration_state: Any


def _unpmap(value):
    return jax.tree_util.tree_map(lambda x: x[0], value)


def _rnd_observation(obs: jax.Array) -> jax.Array:
    """Remove JaxGCRL AntMaze's final ``target_x, target_y`` entries."""

    return obs[..., :-_ANTMAZE_GOAL_DIMS]


def _rnd_input(obs: jax.Array, pixels: jax.Array | None = None) -> jax.Array:
    """Selects pixel RND input, otherwise strips the vector goal entries."""

    return pixels if pixels is not None else _rnd_observation(obs)


def _rnd_normalize_input(
    state: TrainingState, rnd_observation: jax.Array
) -> jax.Array:
    return running_statistics.normalize(
        rnd_observation,
        state.rnd_obs_rms,
        max_abs_value=5.0,
    )


def _rnd_normalize_obs(
    state: TrainingState,
    obs: jax.Array,
    pixels: jax.Array | None = None,
) -> jax.Array:
    return _rnd_normalize_input(state, _rnd_input(obs, pixels))


def _rnd_intrinsic_reward(
    state: TrainingState,
    target_network: RNDTarget,
    predictor_network: RNDPredictor,
    obs: jax.Array,
    pixels: jax.Array | None = None,
) -> jax.Array:
    normalized_obs = _rnd_normalize_obs(state, obs, pixels)
    target = target_network.apply(
        {"params": state.rnd_target_params}, normalized_obs
    )
    prediction = predictor_network.apply(
        {"params": state.rnd_predictor_params}, normalized_obs
    )
    return 0.5 * jnp.sum(jnp.square(target - prediction), axis=-1)


def _apply_gradients(optimizer, params, optimizer_state, grads):
    grads = jax.lax.pmean(grads, axis_name=_PMAP_AXIS_NAME)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    return optax.apply_updates(params, updates), optimizer_state


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    rnd_obs_shape: Tuple[int, ...],
    local_devices_to_use: int,
    local_envs: int,
    sac_network,
    target_network: RNDTarget,
    predictor_network: RNDPredictor,
    alpha_optimizer,
    policy_optimizer,
    q_optimizer,
    rnd_optimizer,
    exploration,
) -> TrainingState:
    policy_key, q_ext_key, q_int_key, target_key, predictor_key = jax.random.split(
        key, 5
    )
    policy_params = sac_network.policy_network.init(policy_key)
    q_params = {
        "ext": sac_network.q_network.init(q_ext_key),
        "int": sac_network.q_network.init(q_int_key),
    }
    dummy_rnd_obs = jnp.zeros((1, *rnd_obs_shape), dtype=jnp.float32)
    rnd_target_params = target_network.init(target_key, dummy_rnd_obs)["params"]
    rnd_predictor_params = predictor_network.init(
        predictor_key, dummy_rnd_obs
    )["params"]
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)

    exploration_state = None
    if exploration is not None:
        exploration_state = exploration.init_state(
            jax.random.fold_in(key, 9173), local_envs
        )

    state = TrainingState(
        policy_optimizer_state=policy_optimizer.init(policy_params),
        policy_params=policy_params,
        q_optimizer_state=q_optimizer.init(q_params),
        q_params=q_params,
        target_q_params=q_params,
        alpha_optimizer_state=alpha_optimizer.init(log_alpha),
        alpha_params=log_alpha,
        rnd_optimizer_state=rnd_optimizer.init(rnd_predictor_params),
        rnd_predictor_params=rnd_predictor_params,
        rnd_target_params=rnd_target_params,
        normalizer_params=running_statistics.init_state(
            specs.Array((obs_size,), jnp.dtype("float32"))
        ),
        rnd_obs_rms=running_statistics.init_state(
            specs.Array(rnd_obs_shape, jnp.dtype("float32"))
        ),
        rnd_rew_rms=running_statistics.init_state(
            specs.Array((), jnp.dtype("float32"))
        ),
        rnd_return=jnp.zeros((local_envs,), dtype=jnp.float32),
        gradient_steps=jnp.zeros((), dtype=jnp.int32),
        env_steps=jnp.zeros((), dtype=jnp.int32),
        exploration_state=exploration_state,
    )
    return jax.device_put_replicated(
        state, jax.local_devices()[:local_devices_to_use]
    )


def train(args=None, rnd=None, environment=None, exploration=None):
    """Trains decomposed SAC-RND and returns the Brax inference interface."""

    if args is None:
        args = ARGS
    if rnd is None:
        rnd = RND
    if args is None or rnd is None:
        raise ValueError("args and rnd are required when SAC_RND is imported in library mode")

    if args.algo.save_video:
        raise NotImplementedError("Set --a_save_video False; video export is not implemented.")
    if args.algo.starts >= args.algo.tot_steps:
        raise ValueError("a_starts must be smaller than a_tot_steps")
    if environment is None:
        environment = make_environment(args)
    # ``train(..., environment=environment)`` is part of the public entry
    # point and callers sometimes build that environment with a factory that
    # predates PixelAntMaze.  Enforce the requested wrapper here as well.
    if getattr(args.env, "pixel_obs", False):
        environment = ensure_pixel_ant_maze(environment, args)

    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if args.algo.max_devices_per_host is not None:
        local_devices_to_use = min(
            local_devices_to_use, args.algo.max_devices_per_host
        )
    device_count = local_devices_to_use * jax.process_count()
    if args.env.n_envs % device_count:
        raise ValueError("e_n_envs must be divisible by the total device count")
    if args.algo.batch_size * args.algo.grad_updates_per_step % device_count:
        raise ValueError(
            "a_batch_size * a_grad_updates_per_step must be divisible by devices"
        )

    env_steps_per_actor_step = args.env.action_repeat * args.env.n_envs
    num_prefill_actor_steps = -(-args.algo.starts // args.env.n_envs)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    if num_prefill_env_steps > args.algo.tot_steps:
        raise ValueError(
            "Rounded replay prefill exceeds a_tot_steps; reduce a_starts or "
            "e_n_envs"
        )
    num_evals_after_init = max(args.algo.n_evals - 1, 1)
    num_training_steps_per_epoch = -(
        -(args.algo.tot_steps - num_prefill_env_steps)
        // (num_evals_after_init * env_steps_per_actor_step)
    )

    env = envs.training.wrap(
        environment,
        episode_length=args.env.max_episode_steps,
        action_repeat=args.env.action_repeat,
    )
    obs_size = env.observation_size
    if isinstance(obs_size, Dict):
        raise NotImplementedError("Dictionary observations are not supported")
    action_size = env.action_size
    pixel_rnd = bool(getattr(args.env, "pixel_obs", False))
    if pixel_rnd:
        if args.env.pixel_size < 36:
            raise ValueError(
                "e_pixel_size must be at least 36 for the SAC_RND convolutional encoder"
            )
        rnd_obs_shape = (args.env.pixel_size, args.env.pixel_size, 3)
    else:
        if obs_size <= _ANTMAZE_GOAL_DIMS:
            raise ValueError(
                "AntMaze observation must contain state plus two goal coordinates"
            )
        rnd_obs_shape = (obs_size - _ANTMAZE_GOAL_DIMS,)

    normalize_fn = lambda observation, _: observation
    if args.env.normalize_obs:
        normalize_fn = running_statistics.normalize
    network_factory = make_network_factory(args)
    sac_network = network_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
    )
    make_policy = sac_networks.make_inference_fn(sac_network)

    target_network = RNDTarget(rnd.dims, rnd.feat_dim, pixel_input=pixel_rnd)
    predictor_network = RNDPredictor(
        rnd.dims, rnd.feat_dim, pixel_input=pixel_rnd
    )
    alpha_optimizer = optax.adam(args.sac.alpha_lr)
    policy_optimizer = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adam(args.net.lr),
    )
    q_optimizer = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adam(args.net.lr),
    )
    rnd_optimizer = optax.chain(
        optax.clip_by_global_norm(rnd.max_grad_norm),
        optax.adam(rnd.lr, eps=1e-5),
    )
    if exploration is not None:
        exploration.configure(
            args=args,
            rnd=rnd,
            obs_size=obs_size,
            action_size=action_size,
            target_network=target_network,
            predictor_network=predictor_network,
        )

    exploration_replay_spec = {}
    if exploration is not None and hasattr(exploration, "replay_extras_spec"):
        exploration_replay_spec = exploration.replay_extras_spec()

    dummy_obs = jnp.zeros((obs_size,), dtype=jnp.float32)
    dummy_action = jnp.zeros((action_size,), dtype=jnp.float32)
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=jnp.float32(0.0),
        discount=jnp.float32(0.0),
        next_observation=dummy_obs,
        extras={
            "state_extras": {"truncation": jnp.float32(0.0)},
            "policy_extras": {},
            "rnd_reward": jnp.float32(0.0),
            **exploration_replay_spec,
        },
    )
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=args.algo.replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=(
            args.algo.batch_size
            * args.algo.grad_updates_per_step
            // device_count
        ),
    )

    action_distribution = sac_network.parametric_action_distribution
    policy_network = sac_network.policy_network
    q_network = sac_network.q_network
    target_entropy = -args.sac.target_entropy_scale * action_size

    def alpha_loss(log_alpha, policy_params, normalizer_params, transitions, key):
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        raw_action = action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = action_distribution.log_prob(dist_params, raw_action)
        alpha = jnp.exp(log_alpha)
        return jnp.mean(
            alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
        )

    def critic_loss(
        q_params,
        policy_params,
        normalizer_params,
        target_q_params,
        alpha,
        transitions,
        intrinsic_reward,
        key,
    ):
        q_ext = q_network.apply(
            normalizer_params,
            q_params["ext"],
            transitions.observation,
            transitions.action,
        )
        q_int = q_network.apply(
            normalizer_params,
            q_params["int"],
            transitions.observation,
            transitions.action,
        )
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        raw_next_action = action_distribution.sample_no_postprocessing(
            next_dist_params, key
        )
        next_log_prob = action_distribution.log_prob(
            next_dist_params, raw_next_action
        )
        next_action = action_distribution.postprocess(raw_next_action)
        next_q_ext = q_network.apply(
            normalizer_params,
            target_q_params["ext"],
            transitions.next_observation,
            next_action,
        )
        next_q_int = q_network.apply(
            normalizer_params,
            target_q_params["int"],
            transitions.next_observation,
            next_action,
        )

        # Entropy belongs to the single SAC objective, so it is included only
        # in the extrinsic decomposition rather than being counted twice.
        next_v_ext = jnp.min(next_q_ext, axis=-1) - alpha * next_log_prob
        ext_target = jax.lax.stop_gradient(
            transitions.reward * args.algo.reward_scale
            + transitions.discount * args.env.gamma * next_v_ext
        )
        # Match PQN_RND: intrinsic value is continuing across env termination.
        int_target = jax.lax.stop_gradient(
            intrinsic_reward
            + rnd.int_gamma * jnp.min(next_q_int, axis=-1)
        )
        ext_error = q_ext - jnp.expand_dims(ext_target, -1)
        truncation = transitions.extras["state_extras"]["truncation"]
        ext_error *= jnp.expand_dims(1.0 - truncation, -1)
        int_error = q_int - jnp.expand_dims(int_target, -1)
        ext_td = 0.5 * jnp.mean(jnp.square(ext_error))
        int_td = 0.5 * jnp.mean(jnp.square(int_error))
        return ext_td + rnd.int_coef * int_td, (
            ext_td,
            int_td,
            jnp.mean(q_ext),
            jnp.mean(q_int),
        )

    def actor_loss(
        policy_params,
        normalizer_params,
        q_params,
        alpha,
        transitions,
        key,
    ):
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        raw_action = action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = action_distribution.log_prob(dist_params, raw_action)
        action = action_distribution.postprocess(raw_action)
        q_ext = q_network.apply(
            normalizer_params, q_params["ext"], transitions.observation, action
        )
        q_int = q_network.apply(
            normalizer_params, q_params["int"], transitions.observation, action
        )
        combined_q = jnp.min(q_ext, axis=-1) + rnd.int_coef * jnp.min(
            q_int, axis=-1
        )
        return jnp.mean(alpha * log_prob - combined_q)

    def predictor_loss(predictor_params, state, observations, key):
        # ``observations`` is already the selected RND input: either a pixel
        # frame or the goal-free vector state.
        normalized_obs = _rnd_normalize_input(state, observations)
        target = target_network.apply(
            {"params": state.rnd_target_params}, normalized_obs
        )
        prediction = predictor_network.apply(
            {"params": predictor_params}, normalized_obs
        )
        per_example = jnp.mean(
            jnp.square(prediction - jax.lax.stop_gradient(target)), axis=-1
        )
        mask = jax.random.bernoulli(key, rnd.upd_prop, per_example.shape)
        return jnp.sum(per_example * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    def update_predictor_once(state, observations, key):
        """Optionally updates RND from one fresh vectorized actor batch."""

        burnin = state.env_steps < rnd.obs_burnin
        actor_steps = state.env_steps // env_steps_per_actor_step
        scheduled = actor_steps % rnd.update_interval == 0
        should_update = (~burnin) & scheduled

        def update(_):
            value, grads = jax.value_and_grad(predictor_loss)(
                state.rnd_predictor_params,
                state,
                observations,
                key,
            )
            params, optimizer_state = _apply_gradients(
                rnd_optimizer,
                state.rnd_predictor_params,
                state.rnd_optimizer_state,
                grads,
            )
            return state.replace(
                rnd_predictor_params=params,
                rnd_optimizer_state=optimizer_state,
            ), value

        def skip(_):
            return state, jnp.float32(0.0)

        new_state, loss = jax.lax.cond(
            should_update, update, skip, operand=None
        )
        return new_state, loss, should_update

    def sgd_step(carry, transitions):
        state, key = carry
        key, alpha_key, critic_key, actor_key, explore_key = jax.random.split(key, 5)
        intrinsic_reward = transitions.extras["rnd_reward"]

        alpha_value, alpha_grads = jax.value_and_grad(alpha_loss)(
            state.alpha_params,
            state.policy_params,
            state.normalizer_params,
            transitions,
            alpha_key,
        )
        alpha_params, alpha_optimizer_state = _apply_gradients(
            alpha_optimizer,
            state.alpha_params,
            state.alpha_optimizer_state,
            alpha_grads,
        )
        alpha = jnp.exp(state.alpha_params)

        (critic_value, critic_aux), q_grads = jax.value_and_grad(
            critic_loss, has_aux=True
        )(
            state.q_params,
            state.policy_params,
            state.normalizer_params,
            state.target_q_params,
            alpha,
            transitions,
            intrinsic_reward,
            critic_key,
        )
        q_params, q_optimizer_state = _apply_gradients(
            q_optimizer, state.q_params, state.q_optimizer_state, q_grads
        )

        actor_value, policy_grads = jax.value_and_grad(actor_loss)(
            state.policy_params,
            state.normalizer_params,
            state.q_params,
            alpha,
            transitions,
            actor_key,
        )
        policy_params, policy_optimizer_state = _apply_gradients(
            policy_optimizer,
            state.policy_params,
            state.policy_optimizer_state,
            policy_grads,
        )

        target_q_params = jax.tree_util.tree_map(
            lambda old, new: old * (1.0 - args.algo.tau) + new * args.algo.tau,
            state.target_q_params,
            q_params,
        )
        ext_td, int_td, q_ext_mean, q_int_mean = critic_aux
        new_state = state.replace(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=target_q_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            gradient_steps=state.gradient_steps + 1,
        )
        metrics = {
            # Preserve vanilla SAC's common critic metric as extrinsic-only.
            "critic_loss": ext_td,
            "actor_loss": actor_value,
            "alpha_loss": alpha_value,
            "alpha": jnp.exp(alpha_params),
            "loss/total_td": critic_value,
            "loss/ext_td": ext_td,
            "loss/int_td": int_td,
            "Q_mean_raw": q_ext_mean,
            "Q_int_mean_raw": q_int_mean,
        }
        if exploration is not None:
            exploration_state, exploration_metrics = exploration.sgd_update(
                new_state.exploration_state,
                new_state,
                transitions,
                explore_key,
            )
            new_state = new_state.replace(exploration_state=exploration_state)
            metrics.update(exploration_metrics)
        return (new_state, key), metrics

    def get_experience(state, env_state, buffer_state, key):
        policy = make_policy((state.normalizer_params, state.policy_params))

        def autoreset_pixels(wrapped_state):
            pixels = wrapped_state.info["pixels"]
            reset_pixels = wrapped_state.info["_pixels_reset"]
            done = wrapped_state.done.reshape(
                wrapped_state.done.shape + (1,) * (pixels.ndim - wrapped_state.done.ndim)
            )
            return jnp.where(done > 0, reset_pixels, pixels)

        previous_pixels = autoreset_pixels(env_state) if pixel_rnd else None
        if exploration is None:
            env_state, transitions = acting.actor_step(
                env, env_state, policy, key, extra_fields=("truncation",)
            )
            behavior_metrics = {}
        else:
            action, policy_extras, exploration_state, behavior_metrics = (
                exploration.select_action(
                    state.exploration_state,
                    state,
                    env_state.obs,
                    policy,
                    key,
                )
            )
            previous_obs = env_state.obs
            env_state = env.step(env_state, action)
            transitions = Transition(
                observation=previous_obs,
                action=action,
                reward=env_state.reward,
                discount=1.0 - env_state.done,
                next_observation=env_state.obs,
                extras={
                    "policy_extras": policy_extras,
                    "state_extras": {
                        "truncation": env_state.info["truncation"],
                    },
                },
            )
            exploration_state = exploration.finish_step(
                exploration_state, transitions
            )
            state = state.replace(exploration_state=exploration_state)

        next_pixels = autoreset_pixels(env_state) if pixel_rnd else None
        rnd_next_observation = _rnd_input(
            transitions.next_observation, next_pixels
        )
        raw_intrinsic = _rnd_intrinsic_reward(
            state,
            target_network,
            predictor_network,
            transitions.next_observation,
            next_pixels,
        )
        next_env_steps = state.env_steps + env_steps_per_actor_step
        burnin = next_env_steps < rnd.obs_burnin

        def update_reward_stats(_):
            rnd_return = state.rnd_return * rnd.int_gamma + raw_intrinsic
            rew_rms = running_statistics.update(
                state.rnd_rew_rms,
                rnd_return,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )
            normalized = raw_intrinsic / jnp.maximum(rew_rms.std, 1e-8)
            return rnd_return, rew_rms, normalized

        def skip_reward_stats(_):
            return (
                state.rnd_return,
                state.rnd_rew_rms,
                jnp.zeros_like(raw_intrinsic),
            )

        rnd_return, rnd_rew_rms, normalized_intrinsic = jax.lax.cond(
            burnin, skip_reward_stats, update_reward_stats, operand=None
        )
        rnd_obs_rms = running_statistics.update(
            state.rnd_obs_rms,
            rnd_next_observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )
        normalizer_params = running_statistics.update(
            state.normalizer_params,
            transitions.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )
        state = state.replace(
            normalizer_params=normalizer_params,
            rnd_obs_rms=rnd_obs_rms,
            rnd_rew_rms=rnd_rew_rms,
            rnd_return=rnd_return,
            env_steps=next_env_steps,
        )
        exploration_replay_extras = {}
        if exploration is not None and hasattr(exploration, "replay_extras"):
            exploration_replay_extras = exploration.replay_extras(
                state.exploration_state,
                transitions,
                previous_pixels,
                next_pixels,
            )
        transitions = transitions._replace(
            extras={
                **transitions.extras,
                "rnd_reward": normalized_intrinsic,
                **exploration_replay_extras,
            }
        )
        buffer_state = replay_buffer.insert(buffer_state, transitions)
        # Full-resolution pixel frames are needed only by the online RND/NEO
        # update.  Attach them after replay insertion; exploration modules may
        # separately store a compact representation through replay_extras.
        fresh_transition = transitions
        if pixel_rnd:
            fresh_transition = transitions._replace(
                extras={
                    **transitions.extras,
                    "rnd_pixels": {
                        "observation": previous_pixels,
                        "next_observation": next_pixels,
                    },
                }
            )
        return (
            state,
            env_state,
            buffer_state,
            normalized_intrinsic,
            rnd_next_observation,
            burnin,
            behavior_metrics,
            fresh_transition,
        )

    def training_step(state, env_state, buffer_state, key):
        experience_key, predictor_key, fresh_key, training_key = jax.random.split(key, 4)
        (
            state,
            env_state,
            buffer_state,
            online_intrinsic,
            fresh_next_observation,
            burnin,
            behavior_metrics,
            fresh_transition,
        ) = get_experience(state, env_state, buffer_state, experience_key)
        fresh_metrics = {}
        if exploration is not None:
            exploration_state, fresh_metrics = exploration.fresh_update(
                state.exploration_state,
                state,
                transitions=fresh_transition,
                key=fresh_key,
            )
            state = state.replace(exploration_state=exploration_state)
        state, rnd_loss, rnd_updated = update_predictor_once(
            state, fresh_next_observation, predictor_key
        )
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x, (args.algo.grad_updates_per_step, -1) + x.shape[1:]
            ),
            transitions,
        )
        (state, _), metrics = jax.lax.scan(
            sgd_step, (state, training_key), transitions
        )
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        metrics["RND/fwd_loss"] = rnd_loss
        metrics["RND/update"] = rnd_updated.astype(jnp.float32)
        metrics["_rnd_rewards"] = online_intrinsic
        metrics["_rnd_burnin"] = burnin.astype(jnp.float32)
        metrics.update(behavior_metrics)
        metrics.update(fresh_metrics)
        return state, env_state, buffer_state, metrics

    def prefill_replay_buffer(state, env_state, buffer_state, key):
        def step(carry, _):
            state, env_state, buffer_state, key = carry
            key, experience_key = jax.random.split(key)
            state, env_state, buffer_state, _, _, _, _, _ = get_experience(
                state, env_state, buffer_state, experience_key
            )
            return (state, env_state, buffer_state, key), ()

        return jax.lax.scan(
            step,
            (state, env_state, buffer_state, key),
            (),
            length=num_prefill_actor_steps,
        )[0]

    prefill_replay_buffer = jax.pmap(
        prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME
    )

    def training_epoch(state, env_state, buffer_state, key):
        def step(carry, _):
            state, env_state, buffer_state, key = carry
            key, training_key = jax.random.split(key)
            state, env_state, buffer_state, metrics = training_step(
                state, env_state, buffer_state, training_key
            )
            return (state, env_state, buffer_state, key), metrics

        (state, env_state, buffer_state, key), metrics = jax.lax.scan(
            step,
            (state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = dict(metrics)
        rnd_rewards = metrics.pop("_rnd_rewards")
        rnd_burnin = metrics.pop("_rnd_burnin")
        # Exploration modules may update a Laplacian less frequently than the
        # actor.  Their skipped-step losses are zeros, so retain meaningful
        # loss scales by averaging those fields over real updates only.
        lap_update_mask = metrics.get("Eigen/update")
        lap_update_metrics = {}
        if lap_update_mask is not None:
            lap_update_count = jnp.maximum(jnp.sum(lap_update_mask), 1.0)
            for name in (
                "Eigen/Total_loss",
                "Eigen/Pos_loss",
                "Eigen/Ortho_loss",
                "Eigen/Dual_loss",
            ):
                if name in metrics:
                    lap_update_metrics[name] = (
                        jnp.sum(metrics[name]) / lap_update_count
                    )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics.update(lap_update_metrics)
        metrics.update(
            {
                "RND/int_reward_mean": jnp.mean(rnd_rewards),
                "RND/int_reward_max": jax.lax.pmax(
                    jnp.max(rnd_rewards), axis_name=_PMAP_AXIS_NAME
                ),
                # PQN_RND uses a 32-step rollout for this metric.  Keep a
                # fixed 32-actor-step window so its scale does not depend on
                # the number of evaluations configured for this SAC run.
                "RND/int_reward_sum_per_env_mean": jnp.mean(
                    jnp.sum(
                        rnd_rewards[
                            -min(32, num_training_steps_per_epoch) :
                        ],
                        axis=0,
                    )
                ),
                "RND/burnin": jnp.mean(rnd_burnin),
            }
        )
        return state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    rng = jax.random.PRNGKey(args.exp.seed)
    rng, init_key = jax.random.split(rng)
    local_key = jax.random.fold_in(rng, process_id)
    local_envs = args.env.n_envs // device_count
    state = _init_training_state(
        init_key,
        obs_size,
        rnd_obs_shape,
        local_devices_to_use,
        local_envs,
        sac_network,
        target_network,
        predictor_network,
        alpha_optimizer,
        policy_optimizer,
        q_optimizer,
        rnd_optimizer,
        exploration,
    )

    local_key, replay_key, env_key, eval_key = jax.random.split(local_key, 4)
    env_keys = jax.random.split(env_key, args.env.n_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)
    if pixel_rnd:
        missing = {
            key for key in ("pixels", "_pixels_reset") if key not in env_state.info
        }
        if missing:
            raise RuntimeError(
                "Pixel RND was requested, but the wrapped environment reset "
                f"did not produce state.info keys {sorted(missing)}. Ensure "
                "PixelAntMazeWrapper is imported from the same package as "
                "this SAC_RND trainer."
            )
    buffer_state = jax.pmap(replay_buffer.init)(
        jax.random.split(replay_key, local_devices_to_use)
    )

    eval_env = envs.training.wrap(
        environment,
        episode_length=args.env.max_episode_steps,
        action_repeat=args.env.action_repeat,
    )
    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=args.algo.deterministic_eval),
        num_eval_envs=args.env.n_test_envs,
        episode_length=args.env.max_episode_steps,
        action_repeat=args.env.action_repeat,
        key=eval_key,
    )
    metric_aliases = dict(SAC_RND_METRIC_ALIASES)
    if exploration is not None and hasattr(exploration, "metric_aliases"):
        metric_aliases.update(exploration.metric_aliases())
    success_auc = SuccessRateAUC()
    progress_fn = make_progress_fn(
        args, metric_aliases, success_auc=success_auc
    )

    metrics = {}
    if process_id == 0 and args.algo.n_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap((state.normalizer_params, state.policy_params)),
            training_metrics={},
        )
        progress_fn(0, metrics)

    start_time = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    state, env_state, buffer_state, _ = prefill_replay_buffer(
        state,
        env_state,
        buffer_state,
        jax.random.split(prefill_key, local_devices_to_use),
    )
    replay_size = (
        jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    )
    if int(replay_size) < args.algo.starts:
        raise RuntimeError("Replay buffer did not reach a_starts during prefill")
    training_walltime = time.time() - start_time

    current_step = int(_unpmap(state.env_steps))
    for _ in range(num_evals_after_init):
        epoch_start = time.time()
        epoch_key, local_key = jax.random.split(local_key)
        state, env_state, buffer_state, training_metrics = training_epoch(
            state,
            env_state,
            buffer_state,
            jax.random.split(epoch_key, local_devices_to_use),
        )
        training_metrics = jax.tree_util.tree_map(jnp.mean, training_metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
        epoch_time = time.time() - epoch_start
        training_walltime += epoch_time
        training_metrics = {
            "training/sps": (
                env_steps_per_actor_step * num_training_steps_per_epoch
            )
            / max(epoch_time, 1e-8),
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in training_metrics.items()},
        }
        current_step = int(_unpmap(state.env_steps))

        if process_id == 0:
            if args.algo.checkpoint_logdir:
                params = _unpmap((state.normalizer_params, state.policy_params))
                os.makedirs(args.algo.checkpoint_logdir, exist_ok=True)
                path = os.path.join(
                    args.algo.checkpoint_logdir,
                    f"{'sac_neo' if exploration is not None else 'sac_rnd'}_"
                    f"{current_step}.pkl",
                )
                model.save_params(path, params)
            metrics = evaluator.run_evaluation(
                _unpmap((state.normalizer_params, state.policy_params)),
                training_metrics,
            )
            progress_fn(current_step, metrics)

    if current_step < args.algo.tot_steps:
        raise AssertionError(
            f"Total steps {current_step} is less than {args.algo.tot_steps}"
        )
    pmap.synchronize_hosts()
    params = _unpmap((state.normalizer_params, state.policy_params))
    auc_metrics = success_auc.print_summary() if process_id == 0 else {}
    metrics = {**metrics, **auc_metrics}
    if process_id == 0 and args.wb.on:
        import wandb

        for name, value in auc_metrics.items():
            wandb.run.summary[name] = value
    return make_policy, params, metrics


def main() -> None:
    print("Available JAX devices:", jax.devices())
    print(
        f"Environment: {ARGS.env.source}:{ARGS.env.id} "
        f"(backend={ARGS.env.backend}, SAC-RND, "
        f"fixed_goal={ARGS.env.fixed_goal})"
    )

    failed = False
    logger_started = False
    try:
        environment = make_environment(ARGS)
        if ARGS.wb.on:
            create_wb_logger(ARGS)
            logger_started = True
        train(ARGS, RND, environment=environment)
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
