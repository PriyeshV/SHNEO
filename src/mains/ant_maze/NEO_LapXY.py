"""Reliable, flat Novel Eigen-Option SAC for JaxGCRL AntMaze.

This is the continuous-action counterpart of ``PQN_NEO.py``/``DQN_SNEO.py``:

* primitive SAC-RND remains unchanged and is the policy used for evaluation;
* RND novelty learns a reliability-weighted source Laplacian;
* one continuous SAC option is learned per source eigenfunction; and
* an epsilon scheduler chooses between an eigen-option and the stochastic
  primitive SAC policy, with no learned meta-controller or option hierarchy.

The existing Brax uniform replay stores independent transitions, so the source
Laplacian update is intentionally one-step.  Treating unrelated replay samples
as an n-step trajectory would silently implement the wrong graph objective.
RND and the XY Laplacian use the shared auxiliary-update interval (16
vectorized actor steps by default for NEO), independently of SAC replay's
gradient-updates-per-step setting.
"""

import os
import sys
from typing import Any, Dict, Tuple

sys.path.append(os.getcwd())

from src.config.ant_maze.sac_laplacian import (  # noqa: E402
    close_logger,
    convert_args_to_dict,
    create_wb_logger,
    parse_args,
)


ARGS = convert_args_to_dict(parse_args("sac_neo"))
RND = ARGS.rnd
NEO = ARGS

os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(ARGS.gpu.id))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(ARGS.gpu.perc))
os.environ["HEO_SAC_RND_LIBRARY_MODE"] = "1"

import flax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from flax import linen as nn  # noqa: E402

from src.losses.laplacian import (  # noqa: E402
    multi_step_ALLO,
    update_ALLO_parameters,
)
from src.mains.ant_maze.RND import (  # noqa: E402
    _PMAP_AXIS_NAME,
    _apply_gradients,
    _rnd_intrinsic_reward,
    train as train_sac_rnd,
)
from src.mains.ant_maze.sac_utils import make_environment  # noqa: E402

os.environ.pop("HEO_SAC_RND_LIBRARY_MODE", None)


_GOAL_DIMS = 2
_PRIMITIVE = -1


class LaplacianMLP(nn.Module):
    hidden_dims: Tuple[int, ...]
    output_dim: int

    @nn.compact
    def __call__(self, xy):
        x = xy
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width)(x))
        return nn.Dense(self.output_dim)(x)


class OptionPolicy(nn.Module):
    hidden_dims: Tuple[int, ...]
    n_options: int
    action_size: int

    @nn.compact
    def __call__(self, observation):
        x = observation
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width)(x))
        output = nn.Dense(self.n_options * self.action_size * 2)(x)
        output = output.reshape(
            output.shape[:-1] + (self.n_options, self.action_size * 2)
        )
        mean, log_std = jnp.split(output, 2, axis=-1)
        return mean, jnp.clip(log_std, -5.0, 2.0)


class OptionCritic(nn.Module):
    hidden_dims: Tuple[int, ...]
    n_options: int

    @nn.compact
    def __call__(self, observation, actions):
        """Returns two Q values for every option: ``[..., option, twin]``."""

        leading = actions.shape[:-2]
        obs = jnp.broadcast_to(
            observation[..., None, :],
            leading + (self.n_options, observation.shape[-1]),
        )
        ids = jnp.broadcast_to(
            jnp.eye(self.n_options, dtype=observation.dtype),
            leading + (self.n_options, self.n_options),
        )
        x = jnp.concatenate([obs, actions, ids], axis=-1)
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width)(x))
        return nn.Dense(2)(x)


@flax.struct.dataclass
class SNEOState:
    lap_params: Any
    lap_optimizer_state: optax.OptState
    barrier_coeffs: jax.Array
    betas: jax.Array
    errors: jax.Array
    quadratic_errors: jax.Array
    option_policy_params: Any
    option_policy_optimizer_state: optax.OptState
    option_q_params: Any
    option_target_q_params: Any
    option_q_optimizer_state: optax.OptState
    option_alpha_params: jax.Array
    option_alpha_optimizer_state: optax.OptState
    active_option: jax.Array
    option_steps_left: jax.Array
    rnd_bonus_max_ema: jax.Array
    max_inv_ema: jax.Array
    mean_scale_ema: jax.Array
    eigenvalues: jax.Array


def _qema(prev, values, tau, quantile):
    estimate = jnp.quantile(values.reshape(-1), quantile)
    estimate = jax.lax.pmax(estimate, axis_name=_PMAP_AXIS_NAME)
    return jnp.maximum(prev, (1.0 - tau) * prev + tau * estimate)


class SNEOExplorer:
    """Exploration module consumed by the shared SAC-RND training loop."""

    def __init__(self, config):
        self.config = config

    def metric_aliases(self):
        """Remove the trainer's transport prefix from Atari-compatible names."""

        names = [
            "Eigen/update",
            "Eigen/Total_loss",
            "Eigen/Pos_loss",
            "Eigen/Ortho_loss",
            "Eigen/Dual_loss",
            "Eigen/Barrier",
            "Eigen/DualNorm",
            "Eigen/ErrorNorm",
            "Eigen/QuadraticErrorNorm",
            "Eigen/Max_EigenValue",
            "Explore/Min_Novelty",
            "Explore/Mean_Novelty",
            "Explore/Max_Novelty",
            "Explore/Min_Novelty_Norm",
            "Explore/Mean_Novelty_Norm",
            "Explore/Max_Novelty_Norm",
            "Explore/bonus_p50",
            "Explore/bonus_p90",
            "Explore/bonus_p99",
            "Counts/MinInv",
            "Counts/MeanInv",
            "Counts/MaxInv",
            "NEO/max_inv_ema",
            "NEO/zeta_min",
            "NEO/zeta_mean",
            "NEO/zeta_max",
            "EigenRew/min",
            "EigenRew/mean",
            "EigenRew/max",
            "RND/fwd_l2",
            "RND/obs_std",
            "RND/rew_std",
            "RND/bonus_max_rollout",
            "RND/bonus_max_ema",
            "Options/actor_loss",
            "Options/critic_loss",
            "Options/alpha_loss",
            "Options/alpha_mean",
            "Options/dur_min",
            "Options/dur_max",
            "Options/dur_mean",
            "behavior/primitive_action_fraction",
            "behavior/random_action_fraction",
            "behavior/option_action_fraction",
            "behavior/epsilon",
        ]
        for index in range(self.config.n_options):
            names.extend(
                [
                    f"Options/TD_loss_{index}",
                    f"Options/TD_ext_{index}",
                    f"Options/TD_int_{index}",
                    f"Options/Q_mean_raw_{index}",
                    f"Options/Q_int_mean_raw_{index}",
                ]
            )
        return {f"training/{name}": name for name in names}

    def configure(
        self,
        *,
        args,
        rnd,
        obs_size,
        action_size,
        target_network,
        predictor_network,
        rnd_input="state",
    ):
        if obs_size <= _GOAL_DIMS:
            raise ValueError("SAC_NEO requires state followed by two goal coordinates")
        self.args = args
        self.rnd = rnd
        # Space the shared RND novelty model is defined over.  ``xy`` puts RND
        # in the same low-dimensional space as this Laplacian, which is what
        # makes the reliability weighting informative about maze coverage.
        self.rnd_input = rnd_input
        self.obs_size = obs_size
        self.option_obs_size = obs_size - _GOAL_DIMS
        self.action_size = action_size
        self.target_network = target_network
        self.predictor_network = predictor_network
        self.lap_network = LaplacianMLP(
            self.config.lap_hidden_dims, self.config.lap_dims
        )
        self.option_policy = OptionPolicy(
            self.config.option_hidden_dims,
            self.config.n_options,
            action_size,
        )
        self.option_q = OptionCritic(
            self.config.option_hidden_dims, self.config.n_options
        )
        self.lap_optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.lap_max_grad_norm),
            optax.adamw(self.config.lap_lr, eps=1.5e-4),
        )
        self.option_policy_optimizer = optax.adam(self.config.option_lr)
        self.option_q_optimizer = optax.adam(self.config.option_lr)
        self.option_alpha_optimizer = optax.adam(self.config.option_alpha_lr)

    def _option_obs(self, observation, normalizer_params=None):
        if normalizer_params is not None and self.args.env.normalize_obs:
            observation = running_statistics.normalize(
                observation, normalizer_params
            )
        return observation[..., :-_GOAL_DIMS]

    def _lap_obs(self, observation):
        # AntMaze's first two state entries are the global torso x/y position.
        # The desired-goal coordinates are deliberately excluded.
        return observation[..., :2]

    def _sample_options(self, params, observation, key, normalizer_params=None):
        mean, log_std = self.option_policy.apply(
            {"params": params},
            self._option_obs(observation, normalizer_params),
        )
        noise = jax.random.normal(key, mean.shape)
        raw_action = mean + jnp.exp(log_std) * noise
        action = jnp.tanh(raw_action)
        gaussian_log_prob = -0.5 * (
            jnp.square(noise) + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
        )
        log_prob = jnp.sum(
            gaussian_log_prob - jnp.log(1.0 - jnp.square(action) + 1e-6),
            axis=-1,
        )
        return action, log_prob

    def init_state(self, key, local_envs):
        lap_key, policy_key, q_eig_key, q_int_key = jax.random.split(key, 4)
        dummy_xy = jnp.zeros((1, 2), dtype=jnp.float32)
        dummy_obs = jnp.zeros((1, self.option_obs_size), dtype=jnp.float32)
        dummy_actions = jnp.zeros(
            (1, self.config.n_options, self.action_size), dtype=jnp.float32
        )
        lap_params = self.lap_network.init(lap_key, dummy_xy)["params"]
        policy_params = self.option_policy.init(policy_key, dummy_obs)["params"]
        q_params = {
            "eig": self.option_q.init(q_eig_key, dummy_obs, dummy_actions)["params"],
            "int": self.option_q.init(q_int_key, dummy_obs, dummy_actions)["params"],
        }
        log_alpha = jnp.zeros((self.config.n_options,), dtype=jnp.float32)
        return SNEOState(
            lap_params=lap_params,
            lap_optimizer_state=self.lap_optimizer.init(lap_params),
            barrier_coeffs=(
                jnp.ones((1, 1), dtype=jnp.float32)
                * (
                    self.config.laplacian.barrier.value
                    if self.config.laplacian.barrier.on
                    else 0.0
                )
            ),
            betas=jnp.zeros(
                (self.config.lap_dims, self.config.lap_dims),
                dtype=jnp.float32,
            ),
            errors=jnp.zeros(
                (self.config.lap_dims, self.config.lap_dims),
                dtype=jnp.float32,
            ),
            quadratic_errors=jnp.zeros(
                (self.config.lap_dims, self.config.lap_dims),
                dtype=jnp.float32,
            ),
            option_policy_params=policy_params,
            option_policy_optimizer_state=self.option_policy_optimizer.init(policy_params),
            option_q_params=q_params,
            option_target_q_params=q_params,
            option_q_optimizer_state=self.option_q_optimizer.init(q_params),
            option_alpha_params=log_alpha,
            option_alpha_optimizer_state=self.option_alpha_optimizer.init(log_alpha),
            active_option=jnp.full((local_envs,), _PRIMITIVE, dtype=jnp.int32),
            option_steps_left=jnp.zeros((local_envs,), dtype=jnp.int32),
            rnd_bonus_max_ema=jnp.asarray(1.0, dtype=jnp.float32),
            max_inv_ema=jnp.asarray(1.0, dtype=jnp.float32),
            mean_scale_ema=jnp.asarray(1.0, dtype=jnp.float32),
            eigenvalues=jnp.full((self.config.lap_dims,), 1e-3, dtype=jnp.float32),
        )

    def select_action(self, extra, train_state, observation, base_policy, key):
        choice_key, id_key, base_key, option_key = jax.random.split(key, 4)
        progress = jnp.clip(
            train_state.env_steps
            / jnp.maximum(self.config.eps_frac * self.args.algo.tot_steps, 1.0),
            0.0,
            1.0,
        )
        epsilon = self.config.eps_start + progress * (
            self.config.eps_end - self.config.eps_start
        )
        draws = jax.random.uniform(choice_key, extra.active_option.shape)
        option_ids = jax.random.randint(
            id_key,
            extra.active_option.shape,
            0,
            self.config.n_options,
        )
        new_choice = extra.option_steps_left <= 0
        proposed = jnp.where(
            draws < epsilon,
            option_ids,
            _PRIMITIVE,
        )
        active = jnp.where(new_choice, proposed, extra.active_option)
        # Use equal-length behavior blocks so epsilon is approximately the
        # fraction of transitions generated by eigen-options.  The primitive
        # SAC policy remains stochastic and samples a fresh action every step.
        duration = jnp.full_like(active, self.config.option_duration)
        steps_left = jnp.where(new_choice, duration, extra.option_steps_left)

        base_action, _ = base_policy(observation, base_key)
        option_actions, _ = self._sample_options(
            extra.option_policy_params,
            observation,
            option_key,
            train_state.normalizer_params,
        )
        gather_ids = jnp.maximum(active, 0)
        option_action = jnp.take_along_axis(
            option_actions,
            gather_ids[..., None, None],
            axis=-2,
        ).squeeze(-2)
        action = jnp.where(
            (active >= 0)[..., None],
            option_action,
            base_action,
        )
        extra = extra.replace(active_option=active, option_steps_left=steps_left)
        metrics = {
            "behavior/primitive_action_fraction": jnp.mean(active == _PRIMITIVE),
            # Preserve the established W&B metric name for old dashboards.
            "behavior/random_action_fraction": jnp.asarray(0.0),
            "behavior/option_action_fraction": jnp.mean(active >= 0),
            "behavior/epsilon": epsilon,
        }
        return action, {}, extra, metrics

    def finish_step(self, extra, transitions):
        steps_left = extra.option_steps_left - 1
        reset = (steps_left <= 0) | (transitions.discount <= 0)
        return extra.replace(
            active_option=jnp.where(reset, _PRIMITIVE, extra.active_option),
            option_steps_left=jnp.where(reset, 0, steps_left),
        )

    def _burnin(self, train_state):
        """True while the RND intrinsic reward is still forced to zero."""

        return train_state.env_steps < self.rnd.obs_burnin

    def _normalized_novelty(self, train_state, observations, pixels=None):
        raw = _rnd_intrinsic_reward(
            train_state,
            self.target_network,
            self.predictor_network,
            observations,
            pixels,
            self.rnd_input,
        )
        novelty = raw / jnp.maximum(train_state.rnd_rew_rms.std, 1e-8)
        return jnp.where(
            self._burnin(train_state), jnp.zeros_like(novelty), novelty
        )

    def _lap_novelty(self, train_state, novelty):
        """Novelty weights for the Laplacian, made safe during RND burn-in.

        ``compute_multi_step_graph_drawing_loss`` normalizes its novelty
        weights both across time and across the batch.  An all-zero novelty
        vector therefore does not merely down-weight the attractive term, it
        annihilates it: ``w_time`` and ``row_w`` both collapse to 0 and the
        Laplacian objective degenerates to the orthogonality barrier alone.
        Feed uniform weights while novelty is identically zero so burn-in
        trains the plain (unweighted) graph-drawing objective instead.
        """

        return jnp.where(
            self._burnin(train_state), jnp.ones_like(novelty), novelty
        )

    def _zeta(self, extra, novelty):
        novelty_n = jnp.clip(
            novelty / (extra.rnd_bonus_max_ema + 1e-8), 0.0, 1.0
        )
        inv = jnp.minimum(
            1.0 / (novelty_n + 1e-6), self.config.neo_inv_cap
        )
        max_inv = _qema(
            extra.max_inv_ema,
            inv,
            self.config.neo_ema_tau,
            self.config.neo_bonus_q,
        )
        x = jnp.clip(inv / jnp.maximum(max_inv, 1e-8), 0.0, 1.0)
        raw_zeta = self.config.neo_delta * jnp.power(x, 1.0 / self.config.neo_k)
        mean_zeta = jax.lax.pmean(jnp.mean(raw_zeta), axis_name=_PMAP_AXIS_NAME)
        desired = jnp.clip(
            self.config.neo_target_mean / (mean_zeta + 1e-8), 0.1, 10.0
        )
        scale = (
            (1.0 - self.config.neo_ema_tau) * extra.mean_scale_ema
            + self.config.neo_ema_tau * desired
        )
        zeta = jnp.clip(raw_zeta * scale, 0.0, self.config.neo_delta)
        return novelty_n, inv, zeta, max_inv, scale

    def fresh_update(self, extra, train_state, transitions, key):
        rnd_pixels = transitions.extras.get("rnd_pixels")
        pixels0 = None if rnd_pixels is None else rnd_pixels["observation"]
        pixels1 = None if rnd_pixels is None else rnd_pixels["next_observation"]
        novelty0 = self._normalized_novelty(
            train_state, transitions.observation, pixels0
        )
        novelty1 = self._normalized_novelty(
            train_state, transitions.next_observation, pixels1
        )
        all_novelty = jnp.concatenate([novelty0, novelty1], axis=0)
        bonus_max = _qema(
            extra.rnd_bonus_max_ema,
            all_novelty,
            self.config.neo_ema_tau,
            self.config.neo_bonus_q,
        )
        extra = extra.replace(rnd_bonus_max_ema=bonus_max)
        novelty_n, inv, zeta, max_inv, scale = self._zeta(extra, all_novelty)
        batch = novelty0.shape[0]
        zeta0, zeta1 = zeta[:batch], zeta[batch:]

        xy0 = self._lap_obs(transitions.observation)
        xy1 = self._lap_obs(transitions.next_observation)
        # Brax's AutoResetWrapper overwrites next_observation with the
        # episode's *initial* observation whenever done, so a done transition
        # is not a graph edge: keeping it would wire every terminal state to
        # the maze start.  JaxGCRL AntMaze terminates on unhealthy z as well
        # as on the time limit, so this is not a rare event early in training
        # and ``1 - truncation`` misses all of it.  ``discount`` is
        # ``1 - done`` and covers both cases.
        edge_valid = transitions.discount
        lap_novelty = self._lap_novelty(train_state, novelty1)

        def lap_loss(params):
            phi0 = self.lap_network.apply({"params": params}, xy0)
            phi1 = self.lap_network.apply({"params": params}, xy1)
            valid = edge_valid
            total, (positive, ortho, dual_diagnostic), (_, error_matrix) = multi_step_ALLO(
                key=key,
                phis_all=phi0,
                future_phis_all=phi1[:, None, :],
                actions=jnp.zeros((phi0.shape[0], 1), dtype=jnp.int32),
                sample_mask=valid.astype(jnp.bool_),
                masks=valid[:, None].astype(jnp.bool_),
                barrier_coeffs=extra.barrier_coeffs,
                betas=extra.betas,
                weights=None,
                novelty_values=lap_novelty[:, None],
                zeta_values=jnp.stack([zeta0, zeta1], axis=1),
                use_novelty=self.config.lap_novel_weight,
                use_zeta=True,
                normalize_zeta=False,
                gamma=self.args.env.gamma,
                state_action=False,
                uni_policy=False,
                n_steps=1,
                use_dual=self.config.laplacian.dual.on,
                use_weights=False,
                average_loss=self.config.lap_average_loss,
                rms_norm=False,
                barrier_dims=1,
            )
            rayleigh = jnp.sum(jnp.square(phi1 - phi0), axis=0) / (
                jnp.sum(0.5 * (jnp.square(phi0) + jnp.square(phi1)), axis=0)
                + 1e-8
            )
            eig_reward = (
                jnp.abs(phi1[:, : self.config.n_options])
                - jnp.abs(phi0[:, : self.config.n_options])
            ) * edge_valid[:, None]
            return total, (
                positive,
                ortho,
                dual_diagnostic,
                rayleigh,
                eig_reward,
                error_matrix,
            )

        actor_steps = train_state.env_steps // (
            self.args.env.n_envs * self.args.env.action_repeat
        )
        should_update_lap = (
            actor_steps % self.config.lap_update_interval == 0
        )

        def update_laplacian(_):
            (loss, aux), grads = jax.value_and_grad(
                lap_loss, has_aux=True
            )(extra.lap_params)
            lap_params, lap_optimizer_state = _apply_gradients(
                self.lap_optimizer,
                extra.lap_params,
                extra.lap_optimizer_state,
                grads,
            )
            (
                positive,
                ortho,
                dual_diagnostic,
                rayleigh,
                eig_reward,
                error_matrix,
            ) = aux
            rayleigh = jax.lax.pmean(
                rayleigh, axis_name=_PMAP_AXIS_NAME
            )
            eigenvalues = 0.9 * extra.eigenvalues + 0.1 * rayleigh
            error_matrix = jax.tree_util.tree_map(
                lambda value: jax.lax.pmean(
                    value, axis_name=_PMAP_AXIS_NAME
                ),
                error_matrix,
            )
            updated_extra = extra.replace(
                lap_params=lap_params,
                lap_optimizer_state=lap_optimizer_state,
            )
            updated_extra = update_ALLO_parameters(
                updated_extra,
                error_matrix,
                self.config.laplacian.rate,
                self.config.laplacian.barrier,
                self.config.laplacian.dual,
                average_loss=self.config.lap_average_loss,
                barrier_dims=1,
            )
            return (
                lap_params,
                lap_optimizer_state,
                updated_extra.barrier_coeffs,
                updated_extra.betas,
                updated_extra.errors,
                updated_extra.quadratic_errors,
                eigenvalues,
                loss,
                positive,
                ortho,
                dual_diagnostic,
                eig_reward,
            )

        def skip_laplacian(_):
            # Still compute the current collection-time eigen-reward for
            # diagnostics.  Only the optimizer update is skipped.
            phi0 = self.lap_network.apply({"params": extra.lap_params}, xy0)
            phi1 = self.lap_network.apply({"params": extra.lap_params}, xy1)
            eig_reward = jax.lax.stop_gradient(
                jnp.abs(phi1[:, : self.config.n_options])
                - jnp.abs(phi0[:, : self.config.n_options])
            ) * edge_valid[:, None]
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return (
                extra.lap_params,
                extra.lap_optimizer_state,
                extra.barrier_coeffs,
                extra.betas,
                extra.errors,
                extra.quadratic_errors,
                extra.eigenvalues,
                zero,
                zero,
                zero,
                zero,
                eig_reward,
            )

        (
            lap_params,
            lap_optimizer_state,
            barrier_coeffs,
            betas,
            errors,
            quadratic_errors,
            eigenvalues,
            loss,
            positive,
            ortho,
            dual_diagnostic,
            eig_reward,
        ) = jax.lax.cond(
            should_update_lap,
            update_laplacian,
            skip_laplacian,
            operand=None,
        )
        extra = extra.replace(
            lap_params=lap_params,
            lap_optimizer_state=lap_optimizer_state,
            barrier_coeffs=barrier_coeffs,
            betas=betas,
            errors=errors,
            quadratic_errors=quadratic_errors,
            max_inv_ema=max_inv,
            mean_scale_ema=scale,
            eigenvalues=eigenvalues,
        )
        metrics = {
            "Eigen/update": should_update_lap.astype(jnp.float32),
            "Eigen/Total_loss": loss,
            "Eigen/Pos_loss": positive,
            "Eigen/Ortho_loss": ortho,
            "Eigen/Dual_loss": dual_diagnostic,
            "Eigen/Barrier": barrier_coeffs[0, 0],
            "Eigen/DualNorm": jnp.linalg.norm(betas),
            "Eigen/ErrorNorm": jnp.linalg.norm(errors),
            "Eigen/QuadraticErrorNorm": jnp.linalg.norm(quadratic_errors),
            "Eigen/Max_EigenValue": jnp.max(eigenvalues),
            "Explore/Min_Novelty": jnp.min(all_novelty),
            "Explore/Mean_Novelty": jnp.mean(all_novelty),
            "Explore/Max_Novelty": jnp.max(all_novelty),
            "Explore/Min_Novelty_Norm": jnp.min(novelty_n),
            "Explore/Mean_Novelty_Norm": jnp.mean(novelty_n),
            "Explore/Max_Novelty_Norm": jnp.max(novelty_n),
            "Explore/bonus_p50": jnp.percentile(all_novelty, 50.0),
            "Explore/bonus_p90": jnp.percentile(all_novelty, 90.0),
            "Explore/bonus_p99": jnp.percentile(all_novelty, 99.0),
            "Counts/MinInv": jnp.min(inv),
            "Counts/MeanInv": jnp.mean(inv),
            "Counts/MaxInv": jnp.max(inv),
            "NEO/max_inv_ema": max_inv,
            "NEO/zeta_min": jnp.min(zeta),
            "NEO/zeta_mean": jnp.mean(zeta),
            "NEO/zeta_max": jnp.max(zeta),
            "EigenRew/min": jnp.min(eig_reward),
            "EigenRew/mean": jnp.mean(eig_reward),
            "EigenRew/max": jnp.max(eig_reward),
            "RND/fwd_l2": jnp.mean(novelty1),
            "RND/obs_std": jnp.mean(train_state.rnd_obs_rms.std),
            "RND/rew_std": train_state.rnd_rew_rms.std,
            "RND/bonus_max_rollout": jnp.max(all_novelty),
            "RND/bonus_max_ema": bonus_max,
            "Options/dur_min": jnp.asarray(self.config.option_duration),
            "Options/dur_max": jnp.asarray(self.config.option_duration),
            "Options/dur_mean": jnp.asarray(self.config.option_duration),
        }
        return extra, metrics

    def sgd_update(self, extra, train_state, transitions, key):
        alpha_key, critic_key, actor_key = jax.random.split(key, 3)
        obs = self._option_obs(
            transitions.observation, train_state.normalizer_params
        )
        next_obs = self._option_obs(
            transitions.next_observation, train_state.normalizer_params
        )
        replay_actions = jnp.broadcast_to(
            transitions.action[..., None, :],
            transitions.action.shape[:-1]
            + (self.config.n_options, transitions.action.shape[-1]),
        )
        if self.rnd_input == "pixels":
            # Frames are not stored in replay, so reuse the value computed from
            # the correct next-state pixels during collection.
            novelty = transitions.extras["rnd_reward"]
        else:
            novelty = self._normalized_novelty(
                train_state,
                transitions.next_observation,
            )
        novelty_n = jnp.clip(
            novelty / (extra.rnd_bonus_max_ema + 1e-8), 0.0, 1.0
        )
        phi0 = self.lap_network.apply(
            {"params": extra.lap_params}, self._lap_obs(transitions.observation)
        )
        phi1 = self.lap_network.apply(
            {"params": extra.lap_params}, self._lap_obs(transitions.next_observation)
        )
        # Zero on a done transition: AutoResetWrapper made next_observation the
        # reset state, so the magnitude difference across it is meaningless.
        eig_reward = jax.lax.stop_gradient(
            jnp.abs(phi1[:, : self.config.n_options])
            - jnp.abs(phi0[:, : self.config.n_options])
        ) * transitions.discount[..., None]

        def alpha_loss(log_alpha):
            _, log_prob = self._sample_options(
                extra.option_policy_params,
                transitions.observation,
                alpha_key,
                train_state.normalizer_params,
            )
            target_entropy = -self.config.option_target_entropy_scale * self.action_size
            alpha = jnp.exp(log_alpha)[None, :]
            return jnp.mean(
                alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
            )

        alpha_value, alpha_grads = jax.value_and_grad(alpha_loss)(
            extra.option_alpha_params
        )
        alpha_params, alpha_optimizer_state = _apply_gradients(
            self.option_alpha_optimizer,
            extra.option_alpha_params,
            extra.option_alpha_optimizer_state,
            alpha_grads,
        )
        alpha = jnp.exp(extra.option_alpha_params)[None, :]

        def critic_loss(q_params):
            q_eig = self.option_q.apply(
                {"params": q_params["eig"]}, obs, replay_actions
            )
            q_int = self.option_q.apply(
                {"params": q_params["int"]}, obs, replay_actions
            )
            next_actions, next_log_prob = self._sample_options(
                extra.option_policy_params,
                transitions.next_observation,
                critic_key,
                train_state.normalizer_params,
            )
            next_q_eig = self.option_q.apply(
                {"params": extra.option_target_q_params["eig"]},
                next_obs,
                next_actions,
            )
            next_q_int = self.option_q.apply(
                {"params": extra.option_target_q_params["int"]},
                next_obs,
                next_actions,
            )
            discount = transitions.discount[..., None]
            eig_target = jax.lax.stop_gradient(
                eig_reward
                + discount
                * self.config.option_gamma
                * (jnp.min(next_q_eig, axis=-1) - alpha * next_log_prob)
            )
            int_target = jax.lax.stop_gradient(
                novelty_n[..., None]
                + discount
                * self.config.option_gamma
                * jnp.min(next_q_int, axis=-1)
            )
            truncation = transitions.extras["state_extras"]["truncation"]
            valid = (1.0 - truncation)[..., None, None]
            eig_error = (q_eig - eig_target[..., None]) * valid
            int_error = (q_int - int_target[..., None]) * valid
            eig_td_by_option = 0.5 * jnp.mean(jnp.square(eig_error), axis=(0, 2))
            int_td_by_option = 0.5 * jnp.mean(jnp.square(int_error), axis=(0, 2))
            total = jnp.sum(
                eig_td_by_option
                + self.config.neo_rnd_coef * int_td_by_option
            )
            return total, (
                eig_td_by_option,
                int_td_by_option,
                jnp.mean(q_eig, axis=(0, 2)),
                jnp.mean(q_int, axis=(0, 2)),
            )

        (critic_value, critic_aux), q_grads = jax.value_and_grad(
            critic_loss, has_aux=True
        )(extra.option_q_params)
        q_params, q_optimizer_state = _apply_gradients(
            self.option_q_optimizer,
            extra.option_q_params,
            extra.option_q_optimizer_state,
            q_grads,
        )

        def actor_loss(policy_params):
            actions, log_prob = self._sample_options(
                policy_params,
                transitions.observation,
                actor_key,
                train_state.normalizer_params,
            )
            q_eig = self.option_q.apply(
                {"params": extra.option_q_params["eig"]}, obs, actions
            )
            q_int = self.option_q.apply(
                {"params": extra.option_q_params["int"]}, obs, actions
            )
            objective = (
                jnp.min(q_eig, axis=-1)
                + self.config.neo_rnd_coef * jnp.min(q_int, axis=-1)
            )
            return jnp.sum(jnp.mean(alpha * log_prob - objective, axis=0))

        actor_value, policy_grads = jax.value_and_grad(actor_loss)(
            extra.option_policy_params
        )
        policy_params, policy_optimizer_state = _apply_gradients(
            self.option_policy_optimizer,
            extra.option_policy_params,
            extra.option_policy_optimizer_state,
            policy_grads,
        )
        target_q_params = jax.tree_util.tree_map(
            lambda old, new: (
                old * (1.0 - self.config.option_tau)
                + new * self.config.option_tau
            ),
            extra.option_target_q_params,
            q_params,
        )
        extra = extra.replace(
            option_policy_params=policy_params,
            option_policy_optimizer_state=policy_optimizer_state,
            option_q_params=q_params,
            option_target_q_params=target_q_params,
            option_q_optimizer_state=q_optimizer_state,
            option_alpha_params=alpha_params,
            option_alpha_optimizer_state=alpha_optimizer_state,
        )
        eig_td, int_td, q_eig, q_int = critic_aux
        metrics: Dict[str, jax.Array] = {
            "Options/actor_loss": actor_value,
            "Options/critic_loss": critic_value,
            "Options/alpha_loss": alpha_value,
            "Options/alpha_mean": jnp.mean(jnp.exp(alpha_params)),
        }
        for index in range(self.config.n_options):
            metrics[f"Options/TD_loss_{index}"] = (
                eig_td[index] + self.config.neo_rnd_coef * int_td[index]
            )
            metrics[f"Options/TD_ext_{index}"] = eig_td[index]
            metrics[f"Options/TD_int_{index}"] = int_td[index]
            metrics[f"Options/Q_mean_raw_{index}"] = q_eig[index]
            metrics[f"Options/Q_int_mean_raw_{index}"] = q_int[index]
        return extra, metrics


def train(args=ARGS, rnd=RND, neo=NEO, environment=None):
    """Runs SAC-NEO through the shared, current SAC-RND training loop.

    Keeping evaluation and primitive learning in ``train_sac_rnd`` means NEO
    automatically uses the same goal-free/pixel RND inputs, observation and
    intrinsic-reward normalization, replay handling, stable metric aliases,
    and final ``test/success_rate`` AUC calculation as SAC_RND.
    """

    if environment is None:
        environment = make_environment(args)
    return train_sac_rnd(
        args,
        rnd,
        environment=environment,
        exploration=SNEOExplorer(neo),
    )


def main() -> None:
    print("Available JAX devices:", jax.devices())
    print(
        f"Environment: {ARGS.env.source}:{ARGS.env.id} "
        f"(backend={ARGS.env.backend}, SAC-SNEO, "
        f"fixed_goal={ARGS.env.fixed_goal}, options={NEO.n_options})"
    )
    failed = False
    logger_started = False
    try:
        environment = make_environment(ARGS)
        if ARGS.wb.on:
            create_wb_logger(ARGS)
            logger_started = True
        train(ARGS, RND, NEO, environment=environment)
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
