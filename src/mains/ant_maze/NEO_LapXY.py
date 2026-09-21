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

Because the Laplacian here consumes torso ``(x, y)`` -- which replay already
stores -- it trains on a replay batch rather than only the transitions
collected this step.  A ``lap_dims``-dimensional embedding needs many more
samples than ``e_n_envs`` for its Gram matrix to be estimable, which is what
both reference agents give it.  The novelty and zeta calibration EMAs still
advance from the online transitions, so the reliability weighting continues to
describe what the agent is currently seeing.
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
from src.mains.algorithms.Laplacian import (  # noqa: E402
    compute_batch_eigenvalues_seq,
)
from src.mains.ant_maze.pixel_ant_maze import ANT_MAZE_LAYOUTS  # noqa: E402
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
# JaxGCRL AntMaze and PixelAntMazeWrapper both default maze_size_scaling to
# 4.0; one cell is 4 world units.
_MAZE_SIZE_SCALING = 4.0
_LAYOUT_BY_ENV_ID = {
    "ant_u_maze": "u_maze",
    "ant_big_maze": "big_maze",
    "ant_hardest_maze": "hardest_maze",
}


class LaplacianMLP(nn.Module):
    hidden_dims: Tuple[int, ...]
    output_dim: int
    batch_norm: bool = False

    @nn.compact
    def __call__(self, xy, train: bool = False):
        x = xy
        for width in self.hidden_dims:
            x = nn.Dense(width)(x)
            if self.batch_norm:
                # Dense -> BatchNorm -> activation on hidden layers, as the
                # reference ``apply_layer`` does.  This is what keeps phi at a
                # controlled scale: the ALLO dual term rewards a larger
                # ||phi|| (its optimal duals are negative), and once the norm
                # outruns the barrier the orthonormality constraint can never
                # be recovered.  Without it the xy Laplacian reached mean
                # |cos| 0.47 against the true eigenvectors and then collapsed
                # to 0.18 in 500k steps as the Gram diagonals grew to ~350.
                x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.relu(x)
        # No bias on the eigenfunction layer, as the reference ``Conv_Encoder``
        # builds its MLP with ``last_bias=False``.  A bias lets the network
        # emit a constant output cheaply, which is the trivial eigenfunction;
        # without it phi_0 has to be a real mode, which is why the reference
        # keeps phi_0 as an option.
        return nn.Dense(self.output_dim, use_bias=False)(x)


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
    lap_batch_stats: Any
    lap_optimizer_state: optax.OptState
    # Rolling per-env window of the last lap_seq_len states, oldest first:
    # the on-policy sequence the reference trains its Laplacian on.  Each
    # entry is the state s_t, whether the transition out of it ended the
    # episode, and s_t's own novelty and zeta.
    lap_seq_xy: jax.Array        # [envs, K, 2]
    lap_seq_done: jax.Array      # [envs, K]
    lap_seq_novelty: jax.Array   # [envs, K]
    lap_seq_zeta: jax.Array      # [envs, K]
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
        # The reference feeds its encoder unit-scaled input (img_norm divides
        # pixels by 255).  Divide torso xy by the maze's world extent so the
        # Laplacian sees coordinates in roughly [0, 1] too.
        layout = ANT_MAZE_LAYOUTS[_LAYOUT_BY_ENV_ID[args.env.id]]
        self.xy_scale = float(len(layout) * _MAZE_SIZE_SCALING)
        self.lap_network = LaplacianMLP(
            self.config.lap_hidden_dims,
            self.config.lap_dims,
            batch_norm=self.config.laplacian.batch_norm,
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
        return observation[..., :2] / self.xy_scale

    def _lap_apply(self, params, batch_stats, xy, train):
        """Evaluates the Laplacian encoder, returning ``(phi, batch_stats)``.

        With ``train=True`` batch norm uses batch statistics and the returned
        ``batch_stats`` are the updated running averages, as the reference's
        ``mutable=["batch_stats"]`` apply does.  With ``train=False`` it uses
        the running averages and returns them unchanged.  Without batch norm
        ``batch_stats`` is an empty collection and passes straight through.
        """

        variables = {"params": params, "batch_stats": batch_stats}
        if train and self.config.laplacian.batch_norm:
            phi, updates = self.lap_network.apply(
                variables, xy, train=True, mutable=["batch_stats"]
            )
            return phi, updates["batch_stats"]
        return self.lap_network.apply(variables, xy, train=False), batch_stats

    def _option_phi(self, phi):
        """The eigenfunction columns that back the options.

        Matches ``_neo_eigen_rewards`` in ``src/mains/algorithms/NEO.py``,
        which takes ``phis[:, :n_options]`` and notes that the ALLO Laplacian
        "generally has no constant first eigenfunction".  So phi_0 is a real
        mode -- the slowest one -- and the reference keeps it as an option.
        Only the *duration* table skips index 0 (see ``_option_durations``).
        """

        return phi[:, : self.config.n_options]

    def _option_durations(self, eigenvalues):
        """Per-option durations, mirroring ``pqn_neo_core`` exactly.

        With ``o_eigdur_options`` off (the reference default) every option
        runs for the fixed ``o_opt_dur``.  With it on, option i runs for
        ceil(1 / lambda_{i+1}) clipped to the configured range, as
        ``eigenvalues_to_option_durations_skip_first`` computes it, from the
        Rayleigh-quotient EMA held in the explorer state.
        """

        if not self.config.option_eigdur:
            return jnp.full(
                (self.config.n_options,),
                self.config.option_duration,
                dtype=jnp.int32,
            )
        eigs = jnp.nan_to_num(
            jnp.reshape(eigenvalues, (-1,)), nan=0.0, posinf=0.0, neginf=0.0
        )
        eigs = jnp.clip(eigs, 0.0, jnp.inf)
        start = 1 if self.config.option_skip_first else 0
        usable = eigs[start : start + self.config.n_options]
        lam = jnp.maximum(usable, 1e-6)
        dur = jnp.ceil(1.0 / lam).astype(jnp.int32)
        return jnp.clip(
            dur,
            self.config.option_duration_min,
            self.config.option_duration_max,
        )

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
        lap_variables = self.lap_network.init(lap_key, dummy_xy, train=False)
        lap_params = lap_variables["params"]
        # Absent without batch norm; an empty collection keeps every apply
        # and lax.cond branch pytree-consistent either way.
        lap_batch_stats = lap_variables.get("batch_stats", {})
        policy_params = self.option_policy.init(policy_key, dummy_obs)["params"]
        q_params = {
            "eig": self.option_q.init(q_eig_key, dummy_obs, dummy_actions)["params"],
            "int": self.option_q.init(q_int_key, dummy_obs, dummy_actions)["params"],
        }
        log_alpha = jnp.zeros((self.config.n_options,), dtype=jnp.float32)
        seq = (local_envs, self.config.lap_seq_len)
        return SNEOState(
            lap_params=lap_params,
            lap_batch_stats=lap_batch_stats,
            lap_optimizer_state=self.lap_optimizer.init(lap_params),
            lap_seq_xy=jnp.zeros(seq + (2,), dtype=jnp.float32),
            # done=1 everywhere until real states shift in, so the sample
            # mask excludes the padding without any separate fill counter.
            lap_seq_done=jnp.ones(seq, dtype=jnp.float32),
            lap_seq_novelty=jnp.ones(seq, dtype=jnp.float32),
            lap_seq_zeta=jnp.zeros(seq, dtype=jnp.float32),
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
        eps_key, mu_key, id_key, base_key, option_key, random_key = (
            jax.random.split(key, 6)
        )
        progress = jnp.clip(
            train_state.env_steps
            / jnp.maximum(self.config.eps_frac * self.args.algo.tot_steps, 1.0),
            0.0,
            1.0,
        )
        epsilon = self.config.eps_start + progress * (
            self.config.eps_end - self.config.eps_start
        )

        # Three-way behaviour choice, matching ``pqn_neo_core``: epsilon decides
        # whether to explore at all, and mu_prob splits that exploration between
        # an eigen-option and a uniform random action.  Folding the two together,
        # as this port used to, makes epsilon the option rate directly, so
        # annealing exploration also anneals the method itself away.
        need_new_choice = (extra.active_option < 0) | (
            extra.option_steps_left <= 0
        )
        u_eps = jax.random.uniform(eps_key, extra.active_option.shape)
        u_mu = jax.random.uniform(mu_key, extra.active_option.shape)
        do_explore = need_new_choice & (u_eps < epsilon)
        explore_option = do_explore & (u_mu < self.config.option_mu_prob)
        explore_primitive = do_explore & (u_mu >= self.config.option_mu_prob)
        continuing_option = jnp.logical_not(need_new_choice)

        option_ids = jax.random.randint(
            id_key,
            extra.active_option.shape,
            0,
            self.config.n_options,
        )
        active = jnp.where(
            explore_option,
            option_ids,
            jnp.where(continuing_option, extra.active_option, _PRIMITIVE),
        )
        # Only an option holds a multi-step block.  A primitive step leaves
        # steps_left at zero so the next step decides afresh, which is what
        # keeps options running a meaningful share of the time once epsilon has
        # annealed: their share is set by their duration, not by epsilon alone.
        durations = self._option_durations(extra.eigenvalues)
        steps_left = jnp.where(
            explore_option,
            durations[option_ids],
            jnp.where(continuing_option, extra.option_steps_left, 0),
        )

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
        # Tanh-squashed policies act in [-1, 1], so that is the random action.
        random_action = jax.random.uniform(
            random_key,
            jnp.shape(base_action),
            minval=-1.0,
            maxval=1.0,
        )
        action = jnp.where(
            (active >= 0)[..., None],
            option_action,
            jnp.where(explore_primitive[..., None], random_action, base_action),
        )
        extra = extra.replace(active_option=active, option_steps_left=steps_left)
        metrics = {
            "behavior/primitive_action_fraction": jnp.mean(active == _PRIMITIVE),
            "behavior/random_action_fraction": jnp.mean(explore_primitive),
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

    def checkpoint_params(self, extra):
        """Parameters worth keeping for post-hoc analysis of a NEO run."""

        return {
            "lap_params": extra.lap_params,
            "lap_batch_stats": extra.lap_batch_stats,
            # The encoder consumes torso xy divided by this; tools that
            # evaluate phi on raw coordinates must apply the same scaling.
            "xy_scale": jnp.asarray(self.xy_scale, dtype=jnp.float32),
            "option_policy_params": extra.option_policy_params,
            "eigenvalues": extra.eigenvalues,
        }

    def fresh_update(self, extra, train_state, transitions, key, replay=None):
        # The Laplacian trains on the on-policy window kept in ``extra`` --
        # the port of the reference's per-env rollout -- so replay is unused.
        del replay
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
        zeta0 = zeta[:batch]

        xy0 = self._lap_obs(transitions.observation)
        xy1 = self._lap_obs(transitions.next_observation)
        # Brax's AutoResetWrapper overwrites next_observation with the
        # episode's *initial* observation whenever done, so a done transition
        # is not a graph edge.  ``discount`` is ``1 - done`` and covers both
        # termination and the time limit.
        done0 = 1.0 - transitions.discount

        # Append s_t to the per-env window (oldest first).  Novelty and zeta
        # are those of s_t itself.  The reference stores the novelty of
        # s_{t+1} at index t and so pairs each future state with its
        # successor's novelty; the intent is plainly the state's own, and
        # that is what is stored here.
        def push(buffer, value):
            return jnp.concatenate([buffer[:, 1:], value[:, None]], axis=1)

        seq_xy = jnp.concatenate(
            [extra.lap_seq_xy[:, 1:], xy0[:, None, :]], axis=1
        )
        seq_done = push(extra.lap_seq_done, done0)
        seq_novelty = push(
            extra.lap_seq_novelty, self._lap_novelty(train_state, novelty0)
        )
        seq_zeta = push(extra.lap_seq_zeta, zeta0)
        extra = extra.replace(
            lap_seq_xy=seq_xy,
            lap_seq_done=seq_done,
            lap_seq_novelty=seq_novelty,
            lap_seq_zeta=seq_zeta,
        )

        # Masks exactly as ``update_neo_laplacian_network`` forms them: the
        # anchor's own transition must not have ended the episode, each
        # future is valid up to the first terminal transition after it, and
        # an eigenvalue pair needs both endpoints alive.
        n_envs, seq_len = seq_done.shape
        n_steps = seq_len - 1
        masks = 1.0 - seq_done
        sample_mask = masks[:, 0]
        future_masks = masks[:, 1:]
        valid_trans = masks[:, :-1] * masks[:, 1:]

        def lap_epoch(carry, _):
            (
                params,
                batch_stats,
                optimizer_state,
                barrier_coeffs,
                betas,
                errors,
                quadratic_errors,
                eigenvalues,
                epoch_key,
            ) = carry
            epoch_key, loss_key = jax.random.split(epoch_key)

            def lap_loss(p):
                # One forward pass over every state in the window, as the
                # reference does over ``obs_flat``, so batch norm sees one
                # set of statistics per update.
                phi_flat, new_batch_stats = self._lap_apply(
                    p, batch_stats, seq_xy.reshape(n_envs * seq_len, 2), train=True
                )
                phis = phi_flat.reshape(n_envs, seq_len, -1)
                total, (positive, ortho, dual_diagnostic), (_, error_matrix) = multi_step_ALLO(
                    key=loss_key,
                    phis_all=phis[:, 0],
                    future_phis_all=phis[:, 1:],
                    actions=jnp.zeros((n_envs, n_steps), dtype=jnp.int32),
                    sample_mask=sample_mask.astype(jnp.bool_),
                    masks=future_masks.astype(jnp.bool_),
                    barrier_coeffs=barrier_coeffs,
                    betas=betas,
                    weights=None,
                    novelty_values=seq_novelty[:, 1:],
                    zeta_values=seq_zeta,
                    use_novelty=self.config.lap_novel_weight,
                    use_zeta=self.config.lap_zeta_weight,
                    normalize_zeta=False,
                    gamma=self.args.env.gamma,
                    state_action=False,
                    uni_policy=False,
                    n_steps=n_steps,
                    use_dual=self.config.laplacian.dual.on,
                    use_weights=False,
                    average_loss=self.config.lap_average_loss,
                    rms_norm=False,
                    barrier_dims=1,
                )
                rayleigh = compute_batch_eigenvalues_seq(phis, valid_trans)
                return total, (
                    positive,
                    ortho,
                    dual_diagnostic,
                    rayleigh,
                    error_matrix,
                    new_batch_stats,
                )

            (loss, aux), grads = jax.value_and_grad(lap_loss, has_aux=True)(params)
            params, optimizer_state = _apply_gradients(
                self.lap_optimizer, params, optimizer_state, grads
            )
            positive, ortho, dual_diagnostic, rayleigh, error_matrix, batch_stats = aux
            pmean = lambda value: jax.lax.pmean(value, axis_name=_PMAP_AXIS_NAME)
            rayleigh = pmean(rayleigh)
            error_matrix = jax.tree_util.tree_map(pmean, error_matrix)
            # Running averages are per-device statistics; average them the
            # same way so every replica carries the same batch_stats.
            batch_stats = jax.tree_util.tree_map(pmean, batch_stats)
            eigenvalues = 0.9 * eigenvalues + 0.1 * rayleigh
            allo = extra.replace(
                barrier_coeffs=barrier_coeffs,
                betas=betas,
                errors=errors,
                quadratic_errors=quadratic_errors,
            )
            if self.config.laplacian.barrier.on or self.config.laplacian.dual.on:
                allo = update_ALLO_parameters(
                    allo,
                    error_matrix,
                    self.config.laplacian.rate,
                    self.config.laplacian.barrier,
                    self.config.laplacian.dual,
                    average_loss=self.config.lap_average_loss,
                    barrier_dims=1,
                )
            carry = (
                params,
                batch_stats,
                optimizer_state,
                allo.barrier_coeffs,
                allo.betas,
                allo.errors,
                allo.quadratic_errors,
                eigenvalues,
                epoch_key,
            )
            return carry, (loss, positive, ortho, dual_diagnostic)

        init_carry = (
            extra.lap_params,
            extra.lap_batch_stats,
            extra.lap_optimizer_state,
            extra.barrier_coeffs,
            extra.betas,
            extra.errors,
            extra.quadratic_errors,
            extra.eigenvalues,
            key,
        )

        def update_laplacian(_):
            # lap_epochs gradient steps on the same window, as the reference
            # makes a_n_epochs passes over each rollout.  Metrics are the
            # last pass's.
            carry, stacked = jax.lax.scan(
                lap_epoch, init_carry, None, length=self.config.lap_epochs
            )
            last = tuple(value[-1] for value in stacked)
            return carry[:-1] + last

        def skip_laplacian(_):
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return init_carry[:-1] + (zero, zero, zero, zero)

        # Once per full window: after actor step lap_seq_len the padding
        # has shifted out entirely, and every lap_seq_len steps after that
        # is a fresh rollout's worth of data, as in the reference.
        actor_steps = train_state.env_steps // (
            self.args.env.n_envs * self.args.env.action_repeat
        )
        should_update_lap = actor_steps % self.config.lap_seq_len == 0
        (
            lap_params,
            lap_batch_stats,
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
        ) = jax.lax.cond(
            should_update_lap,
            update_laplacian,
            skip_laplacian,
            operand=None,
        )
        extra = extra.replace(
            lap_params=lap_params,
            lap_batch_stats=lap_batch_stats,
            lap_optimizer_state=lap_optimizer_state,
            barrier_coeffs=barrier_coeffs,
            betas=betas,
            errors=errors,
            quadratic_errors=quadratic_errors,
            max_inv_ema=max_inv,
            mean_scale_ema=scale,
            eigenvalues=eigenvalues,
        )
        # Collection-time eigen-reward for this step's transition, on the
        # running batch-norm averages, for the diagnostics below.
        phi0, _ = self._lap_apply(lap_params, lap_batch_stats, xy0, train=False)
        phi1, _ = self._lap_apply(lap_params, lap_batch_stats, xy1, train=False)
        eig_reward = jax.lax.stop_gradient(
            jnp.abs(self._option_phi(phi1)) - jnp.abs(self._option_phi(phi0))
        ) * transitions.discount[:, None]
        option_durations = self._option_durations(eigenvalues)
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
            "Options/dur_min": jnp.min(option_durations),
            "Options/dur_max": jnp.max(option_durations),
            "Options/dur_mean": jnp.mean(option_durations.astype(jnp.float32)),
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
        phi0, _ = self._lap_apply(
            extra.lap_params,
            extra.lap_batch_stats,
            self._lap_obs(transitions.observation),
            train=False,
        )
        phi1, _ = self._lap_apply(
            extra.lap_params,
            extra.lap_batch_stats,
            self._lap_obs(transitions.next_observation),
            train=False,
        )
        # Zero on a done transition: AutoResetWrapper made next_observation the
        # reset state, so the magnitude difference across it is meaningless.
        eig_reward = jax.lax.stop_gradient(
            jnp.abs(self._option_phi(phi1)) - jnp.abs(self._option_phi(phi0))
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
