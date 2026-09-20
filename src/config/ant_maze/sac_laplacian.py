"""Unified SAC/RND/NEO configuration for Brax and JaxGCRL.

This is the off-policy SAC counterpart of ``config/atari/pqn_laplacian.py``.
All Brax SAC entry points parse the same CLI and expose the same top-level
configuration groups to W&B:: ``env``, ``sac``, ``rnd``, ``laplacian``,
``options`` and ``neo``.  Compatibility properties named ``net`` and ``algo``
keep the existing trainers readable without duplicating those groups in W&B.

PQN-only quantities such as TD(lambda), rollout epochs, discrete option
temperatures and option minibatches are deliberately absent: SAC learns from
uniform replay with continuous tanh-Gaussian option policies.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from src.config.common.arg_helpers import add_common_base_args
from src.config.common.build_config import close_logger, create_wb_logger
from src.config.common.dataclass_args import ExpConfig, GPUConfig, WBConfig, EpsConfig
from src.utils.utils import int_or_none, str2bool


@dataclass(frozen=True)
class BraxEnvConfig:
    family: str
    source: str
    id: str
    backend: str
    gamma: float
    n_envs: int
    n_test_envs: int
    max_episode_steps: int
    action_repeat: int
    normalize_obs: bool
    terminate_on_success: bool
    dense_reward: bool
    fixed_goal: Optional[Tuple[float, float]]
    pixel_obs: bool
    pixel_size: int


@dataclass(frozen=True)
class SACNetworkConfig:
    lr: float
    c_dims: Tuple[int, ...]
    q_dims: Tuple[int, ...]
    max_grad_norm: float
    layer_norm: bool


@dataclass(frozen=True)
class SACAlgorithmConfig:
    tot_steps: int
    batch_size: int
    starts: int
    replay_size: int
    grad_updates_per_step: int
    tau: float
    reward_scale: float
    n_evals: int
    deterministic_eval: bool
    max_devices_per_host: Optional[int]
    checkpoint_logdir: Optional[str]
    save_video: bool


@dataclass(frozen=True)
class SACGroupConfig:
    network: SACNetworkConfig
    algorithm: SACAlgorithmConfig
    alpha_lr: float
    target_entropy_scale: float
    eps: EpsConfig


@dataclass(frozen=True)
class SACRNDConfig:
    int_coef: float
    int_gamma: float
    obs_burnin: int
    upd_prop: float
    update_interval: int
    lr: float
    dims: Tuple[int, ...]
    feat_dim: int
    max_grad_norm: float
    # Which observation space the novelty model sees: the goal-free ant state
    # ("state"), the torso x/y position ("xy"), or a PixelAntMaze frame
    # ("pixels").  See ``--r_input`` for why this matters on AntMaze.
    input: str


@dataclass(frozen=True)
class BarrierConfig:
    on: bool
    lr: float
    clip: bool
    value: float


@dataclass(frozen=True)
class DualConfig:
    on: bool
    lr: float
    clip: bool


@dataclass(frozen=True)
class RateConfig:
    error: float
    q_error: float


@dataclass(frozen=True)
class SACLaplacianConfig:
    lr: float
    dims: int
    hidden_dims: Tuple[int, ...]
    pixel_size: int
    update_interval: int
    max_grad_norm: float
    batch_norm: bool
    layer_norm: bool
    average_loss: bool
    n_step: int
    seq_len: int
    epochs: int
    barrier: BarrierConfig
    dual: DualConfig
    rate: RateConfig


@dataclass(frozen=True)
class SACOptionConfig:
    gamma: float
    n_options: int
    opt_dur: int
    opt_dur_min: int
    opt_dur_max: int
    mu_prob: float
    skip_first: bool
    eigdur_options: bool
    hidden_dims: Tuple[int, ...]
    lr: float
    alpha_lr: float
    target_entropy_scale: float
    tau: float


@dataclass(frozen=True)
class SACNEOConfig:
    k: float
    delta: float
    target_mean: float
    rnd_coef: float
    lap_novel_weight: bool
    ema_tau: float
    bonus_q: float
    inv_cap: float


@dataclass(frozen=True)
class Config:
    gpu: GPUConfig
    exp: ExpConfig
    wb: WBConfig
    env: BraxEnvConfig
    sac: SACGroupConfig
    rnd: SACRNDConfig
    laplacian: SACLaplacianConfig
    options: SACOptionConfig
    neo: SACNEOConfig

    # Trainer compatibility.  These are properties, not dataclass fields, so
    # W&B records only the new organized ``sac`` group.
    @property
    def net(self):
        return self.sac.network

    @property
    def algo(self):
        return self.sac.algorithm

    # SNEOExplorer compatibility view.  New code should prefer the nested
    # groups, but keeping these aliases makes the config refactor algorithm
    # neutral and easy to audit.
    @property
    def lap_dims(self): return self.laplacian.dims
    @property
    def lap_hidden_dims(self): return self.laplacian.hidden_dims
    @property
    def lap_pixel_size(self): return self.laplacian.pixel_size
    @property
    def lap_update_interval(self): return self.laplacian.update_interval
    @property
    def lap_seq_len(self): return self.laplacian.seq_len
    @property
    def lap_epochs(self): return self.laplacian.epochs
    @property
    def lap_lr(self): return self.laplacian.lr
    @property
    def lap_max_grad_norm(self): return self.laplacian.max_grad_norm
    @property
    def lap_ortho_coef(self):
        return self.laplacian.barrier.value if self.laplacian.barrier.on else 0.0
    @property
    def lap_average_loss(self): return self.laplacian.average_loss
    @property
    def lap_novel_weight(self): return self.neo.lap_novel_weight
    @property
    def neo_k(self): return self.neo.k
    @property
    def neo_delta(self): return self.neo.delta
    @property
    def neo_target_mean(self): return self.neo.target_mean
    @property
    def neo_rnd_coef(self): return self.neo.rnd_coef
    @property
    def neo_ema_tau(self): return self.neo.ema_tau
    @property
    def neo_bonus_q(self): return self.neo.bonus_q
    @property
    def neo_inv_cap(self): return self.neo.inv_cap
    @property
    def n_options(self): return self.options.n_options
    @property
    def option_hidden_dims(self): return self.options.hidden_dims
    @property
    def option_lr(self): return self.options.lr
    @property
    def option_alpha_lr(self): return self.options.alpha_lr
    @property
    def option_target_entropy_scale(self): return self.options.target_entropy_scale
    @property
    def option_gamma(self): return self.options.gamma
    @property
    def option_tau(self): return self.options.tau
    @property
    def option_duration(self): return self.options.opt_dur
    @property
    def option_duration_min(self): return self.options.opt_dur_min
    @property
    def option_duration_max(self): return self.options.opt_dur_max
    @property
    def option_mu_prob(self): return self.options.mu_prob
    @property
    def option_skip_first(self): return self.options.skip_first
    @property
    def option_eigdur(self): return self.options.eigdur_options
    @property
    def eps_start(self): return self.sac.eps.start
    @property
    def eps_end(self): return self.sac.eps.end
    @property
    def eps_frac(self): return self.sac.eps.frac


# Smallest square frame the shared RND / pixel-Laplacian encoder accepts.
# The stack is VALID-padded 8x8 stride 4, 4x4 stride 2, 3x3 stride 1, so the
# spatial size goes 36 -> 8 -> 3 -> 1.  Anything smaller drives the third
# layer to a non-positive width and flax then builds a zero-input Dense.
_MIN_CONV_INPUT = 36


def _positive_int(value: str) -> int:
    parsed = int(float(value))
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _str_or_none(value: str) -> Optional[str]:
    return None if value.lower() in {"none", "null", ""} else value


def build_parser(profile: str = "sac_neo") -> argparse.ArgumentParser:
    if profile not in {"sac", "sac_rnd", "sac_neo"}:
        raise ValueError(f"unknown SAC config profile: {profile}")

    parser = argparse.ArgumentParser(description="SAC on Brax/JaxGCRL")
    add_common_base_args(parser)
    defaults = {
        "sac": ("SAC", "SAC"),
        "sac_rnd": ("SAC_RND", "SAC_RND"),
        "sac_neo": ("SAC_NEO", "SAC_NEO"),
    }
    x_file, wb_model = defaults[profile]
    parser.set_defaults(x_file=x_file, wb_model=wb_model, wb_proj="SAC_BRAX")

    # Environment.
    parser.add_argument("--e_env_source", choices=("jaxgcrl", "brax"), default="jaxgcrl")
    parser.add_argument("--e_id", type=str, default="ant_u_maze")
    parser.add_argument("--e_backend", type=str, default="spring")
    parser.add_argument("--e_gamma", type=float, default=0.99)
    parser.add_argument("--e_n_envs", type=_positive_int, default=128)
    parser.add_argument("--e_n_test_envs", type=_positive_int, default=16)
    parser.add_argument("--e_max_episode_steps", type=_positive_int, default=1000)
    parser.add_argument("--e_action_repeat", type=_positive_int, default=1)
    parser.add_argument("--e_normalize_obs", type=str2bool, default=True)
    parser.add_argument("--e_terminate_on_success", type=str2bool, default=True)
    # AntMaze's sparse reward is exactly `success`, so the agent sees zero
    # signal until it physically touches the goal.  The dense alternative,
    # 10*vel_to_target + healthy - ctrl - contact, is a control: if a goal
    # is unreachable even under dense reward, the problem is not exploration.
    parser.add_argument("--e_dense_reward", type=str2bool, default=False)
    parser.add_argument("--e_fixed_goal", type=float, nargs=2, metavar=("X", "Y"), default=None)
    parser.add_argument("--e_pixel_obs", type=str2bool, default=False)
    parser.add_argument("--e_pixel_size", type=_positive_int, default=84)

    # SAC: continuous actor, twin critics, replay and entropy tuning.
    parser.add_argument("--c_dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--q_dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--layer_norm", type=str2bool, default=False)
    parser.add_argument("--a_tot_steps", type=_positive_int, default=10_000_000)
    parser.add_argument("--a_batch_size", type=_positive_int, default=256)
    parser.add_argument("--a_starts", type=_positive_int, default=10_000)
    parser.add_argument("--a_replay_size", type=_positive_int, default=1_000_000)
    parser.add_argument("--a_grad_updates_per_step", type=_positive_int, default=64)
    parser.add_argument("--a_tau", type=float, default=0.005)
    parser.add_argument("--a_reward_scale", type=float, default=1.0)
    parser.add_argument("--a_n_evals", type=_positive_int, default=20)
    parser.add_argument("--a_deterministic_eval", type=str2bool, default=True)
    parser.add_argument("--a_max_devices_per_host", type=int_or_none, default=None)
    parser.add_argument("--a_checkpoint_logdir", type=_str_or_none, default=None)
    parser.add_argument("--a_save_video", type=str2bool, default=False)
    parser.add_argument("--a_alpha_lr", type=float, default=3e-4)
    parser.add_argument("--a_target_entropy_scale", type=float, default=0.5)
    parser.add_argument("--a_eps_start", type=float, default=1.0)
    parser.add_argument("--a_eps_end", type=float, default=0.01)
    parser.add_argument("--a_eps_frac", type=float, default=0.1)

    # RND.
    parser.add_argument("--a_int_coef", type=float, default=1.0 if profile == "sac_rnd" else 0.0)
    parser.add_argument("--r_int_gamma", type=float, default=0.99)
    # The burn-in only collects observation statistics before the predictor
    # starts training; it does not need a large fraction of the run.  The
    # PQN default of 500k assumes a 25M-step Atari budget *and* 4096
    # transitions per rollout.  Here one vectorized actor step already yields
    # e_n_envs transitions, so 500k env steps freezes the novelty signal --
    # and therefore the whole NEO Laplacian weighting -- for the first 5% of
    # training.  Match the replay prefill instead.
    parser.add_argument("--r_obs_burnin", type=int, default=10_000)
    parser.add_argument("--r_upd_prop", type=float, default=0.25)
    # Observation space seen by the RND target/predictor.
    #   state  - goal-free ant state (31 - 2 = 29 dims on JaxGCRL AntMaze).
    #   xy     - torso x/y only.  On AntMaze the full state is dominated by
    #            joint angles and velocities, which stay novel forever, so
    #            "state" novelty says almost nothing about *where* the ant has
    #            been.  NEO's reliability weighting needs positional novelty.
    #   pixels - PixelAntMaze frame; requires --e_pixel_obs True.
    #   auto   - "pixels" when --e_pixel_obs True, otherwise "state" (the
    #            historical behaviour of this file).
    parser.add_argument(
        "--r_input", choices=("auto", "state", "xy", "pixels"), default="auto"
    )
    parser.add_argument("--r_update_interval", type=_positive_int, default=16 if profile == "sac_neo" else 1)
    parser.add_argument("--r_lr", type=float, default=1e-4)
    parser.add_argument("--r_dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--r_feat_dim", type=_positive_int, default=128)
    parser.add_argument("--r_max_grad_norm", type=float, default=10.0)

    # Source Laplacian.  It remains one-step because SAC replay samples are
    # independent transitions, unlike PQN's contiguous 32-step rollout.
    parser.add_argument("--lap_lr", type=float, default=2.5e-4)
    parser.add_argument("--lap_dims", type=_positive_int, default=32)
    # One 512-wide hidden layer, as the reference Conv_Encoder's MLP (c_dims).
    parser.add_argument("--lap_hidden_dims", type=int, nargs="+", default=[512])
    # The vector Laplacian (NEO_LapXY) trains on per-env sequences of the
    # last lap_seq_len states, so the graph-drawing term is the gamma-
    # discounted multi-step objective over lap_seq_len - 1 futures, exactly
    # as the discrete agents train on their e_n_steps rollout.  It updates
    # once per full window with lap_epochs gradient steps, matching the
    # reference's a_n_epochs passes per rollout; lap_update_interval does not
    # apply to it.  The pixel Laplacian (NEO.py) is still 1-step.
    parser.add_argument("--lap_seq_len", type=_positive_int, default=32)
    parser.add_argument("--lap_epochs", type=_positive_int, default=2)
    # PixelLaplacianEncoder runs the same VALID-padded 8x8/4, 4x4/2, 3x3/1
    # stack as the RND encoders, which cannot consume anything below 36 (see
    # _MIN_CONV_INPUT).  The previous default of 21 collapsed the third
    # convolution to a zero-width feature map, so the pixel Laplacian could
    # never initialize.  48 keeps the frame cheap while leaving a 2x2x64
    # feature map rather than the single spatial cell that 36 produces.
    parser.add_argument("--lap_pixel_size", type=_positive_int, default=48)
    # One vectorized actor step provides a single fresh transition per env, so
    # the Laplacian batch is e_n_envs.  Tying this to r_update_interval (16)
    # left roughly a_tot_steps / (e_n_envs * 16) ~= 4.9k Laplacian gradient
    # steps for a 10M-step run, which is far too few to resolve lap_dims
    # eigenfunctions.  Default to every actor step; pass an explicit value to
    # restore the sparse schedule.
    parser.add_argument("--lap_update_interval", type=_positive_int, default=1)
    parser.add_argument("--lap_max_grad_norm", type=float, default=10.0)
    parser.add_argument("--lap_batch_norm", type=str2bool, default=False)
    parser.add_argument("--lap_layer_norm", type=str2bool, default=False)
    parser.add_argument("--lap_average", type=str2bool, default=True)
    parser.add_argument("--lap_br", type=str2bool, default=True)
    parser.add_argument("--lap_br_lr", type=float, default=0.01)
    parser.add_argument("--lap_br_clip", type=str2bool, default=True)
    parser.add_argument("--lap_br_val", "--lap_ortho_coef", dest="lap_br_val", type=float, default=100.0)
    parser.add_argument("--lap_dual", type=str2bool, default=True)
    parser.add_argument("--lap_dual_lr", type=float, default=0.01)
    parser.add_argument("--lap_dual_clip", type=str2bool, default=True)
    parser.add_argument("--lap_err_rate", type=float, default=1.0)
    parser.add_argument("--lap_q_err_rate", type=float, default=0.1)

    # Continuous SAC options.
    parser.add_argument("--o_gamma", type=float, default=0.95)
    parser.add_argument("--o_n_options", type=_positive_int, default=5)
    # Option durations, mirroring the discrete agents exactly.  With
    # o_eigdur_options False (the reference default) every option runs for
    # o_opt_dur steps.  With it True, option i runs for ceil(1 / lambda_{i+1})
    # clipped to [o_opt_dur_min, o_opt_dur_max], as
    # ``eigenvalues_to_option_durations_skip_first`` computes it.
    parser.add_argument("--o_opt_dur", type=_positive_int, default=5)
    parser.add_argument("--o_opt_dur_min", type=_positive_int, default=5)
    parser.add_argument("--o_opt_dur_max", type=_positive_int, default=50)
    parser.add_argument("--o_eigdur_options", type=str2bool, default=False)
    # Probability that an exploratory decision picks an option rather than a
    # random primitive action.  The discrete agents split epsilon this way
    # (``explore_option = do_explore & (u_mu < mu_prob)``); the continuous port
    # had collapsed the two, which made epsilon alone the option rate.
    parser.add_argument("--o_mu_prob", type=float, default=0.5)
    # Governs the DURATION table only, mirroring
    # ``eigenvalues_to_option_durations_skip_first``: option i's duration comes
    # from lambda_{i+1}.  Option i's REWARD always comes from phi_i, including
    # phi_0, exactly as ``_neo_eigen_rewards`` does -- the ALLO Laplacian has
    # no constant first eigenfunction, so phi_0 is the slowest real mode and
    # the most useful option for long-range exploration.
    parser.add_argument("--o_skip_first", type=str2bool, default=True)
    parser.add_argument("--o_dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--o_lr", type=float, default=1e-4)
    parser.add_argument("--o_alpha_lr", type=float, default=3e-4)
    parser.add_argument("--o_target_entropy_scale", type=float, default=0.5)
    parser.add_argument("--o_tau", type=float, default=0.005)

    # Reliable NEO scaling.
    parser.add_argument("--neo_k", type=float, default=1.0)
    parser.add_argument("--neo_delta", type=float, default=0.5)
    parser.add_argument("--neo_targ_mean", type=float, default=0.3)
    # Weight of the RND novelty term in the *option* reward: the novelty bonus
    # that makes this Reliable NEO rather than plain NEO.  Keep the default at
    # 1.0 -- it is the method.  `add_common_neo_args` and DQN_SNEO's
    # --sneo_rnd_beta use 0.0 because they implement base NEO, which is the
    # ablation here, so pass --neo_rnd_coef 0 to obtain that baseline.
    parser.add_argument("--neo_rnd_coef", type=float, default=1.0)
    parser.add_argument("--neo_lap_wt", type=str2bool, default=True)
    parser.add_argument("--neo_ema_tau", type=float, default=1e-3)
    parser.add_argument("--neo_bonus_q", type=float, default=0.99)
    parser.add_argument("--neo_inv_cap", type=float, default=1e4)
    return parser


def parse_args(profile: str = "sac_neo") -> Config:
    raw = build_parser(profile).parse_args()
    lap_update_interval = raw.lap_update_interval

    rnd_input = raw.r_input
    if rnd_input == "auto":
        rnd_input = "pixels" if raw.e_pixel_obs else "state"
    if rnd_input == "pixels" and not raw.e_pixel_obs:
        raise ValueError("r_input=pixels requires --e_pixel_obs True")

    if not 0.0 < raw.e_gamma <= 1.0:
        raise ValueError("e_gamma must be in (0, 1]")
    if not 0.0 < raw.a_tau <= 1.0 or not 0.0 < raw.o_tau <= 1.0:
        raise ValueError("a_tau and o_tau must be in (0, 1]")
    if raw.a_starts >= raw.a_tot_steps or raw.a_replay_size < raw.a_starts:
        raise ValueError("require a_starts < a_tot_steps and a_replay_size >= a_starts")
    if tuple(raw.c_dims) != tuple(raw.q_dims):
        raise ValueError("Brax SAC requires c_dims and q_dims to be equal")
    positive_dims = (*raw.c_dims, *raw.r_dims, *raw.lap_hidden_dims, *raw.o_dims)
    if any(width <= 0 for width in positive_dims):
        raise ValueError("all network dimensions must be positive")
    # Rewards use phi_0..phi_{n-1}; eigenvalue durations use lambda_1..lambda_n
    # when both o_eigdur_options and o_skip_first are set, which needs one
    # more Laplacian dimension.
    needs_skip = raw.o_eigdur_options and raw.o_skip_first
    required_dims = raw.o_n_options + (1 if needs_skip else 0)
    if raw.lap_dims < required_dims:
        raise ValueError(
            "lap_dims must be at least o_n_options, plus one more when "
            "o_eigdur_options and o_skip_first are both set"
        )
    if raw.lap_seq_len < 2:
        raise ValueError("lap_seq_len must be at least 2 (one anchor, one future)")
    if raw.o_opt_dur_min > raw.o_opt_dur_max:
        raise ValueError("require o_opt_dur_min <= o_opt_dur_max")
    if not 0.0 <= raw.o_mu_prob <= 1.0:
        raise ValueError("o_mu_prob must be in [0, 1]")
    if raw.e_pixel_obs and raw.e_pixel_size < _MIN_CONV_INPUT:
        raise ValueError(
            f"pixel RND requires e_pixel_size >= {_MIN_CONV_INPUT}"
        )
    if raw.lap_pixel_size < _MIN_CONV_INPUT:
        raise ValueError(
            f"lap_pixel_size must be at least {_MIN_CONV_INPUT}: the pixel "
            "Laplacian shares the RND convolutional stack, whose third layer "
            "produces an empty feature map on smaller frames"
        )
    # Batch norm is implemented by the vector Laplacian (NEO_LapXY), which
    # threads Flax batch_stats through its state exactly as the discrete
    # agents' Conv_Encoder does; the pixel encoder in NEO.py rejects it in
    # configure().  Layer norm is not implemented by either.
    if raw.lap_layer_norm:
        raise NotImplementedError(
            "The SAC Laplacian encoders do not implement layer normalization; "
            "keep lap_layer_norm=False rather than logging an unused setting"
        )
    if raw.lap_br_lr < 0.0 or raw.lap_dual_lr < 0.0:
        raise ValueError("lap_br_lr and lap_dual_lr must be non-negative")
    if not 0.0 <= raw.lap_err_rate <= 1.0:
        raise ValueError("lap_err_rate must be in [0, 1]")
    if not 0.0 <= raw.lap_q_err_rate <= 1.0:
        raise ValueError("lap_q_err_rate must be in [0, 1]")
    if not 0.0 < raw.r_upd_prop <= 1.0:
        raise ValueError("r_upd_prop must be in (0, 1]")
    if raw.r_obs_burnin < 0 or raw.r_obs_burnin >= raw.a_tot_steps:
        raise ValueError("require 0 <= r_obs_burnin < a_tot_steps")
    for name, value in (
        ("r_int_gamma", raw.r_int_gamma), ("o_gamma", raw.o_gamma),
        ("a_eps_start", raw.a_eps_start), ("a_eps_end", raw.a_eps_end),
        ("a_eps_frac", raw.a_eps_frac), ("neo_bonus_q", raw.neo_bonus_q),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if raw.a_eps_end > raw.a_eps_start:
        raise ValueError("a_eps_end must not exceed a_eps_start")

    network = SACNetworkConfig(
        lr=raw.lr, c_dims=tuple(raw.c_dims), q_dims=tuple(raw.q_dims),
        max_grad_norm=raw.max_grad_norm, layer_norm=raw.layer_norm,
    )
    algorithm = SACAlgorithmConfig(
        tot_steps=raw.a_tot_steps, batch_size=raw.a_batch_size,
        starts=raw.a_starts, replay_size=raw.a_replay_size,
        grad_updates_per_step=raw.a_grad_updates_per_step, tau=raw.a_tau,
        reward_scale=raw.a_reward_scale, n_evals=raw.a_n_evals,
        deterministic_eval=raw.a_deterministic_eval,
        max_devices_per_host=raw.a_max_devices_per_host,
        checkpoint_logdir=raw.a_checkpoint_logdir,
        save_video=raw.a_save_video,
    )
    return Config(
        gpu=GPUConfig(raw.gpu_id, raw.gpu_perc),
        exp=ExpConfig(raw.x_seed, raw.x_file, int(os.environ.get("SLURM_JOB_ID", -1)), os.getpid()),
        wb=WBConfig(raw.wb_on, raw.wb_proj, raw.wb_model, raw.wb_entity),
        env=BraxEnvConfig(
            family="brax", source=raw.e_env_source, id=raw.e_id,
            backend=raw.e_backend, gamma=raw.e_gamma, n_envs=raw.e_n_envs,
            n_test_envs=raw.e_n_test_envs,
            max_episode_steps=raw.e_max_episode_steps,
            action_repeat=raw.e_action_repeat,
            normalize_obs=raw.e_normalize_obs,
            terminate_on_success=raw.e_terminate_on_success,
            dense_reward=raw.e_dense_reward,
            fixed_goal=tuple(raw.e_fixed_goal) if raw.e_fixed_goal is not None else None,
            pixel_obs=raw.e_pixel_obs, pixel_size=raw.e_pixel_size,
        ),
        sac=SACGroupConfig(
            network=network, algorithm=algorithm,
            alpha_lr=raw.a_alpha_lr,
            target_entropy_scale=raw.a_target_entropy_scale,
            eps=EpsConfig(raw.a_eps_frac, raw.a_eps_start, raw.a_eps_end),
        ),
        rnd=SACRNDConfig(
            int_coef=raw.a_int_coef, int_gamma=raw.r_int_gamma,
            obs_burnin=raw.r_obs_burnin, upd_prop=raw.r_upd_prop,
            update_interval=raw.r_update_interval, lr=raw.r_lr,
            dims=tuple(raw.r_dims), feat_dim=raw.r_feat_dim,
            max_grad_norm=raw.r_max_grad_norm, input=rnd_input,
        ),
        laplacian=SACLaplacianConfig(
            lr=raw.lap_lr, dims=raw.lap_dims,
            hidden_dims=tuple(raw.lap_hidden_dims), pixel_size=raw.lap_pixel_size,
            update_interval=lap_update_interval,
            max_grad_norm=raw.lap_max_grad_norm,
            batch_norm=raw.lap_batch_norm, layer_norm=raw.lap_layer_norm,
            average_loss=raw.lap_average, n_step=raw.lap_seq_len - 1,
            seq_len=raw.lap_seq_len, epochs=raw.lap_epochs,
            barrier=BarrierConfig(
                on=raw.lap_br, lr=raw.lap_br_lr,
                clip=raw.lap_br_clip, value=raw.lap_br_val,
            ),
            dual=DualConfig(
                on=raw.lap_dual, lr=raw.lap_dual_lr,
                clip=raw.lap_dual_clip,
            ),
            rate=RateConfig(
                error=raw.lap_err_rate, q_error=raw.lap_q_err_rate,
            ),
        ),
        options=SACOptionConfig(
            gamma=raw.o_gamma, n_options=raw.o_n_options,
            opt_dur=raw.o_opt_dur, opt_dur_min=raw.o_opt_dur_min,
            opt_dur_max=raw.o_opt_dur_max, mu_prob=raw.o_mu_prob,
            skip_first=raw.o_skip_first, eigdur_options=raw.o_eigdur_options,
            hidden_dims=tuple(raw.o_dims),
            lr=raw.o_lr, alpha_lr=raw.o_alpha_lr,
            target_entropy_scale=raw.o_target_entropy_scale, tau=raw.o_tau,
        ),
        neo=SACNEOConfig(
            k=raw.neo_k, delta=raw.neo_delta,
            target_mean=raw.neo_targ_mean, rnd_coef=raw.neo_rnd_coef,
            lap_novel_weight=raw.neo_lap_wt, ema_tau=raw.neo_ema_tau,
            bonus_q=raw.neo_bonus_q, inv_cap=raw.neo_inv_cap,
        ),
    )


def convert_args_to_dict(args):
    """Compatibility identity: the unified parser already returns Config."""
    return args


__all__ = [
    "Config", "SACRNDConfig", "SACLaplacianConfig", "SACOptionConfig",
    "BarrierConfig", "DualConfig", "RateConfig",
    "SACNEOConfig", "build_parser", "parse_args", "convert_args_to_dict",
    "create_wb_logger", "close_logger",
]
