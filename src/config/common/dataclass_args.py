from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------
# Small nested configs
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class GPUConfig:
    id: int
    perc: float


@dataclass(frozen=True)
class ExpConfig:
    seed: int
    file: str
    job_id: int
    p_id: int


@dataclass(frozen=True)
class WBConfig:
    on: bool
    proj: str
    model: str
    entity: str


@dataclass(frozen=True)
class EpsConfig:
    frac: float
    start: float
    end: float


@dataclass(frozen=True)
class DualArgs:
    on: bool
    lr: float
    clip: bool
    eig_clip: Optional[float] = None


@dataclass(frozen=True)
class RateArgs:
    error: float
    q_error: float


@dataclass(frozen=True)
class BarrierArgs:
    on: bool
    lr: float
    clip: bool
    value: float


@dataclass(frozen=True)
class LaplacianConfig:
    dims: int
    lr: float
    layer_norm: bool
    batch_norm: bool
    max_grad_norm: float
    dual: DualArgs
    rate: RateArgs
    barrier: BarrierArgs
    n_step: int
    average_loss: bool


@dataclass(frozen=True)
class RNDArgs:
    int_gamma: float
    obs_burnin: int
    upd_prop: float


@dataclass(frozen=True)
class OptionArgs:
    m_steps: int
    mu_prob: float
    meta_td_lambda: float
    n_mini_batches: int
    n_options: int
    bidir_options: bool
    opt_dur: int
    opt_dur_min: int
    opt_dur_max: int
    eigdur_options: bool
    ioq_deter: bool

    gamma: float = 0.99
    td_lambda: float = 0.95
    stall_coef: float = 0.0
    policy_greedy: bool = True
    policy_temp: float = 1.0


@dataclass(frozen=True)
class NEOArgs:
    k: int
    delta: float
    target_mean: float
    rnd_coef: float
    lap_novel_weight: bool


@dataclass(frozen=True)
class NetConfig:
    lr: float
    c_dims: Tuple[int, ...]
    q_dims: Tuple[int, ...]
    max_grad_norm: float
    batch_norm: bool
    layer_norm: bool
    impala: bool = False


@dataclass(frozen=True)
class AlgoConfig:
    n_epochs: int
    n_upds: int
    n_mini_batches: int
    batch_size: int
    mini_batch_size: int
    num_iterations: int
    tot_steps: int
    anneal_lr: bool
    td_lambda: float
    int_coef: float
    rew_scale: bool
    eps: EpsConfig
    save_video: bool

    use_rnd_eps: bool = False
    rnd_eps_max: float = 0.1


@dataclass(frozen=True)
class EnvConfig:
    family: str
    id: str
    gamma: float
    n_frames: int
    n_envs: int
    n_steps: int

    n_test_envs: int = 8
    n_total_envs: int = 0

    # Procgen extras
    start_level: int = 0
    distribution_mode: str = "hard"
    center_agent: bool = True
    use_backgrounds: bool = True
    use_monochrome_assets: bool = False
    restrict_themes: bool = False
    use_generated_assets: bool = False
    paint_vel_info: bool = False
    use_easy_jump: bool = False

    max_episode_steps: Optional[int] = None
    reward_clip: bool = False


@dataclass(frozen=True)
class Config:
    gpu: GPUConfig
    exp: ExpConfig
    wb: WBConfig
    env: EnvConfig
    net: NetConfig
    algo: AlgoConfig
    laplacian: LaplacianConfig
    rnd: RNDArgs
    options: OptionArgs
    neo: NEOArgs