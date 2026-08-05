from dataclasses import dataclass
from typing import Tuple, Any, Optional, List

@dataclass(frozen=True)
class EnvConfig:
    id: str
    gamma: float
    n_frames: int
    n_envs: int
    n_steps: int
    n_test_envs: int
    n_total_envs: int


@dataclass(frozen=True)
class AlgoConfig:
    n_epochs: int
    upd_epochs: int
    n_mini_batches: int
    batch_size: int
    mini_batch_size: int
    num_iterations: int
    tot_steps: int
    anneal_lr: bool
    gae_lambda: float
    norm_adv: bool
    clip_coef: float
    ent_coef: float
    vf_coef: float
    clip_vloss: bool
    target_kl: float
    save_video: bool

    int_coef: float
    ext_coef: float
    int_gamma: float
    obs_burnin: int
    upd_prop: int

    n_options: int
    bidir_options: bool
    opt_dur: int
    opt_dur_min: int
    opt_dur_max: int
    eigdur_options: bool
    ioq_deter: bool


@dataclass(frozen=True)
class DualArgs:
    on: Optional[bool] = None
    lr: Optional[float] = None
    clip: Optional[bool] = None
    eig_clip: Optional[bool] = None


@dataclass(frozen=True)
class BarrierArgs:
    on: Optional[bool] = None
    lr: Optional[float] = None
    clip: Optional[bool] = None
    value: Optional[float] = None


@dataclass(frozen=True)
class RateArgs:
    error: Optional[float] = None
    q_error: Optional[float] = None


@dataclass(frozen=True)
class LaplacianConfig:
    dims: int
    lr: float
    layer_norm: bool
    batch_norm: bool
    max_grad_norm: float 

    dual: Optional[DualArgs] = None
    rate: Optional[RateArgs] = None
    barrier: Optional[BarrierArgs] = None

    n_step: Optional[int] = 1
    average_loss: Optional[bool] = True


@dataclass(frozen=True)
class NetConfig:
    lr: float
    c_dims: Tuple[int] = ()
    q_dims: Tuple[int] = ()
    max_grad_norm: float = 0.5
    batch_norm: bool = False
    layer_norm: bool = False


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
class Config:
    gpu: GPUConfig
    net: NetConfig
    exp: ExpConfig
    wb: WBConfig
    env: EnvConfig
    algo: AlgoConfig
    laplacian: LaplacianConfig