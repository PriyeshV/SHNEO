from dataclasses import dataclass
from typing import Tuple, Any, Optional, List


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
class MagnetArgs:
    n_q: int = 8
    q_init: float = 0.10
    head_hidden: int = 512
    include_reim: bool = True


@dataclass(frozen=True)
class LaplacianArgs:
    # Required
    dims: int = 16
    lr: float = 1e-4
    layer_norm: bool = False
    batch_norm: bool = False
    max_grad_norm: float = 10
    bootstrap: bool = False

    dual: Optional[DualArgs] = None
    rate: Optional[RateArgs] = None
    barrier: Optional[BarrierArgs] = None

    n_step: Optional[int] = 1
    sample_n_step: bool = True
    sa: Optional[bool] = True
    uni: Optional[bool] = True
    rms_norm: Optional[bool] = True
    one_hot: Optional[bool] = True
    sa_interact: Optional[str] = 'project'
    sa_combine: Optional[str] = 'concat'
    sa_center: Optional[bool] = False

    upd_freq: Optional[int] = 1
    tgt_net: Optional[bool] = False
    imp_samp: Optional[bool] = True
    q_tgt_net: Optional[bool] = True
    average_loss: Optional[bool] = True
    save_metrics: Optional[bool] = False

    magnet: Optional[MagnetArgs] = None


@dataclass(frozen=True)
class EigenTransformConfig:
    on: bool
    gamma: float
    inv_pow: float
    poly_deg: int
    non_param: Optional[bool]


@dataclass(frozen=True)
class LRConfig:
    enc: float
    ffn: float


@dataclass(frozen=True)
class NetConfig:
    lr: LRConfig
    c_dims: Tuple[int] = ()
    q_dims: Tuple[int] = ()
    sa_h_dims: Tuple[int] = ()
    fusion_ip_norm: bool = False
    max_grad_norm: float = 10
    ip_batch_norm: bool = False
    ip_layer_norm: bool = False
    ip_l2_norm: bool = False
    batch_norm: bool = False
    layer_norm: bool = False
    impala_cnn: bool = False
    cnn_width_mul: int = 1


@dataclass(frozen=True)
class GPUConfig:
    id: int
    perc: float


@dataclass(frozen=True)
class CKPTConfig:
    dir: str
    keep: int
    interval: int


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