from dataclasses import dataclass
from typing import Tuple, Any, Optional
from src.config.customgrids.common_dataclass import *


@dataclass(frozen=True)
class EnvConfig:
    maze_type: str
    n_states: int
    obs_mode: str
    screen_size: Any  # e.g., tuple (width, height)
    rand_reset: bool
    eval_rand_reset: bool
    gamma: float


@dataclass(frozen=True)
class STauConfig:
    frac: float
    start: float
    end: float


@dataclass(frozen=True)
class BufferConfig:
    prioritized: bool
    lap_prioritized: bool
    size: int
    beta: float
    device: str
    q_batch_size: int
    lap_batch_size: int


@dataclass(frozen=True)
class AlgoConfig:
    fusion: bool
    lr_alpha: float
    ddqn: bool
    enc_allow_grad: bool
    lap_loss_wt: float
    starts: int
    sample_action: bool
    n_step: int
    max_eps_len: int
    tot_steps: int
    eps: EpsConfig
    s_tau: STauConfig
    clip_q_pred: bool
    tau: float
    train_freq: int
    tgt_upd_freq: int
    parseval_mul: float
    n_options: int

    m_step: int
    bidir_opt:bool
    opt_dur: int
    eig_dur: bool
    phi_norm: bool
    intr_norm: bool


@dataclass(frozen=True)
class FeaturesConfig:
    pi: bool
    phi_comb: bool
    n_dims: int
    p_dims: Tuple[int]


@dataclass(frozen=True)
class GraphConfig:
    state_action: bool
    undirected: bool
    normalized_P: bool
    laplacian: bool


@dataclass(frozen=True)
class VisualConfig:
    on: bool
    emb_annot: bool


@dataclass(frozen=True)
class Config:
    gpu: GPUConfig
    net: NetConfig
    laplacian: Any
    eigen_transform: EigenTransformConfig
    ckpt: CKPTConfig
    buffer: BufferConfig
    exp: ExpConfig
    wb: WBConfig
    env: EnvConfig
    algo: AlgoConfig
    features: FeaturesConfig
    graph: GraphConfig
    visual: VisualConfig
