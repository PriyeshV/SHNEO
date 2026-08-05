import argparse
import os

from src.utils.utils import str2bool, int_or_none, float_or_none


# ---------------------------------------------------------------------
# Common args shared by ALL agents / methods
# ---------------------------------------------------------------------
def add_common_base_args(parser: argparse.ArgumentParser) -> None:
    # GPU
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--gpu_perc", type=float, default=0.30)

    # Experiment / logging
    parser.add_argument("--x_seed", type=int, default=0, help="experiment seed")
    parser.add_argument(
        "--x_file",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="file name"
    )

    parser.add_argument("--wb_on", type=str2bool, default=False, help="logger on")
    parser.add_argument("--wb_proj", type=str, default="RL_RESULTS", help="wandb project")
    parser.add_argument("--wb_model", type=str, default="PQN", help="wandb group/model")
    parser.add_argument("--wb_entity", type=str, default="revisit-spectral-rl", help="wandb entity")


def add_common_env_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--e_gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--e_n_frames", type=int, default=4, help="logical stacked frames")
    parser.add_argument("--e_n_envs", type=int, default=128, help="# parallel train envs")
    parser.add_argument("--e_n_steps", type=int, default=32, help="# steps per env per update")
    parser.add_argument("--e_max_episode_steps", type=int_or_none, default=None)
    parser.add_argument("--e_reward_clip", type=str2bool, default=False)


def add_common_net_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--c_dims", type=int, nargs="*", default=[512], help="encoder dims")
    parser.add_argument("--q_dims", type=int, nargs="*", default=[], help="Q head dims")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--batch_norm", type=str2bool, default=False)
    parser.add_argument("--layer_norm", type=str2bool, default=True)
    parser.add_argument("--impala", type=str2bool, default=False)


def add_common_algo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--a_n_epochs", type=int, default=2, help="# epochs per update")
    parser.add_argument("--a_n_mini_batches", type=int, default=32, help="# mini-batches per epoch")
    parser.add_argument("--a_anneal_lr", type=str2bool, default=False, help="anneal learning rate")
    parser.add_argument("--a_rew_scale", type=str2bool, default=True, help="reward scaling / transformed target switch")
    parser.add_argument("--a_td_lambda", type=float, default=0.95, help="TD lambda")
    parser.add_argument("--a_tot_steps", type=lambda x: int(float(x)), default=25e6, help="total environment steps")
    parser.add_argument("--a_save_video", type=str2bool, default=False)
    parser.add_argument("--a_int_coef", type=float, default=0.)

    # Exploration
    parser.add_argument("--a_eps_start", type=float, default=1.0)
    parser.add_argument("--a_eps_end", type=float, default=0.01)
    parser.add_argument("--a_eps_frac", type=float, default=0.10)

    # Optional RND epsilon controller
    parser.add_argument("--a_use_rnd_eps", type=str2bool, default=False)
    parser.add_argument("--a_rnd_eps_max", type=float, default=0.1)


def add_common_laplacian_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lap_lr", type=float_or_none, default=2.5e-4)
    parser.add_argument("--lap_dims", type=int, default=32)
    parser.add_argument("--lap_max_grad_norm", type=float, default=10.0)
    parser.add_argument("--lap_batch_norm", type=str2bool, default=True)
    parser.add_argument("--lap_layer_norm", type=str2bool, default=False)
    parser.add_argument("--lap_average", type=str2bool, default=True)

    parser.add_argument("--lap_br", type=str2bool, default=True)
    parser.add_argument("--lap_br_lr", type=float_or_none, default=0.01)
    parser.add_argument("--lap_br_val", type=float_or_none, default=100.0)
    parser.add_argument("--lap_br_clip", type=str2bool, default=True)

    parser.add_argument("--lap_dual", type=str2bool, default=True)
    parser.add_argument("--lap_dual_lr", type=float_or_none, default=0.01)
    parser.add_argument("--lap_dual_clip", type=str2bool, default=True)

    parser.add_argument("--lap_err_rate", type=float_or_none, default=1.0)
    parser.add_argument("--lap_q_err_rate", type=float_or_none, default=0.1)


def add_common_rnd_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--r_int_gamma", type=float, default=0.99)
    parser.add_argument("--r_obs_burnin", type=int, default=500_000)
    parser.add_argument("--r_upd_prop", type=float, default=0.25)


def add_common_option_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--o_gamma", type=float, default=0.95)
    parser.add_argument("--o_td_lambda", type=float, default=0.95)

    parser.add_argument("--o_m_steps", type=int, default=8)
    parser.add_argument("--o_mu_prob", type=float, default=0.5)
    parser.add_argument("--o_stall_coef", type=float, default=0.0)

    parser.add_argument("--o_n_options", type=int, default=5)
    parser.add_argument("--o_bidir_options", type=str2bool, default=False)

    parser.add_argument("--o_opt_dur", type=int, default=5)
    parser.add_argument("--o_opt_dur_min", type=int, default=5)
    parser.add_argument("--o_opt_dur_max", type=int, default=50)
    parser.add_argument("--o_eigdur_options", type=str2bool, default=False)

    parser.add_argument("--o_policy_greedy", type=str2bool, default=True)
    parser.add_argument("--o_policy_temp", type=float, default=1.0)

    parser.add_argument("--o_n_mini_batches", type=int, default=32)
    parser.add_argument("--o_meta_td_lambda", type=float, default=0.95)
    parser.add_argument("--o_ioq_deter", type=str2bool, default=False)


def add_common_neo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--neo_k", type=int, default=1)
    parser.add_argument("--neo_delta", type=float, default=0.5)
    parser.add_argument("--neo_targ_mean", type=float, default=0.3)
    parser.add_argument("--neo_rnd_coef", type=float, default=0.0)
    parser.add_argument("--neo_lap_wt", type=str2bool, default=True)