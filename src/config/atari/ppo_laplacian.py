import os
import time
import wandb
import argparse
from src.utils.utils import DotDict
from distutils.util import strtobool

from src.config.atari.ppo_dataclass_args import *
from src.utils.utils import str2bool, int_or_none, float_or_none, float_range


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment parameters")
    # GPU
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--gpu_perc", type=float, default=0.3)

    # Experiment parameters
    parser.add_argument("--x_seed", type=int, default=1, help="experiment seed")
    parser.add_argument("--x_file", type=str, default=os.path.basename(__file__).rstrip(".py"), help="File name")
    parser.add_argument("--wb_on", type=str2bool, default=False, help="Logger on")
    parser.add_argument("--wb_proj", type=str, default="ATARI_PPO", help="project name")
    parser.add_argument("--wb_model", type=str, default="PPO", help="model name")
    parser.add_argument("--wb_entity", type=str, default="revisit-spectral-rl", help="project entity")

    # Environment parameters
    parser.add_argument('--e_id', type=str, default='Breakout', help='ATARI ENVIRONMENTS')
    parser.add_argument('--e_gamma', type=float, default=0.99, help='Discount factor gamma')
    parser.add_argument("--e_n_frames", type=int, default=4, help="Frames to Stack as input")
    parser.add_argument("--e_n_envs", type=int, default=128, help="#parallel environments")
    parser.add_argument("--e_n_test_envs", type=int, default=0, help="#parallel environments")
    parser.add_argument("--e_n_steps", type=int, default=128, help="#steps per environment in each update")

    # Network
    parser.add_argument('--c_dims', type=int, default=[512], nargs='*', help='Number of dimensions for Conv Encoder')
    parser.add_argument('--q_dims', type=int, default=[], nargs='*', help='Number of dimensions for Q network')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Encoder Learning Rate')
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--batch_norm", type=str2bool, default=False)
    parser.add_argument("--layer_norm", type=str2bool, default=False)

    # Laplacian Network
    parser.add_argument("--lap_lr", type=float_or_none, default=2.5e-4, help="learning rate of the Lap network")
    parser.add_argument("--lap_dims", type=int, default=512, help="laplacian dims")
    parser.add_argument("--lap_max_grad_norm", type=float, default=10)
    parser.add_argument("--lap_batch_norm", type=str2bool, default=False)
    parser.add_argument("--lap_layer_norm", type=str2bool, default=False)
    parser.add_argument('--lap_average', type=str2bool, default=True, help="Dimension wise average")

    parser.add_argument("--lap_br", type=str2bool, default=True, help="Learn Barrier Coefficient")
    parser.add_argument("--lap_br_lr", type=float_or_none, default=0.01, help="learning rate of the barrier coeff")
    parser.add_argument("--lap_br_val", type=float_or_none, default=100)
    parser.add_argument("--lap_br_clip", type=str2bool, default=True, help="clip barrier coefficient update grad")

    parser.add_argument("--lap_dual", type=str2bool, default=True, help="learning rate of the dual variables")
    parser.add_argument("--lap_dual_lr", type=float_or_none, default=0.01, help="learning rate of the dual variables")
    parser.add_argument("--lap_dual_clip", type=str2bool, default=True, help="clip dual update grad")

    parser.add_argument("--lap_err_rate", type=float_or_none, default=1., help="Error Update Rate")
    parser.add_argument("--lap_q_err_rate", type=float_or_none, default=0.1, help="Quadratic Error Update Rate")
    parser.add_argument("--lap_save_metrics", type=str2bool, default=False)

    # Algorithm parameters
    parser.add_argument("--a_n_epochs", type=int, default=2, help='#epochs per update')
    parser.add_argument("--a_upd_epochs", type=int, default=4, help='K epochs to update the policy')
    parser.add_argument("--a_n_mini_batches", type=int, default=4, help="mini-batches per epoch")
    parser.add_argument("--a_anneal_lr", type=str2bool, default=True, help="anneal learning rate")
    parser.add_argument("--a_gae_lambda", type=float, default=0.95, help="lambda for GAE")
    parser.add_argument("--a_norm_adv", type=str2bool, default=True, help="advantage normalization")
    parser.add_argument("--a_clip_coef", type=float, default=0.1, help="surrogate clipping coefficient")
    parser.add_argument("--a_clip_vloss", type=str2bool, default=True, help="clipped loss for value function")
    parser.add_argument("--a_ent_coef", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--a_vf_coef", type=float, default=0.5, help="value function coefficient")
    parser.add_argument("--a_target_kl", type=float_or_none, default=None, help="target kl divergence threshold")
    parser.add_argument('--a_tot_steps', type=lambda x: int(float(x)), default=5e7,
                        help="Environment steps - with 4 frame skip corresponds to 200M frames")
    parser.add_argument("--a_save_video", type=str2bool, default=False, help="Save final video")

    parser.add_argument("--a_int_coef", type=float, default=1.0, help="coefficient of intrinsic reward")
    parser.add_argument("--a_ext_coef", type=float, default=2.0, help="coefficient of extrinsic reward")
    parser.add_argument('--a_int_gamma', type=float, default=0.99, help='Discount factor gamma')
    parser.add_argument('--a_obs_burnin', type=int, default=500_000, help="Obervation burnin steps")
    parser.add_argument('--a_upd_prop', type=float, default=1.0, help='RND update proportion')

    parser.add_argument("--a_n_options", type=int, default=10, help='number of options')
    parser.add_argument("--a_bidir_options", type=str2bool, default=False, help='Bidirectional Options')
    parser.add_argument("--a_opt_dur", type=int, default=10, help='option duration')
    parser.add_argument("--a_opt_dur_min", type=int, default=5, help='option duration')
    parser.add_argument("--a_opt_dur_max", type=int, default=50, help='option duration')
    parser.add_argument("--a_eigdur_options", type=str2bool, default=False, help='eigenvalue-based option durations')

    parser.add_argument("--a_ioq_deter", type=str2bool, default=False, help='Bidirectional Options')
    return parser.parse_args()


def convert_args_to_dict(args):

    args.a_batch_size = int(args.e_n_envs * args.e_n_steps)
    args.a_minibatch_size = int(args.a_batch_size // args.a_n_mini_batches)
    args.a_num_iterations = args.a_tot_steps // args.a_batch_size

    args.lap_dual_lr = 0 if not args.lap_dual else args.lap_dual_lr
    args.lap_br_lr = 0 if not args.lap_br else args.lap_br_lr

    LapConfig = LaplacianConfig(lr=args.lap_lr,
                              dims=args.lap_dims, average_loss=args.lap_average,
                              max_grad_norm=args.lap_max_grad_norm,
                              dual=DualArgs(on=args.lap_dual, lr=args.lap_dual_lr, clip=args.lap_dual_clip),
                              barrier=BarrierArgs(on=args.lap_br, value=args.lap_br_val, lr=args.lap_br_lr,
                                                  clip=args.lap_br_clip),
                              rate=RateArgs(error=args.lap_err_rate, q_error=args.lap_q_err_rate),
                              batch_norm=args.lap_batch_norm, layer_norm=args.lap_layer_norm)

    config = Config(
        gpu=GPUConfig(id=args.gpu_id, perc=args.gpu_perc),
        net=NetConfig(lr=args.lr, max_grad_norm=args.max_grad_norm,
                      c_dims=tuple(args.c_dims), q_dims=tuple(args.q_dims),
                      batch_norm=args.batch_norm, layer_norm=args.layer_norm,
                      ),
        laplacian=LapConfig,
        exp=ExpConfig(seed=args.x_seed, file=args.x_file,
                      job_id=int(os.environ.get("SLURM_JOB_ID", -1)),
                      p_id=os.getpid()),
        wb=WBConfig(on=args.wb_on, proj=args.wb_proj, model=args.wb_model, entity=args.wb_entity),
        env=EnvConfig(id=args.e_id, gamma=args.e_gamma, n_frames=args.e_n_frames,
                      n_envs=args.e_n_envs, n_test_envs=args.e_n_test_envs,
                      n_total_envs=args.e_n_envs + args.e_n_test_envs, n_steps=args.e_n_steps),
        algo=AlgoConfig(n_epochs=args.a_n_epochs, upd_epochs=args.a_upd_epochs, n_mini_batches=args.a_n_mini_batches,
                        batch_size=args.a_batch_size, mini_batch_size=args.a_minibatch_size,
                        num_iterations=args.a_num_iterations, tot_steps=args.a_tot_steps,
                        anneal_lr=args.a_anneal_lr, gae_lambda=args.a_gae_lambda, norm_adv=args.a_norm_adv,
                        clip_coef=args.a_clip_coef, ent_coef=args.a_ent_coef, vf_coef=args.a_vf_coef,
                        clip_vloss=args.a_clip_vloss, target_kl=args.a_target_kl,
                        save_video=args.a_save_video,
                        int_coef=args.a_int_coef, ext_coef=args.a_ext_coef, int_gamma=args.a_int_gamma,
                        obs_burnin=args.a_obs_burnin, upd_prop=args.a_upd_prop,
                        n_options=args.a_n_options, bidir_options=args.a_bidir_options,
                        eigdur_options=args.a_eigdur_options,
                        opt_dur=args.a_opt_dur, opt_dur_min=args.a_opt_dur_min, opt_dur_max=args.a_opt_dur_max,
                        ioq_deter=args.a_ioq_deter,))

    return config


def create_wb_logger(params):
    if params.wb.on:
        run_name = f"{params.env.id}__{params.exp.file}__{params.exp.seed}__{int(time.time())}"
        wandb.init(
            project=params.wb.proj,
            entity=params.wb.entity,
            config=params,
            name=run_name,
            monitor_gym=False,
            save_code=True,
            group=params.wb.model,
            job_type=str(params.exp.seed),
        )


def close_logger():
    wandb.finish()

