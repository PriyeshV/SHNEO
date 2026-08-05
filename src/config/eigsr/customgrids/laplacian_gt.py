import os
import time
import wandb
import argparse
from src.utils.utils import DotDict
from distutils.util import strtobool
from custom_gridworlds import all_mazes, n_states
from src.utils.utils import str2bool, int_or_none, float_or_none
from src.utils.laplacian_args import LaplacianArgs, BarrierArgs, DualArgs, RateArgs


max_steps = {
            # 'GridMaze-11': 200_000, 'GridRoom-4': 200_000, 'GridRoom-16': 500_000,
            # 'GridMaze-26': 500_000, 'GridRoom-32': 750_000, 'GridRoom-64': 750_000,
            # 'HardMaze': 200_000
            }


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment parameters")
    # GPU
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--gpu_perc", type=float, default=0.3)

    # Experiment parameters
    parser.add_argument("--x_seed", type=int, default=0, help="experiment seed")
    parser.add_argument("--x_file", type=str, default=os.path.basename(__file__).rstrip(".py"), help="File name")
    parser.add_argument("--wb_on", type=str2bool, default=False, help="Logger on")
    parser.add_argument("--wb_proj", type=str, default="GT_EigVec", help="project name")
    parser.add_argument("--wb_model", type=str, default="ALLO", help="model name")
    parser.add_argument("--wb_entity", type=str, default="revisit-spectral-rl", help="project entity")
    # Environment parameters
    parser.add_argument('--e_maze_type', type=str, default='GridMaze-19', choices=all_mazes,
                        help='Type of maze environment')
    parser.add_argument('--e_obs_mode', type=str, default='pixels', choices=['pixels', 'xy', 'one-hot'])
    parser.add_argument("--e_screen_size", type=int, default=64, choices=[32, 64, 84])
    parser.add_argument('--e_rand_reset', type=str2bool, default=False, help='Random Restarts')
    parser.add_argument('--e_eval_rand_reset', type=str2bool, default=False, help='Eval: Random Restarts')
    parser.add_argument('--e_gamma', type=float, default=0.99, help='Discount factor gamma')

    # Bax Buffer
    parser.add_argument("--buff_prior", type=str2bool, default=True, help="Prioritized Buffer")
    parser.add_argument("--buff_beta", type=float_or_none, default=0.4, help="Prioritized Buffer")
    parser.add_argument("--buff_lap_prior", type=str2bool, default=False, help="Prioritized Neg Sampling")
    parser.add_argument("--buff_size", type=int_or_none, default=None, help="the replay memory buffer size")
    parser.add_argument("--buff_device", type=str, default='cpu', choices=['gpu', 'cpu'])
    parser.add_argument('--buff_q_batch_size', type=int, default=64)
    parser.add_argument('--buff_lap_batch_size', type=int, default=64)

    # [TODO] Checkpoint
    parser.add_argument("--ckpt_dir", type=str,
                        default="/Users/priyeshvijayan/Desktop/Git_Codes/EigSR/pt_ckpt")
    parser.add_argument("--ckpt_keep", type=int, default=1)
    parser.add_argument("--ckpt_interval", type=int, default=25_000)

    # Network
    parser.add_argument('--c_dims', type=int, default=[128], nargs='*', help='Number of dimensions for Conv Encoder')
    parser.add_argument('--q_dims', type=int, default=[16, 8], nargs='*', help='Number of dimensions for Q network')
    parser.add_argument('--lr_enc', type=float_or_none, default=None, help='Encoder Learning Rate')
    parser.add_argument('--lr_ffn', type=float, default=1e-3, help='QNet_FFN Learning Rate')
    parser.add_argument("--max_grad_norm", type=float, default=5)
    parser.add_argument("--ip_batch_norm", type=str2bool, default=False)
    parser.add_argument("--batch_norm", type=str2bool, default=False)
    parser.add_argument("--layer_norm", type=str2bool, default=False)
    parser.add_argument("--huber_loss", type=str2bool, default=False)
    # Laplacian Network
    parser.add_argument("--lap_sa", type=str2bool, default=False, help="State Action Representations")
    parser.add_argument("--lap_uni", type=str2bool, default=True, help="Uniform Random Policy Representations")
    parser.add_argument("--lap_opt", type=str, default='ALLO', choices=['ALLO', 'GGDO'], help="GGDO or ALLO Optimizer")
    parser.add_argument("--lap_lr", type=float, default=1e-4, help="learning rate of the Lap network")
    parser.add_argument("--lap_lr_alpha", type=float, default=0., help="learning rate decay factor")
    parser.add_argument('--lap_riemann_grad', type=str2bool, default=False, help='Riemannian Gradient Update')
    parser.add_argument("--lap_max_grad_clip", type=str2bool, default=True)
    parser.add_argument("--lap_max_grad_norm", type=float, default=10)
    parser.add_argument("--lap_dims", type=int, default=11, help="laplacian dims")
    parser.add_argument("--lap_batch_norm", type=str2bool, default=True)

    parser.add_argument('--lap_average', type=str2bool, default=False, help="Dimension wise average")
    parser.add_argument('--lap_next_k', type=int, default=1, help="Step size of future state")

    parser.add_argument("--lap_br", type=str2bool, default=True, help="Learn Barrier Coefficient")
    parser.add_argument("--lap_br_lr", type=float_or_none, default=1e-2, help="learning rate of the barrier coeff")
    parser.add_argument("--lap_br_val", type=float_or_none, default=1)
    parser.add_argument("--lap_br_clip", type=str2bool, default=False, help="clip barrier coefficient update grad")

    parser.add_argument("--lap_dual", type=str2bool, default=False, help="learning rate of the dual variables")
    parser.add_argument("--lap_dual_lr", type=float_or_none, default=1e-2, help="learning rate of the dual variables")
    parser.add_argument("--lap_dual_clip", type=str2bool, default=False, help="clip dual update grad")
    parser.add_argument("--lap_eig_clip", type=str2bool, default=False,
                        help="Clip Eigen (dual) value range by #Actions")

    parser.add_argument("--lap_err_rate", type=float, default=1., help="Error Update Rate")
    parser.add_argument("--lap_q_err_rate", type=float, default=0.1, help="Quadratic Error Update Rate")

    parser.add_argument("--lap_diff_ortho", type=str2bool, default=False,
                        help="Separate batch for enforcing orthogonal constraints")
    parser.add_argument("--lap_upd_freq", type=int, default=1, choices=range(1, 6), help="Lap network update frequency")
    parser.add_argument("--lap_q_tgt_net", type=str2bool, default=True,
                        help="use lap target network for computing q targets")
    parser.add_argument("--lap_save_metrics", type=str2bool, default=True)

    # Algorithm parameters
    parser.add_argument('--a_tot_steps', type=int, default=50_000, help='Training Steps')

    parser.add_argument('--f_n_dims', type=int_or_none, default=None, help='Number of dimensions for features')
    parser.add_argument('--f_p_dims', type=float_or_none, default=None, help='perc of dimensions for features')
    parser.add_argument('--f_pi', type=str, default=None, choices=['r', 'pi'], help='representation''s policy')
    parser.add_argument('--f_phi_comb', type=str, default=None, choices=['sum', 'concat'],
                        help='representation''s policy')
    # Graph parameters
    parser.add_argument('--g_state_action', type=str2bool, default=None, help='Use state-action graph')
    parser.add_argument('--g_undirected', type=str2bool, default=None, help='Use undirected graph')
    parser.add_argument('--g_normalized_P', type=str2bool, default=False,
                        help='Normalize transition probability matrix')
    parser.add_argument('--g_laplacian', type=str2bool, default=None, help='Use graph Laplacian')
    # Visualization parameters
    parser.add_argument('--v_on', type=str2bool, default=True, help='Turn on Visualizations')
    parser.add_argument('--v_emb_annot', type=str2bool, default=False, help='Annotate Embedding visualization')
    return parser.parse_args()


def convert_args_to_dict(args):
    assert args.lap_upd_freq >= 1

    args.buff_size = args.buff_size or int(0.2*args.a_tot_steps)

    assert args.f_p_dims is not None or args.lap_dims is not None
    args.f_p_dims = args.f_p_dims or args.lap_dims/n_states[args.e_maze_type]
    args.lap_dims = args.lap_dims or args.f_p_dims * n_states[args.e_maze_type]

    args.lap_dual_lr = 0 if not args.lap_dual else args.lap_dual_lr
    args.lap_br_lr = 0 if not args.lap_br else args.lap_br_lr
    frozen_lap_args = LaplacianArgs(lr=args.lap_lr,
                                    sa=args.lap_sa, uni=args.lap_uni, dims=args.lap_dims,
                                    average_loss=args.lap_average, n_step=args.lap_next_k,
                                    # max_grad_clip=args.lap_max_grad_clip,
                                    max_grad_norm=args.lap_max_grad_norm,
                                    # opt=args.lap_opt,
                                    # diff_ortho=args.lap_diff_ortho,
                                    q_tgt_net=args.lap_q_tgt_net, upd_freq=args.lap_upd_freq,
                                    dual=DualArgs(on=args.lap_dual, lr=args.lap_dual_lr, eig_clip=args.lap_eig_clip,
                                                  clip=args.lap_dual_clip),
                                    barrier=BarrierArgs(on=args.lap_br, value=args.lap_br_val, lr=args.lap_br_lr,
                                                        clip=args.lap_br_clip),
                                    rate=RateArgs(error=args.lap_err_rate, q_error=args.lap_q_err_rate),
                                    batch_norm=args.lap_batch_norm,
                                    # riemannian_grad=args.lap_riemann_grad,
                                    save_metrics=args.lap_save_metrics)
    return {
        'gpu': {
            'id': args.gpu_id,
            'perc': args.gpu_perc,
        },
        'net': {
            'c_dims': args.c_dims,
            'q_dims': args.q_dims,
            'max_grad_norm': args.max_grad_norm,
            'ip_batch_norm': args.ip_batch_norm,
            'batch_norm': args.batch_norm,
            'layer_norm': args.layer_norm,
            'huber_loss': args.huber_loss,
            'lr': {
                'enc': args.lr_enc,
                'ffn': args.lr_ffn,
            },
        },
        'laplacian': frozen_lap_args,
        'ckpt': {
            'dir': args.ckpt_dir,
            'keep': args.ckpt_keep,
            'interval': args.ckpt_interval,
        },
        'buffer': {
            'prioritized': args.buff_prior,
            'lap_prioritized': args.buff_lap_prior,
            'size': args.buff_size,
            'beta': args.buff_beta,
            'device': args.buff_device,
            'q_batch_size': args.buff_q_batch_size,
            'lap_batch_size': args.buff_lap_batch_size,
        },
        'exp': {
            'seed': args.x_seed,
            'file': args.x_file,
            'job_id': int(os.environ.get("SLURM_JOB_ID", -1)),
            'p_id': os.getpid(),
        },
        'wb': {
            'on': args.wb_on,
            'proj': args.wb_proj,
            'model': args.wb_model,
            'entity': args.wb_entity,
        },
        'env': {
            'maze_type': args.e_maze_type,
            'obs_mode': args.e_obs_mode,
            'screen_size': args.e_screen_size,
            'rand_reset': args.e_rand_reset,
            'eval_rand_reset': args.e_eval_rand_reset,
            'gamma': args.e_gamma
        },
        'algo': {
            'tot_steps': args.a_tot_steps,
        },
        'features': {
            'pi': args.f_pi,
            'phi_comb': args.f_phi_comb,
            'n_dims': args.f_n_dims,
            'p_dims': args.f_p_dims,
        },
        'graph': {
            'state_action': args.g_state_action,
            'undirected': args.g_undirected,
            'normalized_P': args.g_normalized_P,
            'laplacian': args.g_laplacian
        },
        'visual': {
            'on': args.v_on,
            'emb_annot': args.v_emb_annot,
        }
    }


def create_wb_logger(params):
    if params.wb.on:
        run_name = f"{params.env.maze_type}__{params.exp.file}__{params.exp.seed}__{int(time.time())}"
        run = wandb.init(
            project=params.wb.proj,
            entity=params.wb.entity,
            config=params,
            name=run_name,
            monitor_gym=False,
            save_code=True,
            group=params.wb.model,
            job_type=str(params.exp.seed),
        )
        # run.config.update(params)
        return run


def close_logger(run):
    run.finish()

