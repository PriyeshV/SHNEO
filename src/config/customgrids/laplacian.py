import os
import time
import wandb
import argparse
from src.utils.utils import DotDict
from distutils.util import strtobool
from custom_gridworlds import all_mazes, n_states

from src.config.customgrids.dataclass_args import *
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
    parser.add_argument("--wb_proj", type=str, default="HEO_CustomGrids", help="project name")
    parser.add_argument("--wb_model", type=str, default="DQN", help="model name")
    parser.add_argument("--wb_entity", type=str, default="revisit-spectral-rl", help="project entity")
    # Environment parameters
    parser.add_argument('--e_maze_type', type=str, default='GridRoom-4', choices=all_mazes,
                        help='Type of maze environment')
    parser.add_argument('--e_obs_mode', type=str, default='pixels', choices=['pixels', 'xy', 'one-hot'])
    parser.add_argument("--e_screen_size", type=int_or_none, default=None, choices=[32, 64, 84])
    parser.add_argument('--e_rand_reset', type=str2bool, default=False, help='Random Restarts')
    parser.add_argument('--e_eval_rand_reset', type=str2bool, default=False, help='Eval: Random Restarts')
    parser.add_argument('--e_gamma', type=float, default=0.99, help='Discount factor gamma')

    # FlashBax Buffer
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
    parser.add_argument("--ckpt_interval", type=int, default=50_000)

    # DQN
    parser.add_argument("--dqn", type=str2bool, default=False, help="DQN")

    # Network
    parser.add_argument('--c_dims', type=int, default=[128], nargs='*', help='Number of dimensions for Conv Encoder')
    parser.add_argument('--q_dims', type=int, default=[], nargs='*', help='Number of dimensions for Q network')
    parser.add_argument('--lr_enc', type=float_or_none, default=None, help='Encoder Learning Rate')
    parser.add_argument('--lr_ffn', type=float, default=1e-4, help='QNet_FFN Learning Rate')
    parser.add_argument("--max_grad_norm", type=float, default=10)
    parser.add_argument("--ip_batch_norm", type=str2bool, default=True)
    parser.add_argument("--ip_layer_norm", type=str2bool, default=False)
    parser.add_argument("--ip_l2_norm", type=str2bool, default=False)
    parser.add_argument("--batch_norm", type=str2bool, default=False)
    parser.add_argument("--layer_norm", type=str2bool, default=True)
    parser.add_argument("--fusion_ip_norm", type=str2bool, default=True)
    parser.add_argument("--impala_cnn", type=str2bool, default=False)
    parser.add_argument("--cnn_width_mul", type=float, default=1.0)

    # Laplacian Network
    parser.add_argument("--lap_lr", type=float_or_none, default=1e-4, help="learning rate of the Lap network")
    parser.add_argument("--lap_dims", type=int, default=128, help="laplacian dims")
    parser.add_argument("--lap_boot", type=str2bool, default=False, help="Bootstrap laplacian embeddings")
    parser.add_argument("--lap_rms_norm", type=str2bool, default=False, help='Neural Eigen L2 Batch Norm to g   et unit eigen vectors')
    parser.add_argument("--lap_max_grad_norm", type=float, default=0.5)

    parser.add_argument("--lap_sa", type=str2bool, default=False, help="State Action Representations")
    parser.add_argument("--lap_uni", type=str2bool, default=False, help="Uniform Random Policy Representations")
    parser.add_argument("--lap_sa_one_hot", type=str2bool, default=True, help="One hot or embedding for actions")
    parser.add_argument("--lap_sa_interact", type=str, default='project', choices=['project', 'concat', 'film', 'None'])
    parser.add_argument("--lap_imp_samp", type=str2bool, default=False, help="Importance sampling for encoder ")
    parser.add_argument("--lap_batch_norm", type=str2bool, default=False)
    parser.add_argument("--lap_layer_norm", type=str2bool, default=False)

    parser.add_argument('--lap_average', type=str2bool, default=True, help="Dimension wise average")
    parser.add_argument('--lap_n_step', type=int_or_none, default=1, help="Step size of future state")
    parser.add_argument('--lap_sample_n_step', type=str2bool, default=True, help="Sample future state")
    parser.add_argument("--lap_upd_freq", type=int_or_none, default=1, choices=range(1, 6), help="Lap network update frequency")
    parser.add_argument("--lap_tgt_net", type=str2bool, default=False, help="use target network for the encoder ")
    parser.add_argument("--lap_q_tgt_net", type=str2bool, default=True, help="use lap target network for computing q targets")

    parser.add_argument("--lap_br", type=str2bool, default=True, help="Learn Barrier Coefficient")
    parser.add_argument("--lap_br_lr", type=float_or_none, default=0.01, help="learning rate of the barrier coeff")
    parser.add_argument("--lap_br_val", type=float_or_none, default=100)
    parser.add_argument("--lap_br_clip", type=str2bool, default=False, help="clip barrier coefficient update grad")

    parser.add_argument("--lap_dual", type=str2bool, default=True, help="learning rate of the dual variables")
    parser.add_argument("--lap_dual_lr", type=float_or_none, default=0.01, help="learning rate of the dual variables")
    parser.add_argument("--lap_dual_clip", type=str2bool, default=False, help="clip dual update grad")
    parser.add_argument("--lap_eig_clip", type=str2bool, default=False,
                        help="Clip Eigen (dual) value range by #Actions")

    parser.add_argument("--lap_err_rate", type=float_or_none, default=1., help="Error Update Rate")
    parser.add_argument("--lap_q_err_rate", type=float_or_none, default=0.1, help="Quadratic Error Update Rate")
    parser.add_argument("--lap_save_metrics", type=str2bool, default=False)

    # Eigenvalue scaling parameters
    parser.add_argument("--eig_on", type=str2bool, default=False)
    parser.add_argument("--eig_inv_pow", type=int, default=0, help="1/sqrt(x^pow))")
    parser.add_argument("--eig_poly_deg", type=int, default=0, help="Polynomial degree: x + ax^2 + ... + ax^deg")
    parser.add_argument("--eig_non_param", type=str2bool, default=None, help="weights for each eigenvalue")

    # Algorithm parameters
    parser.add_argument("--a_n_options", type=int, default=10, help="Number of Options/Meta-Actions")
    parser.add_argument("--a_fusion", type=str2bool, default=False, help="Mixture of Experts: DQN-Q and Lap-Q values")
    parser.add_argument("--a_lr_alpha", type=float, default=0.1, help="learning rate decay factor")
    parser.add_argument("--a_enc_allow_grad", type=str2bool, default=False, help="Allow or regularized laplacian encoder")
    parser.add_argument("--a_lap_loss_wt", type=float_or_none, default=None)
    parser.add_argument('--a_ddqn', type=str2bool, default=True)
    parser.add_argument('--a_n_step', type=int, default=1, help="N-Step DQN")
    parser.add_argument('--a_m_step', type=int, default=1, help="M-Step H-DQN")
    parser.add_argument('--a_opt_dur', type=int, default=10, help="Option Duration for H-DQN")
    parser.add_argument('--a_sample_act', type=str2bool, default=False, help="Sample Actions")
    parser.add_argument('--a_q_clip_q_pred', type=str2bool, default=False,
                        help='Clip TD-Target: jnp.clip(q_next_target, jnp.min(q_pred), jnp.max(q_pred))')
    parser.add_argument('--a_tot_steps', type=int, default=200_000, help='Training Steps')
    parser.add_argument('--a_max_eps_len', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument("--a_starts", type=int, default=20_000, help="timestep to start when buffer size reaches this value")
    parser.add_argument("--a_train_freq", type=int, default=1, help="the frequency of training")
    parser.add_argument("--a_tau", type=float, default=1, help="the target network update rate")
    parser.add_argument("--a_tgt_upd_freq", type=int, default=500, help="target network update frequency")
    parser.add_argument('--a_parseval_mul', type=float, default=0.0, help='Learning Rate')

    parser.add_argument("--a_bidir_opt", type=str2bool, default=False, help="Eigenvalue based Option Duration")
    parser.add_argument("--a_eig_dur", type=str2bool, default=False, help="Eigenvalue based Option Duration")
    parser.add_argument("--a_phi_norm", type=str2bool, default=True, help="Intrinsic representation normalization")
    parser.add_argument("--a_intr_norm", type=str2bool, default=True, help="Intrinsic reward normalization")

    # Exploration
    parser.add_argument('--a_eps_start', type=float, default=1.0, help='Exploration Starting Epsilon')
    parser.add_argument('--a_eps_end', type=float, default=0.05, help='Exploration Ending Epsilon')
    parser.add_argument('--a_eps_frac', type=float, default=0.5,
                        help='the fraction of `total-timesteps` it takes from start-e to go end-e')
    parser.add_argument('--a_s_tau_start', type=float, default=0., help='Exploration Starting Epsilon')
    parser.add_argument('--a_s_tau_end', type=float, default=0., help='Exploration Ending Epsilon')
    parser.add_argument('--a_s_tau_frac', type=float, default=0.,
                        help='the fraction of `total-timesteps` it takes from start-e to go end-e')

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
    parser.add_argument('--v_on', type=str2bool, default=False, help='Turn on Visualizations')
    parser.add_argument('--v_emb_annot', type=str2bool, default=False, help='Annotate Embedding visualization')
    return parser.parse_args()


def convert_args_to_dict(args):

    if not args.lap_sa or args.lap_sa_interact.lower() == 'none':
        args.lap_sa_interact = None
    if args.lap_sa:
        assert args.lap_sa_interact is not None

    # assert not (args.ip_batch_norm and args.ip_layer_norm), 'Only one of the ip norms can be true'

    if not args.a_enc_allow_grad:
        args.a_lap_loss_wt = None
    assert args.a_lap_loss_wt is not None if args.a_enc_allow_grad else True, 'Need to specify a_lap_loss_wt with a_enc_allow_grad'
    assert args.a_lap_loss_wt >= 0 if args.a_enc_allow_grad else True, 'Laplacian loss weights can only be non-negative'
    if args.a_enc_allow_grad and args.a_lap_loss_wt == 0:
        args.dqn = True

    assert args.lap_n_step >= 1, 'Takes only positive values'
    assert args.a_n_step >= 1, 'Takes only positive values'
    assert args.a_m_step >= 1, 'Takes only positive values'
    assert args.a_n_options <= args.lap_dims, 'Number of options should be <= Number of Laplacian Dimensions'

    assert args.lap_upd_freq >= 1
    args.buff_size = args.buff_size if args.buff_size is not None else int(0.2*args.a_tot_steps)
    print('Buffer Size: ', args.buff_size)

    assert not args.lap_tgt_net or args.lap_q_tgt_net  # lap_tgt_net can be true only when lap_q_tgt_net is True
    args.lap_tgt_net &= args.lap_q_tgt_net

    assert args.f_p_dims is not None or args.lap_dims is not None
    args.f_p_dims = args.f_p_dims or args.lap_dims/n_states[args.e_maze_type]
    args.lap_dims = args.lap_dims or args.f_p_dims * n_states[args.e_maze_type]
    # assert args.lap_dims == args.q_dims[0], "Ensure that the first dim of Q-Network matches the laplacian dim"

    assert (not args.lap_uni) or args.lap_sa, "Uniform Random Policy (URP) is only supported for state-action case"
    assert (not args.lap_imp_samp) or args.lap_uni, "Importance sampling for encoder is only supported for URP"

    args.lap_dual_lr = 0 if not args.lap_dual else args.lap_dual_lr
    args.lap_br_lr = 0 if not args.lap_br else args.lap_br_lr

    if args.dqn:
        reg_dqn = args.a_enc_allow_grad and args.a_lap_loss_wt == 0  # SA version of DQN is possible
        args.ip_batch_norm = False
        args.ip_layer_norm = False
        args.ip_l2_norm = False
        args.lap_sa = None if not reg_dqn else args.lap_sa
        args.lap_sa_interact = None if not reg_dqn else args.lap_sa_interact
        args.lap_uni = None if not reg_dqn else args.lap_uni
        args.lap_imp_samp = None
        args.lap_average = None
        args.lap_n_step = None
        args.lap_sample_n_step = None
        args.lap_upd_freq = None
        args.lap_tgt_net = None
        args.lap_q_tgt_net = None if not (args.a_enc_allow_grad and args.a_lap_loss_wt == 0) else args.lap_q_tgt_net
        args.lap_br = None
        args.lap_rms_norm = None
        args.lap_br_lr = None
        args.lap_br_val = None
        args.lap_br_clip = None
        args.lap_dual = None
        args.lap_dual_lr = None
        args.lap_dual_clip = None
        args.lap_eig_clip = None
        args.lap_err_rate = None
        args.lap_q_err_rate = None
        args.lap_save_metrics = None

    LapConfig = LaplacianArgs(lr=args.lap_lr, sa=args.lap_sa,
                              one_hot=args.lap_sa_one_hot, sa_interact=args.lap_sa_interact,
                              rms_norm=args.lap_rms_norm, bootstrap=args.lap_boot,
                              uni=args.lap_uni, imp_samp=args.lap_imp_samp,
                              dims=args.lap_dims, average_loss=args.lap_average,
                              n_step=args.lap_n_step, sample_n_step=args.lap_sample_n_step,
                              max_grad_norm=args.lap_max_grad_norm,
                              upd_freq=args.lap_upd_freq, tgt_net=None, q_tgt_net=None,
                              dual=DualArgs(on=args.lap_dual, lr=args.lap_dual_lr, eig_clip=args.lap_eig_clip,
                                            clip=args.lap_dual_clip),
                              barrier=BarrierArgs(on=args.lap_br, value=args.lap_br_val, lr=args.lap_br_lr,
                                                  clip=args.lap_br_clip),
                              rate=RateArgs(error=args.lap_err_rate, q_error=args.lap_q_err_rate),
                              batch_norm=args.lap_batch_norm, layer_norm=args.lap_layer_norm,
                              save_metrics=args.lap_save_metrics)

    config = Config(
        gpu=GPUConfig(id=args.gpu_id, perc=args.gpu_perc),
        net=NetConfig(c_dims=tuple(args.c_dims), q_dims=tuple(args.q_dims),
                      fusion_ip_norm=args.fusion_ip_norm,
                      max_grad_norm=args.max_grad_norm,
                      ip_batch_norm=args.ip_batch_norm, ip_layer_norm=args.ip_layer_norm, ip_l2_norm=args.ip_l2_norm,
                      batch_norm=args.batch_norm, layer_norm=args.layer_norm,
                      lr=LRConfig(enc=args.lr_enc, ffn=args.lr_ffn),
                      impala_cnn=args.impala_cnn, cnn_width_mul=args.cnn_width_mul),
        laplacian=LapConfig,
        eigen_transform=EigenTransformConfig(on=args.eig_on, gamma=args.e_gamma, inv_pow=args.eig_inv_pow,
                                             poly_deg=args.eig_poly_deg, non_param=args.eig_non_param),
        exp=ExpConfig(seed=args.x_seed, file=args.x_file,
                      job_id=int(os.environ.get("SLURM_JOB_ID", -1)),
                      p_id=os.getpid()),
        wb=WBConfig(on=args.wb_on, proj=args.wb_proj, model=args.wb_model, entity=args.wb_entity),
        algo=AlgoConfig(n_options=args.a_n_options, fusion=args.a_fusion, lr_alpha=args.a_lr_alpha, ddqn=args.a_ddqn,
                        enc_allow_grad=args.a_enc_allow_grad, lap_loss_wt=args.a_lap_loss_wt,
                        starts=args.a_starts, sample_action=args.a_sample_act,
                        n_step=args.a_n_step, m_step=args.a_m_step, opt_dur=args.a_opt_dur,
                        max_eps_len=args.a_max_eps_len, tot_steps=args.a_tot_steps,
                        eps=EpsConfig(frac=args.a_eps_frac, start=args.a_eps_start, end=args.a_eps_end),
                        s_tau=STauConfig(frac=args.a_s_tau_frac, start=args.a_s_tau_start, end=args.a_s_tau_end),
                        clip_q_pred=args.a_q_clip_q_pred,
                        tau=args.a_tau,
                        train_freq=args.a_train_freq, tgt_upd_freq=args.a_tgt_upd_freq,
                        parseval_mul=args.a_parseval_mul,
                        bidir_opt=args.a_bidir_opt,
                        eig_dur=args.a_eig_dur, phi_norm=args.a_phi_norm, intr_norm=args.a_intr_norm),
        ckpt=CKPTConfig(dir=args.ckpt_dir, keep=args.ckpt_keep, interval=args.ckpt_interval),
        buffer=BufferConfig(prioritized=args.buff_prior, lap_prioritized=args.buff_lap_prior,
                            size=args.buff_size, beta=args.buff_beta, device=args.buff_device,
                            q_batch_size=args.buff_q_batch_size, lap_batch_size=args.buff_lap_batch_size),
        env=EnvConfig(maze_type=args.e_maze_type, obs_mode=args.e_obs_mode, screen_size=args.e_screen_size,
                      rand_reset=args.e_rand_reset, eval_rand_reset=args.e_eval_rand_reset, gamma=args.e_gamma,
                      n_states=n_states[args.e_maze_type]),
        features=FeaturesConfig(pi=args.f_pi, phi_comb=args.f_phi_comb, n_dims=args.f_n_dims, p_dims=args.f_p_dims),
        graph=GraphConfig(state_action=args.g_state_action, undirected=args.g_undirected,
                          normalized_P=args.g_normalized_P, laplacian=args.g_laplacian),
        visual=VisualConfig(on=args.v_on, emb_annot=args.v_emb_annot))

    return config


def create_wb_logger(params):
    if params.wb.on:
        run_name = f"{params.env.maze_type}__{params.exp.file}__{params.exp.seed}__{int(time.time())}"
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

