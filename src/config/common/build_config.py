import os
import time
import wandb

from src.config.common.dataclass_args import (
    Config,
    GPUConfig,
    ExpConfig,
    WBConfig,
    EnvConfig,
    NetConfig,
    AlgoConfig,
    EpsConfig,
    LaplacianConfig,
    DualArgs,
    RateArgs,
    BarrierArgs,
    RNDArgs,
    OptionArgs,
    NEOArgs,
)


def convert_args_to_config(args, env_family: str) -> Config:
    # -----------------------------------------------------------------
    # Derived sizes
    # -----------------------------------------------------------------
    batch_size = int(args.e_n_steps * args.e_n_envs)
    mini_batch_size = int(batch_size // args.a_n_mini_batches)
    num_iterations = int(args.a_tot_steps // batch_size)
    n_upds = num_iterations

    # -----------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------
    assert batch_size % args.a_n_mini_batches == 0, \
        "# mini-batches must divide e_n_steps * e_n_envs"

    assert (args.o_m_steps * args.e_n_envs) % args.o_n_mini_batches == 0, \
        "# option mini-batches must divide o_m_steps * e_n_envs"

    lap_dual_lr = 0.0 if not args.lap_dual else args.lap_dual_lr
    lap_br_lr = 0.0 if not args.lap_br else args.lap_br_lr

    # -----------------------------------------------------------------
    # Shared sub-configs
    # -----------------------------------------------------------------
    lap_config = LaplacianConfig(
        dims=args.lap_dims,
        lr=args.lap_lr,
        layer_norm=args.lap_layer_norm,
        batch_norm=args.lap_batch_norm,
        max_grad_norm=args.lap_max_grad_norm,
        dual=DualArgs(
            on=args.lap_dual,
            lr=lap_dual_lr,
            clip=args.lap_dual_clip,
            eig_clip=None,
        ),
        rate=RateArgs(
            error=args.lap_err_rate,
            q_error=args.lap_q_err_rate,
        ),
        barrier=BarrierArgs(
            on=args.lap_br,
            lr=lap_br_lr,
            clip=args.lap_br_clip,
            value=args.lap_br_val,
        ),
        n_step=args.e_n_steps - 1,
        average_loss=args.lap_average,
    )

    rnd_config = RNDArgs(
        int_gamma=args.r_int_gamma,
        obs_burnin=args.r_obs_burnin,
        upd_prop=args.r_upd_prop,
    )

    option_config = OptionArgs(
        gamma=args.o_gamma,
        td_lambda=args.o_td_lambda,
        m_steps=args.o_m_steps,
        mu_prob=args.o_mu_prob,
        meta_td_lambda=args.o_meta_td_lambda,
        n_mini_batches=args.o_n_mini_batches,
        n_options=args.o_n_options,
        bidir_options=args.o_bidir_options,
        opt_dur=args.o_opt_dur,
        opt_dur_min=args.o_opt_dur_min,
        opt_dur_max=args.o_opt_dur_max,
        eigdur_options=args.o_eigdur_options,
        ioq_deter=args.o_ioq_deter,
        stall_coef=args.o_stall_coef,
        policy_greedy=args.o_policy_greedy,
        policy_temp=args.o_policy_temp,
    )

    # -----------------------------------------------------------------
    # Env config by family
    # -----------------------------------------------------------------
    if env_family == "atari":
        env_config = EnvConfig(
            family="atari",
            id=args.e_id,
            gamma=args.e_gamma,
            n_frames=args.e_n_frames,
            n_envs=args.e_n_envs,
            n_steps=args.e_n_steps,
            n_test_envs=args.e_n_test_envs,
            n_total_envs=args.e_n_envs + args.e_n_test_envs,
            max_episode_steps=args.e_max_episode_steps,
            reward_clip=args.e_reward_clip,
        )
    elif env_family == "procgen":
        env_config = EnvConfig(
            family="procgen",
            id=args.e_id,
            gamma=args.e_gamma,
            n_frames=args.e_n_frames,
            n_envs=args.e_n_envs,
            n_steps=args.e_n_steps,
            n_test_envs=args.e_n_test_envs,
            n_total_envs=args.e_n_envs + args.e_n_test_envs,
            start_level=args.e_start_level,
            distribution_mode=args.e_distribution_mode,
            center_agent=args.e_center_agent,
            use_backgrounds=args.e_use_backgrounds,
            use_monochrome_assets=args.e_use_monochrome_assets,
            restrict_themes=args.e_restrict_themes,
            use_generated_assets=args.e_use_generated_assets,
            paint_vel_info=args.e_paint_vel_info,
            use_easy_jump=args.e_use_easy_jump,
            max_episode_steps=args.e_max_episode_steps,
            reward_clip=args.e_reward_clip,
        )
    else:
        raise ValueError(f"Unknown env_family: {env_family}")

    # -----------------------------------------------------------------
    # Top-level config
    # -----------------------------------------------------------------
    config = Config(
        gpu=GPUConfig(
            id=args.gpu_id,
            perc=args.gpu_perc,
        ),
        exp=ExpConfig(
            seed=args.x_seed,
            file=args.x_file,
            job_id=int(os.environ.get("SLURM_JOB_ID", -1)),
            p_id=os.getpid(),
        ),
        wb=WBConfig(
            on=args.wb_on,
            proj=args.wb_proj,
            model=args.wb_model,
            entity=args.wb_entity,
        ),
        env=env_config,
        net=NetConfig(
            lr=args.lr,
            c_dims=tuple(args.c_dims),
            q_dims=tuple(args.q_dims),
            max_grad_norm=args.max_grad_norm,
            batch_norm=args.batch_norm,
            layer_norm=args.layer_norm,
            impala=args.impala,
        ),
        algo=AlgoConfig(
            n_epochs=args.a_n_epochs,
            n_upds=n_upds,
            n_mini_batches=args.a_n_mini_batches,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            num_iterations=num_iterations,
            tot_steps=args.a_tot_steps,
            anneal_lr=args.a_anneal_lr,
            td_lambda=args.a_td_lambda,
            int_coef=args.a_int_coef,
            rew_scale=args.a_rew_scale,
            eps=EpsConfig(
                frac=args.a_eps_frac,
                start=args.a_eps_start,
                end=args.a_eps_end,
            ),
            save_video=args.a_save_video,
            use_rnd_eps=args.a_use_rnd_eps,
            rnd_eps_max=args.a_rnd_eps_max,
        ),
        laplacian=lap_config,
        rnd=rnd_config,
        options=option_config,
        neo=NEOArgs(
            k=args.neo_k,
            delta=args.neo_delta,
            target_mean=args.neo_targ_mean,
            rnd_coef=args.neo_rnd_coef,
            lap_novel_weight=args.neo_lap_wt,
        ),
    )
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