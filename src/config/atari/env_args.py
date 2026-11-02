import argparse


def add_atari_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--e_id", type=str, default="Breakout", help="Atari game id without -v5 suffix")
    parser.add_argument("--e_n_test_envs", type=int, default=8, help="# greedy test envs packed into same envpool instance")


def set_atari_defaults(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        # Environment
        e_n_frames=4,

        # Agent
        a_td_lambda=0.65,

        # Options
        o_td_lambda=0.65,
        o_meta_td_lambda=0.65,
        o_mu_prob=0.9,
        o_opt_dur=10,
        o_n_options=10,

        # Logging
        wb_proj="SHNEO-Atari",
    )