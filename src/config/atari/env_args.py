import argparse


def add_atari_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--e_id", type=str, default="Breakout", help="Atari game id without -v5 suffix")
    parser.add_argument("--e_n_test_envs", type=int, default=8, help="# greedy test envs packed into same envpool instance")