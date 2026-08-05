import argparse

from src.utils.utils import str2bool


def add_procgen_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--e_id", type=str, default="Coinrun", help="Procgen game id")
    parser.add_argument("--e_n_test_envs", type=int, default=8,
                        help="# greedy test envs packed into same envpool instance")
    parser.add_argument("--e_start_level", type=int, default=0, help="Environment Seed")
    parser.add_argument("--e_distribution_mode", type=str, default="hard")
    parser.add_argument("--e_center_agent", type=str2bool, default=True)
    parser.add_argument("--e_use_backgrounds", type=str2bool, default=True)
    parser.add_argument("--e_use_monochrome_assets", type=str2bool, default=False)
    parser.add_argument("--e_restrict_themes", type=str2bool, default=False)
    parser.add_argument("--e_use_generated_assets", type=str2bool, default=False)
    parser.add_argument("--e_paint_vel_info", type=str2bool, default=False)
    parser.add_argument("--e_use_easy_jump", type=str2bool, default=False)