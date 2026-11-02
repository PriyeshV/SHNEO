import argparse

from src.config.common.arg_helpers import (
    add_common_base_args,
    add_common_env_runtime_args,
    add_common_net_args,
    add_common_algo_args,
    add_common_laplacian_args,
    add_common_rnd_args,
    add_common_option_args,
    add_common_neo_args,
)
from src.config.atari.env_args import add_atari_env_args, set_atari_defaults
from src.config.common.build_config import convert_args_to_config, create_wb_logger, close_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Atari RL config")

    add_common_base_args(parser)
    add_common_env_runtime_args(parser)
    add_atari_env_args(parser)

    add_common_net_args(parser)
    add_common_algo_args(parser)
    add_common_laplacian_args(parser)
    add_common_rnd_args(parser)
    add_common_option_args(parser)
    add_common_neo_args(parser)

    set_atari_defaults(parser)

    return parser.parse_args()


def convert_args_to_dict(args):
    return convert_args_to_config(args, env_family="atari")