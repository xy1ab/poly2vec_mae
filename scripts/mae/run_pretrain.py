import os
import sys
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mae_core.trainer import run_cli
from utils.config.loader import load_yaml_config

def _build_cli_args_from_config(config_dict):
    cli_args = []
    for key, value in config_dict.items():
        arg_name = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli_args.append(arg_name)
        else:
            cli_args.extend([arg_name, str(value)])
    return cli_args

if __name__ == '__main__':
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "configs", "mae", "pretrain.yaml"),
        type=str
    )
    pre_args, remaining = pre_parser.parse_known_args()

    config = load_yaml_config(pre_args.config)

    config_cli_args = _build_cli_args_from_config(config)
    run_cli(config_cli_args + remaining)
