# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import argparse
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


def main():c
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        # default="default_thermal",
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()


