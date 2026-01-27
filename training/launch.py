# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
import os



def setup_torchrun_environment():
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29555"
        
        print("Setting torchrun environment variables for single-GPU mode:")
        print(f"  RANK={os.environ['RANK']}")
        print(f"  LOCAL_RANK={os.environ['LOCAL_RANK']}")
        print(f"  WORLD_SIZE={os.environ['WORLD_SIZE']}")
        print(f"  MASTER_ADDR={os.environ['MASTER_ADDR']}")
        print(f"  MASTER_PORT={os.environ['MASTER_PORT']}")

def main():
    setup_torchrun_environment()
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
#        default="thermal_df_cf",
        default="thermal_df",
        #default="thermal_df_only",
        #default="default_thermal",
        # default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()


