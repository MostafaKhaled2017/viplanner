# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import argparse

import torch

torch.set_default_dtype(torch.float32)

# imperative-planning-learning
from viplanner.config import TrainCfg
from viplanner.utils.trainer import Trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VIPlanner from a YAML config file.")
    parser.add_argument(
        "--config",
        default="viplanner/config/train.yaml",
        help="Path to a training YAML config file.",
    )
    visual_group = parser.add_mutually_exclusive_group()
    visual_group.add_argument(
        "--show-test-visualizations",
        dest="show_test_visualizations",
        action="store_true",
        default=None,
        help="Show visualizations during the post-training test run.",
    )
    visual_group.add_argument(
        "--no-test-visualizations",
        dest="show_test_visualizations",
        action="store_false",
        help="Disable visualizations during the post-training test run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg: TrainCfg = TrainCfg.from_yaml(args.config)
    trainer = Trainer(cfg)
    trainer.train()
    trainer.test(show_visualizations=args.show_test_visualizations)
    trainer.save_config()
    torch.cuda.empty_cache()
