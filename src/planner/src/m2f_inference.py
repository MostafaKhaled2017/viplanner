# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import rospy
from viplanner.utils import Mask2FormerPredictor


class Mask2FormerInference(Mask2FormerPredictor):
    """Run Inference on Mask2Former model to estimate semantic segmentation"""

    debug: bool = False

    def __init__(
        self,
        config_file="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        checkpoint_file="model_final.pth",
        device="cuda:0",
    ) -> None:
        super().__init__(config_file=config_file, checkpoint_file=checkpoint_file, device=device, warn_fn=rospy.logwarn)


# EoF
