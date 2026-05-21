import argparse
import pickle
from typing import Optional

import torch
import torch.nn as nn

try:
    from detectron2.config import get_cfg
    from detectron2.modeling.backbone import build_resnet_backbone
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config

    PRE_TRAIN_POSSIBLE = True
except ImportError as exc:
    get_cfg = None
    build_resnet_backbone = None
    add_deeplab_config = None
    add_maskformer2_config = None
    PRE_TRAIN_POSSIBLE = False
    PRE_TRAIN_IMPORT_ERROR = exc
else:
    PRE_TRAIN_IMPORT_ERROR = None


def get_m2f_cfg(cfg_path: str):
    if not PRE_TRAIN_POSSIBLE:
        raise ImportError(
            "RGB pre-trained backbone support requires detectron2 and an installed mask2former package. "
            f"Original import error: {PRE_TRAIN_IMPORT_ERROR}"
        )
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    return cfg


class RGBEncoder(nn.Module):
    def __init__(self, cfg, weight_path: Optional[str] = None, freeze: bool = True) -> None:
        super().__init__()
        if not PRE_TRAIN_POSSIBLE:
            raise ImportError(
                "RGBEncoder requires detectron2 and an installed mask2former package. "
                f"Original import error: {PRE_TRAIN_IMPORT_ERROR}"
            )

        input_shape = argparse.Namespace(channels=3)
        self.backbone = build_resnet_backbone(cfg, input_shape)

        if weight_path is not None:
            with open(weight_path, "rb") as file:
                model_file = pickle.load(file, encoding="latin1")
            model_file["model"] = {key.replace("backbone.", ""): torch.tensor(value) for key, value in model_file["model"].items()}
            self.backbone.load_state_dict(model_file["model"], strict=False)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(self.backbone(x)["res5"])
