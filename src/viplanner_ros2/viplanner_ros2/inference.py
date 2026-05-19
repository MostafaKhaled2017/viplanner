from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms

from .autoencoder import AutoEncoder, DualAutoEncoder
from .learning_cfg import TrainCfg
from .rgb_encoder import get_m2f_cfg
from .traj_opt import TrajOpt

torch.set_default_dtype(torch.float32)


def resolve_model_files(model_save: str) -> Tuple[Path, Path]:
    model_dir = Path(model_save).expanduser()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model_save must be a trained model directory: {model_dir}")
    model_path = model_dir / "model.pt"
    config_path = model_dir / "model.yaml"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing VIPlanner checkpoint: {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing VIPlanner training config: {config_path}")
    return model_path, config_path


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, tuple) and checkpoint:
        first = checkpoint[0]
        if isinstance(first, dict):
            return first
        if hasattr(first, "state_dict"):
            return first.state_dict()
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    raise RuntimeError(f"Unsupported checkpoint format: {type(checkpoint)}")


class VIPlannerInference:
    def __init__(
        self,
        model_save: str,
        m2f_config_path: str = "",
        m2f_model_path: str = "",
        device: str | None = None,
    ) -> None:
        model_path, config_path = resolve_model_files(model_save)
        self.train_cfg = TrainCfg.from_yaml(str(config_path))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if self.train_cfg.rgb:
            cfg_path = m2f_config_path or self._resolve_model_relative_path(self.train_cfg.pre_train_cfg, model_path.parent)
            weight_path = m2f_model_path or self._resolve_model_relative_path(self.train_cfg.pre_train_weights, model_path.parent)
            m2f_cfg = get_m2f_cfg(cfg_path)
            self.pixel_mean = np.asarray(m2f_cfg.MODEL.PIXEL_MEAN, dtype=np.float32)
            self.pixel_std = np.asarray(m2f_cfg.MODEL.PIXEL_STD, dtype=np.float32)
        else:
            m2f_cfg = None
            weight_path = None
            self.pixel_mean = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            self.pixel_std = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

        if self.train_cfg.rgb or self.train_cfg.sem:
            self.net = DualAutoEncoder(self.train_cfg, m2f_cfg=m2f_cfg, weight_path=weight_path)
        else:
            self.net = AutoEncoder(encoder_channel=self.train_cfg.in_channel, k=self.train_cfg.knodes)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(extract_state_dict(checkpoint))
        self.net.to(self.device)
        self.net.eval()

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(tuple(self.train_cfg.img_input_size), antialias=True),
            ]
        )
        self.traj_generate = TrajOpt()

    @staticmethod
    def _resolve_model_relative_path(path_value: str | None, model_dir: Path) -> str | None:
        if not path_value:
            return None
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return str(path)
        candidate = model_dir / path
        if candidate.exists():
            return str(candidate)
        models_candidate = model_dir.parent / path
        return str(models_candidate if models_candidate.exists() else path)

    @property
    def requires_rgb(self) -> bool:
        return bool(self.train_cfg.rgb or self.train_cfg.sem)

    @property
    def requires_semantic_prediction(self) -> bool:
        return bool(self.train_cfg.sem)

    def img_converter(self, image: np.ndarray) -> torch.Tensor:
        image_t = self.transforms(image)
        return image_t.unsqueeze(0).to(self.device)

    def plan_depth(self, depth_image: np.ndarray, goal_robot_frame: torch.Tensor):
        with torch.no_grad():
            depth = self.img_converter(depth_image).float()
            keypoints, fear = self.net(depth, goal_robot_frame.to(self.device))
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        return keypoints, traj, fear

    def plan(self, depth_image: np.ndarray, sem_rgb_image: np.ndarray, goal_robot_frame: torch.Tensor):
        with torch.no_grad():
            depth = self.img_converter(depth_image).float()
            if self.train_cfg.rgb:
                sem_rgb_image = (sem_rgb_image.astype(np.float32) - self.pixel_mean) / self.pixel_std
            sem_rgb = self.img_converter(sem_rgb_image.astype(np.uint8)).float()
            keypoints, fear = self.net(depth, sem_rgb, goal_robot_frame.to(self.device))
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        return keypoints, traj, fear
