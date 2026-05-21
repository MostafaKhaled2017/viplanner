from typing import Optional

import torch
import torch.nn as nn

from .learning_cfg import TrainCfg
from .planner_net import PlannerNet
from .rgb_encoder import PRE_TRAIN_POSSIBLE, RGBEncoder


class AutoEncoder(nn.Module):
    def __init__(self, encoder_channel=64, k=5):
        super().__init__()
        self.encoder = PlannerNet(layers=[2, 2, 2, 2])
        self.decoder = Decoder(512, encoder_channel, k)

    def forward(self, x: torch.Tensor, goal: torch.Tensor):
        x = x.expand(-1, 3, -1, -1)
        x = self.encoder(x)
        return self.decoder(x, goal)


class DualAutoEncoder(nn.Module):
    def __init__(self, train_cfg: TrainCfg, m2f_cfg=None, weight_path: Optional[str] = None):
        super().__init__()
        self.encoder_depth = PlannerNet(layers=[2, 2, 2, 2])
        if train_cfg.rgb and train_cfg.pre_train_sem:
            if not PRE_TRAIN_POSSIBLE:
                raise ImportError("The RGB model was trained with a pre-trained semantic backbone, but detectron2/mask2former are unavailable.")
            self.encoder_sem = RGBEncoder(m2f_cfg, weight_path, freeze=train_cfg.pre_train_freeze)
        else:
            self.encoder_sem = PlannerNet(layers=[2, 2, 2, 2])

        if train_cfg.decoder_small:
            self.decoder = DecoderS(1024, train_cfg.in_channel, train_cfg.knodes)
        else:
            self.decoder = Decoder(1024, train_cfg.in_channel, train_cfg.knodes)

    def forward(self, x_depth: torch.Tensor, x_sem: torch.Tensor, goal: torch.Tensor):
        x_depth = self.encoder_depth(x_depth.expand(-1, 3, -1, -1))
        x_sem = self.encoder_sem(x_sem)
        return self.decoder(torch.cat((x_depth, x_sem), dim=1), goal)


class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu = nn.ReLU(inplace=True)
        self.fg = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels + goal_channels, 512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(256 * 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, k * 3)
        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)

    def forward(self, x, goal):
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat((x, goal), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        features = self.relu(self.fc1(x))
        points = self.fc3(self.relu(self.fc2(features))).reshape(-1, self.k, 3)
        fear = self.sigmoid(self.frc2(self.relu(self.frc1(features))))
        return points, fear


class DecoderS(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu = nn.ReLU(inplace=True)
        self.fg = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels + goal_channels, 512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(64 * 48, 256)
        self.fc2 = nn.Linear(256, k * 3)
        self.frc1 = nn.Linear(256, 1)

    def forward(self, x, goal):
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat((x, goal), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        features = self.relu(self.fc1(torch.flatten(x, 1)))
        points = self.fc2(features).reshape(-1, self.k, 3)
        fear = self.sigmoid(self.frc1(features))
        return points, fear
