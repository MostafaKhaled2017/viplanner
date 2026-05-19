from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FearState:
    buffer_size: int
    threshold: float
    buffer: int = 0
    active: bool = False

    def update(self, fear_value: float, is_forward: bool) -> bool:
        if fear_value > self.threshold and is_forward:
            if not self.active:
                self.buffer += 1
        elif self.active:
            self.buffer -= 1

        if self.buffer > self.buffer_size:
            self.active = True
        elif self.buffer <= 0:
            self.active = False
        return self.active

    def reset(self) -> None:
        self.buffer = 0
        self.active = False


def fear_scalar(fear) -> float:
    if isinstance(fear, torch.Tensor):
        return float(fear.detach().cpu().squeeze().item())
    try:
        return float(fear)
    except (TypeError, ValueError):
        return 0.0


def is_forward_tracking(waypoints, track_dist: float, angular_threshold: float) -> bool:
    xhead = np.array([1.0, 0.0])
    points = waypoints.detach().cpu().numpy() if isinstance(waypoints, torch.Tensor) else np.asarray(waypoints)
    points = np.squeeze(points)
    if points.ndim != 2:
        return True
    for point in points:
        if np.linalg.norm(point[0:2]) > track_dist:
            heading = point[0:2] / np.linalg.norm(point[0:2])
            return float(heading.dot(xhead)) > 1.0 - angular_threshold
    return True


def clip_goal_xy(goal_xyz, max_distance: float):
    gx, gy, gz = map(float, goal_xyz)
    dist_xy = math.hypot(gx, gy)
    if dist_xy > max_distance and dist_xy > 1e-6:
        scale = max_distance / dist_xy
        gx *= scale
        gy *= scale
    return gx, gy, gz
