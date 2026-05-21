from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


def _finite_values(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.size == 0:
        return values.reshape(-1)
    if not np.issubdtype(values.dtype, np.number):
        return values.reshape(-1)
    return values[np.isfinite(values)]


def array_stats(array: Any) -> Dict[str, Any]:
    values = np.asarray(array)
    finite = _finite_values(values)
    stats: Dict[str, Any] = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.number) else 0,
        "inf_count": int(np.isinf(values).sum()) if np.issubdtype(values.dtype, np.number) else 0,
    }
    if finite.size > 0:
        stats.update(
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
            }
        )
    return stats


def tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    return array_stats(tensor.detach().cpu().numpy())


def xyz_summary(points: Any) -> Dict[str, Any]:
    if isinstance(points, torch.Tensor):
        values = points.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        values = np.asarray(points, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.size == 0 or values.shape[-1] < 3:
        return {"shape": list(values.shape)}

    xyz = values.reshape(-1, values.shape[-1])[:, :3]
    return {
        "shape": list(values.shape),
        "x": array_stats(xyz[:, 0]),
        "y": array_stats(xyz[:, 1]),
        "z": array_stats(xyz[:, 2]),
    }


def tensor_to_list(tensor: torch.Tensor, decimals: Optional[int] = 6):
    values = tensor.detach().cpu().numpy()
    if decimals is not None:
        values = np.round(values.astype(np.float64), decimals=decimals)
    return values.tolist()


def loss_frame_tensors(preds: torch.Tensor, goal: torch.Tensor, augment: bool):
    preds_loss = preds.clone()
    goal_loss = goal.clone()
    if augment:
        preds_loss[:, :, 1] *= -1.0
        goal_loss[:, 1] *= -1.0
    return preds_loss, goal_loss


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_to_list(value)
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value
