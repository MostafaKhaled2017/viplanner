from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def array_stats(array: Any) -> Dict[str, Any]:
    values = np.asarray(array)
    finite = values[np.isfinite(values)] if values.size > 0 and np.issubdtype(values.dtype, np.number) else values.reshape(-1)
    stats: Dict[str, Any] = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.number) else 0,
        "inf_count": int(np.isinf(values).sum()) if np.issubdtype(values.dtype, np.number) else 0,
    }
    if finite.size > 0 and np.issubdtype(values.dtype, np.number):
        stats.update(
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
            }
        )
    return stats


def tensor_to_list(tensor: torch.Tensor, decimals: Optional[int] = 6):
    values = tensor.detach().cpu().numpy()
    if decimals is not None:
        values = np.round(values.astype(np.float64), decimals=decimals)
    return values.tolist()


def xyz_summary(points: Any) -> Dict[str, Any]:
    values = np.asarray(points.detach().cpu().numpy() if isinstance(points, torch.Tensor) else points, dtype=np.float32)
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


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_to_list(value)
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_for_json(payload), indent=2, sort_keys=True), encoding="utf-8")
