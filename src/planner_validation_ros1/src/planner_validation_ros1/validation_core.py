from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


Point3 = Tuple[float, float, float]
REQUIRED_MODEL_FILES = ("model.pt", "model.yaml")
SWEEP_MODEL_COLUMNS = ["model_name", "model_path"]
SWEEP_SUMMARY_COLUMNS = [
    "model_name",
    "model_path",
    "num_trials",
    "reached_count",
    "collision_count",
    "success_rate",
    "mean_elapsed_sec",
    "mean_distance_to_goal",
    "status",
    "error",
]


@dataclass(frozen=True)
class Scenario:
    pair_id: int
    start: Point3
    goal: Point3


def resolve_package_path(path: str, package_root: Path) -> str:
    if not path:
        return str(package_root)
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return str(package_root / expanded)


def load_scenarios_txt(path: str, warn_fn: Optional[Callable[[str], None]] = None) -> List[Scenario]:
    scenarios = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 7:
                if warn_fn is not None:
                    warn_fn(f"[validation] Bad line (expected 7 columns): {line}")
                continue
            pair_id = int(parts[0])
            sx, sy, sz = map(float, parts[1:4])
            gx, gy, gz = map(float, parts[4:7])
            scenarios.append(Scenario(pair_id=pair_id, start=(sx, sy, sz), goal=(gx, gy, gz)))
    return scenarios


def discover_model_directories(parent_dir: str, warn_fn: Optional[Callable[[str], None]] = None) -> List[Path]:
    parent = Path(parent_dir).expanduser()
    if not parent.is_dir():
        raise FileNotFoundError(f"Models parent directory does not exist: {parent}")

    models = []
    for child in sorted(parent.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        missing = [name for name in REQUIRED_MODEL_FILES if not (child / name).is_file()]
        if missing:
            if warn_fn is not None:
                warn_fn(f"[model-sweep] Skipping {child}: missing {', '.join(missing)}")
            continue
        models.append(child)
    return models


def distance3(a_xyz: Sequence[float], b_xyz: Sequence[float]) -> float:
    ax, ay, az = a_xyz
    bx, by, bz = b_xyz
    dx, dy, dz = ax - bx, ay - by, az - bz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def has_reached(current_xyz: Sequence[float], target_xyz: Sequence[float], threshold: float) -> bool:
    return distance3(current_xyz, target_xyz) <= threshold


def yaw_from_start_to_goal(start_xyz: Sequence[float], goal_xyz: Sequence[float], offset_rad: float = 0.0) -> float:
    sx, sy, _ = start_xyz
    gx, gy, _ = goal_xyz
    return math.atan2(gy - sy, gx - sx) + offset_rad


def yaw_to_quaternion_zw(yaw_rad: float) -> Tuple[float, float]:
    return math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)


DEBUG_RESULT_COLUMNS = [
    "start_z",
    "goal_z",
    "robot_start_z",
    "robot_end_z",
    "final_distance",
    "planner_status",
    "outcome",
]


def init_results_csv(log_dir: str, timestamp: Optional[datetime] = None, include_debug_columns: bool = False) -> str:
    ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"validation_results_{ts}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = [
            "pair_id",
            "planner_id",
            "collided",
            "reached_goal",
            "distance_to_goal",
            "elapsed_sec",
        ]
        if include_debug_columns:
            columns.extend(DEBUG_RESULT_COLUMNS)
        writer.writerow(columns)
    return csv_path


def append_result_row(
    csv_path: str,
    pair_id: int,
    planner_id: str,
    collided: bool,
    reached_goal: bool,
    distance_to_goal: float,
    elapsed_sec: float,
    debug_values: Optional[dict] = None,
) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        row = [
            int(pair_id),
            planner_id,
            bool(collided),
            bool(reached_goal),
            float(distance_to_goal),
            float(elapsed_sec),
        ]
        if debug_values is not None:
            row.extend(debug_values.get(name, "") for name in DEBUG_RESULT_COLUMNS)
        writer.writerow(row)


def read_csv_dicts(csv_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def append_sweep_trial_rows(sweep_csv_path: str, validation_csv_path: str, model_name: str, model_path: str) -> int:
    validation_columns, validation_rows = read_csv_dicts(validation_csv_path)
    output_path = Path(sweep_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = SWEEP_MODEL_COLUMNS + validation_columns
    needs_header = not output_path.exists() or output_path.stat().st_size == 0

    with open(output_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if needs_header:
            writer.writeheader()
        for row in validation_rows:
            writer.writerow({"model_name": model_name, "model_path": model_path, **row})
    return len(validation_rows)


def _bool_from_csv(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_sweep_summary(
    validation_rows: Iterable[Dict[str, str]],
    model_name: str,
    model_path: str,
    status: str = "ok",
    error: str = "",
) -> Dict[str, object]:
    rows = list(validation_rows)
    num_trials = len(rows)
    reached_count = sum(1 for row in rows if _bool_from_csv(row.get("reached_goal", False)))
    collision_count = sum(1 for row in rows if _bool_from_csv(row.get("collided", False)))
    elapsed_values = [_float_or_none(row.get("elapsed_sec")) for row in rows]
    distance_values = [_float_or_none(row.get("distance_to_goal")) for row in rows]
    elapsed_values = [value for value in elapsed_values if value is not None]
    distance_values = [value for value in distance_values if value is not None]

    return {
        "model_name": model_name,
        "model_path": model_path,
        "num_trials": num_trials,
        "reached_count": reached_count,
        "collision_count": collision_count,
        "success_rate": (reached_count / num_trials) if num_trials else 0.0,
        "mean_elapsed_sec": (sum(elapsed_values) / len(elapsed_values)) if elapsed_values else "",
        "mean_distance_to_goal": (sum(distance_values) / len(distance_values)) if distance_values else "",
        "status": status,
        "error": error,
    }


def append_sweep_summary_row(summary_csv_path: str, summary_row: Dict[str, object]) -> None:
    output_path = Path(summary_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_path.exists() or output_path.stat().st_size == 0
    with open(output_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_SUMMARY_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow({name: summary_row.get(name, "") for name in SWEEP_SUMMARY_COLUMNS})


def build_sweep_configs(
    viplanner_config: Dict[str, object],
    validation_config: Dict[str, object],
    model_path: str,
    model_name: str,
    model_log_dir: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    viplanner_run_config = dict(viplanner_config)
    validation_run_config = dict(validation_config)

    viplanner_run_config["model_save"] = model_path
    validation_run_config["log_dir"] = model_log_dir
    validation_run_config["planner_id"] = model_name
    validation_run_config["planner_status_topic"] = "/viplanner/status"
    validation_run_config["shutdown_on_complete"] = True
    return viplanner_run_config, validation_run_config


def scenario_dicts(scenarios: Iterable[Scenario]):
    return [{"pair_id": item.pair_id, "start": item.start, "goal": item.goal} for item in scenarios]
