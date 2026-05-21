from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


Point3 = Tuple[float, float, float]


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


def scenario_dicts(scenarios: Iterable[Scenario]):
    return [{"pair_id": item.pair_id, "start": item.start, "goal": item.goal} for item in scenarios]
