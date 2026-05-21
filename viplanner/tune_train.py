# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import copy
import csv
import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import yaml


@dataclass
class SweepSpec:
    base_config: Path
    output_root: Path
    defaults: Dict[str, Any] = field(default_factory=dict)
    fixed: Dict[str, Any] = field(default_factory=dict)
    grid: Dict[str, List[Any]] = field(default_factory=dict)
    seed: int = 0
    max_trials: Optional[int] = None


@dataclass
class Trial:
    name: str
    params: Dict[str, Any]
    config_path: Path
    model_dir_name: str


@dataclass
class RunningTrial:
    trial: Trial
    gpu_id: int
    process: subprocess.Popen
    log_file: Any
    log_path: Path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _require_mapping(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{key}` must be a mapping")
    return value


def load_sweep_config(path: str) -> SweepSpec:
    sweep_path = Path(path)
    data = _load_yaml(sweep_path)

    if "base_config" not in data:
        raise ValueError("Sweep YAML must define `base_config`")
    if "output_root" not in data:
        raise ValueError("Sweep YAML must define `output_root`")

    grid = _require_mapping(data, "grid")
    for key, values in grid.items():
        if not isinstance(values, list):
            raise ValueError(f"Grid entry `{key}` must be a list")

    defaults = _require_mapping(data, "defaults")
    seed = int(data.get("seed", defaults.get("seed", 0)))
    max_trials = data.get("max_trials")
    if max_trials is not None:
        max_trials = int(max_trials)

    return SweepSpec(
        base_config=Path(data["base_config"]),
        output_root=Path(data["output_root"]),
        defaults=defaults,
        fixed=_require_mapping(data, "fixed"),
        grid=grid,
        seed=seed,
        max_trials=max_trials,
    )


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]

    keys = list(grid.keys())
    combinations = []
    for values in itertools.product(*(grid[key] for key in keys)):
        combinations.append(dict(zip(keys, values)))
    return combinations


def limit_trials(trials: List[Dict[str, Any]], max_trials: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if max_trials is None or max_trials >= len(trials):
        return list(trials)
    if max_trials < 1:
        raise ValueError("`max_trials` must be greater than zero")

    selected_indices = sorted(random.Random(seed).sample(range(len(trials)), max_trials))
    return [trials[idx] for idx in selected_indices]


def _set_nested_value(config: Dict[str, Any], key: str, value: Any) -> None:
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if next_value is None:
            next_value = {}
            cursor[part] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set `{key}` because `{part}` is not a mapping")
        cursor = next_value
    cursor[parts[-1]] = value


def _apply_params(config: Dict[str, Any], params: Dict[str, Any]) -> None:
    for key, value in params.items():
        _set_nested_value(config, key, value)


def _write_trial_config(base_config: Dict[str, Any], trial: Trial, params: Dict[str, Any], gpu_id: int) -> None:
    config = copy.deepcopy(base_config)
    if "config" not in config or not isinstance(config["config"], dict):
        raise ValueError("Base training YAML must contain a top-level `config` mapping")

    _apply_params(config["config"], params)
    config["config"]["gpu_id"] = gpu_id
    config["config"]["model_dir_name"] = trial.model_dir_name

    trial.config_path.parent.mkdir(parents=True, exist_ok=True)
    with trial.config_path.open("w") as file:
        yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)


def create_trials(spec: SweepSpec, max_trials: Optional[int] = None) -> List[Trial]:
    requested_max_trials = spec.max_trials if max_trials is None else max_trials
    grid_trials = limit_trials(expand_grid(spec.grid), requested_max_trials, spec.seed)

    trials = []
    for idx, grid_params in enumerate(grid_trials):
        params = {}
        params.update(spec.defaults)
        params.update(spec.fixed)
        params.update(grid_params)

        name = f"trial_{idx:04d}"
        trials.append(
            Trial(
                name=name,
                params=params,
                config_path=spec.output_root / "configs" / f"{name}.yaml",
                model_dir_name=name,
            )
        )
    return trials


class GpuScheduler:
    def __init__(self, gpu_ids: Iterable[int], max_parallel: int) -> None:
        self._gpu_ids = list(gpu_ids)
        if not self._gpu_ids:
            raise ValueError("At least one GPU ID must be provided")
        if max_parallel < 1:
            raise ValueError("`max_parallel` must be greater than zero")

        self._max_parallel = max_parallel
        self._active: Dict[str, int] = {}
        self._next_gpu_index = 0

    @property
    def active_count(self) -> int:
        return len(self._active)

    def can_start(self) -> bool:
        return self.active_count < self._max_parallel

    def available_gpus(self) -> List[int]:
        return list(self._gpu_ids)

    def acquire(self, trial_name: str) -> int:
        if not self.can_start():
            raise RuntimeError("No GPU slot is available")
        if trial_name in self._active:
            raise KeyError(f"Trial {trial_name} is already active")
        gpu_id = self._gpu_ids[self._next_gpu_index]
        self._next_gpu_index = (self._next_gpu_index + 1) % len(self._gpu_ids)
        self._active[trial_name] = gpu_id
        return gpu_id

    def release(self, trial_name: str) -> None:
        if trial_name not in self._active:
            raise KeyError(f"Trial {trial_name} is not active")
        del self._active[trial_name]


def parse_gpu_ids(value: str) -> List[int]:
    gpu_ids = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        gpu_ids.append(int(entry))
    if not gpu_ids:
        raise ValueError("At least one GPU ID must be provided")
    return gpu_ids


def _model_root(config: Dict[str, Any]) -> Path:
    file_path = config.get("file_path")
    if file_path is None:
        raise ValueError("Training config must define `file_path` to locate model outputs")
    return Path(os.getenv("EXPERIMENT_DIRECTORY", file_path)) / "models"


def _find_model_yaml(model_root: Path, model_dir_name: str) -> Optional[Path]:
    expected = model_root / model_dir_name / "model.yaml"
    if expected.is_file():
        return expected

    candidates = list(model_root.glob(f"{model_dir_name}_*/model.yaml"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_trial_result(trial: Trial, return_code: int, log_path: Path) -> Dict[str, Any]:
    config_data = _load_yaml(trial.config_path)
    train_config = config_data.get("config", {})
    model_root = _model_root(train_config)
    model_yaml = _find_model_yaml(model_root, trial.model_dir_name)

    val_loss = None
    test_loss = None
    model_dir = None
    model_path = None
    if model_yaml is not None:
        model_data = _load_yaml(model_yaml)
        loss = model_data.get("loss", {}) or {}
        val_loss = loss.get("val_loss")
        test_loss = loss.get("test_loss")
        model_dir = str(model_yaml.parent)
        model_path = str(model_yaml.parent / "model.pt")

    return {
        "trial_name": trial.name,
        "params": trial.params,
        "config_path": str(trial.config_path),
        "model_dir": model_dir,
        "model_path": model_path,
        "log_path": str(log_path),
        "return_code": return_code,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "status": "success" if return_code == 0 and val_loss is not None else "failed",
    }


def _result_sort_key(result: Dict[str, Any]) -> Any:
    val_loss = result.get("val_loss")
    test_loss = result.get("test_loss")
    has_loss = result.get("status") == "success" and val_loss is not None
    return (
        0 if has_loss else 1,
        float(val_loss) if val_loss is not None else float("inf"),
        float(test_loss) if test_loss is not None else float("inf"),
        result["trial_name"],
    )


def write_summary(results: List[Dict[str, Any]], output_root: Path) -> List[Dict[str, Any]]:
    ranked = sorted(results, key=_result_sort_key)
    rank = 1
    for result in ranked:
        if result["status"] == "success":
            result["rank"] = rank
            rank += 1
        else:
            result["rank"] = None

    output_root.mkdir(parents=True, exist_ok=True)
    summary_yaml = output_root / "summary.yaml"
    summary_csv = output_root / "summary.csv"

    with summary_yaml.open("w") as file:
        yaml.safe_dump({"trials": ranked}, file, default_flow_style=False, sort_keys=False)

    fieldnames = [
        "rank",
        "status",
        "trial_name",
        "return_code",
        "val_loss",
        "test_loss",
        "config_path",
        "model_dir",
        "model_path",
        "log_path",
        "params",
    ]
    with summary_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in ranked:
            row = dict(result)
            row["params"] = yaml.safe_dump(result["params"], default_flow_style=True).strip()
            writer.writerow({key: row.get(key) for key in fieldnames})

    return ranked


def _start_trial(
    trial: Trial,
    gpu_id: int,
    train_script: Path,
    show_test_visualizations: bool,
    log_dir: Path,
) -> RunningTrial:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{trial.name}.log"
    log_file = log_path.open("w")
    command = [
        sys.executable,
        str(train_script),
        "--config",
        str(trial.config_path),
        "--show-test-visualizations" if show_test_visualizations else "--no-test-visualizations",
    ]
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    return RunningTrial(trial=trial, gpu_id=gpu_id, process=process, log_file=log_file, log_path=log_path)


def run_trials(
    trials: List[Trial],
    base_config: Dict[str, Any],
    gpu_ids: List[int],
    max_parallel: int,
    output_root: Path,
    train_script: Path,
    show_test_visualizations: bool,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    scheduler = GpuScheduler(gpu_ids, max_parallel)
    pending = list(trials)
    active: List[RunningTrial] = []
    results: List[Dict[str, Any]] = []
    log_dir = output_root / "logs"

    if dry_run:
        for idx, trial in enumerate(pending):
            gpu_id = gpu_ids[idx % len(gpu_ids)]
            _write_trial_config(base_config, trial, trial.params, gpu_id)
            results.append(
                {
                    "trial_name": trial.name,
                    "params": trial.params,
                    "config_path": str(trial.config_path),
                    "model_dir": None,
                    "model_path": None,
                    "log_path": None,
                    "return_code": None,
                    "val_loss": None,
                    "test_loss": None,
                    "status": "dry_run",
                }
            )
        return write_summary(results, output_root)

    while pending or active:
        while pending and scheduler.can_start():
            trial = pending.pop(0)
            gpu_id = scheduler.acquire(trial.name)
            _write_trial_config(base_config, trial, trial.params, gpu_id)

            running = _start_trial(trial, gpu_id, train_script, show_test_visualizations, log_dir)
            active.append(running)
            print(f"[INFO] Started {trial.name} on GPU {gpu_id}")

        time.sleep(5)
        still_active = []
        for running in active:
            return_code = running.process.poll()
            if return_code is None:
                still_active.append(running)
                continue

            running.log_file.close()
            scheduler.release(running.trial.name)
            result = parse_trial_result(running.trial, return_code, running.log_path)
            results.append(result)
            write_summary(results, output_root)
            print(
                "[INFO] Finished "
                f"{running.trial.name} on GPU {running.gpu_id} "
                f"with return code {return_code}"
            )

        active = still_active

    return write_summary(results, output_root)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a YAML-defined VIPlanner training parameter sweep.")
    parser.add_argument("--sweep-config", required=True, help="Path to the sweep YAML file.")
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU IDs available for tuning runs.")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum total number of concurrent training runs; values above the GPU count oversubscribe GPUs.",
    )
    parser.add_argument("--max-trials", type=int, default=None, help="Limit the number of expanded grid trials.")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs and summaries without training.")
    parser.add_argument(
        "--train-script",
        default="viplanner/train.py",
        help="Training script to execute for each trial.",
    )
    parser.add_argument(
        "--show-test-visualizations",
        action="store_true",
        help="Pass --show-test-visualizations to training runs.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    spec = load_sweep_config(args.sweep_config)
    base_config = _load_yaml(spec.base_config)
    gpu_ids = parse_gpu_ids(args.gpus)
    trials = create_trials(spec, max_trials=args.max_trials)

    print(f"[INFO] Prepared {len(trials)} tuning trials")
    ranked = run_trials(
        trials=trials,
        base_config=base_config,
        gpu_ids=gpu_ids,
        max_parallel=args.max_parallel,
        output_root=spec.output_root,
        train_script=Path(args.train_script),
        show_test_visualizations=args.show_test_visualizations,
        dry_run=args.dry_run,
    )

    best = next((result for result in ranked if result.get("status") == "success"), None)
    if best is not None:
        print(
            "[INFO] Best trial: "
            f"{best['trial_name']} val_loss={best['val_loss']} test_loss={best['test_loss']}"
        )
    print(f"[INFO] Wrote tuning summary to {spec.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
