#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from planner_validation_ros1.validation_core import (
    append_sweep_summary_row,
    append_sweep_trial_rows,
    build_sweep_configs,
    compute_sweep_summary,
    discover_model_directories,
    read_csv_dicts,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run planner validation across VIPlanner model directories.")
    parser.add_argument("--models-parent", required=True, help="Parent directory containing model subdirectories.")
    parser.add_argument("--output-dir", required=True, help="Directory where sweep reports and run configs are written.")
    parser.add_argument("--viplanner-config", required=True, help="Base viplanner_ros1 YAML config.")
    parser.add_argument("--validation-config", required=True, help="Base planner_validation_ros1 YAML config.")
    parser.add_argument("--planner-start-delay-sec", type=float, default=5.0, help="Delay after starting VIPlanner.")
    parser.add_argument("--run-timeout-sec", type=float, default=0.0, help="Optional timeout for each model run. Zero disables it.")
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a flat YAML mapping: {path}")
    return data


def write_yaml_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)


def start_roslaunch(package: str, launch_file: str, config_file: Path) -> subprocess.Popen:
    command = ["roslaunch", package, launch_file, f"config_file:={config_file}"]
    return subprocess.Popen(command, preexec_fn=os.setsid)


def stop_process(process: Optional[subprocess.Popen], timeout_sec: float = 10.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        process.wait(timeout=timeout_sec)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass

    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=timeout_sec)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()


def wait_for_validation(viplanner_proc: subprocess.Popen, validation_proc: subprocess.Popen, timeout_sec: float) -> int:
    start_time = time.time()
    while True:
        validation_code = validation_proc.poll()
        if validation_code is not None:
            return validation_code

        viplanner_code = viplanner_proc.poll()
        if viplanner_code is not None:
            raise RuntimeError(f"viplanner_ros1 exited before validation completed (exit code {viplanner_code})")

        if timeout_sec > 0.0 and (time.time() - start_time) >= timeout_sec:
            raise TimeoutError(f"validation run exceeded timeout of {timeout_sec:.1f} seconds")

        time.sleep(0.5)


def latest_validation_csv(model_log_dir: Path) -> Path:
    candidates = sorted(model_log_dir.glob("validation_results_*.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No validation_results_*.csv found in {model_log_dir}")
    return candidates[-1]


def run_one_model(
    model_dir: Path,
    sweep_dir: Path,
    base_viplanner_config: dict,
    base_validation_config: dict,
    sweep_trials_csv: Path,
    sweep_summary_csv: Path,
    planner_start_delay_sec: float,
    run_timeout_sec: float,
) -> bool:
    model_name = model_dir.name
    model_path = str(model_dir.resolve())
    model_run_dir = sweep_dir / model_name
    model_log_dir = model_run_dir / "validation_logs"
    model_run_dir.mkdir(parents=True, exist_ok=True)

    viplanner_run_config, validation_run_config = build_sweep_configs(
        base_viplanner_config,
        base_validation_config,
        model_path=model_path,
        model_name=model_name,
        model_log_dir=str(model_log_dir),
    )
    viplanner_config_path = model_run_dir / "viplanner.yaml"
    validation_config_path = model_run_dir / "planner_validation.yaml"
    write_yaml_config(viplanner_config_path, viplanner_run_config)
    write_yaml_config(validation_config_path, validation_run_config)

    viplanner_proc = None
    validation_proc = None
    try:
        print(f"[model-sweep] Starting model {model_name}: {model_path}", flush=True)
        viplanner_proc = start_roslaunch("viplanner_ros1", "viplanner.launch", viplanner_config_path)
        time.sleep(max(0.0, planner_start_delay_sec))
        if viplanner_proc.poll() is not None:
            raise RuntimeError(f"viplanner_ros1 exited during startup (exit code {viplanner_proc.returncode})")

        validation_proc = start_roslaunch("planner_validation_ros1", "planner_validation.launch", validation_config_path)
        validation_code = wait_for_validation(viplanner_proc, validation_proc, run_timeout_sec)
        if validation_code != 0:
            raise RuntimeError(f"planner_validation_ros1 exited with code {validation_code}")

        validation_csv = latest_validation_csv(model_log_dir)
        _, validation_rows = read_csv_dicts(str(validation_csv))
        append_sweep_trial_rows(str(sweep_trials_csv), str(validation_csv), model_name, model_path)
        append_sweep_summary_row(
            str(sweep_summary_csv),
            compute_sweep_summary(validation_rows, model_name, model_path, status="ok", error=""),
        )
        print(f"[model-sweep] Completed model {model_name}: {len(validation_rows)} trials", flush=True)
        return True
    except Exception as exc:
        print(f"[model-sweep] Model {model_name} failed: {exc}", file=sys.stderr, flush=True)
        append_sweep_summary_row(
            str(sweep_summary_csv),
            compute_sweep_summary([], model_name, model_path, status="failed", error=str(exc)),
        )
        return False
    finally:
        stop_process(validation_proc)
        stop_process(viplanner_proc)


def main() -> int:
    args = parse_args()
    base_viplanner_config = load_yaml_config(Path(args.viplanner_config).expanduser())
    base_validation_config = load_yaml_config(Path(args.validation_config).expanduser())
    models = discover_model_directories(args.models_parent, warn_fn=lambda message: print(message, file=sys.stderr))
    if not models:
        print(f"[model-sweep] No valid model directories found in {args.models_parent}", file=sys.stderr)
        return 2

    sweep_dir = Path(args.output_dir).expanduser() / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(parents=True, exist_ok=False)
    sweep_trials_csv = sweep_dir / "sweep_trials.csv"
    sweep_summary_csv = sweep_dir / "sweep_summary.csv"
    print(f"[model-sweep] Sweep output: {sweep_dir}", flush=True)
    print(f"[model-sweep] Found {len(models)} valid model directories", flush=True)

    all_ok = True
    for model_dir in models:
        ok = run_one_model(
            model_dir=model_dir,
            sweep_dir=sweep_dir,
            base_viplanner_config=base_viplanner_config,
            base_validation_config=base_validation_config,
            sweep_trials_csv=sweep_trials_csv,
            sweep_summary_csv=sweep_summary_csv,
            planner_start_delay_sec=args.planner_start_delay_sec,
            run_timeout_sec=args.run_timeout_sec,
        )
        all_ok = all_ok and ok

    print(f"[model-sweep] Trial report: {sweep_trials_csv}", flush=True)
    print(f"[model-sweep] Summary report: {sweep_summary_csv}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
