# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

from viplanner.utils import Mask2FormerPredictor

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class EnvironmentProcessingStats:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    non_image_files: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VIPlanner semantic images from recorded RGB images.")
    parser.add_argument("--input", required=True, help="Environment directory or parent dataset directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively process all environment directories.")
    parser.add_argument("--config", required=True, help="Path to the Mask2Former config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the Mask2Former checkpoint file.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing semantic images.")
    parser.add_argument("--device", default="cuda:0", help="Inference device, for example 'cuda:0' or 'cpu'.")
    return parser.parse_args()


def discover_environment_dirs(root: Path, recursive: bool) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    if (root / "rgb").is_dir():
        return [root]

    if not recursive:
        raise FileNotFoundError(f"Input path '{root}' does not contain an 'rgb' directory.")

    environments = sorted({rgb_dir.parent for rgb_dir in root.rglob("rgb") if rgb_dir.is_dir()})
    if not environments:
        raise FileNotFoundError(f"No environment directories containing an 'rgb' folder were found under '{root}'.")
    return environments


def iter_rgb_images(rgb_dir: Path) -> Iterable[Path]:
    for path in sorted(rgb_dir.iterdir()):
        if path.is_file():
            yield path


def process_environment(
    env_dir: Path,
    predictor: Mask2FormerPredictor,
    force: bool = False,
) -> EnvironmentProcessingStats:
    rgb_dir = env_dir / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Environment directory '{env_dir}' does not contain an 'rgb' folder.")

    semantics_dir = env_dir / "semantics"
    semantics_dir.mkdir(exist_ok=True)

    stats = EnvironmentProcessingStats()

    for image_path in iter_rgb_images(rgb_dir):
        if image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            stats.non_image_files += 1
            print(f"[WARN] Skipping non-image file: {image_path}")
            continue

        output_path = semantics_dir / f"{image_path.stem}.png"
        if output_path.exists() and not force:
            stats.skipped += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            stats.failed += 1
            print(f"[WARN] Failed to read image: {image_path}")
            continue

        try:
            semantic_image = predictor.predict(image)
        except Exception as exc:  # noqa: BLE001
            stats.failed += 1
            print(f"[WARN] Failed to generate semantics for '{image_path}': {exc}")
            continue

        if not cv2.imwrite(str(output_path), cv2.cvtColor(semantic_image, cv2.COLOR_RGB2BGR)):
            stats.failed += 1
            print(f"[WARN] Failed to write semantic image: {output_path}")
            continue

        stats.processed += 1

    print(
        f"[INFO] {env_dir}: processed={stats.processed}, skipped={stats.skipped},"
        f" failed={stats.failed}, non_image_files={stats.non_image_files}"
    )
    return stats


def main() -> int:
    args = parse_args()

    predictor = Mask2FormerPredictor(
        config_file=args.config,
        checkpoint_file=args.checkpoint,
        device=args.device,
        warn_fn=lambda msg: print(f"[WARN] {msg}"),
    )

    environments = discover_environment_dirs(Path(args.input).expanduser().resolve(), recursive=args.recursive)

    totals = EnvironmentProcessingStats()
    for env_dir in environments:
        env_stats = process_environment(env_dir=env_dir, predictor=predictor, force=args.force)
        totals.processed += env_stats.processed
        totals.skipped += env_stats.skipped
        totals.failed += env_stats.failed
        totals.non_image_files += env_stats.non_image_files

    print(
        "[INFO] Finished semantic generation:"
        f" processed={totals.processed}, skipped={totals.skipped},"
        f" failed={totals.failed}, non_image_files={totals.non_image_files},"
        f" environments={len(environments)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
