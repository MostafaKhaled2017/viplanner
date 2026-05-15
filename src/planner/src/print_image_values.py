#!/usr/bin/env python3

# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write image pixel values to a text file while preserving the image row/column layout."
    )
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output text file path. Defaults to <image_stem>_values.txt next to the image.",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=" ",
        help="Delimiter used between scalar pixel values on the same row",
    )
    return parser.parse_args()


def default_output_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_values.txt")


def format_stats(image: np.ndarray) -> List[str]:
    flat = image.reshape(-1, image.shape[2]) if image.ndim == 3 else image.reshape(-1)
    nonzero_count = int(np.count_nonzero(image))
    total_count = int(image.size)

    lines = [
        f"# rows: {image.shape[0]}",
        f"# cols: {image.shape[1]}",
        f"# dtype: {image.dtype}",
    ]
    if image.ndim == 3:
        lines.append(f"# channels: {image.shape[2]}")
    lines.extend(
        [
            f"# min: {np.min(flat, axis=0).tolist() if image.ndim == 3 else int(np.min(flat))}",
            f"# max: {np.max(flat, axis=0).tolist() if image.ndim == 3 else int(np.max(flat))}",
            f"# mean: {np.mean(flat, axis=0).tolist() if image.ndim == 3 else float(np.mean(flat))}",
            f"# nonzero: {nonzero_count}",
            f"# total_values: {total_count}",
            f"# zero_values: {total_count - nonzero_count}",
            "# values:",
        ]
    )
    return lines


def format_scalar_image(image: np.ndarray, delimiter: str) -> str:
    lines = format_stats(image)
    for row in image:
        lines.append(delimiter.join(str(int(value)) for value in row))
    return "\n".join(lines) + "\n"


def format_multi_channel_image(image: np.ndarray) -> str:
    lines = format_stats(image)
    for row in image:
        formatted_row = ["(" + ",".join(str(int(value)) for value in pixel) + ")" for pixel in row]
        lines.append(" ".join(formatted_row))
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    output_path = args.output.expanduser() if args.output is not None else default_output_path(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim == 2:
        content = format_scalar_image(image, args.delimiter)
    elif image.ndim == 3:
        content = format_multi_channel_image(image)
    else:
        raise RuntimeError(f"Unsupported image shape: {image.shape}")

    output_path.write_text(content)
    print(f"Wrote pixel values for {image.shape} image to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
