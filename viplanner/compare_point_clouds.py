# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare two PLY point clouds and report similarity metrics."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d


class PointCloudComparisonError(RuntimeError):
    """Raised when point-cloud comparison inputs are invalid."""


@dataclass
class CloudData:
    label: str
    path: Path
    pcd: o3d.geometry.PointCloud
    points: np.ndarray
    has_colors: bool
    has_normals: bool
    min_bound: np.ndarray
    max_bound: np.ndarray
    extent: np.ndarray
    bounding_box_diagonal: float
    centroid: np.ndarray


@dataclass
class ComparisonResult:
    ordered_metrics: Optional[Dict[str, float]]
    a_to_b: np.ndarray
    b_to_a: np.ndarray
    similarity_score: float
    similarity_label: str


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def resolve_cloud_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.cloud_a is not None or args.cloud_b is not None:
        if args.cloud_a is None or args.cloud_b is None:
            raise PointCloudComparisonError("--cloud-a and --cloud-b must be provided together")
        return Path(args.cloud_a), Path(args.cloud_b)

    if args.env_dir is None:
        raise PointCloudComparisonError("Use --env-dir or provide both --cloud-a and --cloud-b")

    env_dir = Path(args.env_dir)
    return env_dir / args.cloud_a_name, env_dir / args.cloud_b_name


def load_cloud(path: Path, label: str) -> CloudData:
    if not path.is_file():
        raise PointCloudComparisonError(f"{label} file does not exist: {path}")

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=float)
    if points.size == 0:
        raise PointCloudComparisonError(f"{label} point cloud is empty or unreadable: {path}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise PointCloudComparisonError(f"{label} point cloud does not contain XYZ points: {path}")

    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    extent = max_bound - min_bound
    return CloudData(
        label=label,
        path=path,
        pcd=pcd,
        points=points,
        has_colors=bool(pcd.has_colors()),
        has_normals=bool(pcd.has_normals()),
        min_bound=min_bound,
        max_bound=max_bound,
        extent=extent,
        bounding_box_diagonal=float(np.linalg.norm(extent)),
        centroid=np.mean(points, axis=0),
    )


def sample_cloud(cloud: CloudData, sample_points: Optional[int]) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    if sample_points is None or cloud.points.shape[0] <= sample_points:
        return cloud.pcd, cloud.points

    indices = np.linspace(0, cloud.points.shape[0] - 1, sample_points, dtype=np.int64)
    sampled = cloud.pcd.select_by_index(indices.tolist())
    return sampled, np.asarray(sampled.points, dtype=float)


def ordered_abs_error_metrics(points_a: np.ndarray, points_b: np.ndarray) -> Optional[Dict[str, float]]:
    if points_a.shape != points_b.shape:
        return None

    abs_error = np.abs(points_a - points_b)
    point_error = np.linalg.norm(points_a - points_b, axis=1)
    return {
        "ordered_abs_error_mean": float(np.mean(abs_error)),
        "ordered_abs_error_median": float(np.median(abs_error)),
        "ordered_abs_error_p95": float(np.percentile(abs_error, 95)),
        "ordered_abs_error_p99": float(np.percentile(abs_error, 99)),
        "ordered_abs_error_max": float(np.max(abs_error)),
        "ordered_point_distance_mean": float(np.mean(point_error)),
        "ordered_point_distance_p95": float(np.percentile(point_error, 95)),
        "ordered_point_distance_max": float(np.max(point_error)),
    }


def distance_metrics(distances: np.ndarray) -> Dict[str, float]:
    if distances.size == 0:
        raise PointCloudComparisonError("Cannot compute metrics for empty distance array")
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
        "max": float(np.max(distances)),
    }


def similarity_label(score: float) -> str:
    if score >= 99.9:
        return "nearly_identical"
    if score >= 95.0:
        return "very_similar"
    if score >= 80.0:
        return "similar"
    if score >= 50.0:
        return "partially_similar"
    return "different"


def compare_clouds(
    cloud_a: CloudData,
    cloud_b: CloudData,
    similarity_distance_scale: float,
    sample_points: Optional[int],
) -> ComparisonResult:
    sampled_a, points_a = sample_cloud(cloud_a, sample_points)
    sampled_b, points_b = sample_cloud(cloud_b, sample_points)

    ordered_metrics = ordered_abs_error_metrics(points_a, points_b)
    a_to_b = np.asarray(sampled_a.compute_point_cloud_distance(sampled_b), dtype=float)
    b_to_a = np.asarray(sampled_b.compute_point_cloud_distance(sampled_a), dtype=float)
    all_distances = np.concatenate([a_to_b, b_to_a])
    per_point_similarity = np.maximum(0.0, 1.0 - all_distances / similarity_distance_scale)
    score = float(100.0 * np.mean(per_point_similarity))
    return ComparisonResult(
        ordered_metrics=ordered_metrics,
        a_to_b=a_to_b,
        b_to_a=b_to_a,
        similarity_score=score,
        similarity_label=similarity_label(score),
    )


def format_bool(value: bool) -> str:
    return "true" if value else "false"


def format_float(value: float) -> str:
    return f"{value:.9g}"


def format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(format_float(float(value)) for value in values) + "]"


def print_cloud_metadata(cloud: CloudData) -> None:
    print(f"Point cloud {cloud.label}")
    print(f"  path: {cloud.path}")
    print(f"  point_count: {cloud.points.shape[0]}")
    print(f"  has_colors: {format_bool(cloud.has_colors)}")
    print(f"  has_normals: {format_bool(cloud.has_normals)}")
    print(f"  min_xyz: {format_vector(cloud.min_bound)}")
    print(f"  max_xyz: {format_vector(cloud.max_bound)}")
    print(f"  extent_xyz: {format_vector(cloud.extent)}")
    print(f"  bounding_box_diagonal: {format_float(cloud.bounding_box_diagonal)}")
    print(f"  centroid_xyz: {format_vector(cloud.centroid)}")


def print_metric_dict(prefix: str, metrics: Dict[str, float]) -> None:
    for name in ["mean", "median", "p95", "p99", "max"]:
        print(f"  {prefix}_{name}: {format_float(metrics[name])}")


def print_results(
    cloud_a: CloudData,
    cloud_b: CloudData,
    result: ComparisonResult,
    similarity_distance_scale: float,
    sample_points: Optional[int],
) -> None:
    print_cloud_metadata(cloud_a)
    print_cloud_metadata(cloud_b)

    comparison_count_a = min(cloud_a.points.shape[0], sample_points) if sample_points is not None else cloud_a.points.shape[0]
    comparison_count_b = min(cloud_b.points.shape[0], sample_points) if sample_points is not None else cloud_b.points.shape[0]

    print("Comparison metadata")
    print(f"  point_count_difference: {cloud_a.points.shape[0] - cloud_b.points.shape[0]}")
    print(f"  comparison_point_count_a: {comparison_count_a}")
    print(f"  comparison_point_count_b: {comparison_count_b}")
    print(f"  min_delta_xyz: {format_vector(cloud_a.min_bound - cloud_b.min_bound)}")
    print(f"  max_delta_xyz: {format_vector(cloud_a.max_bound - cloud_b.max_bound)}")
    print(f"  extent_delta_xyz: {format_vector(cloud_a.extent - cloud_b.extent)}")
    print(f"  centroid_delta_xyz: {format_vector(cloud_a.centroid - cloud_b.centroid)}")
    print(f"  bounding_box_diagonal_delta: {format_float(cloud_a.bounding_box_diagonal - cloud_b.bounding_box_diagonal)}")

    print("Ordered XYZ absolute errors")
    if result.ordered_metrics is None:
        print("  status: skipped_point_count_mismatch")
    else:
        print("  status: computed")
        for name in [
            "ordered_abs_error_mean",
            "ordered_abs_error_median",
            "ordered_abs_error_p95",
            "ordered_abs_error_p99",
            "ordered_abs_error_max",
            "ordered_point_distance_mean",
            "ordered_point_distance_p95",
            "ordered_point_distance_max",
        ]:
            print(f"  {name}: {format_float(result.ordered_metrics[name])}")

    print("Nearest-neighbor distances")
    print_metric_dict("a_to_b", distance_metrics(result.a_to_b))
    print_metric_dict("b_to_a", distance_metrics(result.b_to_a))
    print_metric_dict("symmetric", distance_metrics(np.concatenate([result.a_to_b, result.b_to_a])))

    print("Similarity")
    print(f"  similarity_distance_scale: {format_float(similarity_distance_scale)}")
    print(f"  similarity_score: {result.similarity_score:.6f}")
    print(f"  similarity_label: {result.similarity_label}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two PLY point clouds and print metadata, distance metrics, and a similarity score.",
    )
    parser.add_argument("--env-dir", help="Environment directory containing the point clouds.")
    parser.add_argument("--cloud-a-name", default="cloud.ply", help="Cloud A filename within --env-dir.")
    parser.add_argument("--cloud-b-name", default="cloud_original.ply", help="Cloud B filename within --env-dir.")
    parser.add_argument("--cloud-a", help="Explicit Cloud A PLY path. Must be used with --cloud-b.")
    parser.add_argument("--cloud-b", help="Explicit Cloud B PLY path. Must be used with --cloud-a.")
    parser.add_argument(
        "--similarity-distance-scale",
        required=True,
        type=positive_float,
        help="Distance in meters that maps a point's contribution to zero similarity.",
    )
    parser.add_argument(
        "--sample-points",
        type=positive_int,
        help="Deterministically subsample each cloud to at most this many points for comparison.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        path_a, path_b = resolve_cloud_paths(args)
        cloud_a = load_cloud(path_a, "A")
        cloud_b = load_cloud(path_b, "B")
        result = compare_clouds(
            cloud_a,
            cloud_b,
            similarity_distance_scale=args.similarity_distance_scale,
            sample_points=args.sample_points,
        )
        print_results(
            cloud_a,
            cloud_b,
            result,
            similarity_distance_scale=args.similarity_distance_scale,
            sample_points=args.sample_points,
        )
    except PointCloudComparisonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
