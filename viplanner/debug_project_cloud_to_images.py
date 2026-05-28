# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project a reconstructed vIPlanner point cloud back into recorded images."""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import scipy.spatial.transform as tf

from viplanner.config import ReconstructionCfg
from viplanner.depth_reconstruct import DepthReconstruction

DEFAULT_ALPHA_VALUES = "0.0,0.20,0.40,0.60,0.75"


@dataclass
class ProjectionResult:
    pixels: np.ndarray
    depths: np.ndarray
    colors: np.ndarray
    mask: np.ndarray
    source_indices: Optional[np.ndarray] = None


@dataclass
class DebugContext:
    cfg: ReconstructionCfg
    env_path: str
    rgb_dir: str
    output_dir: str
    depth_scale: float
    K_depth: np.ndarray
    K_rgb: np.ndarray
    depth_extrinsics: np.ndarray
    rgb_extrinsics: np.ndarray
    points: np.ndarray
    colors: Optional[np.ndarray]
    total_cloud_points: int
    frame_ids: list
    overlay_alpha: float
    overlay_alpha_values: list


def projection_rows(intrinsic_path: str) -> np.ndarray:
    return DepthReconstruction._projection_rows(np.loadtxt(intrinsic_path, delimiter=","), intrinsic_path)


def extrinsic_rows(extrinsic_path: str) -> np.ndarray:
    values = np.asarray(np.loadtxt(extrinsic_path, delimiter=","), dtype=float)
    if values.ndim == 1 and values.size == 7:
        return values.reshape(1, 7)
    if values.ndim == 2 and values.shape[1] == 7:
        return values
    raise ValueError(
        "Expected camera extrinsics with rows of 7 values "
        f"(x, y, z, qx, qy, qz, qw): {extrinsic_path}"
    )


def project_points_to_image(
    points_world: np.ndarray,
    pose: np.ndarray,
    K: np.ndarray,
    image_shape: Tuple[int, int],
    point_colors: Optional[np.ndarray] = None,
) -> ProjectionResult:
    """Project world points into an image using vIPlanner camera-frame conventions."""
    height, width = image_shape
    rot = tf.Rotation.from_quat(pose[3:]).as_matrix()
    points_cam = (rot.T @ (points_world - pose[:3]).T).T

    forward = points_cam[:, 0]
    forward_mask = forward > 1e-6

    optical_norm = np.empty((points_cam.shape[0], 3), dtype=float)
    optical_norm[:, 0] = -points_cam[:, 1] / np.maximum(forward, 1e-6)
    optical_norm[:, 1] = -points_cam[:, 2] / np.maximum(forward, 1e-6)
    optical_norm[:, 2] = 1.0

    pixels_float = (K @ optical_norm.T).T
    pixels = np.round(pixels_float[:, :2]).astype(int)
    in_fov = (
        forward_mask
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )

    if point_colors is None or len(point_colors) == 0:
        colors = depth_colors(forward)
    else:
        colors = np.clip(np.asarray(point_colors) * 255.0, 0, 255).astype(np.uint8)

    return ProjectionResult(
        pixels=pixels[in_fov],
        depths=forward[in_fov],
        colors=colors[in_fov],
        mask=in_fov,
        source_indices=np.flatnonzero(in_fov),
    )


def zbuffer_projection(projection: ProjectionResult, image_shape: Tuple[int, int]) -> ProjectionResult:
    """Keep the nearest projected point for each image pixel."""
    if len(projection.pixels) == 0:
        return ProjectionResult(
            pixels=projection.pixels,
            depths=projection.depths,
            colors=projection.colors,
            mask=np.zeros(0, dtype=bool),
            source_indices=projection.source_indices,
        )

    _height, width = image_shape
    linear_idx = projection.pixels[:, 1] * width + projection.pixels[:, 0]
    order = np.lexsort((projection.depths, linear_idx))
    sorted_linear_idx = linear_idx[order]
    first_for_pixel = np.ones(sorted_linear_idx.shape[0], dtype=bool)
    first_for_pixel[1:] = sorted_linear_idx[1:] != sorted_linear_idx[:-1]
    keep_idx = order[first_for_pixel]

    return ProjectionResult(
        pixels=projection.pixels[keep_idx],
        depths=projection.depths[keep_idx],
        colors=projection.colors[keep_idx],
        mask=np.ones(keep_idx.shape[0], dtype=bool),
        source_indices=None if projection.source_indices is None else projection.source_indices[keep_idx],
    )


def depth_colors(depths: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depths) & (depths > 0)
    norm = np.zeros(depths.shape, dtype=np.uint8)
    if np.any(finite):
        finite_depths = depths[finite]
        near, far = np.percentile(finite_depths, [2, 98])
        if far <= near:
            far = near + 1.0
        norm[finite] = np.clip((finite_depths - near) / (far - near) * 255.0, 0, 255).astype(np.uint8)
    colorized = cv2.cvtColor(cv2.applyColorMap(norm.reshape(-1, 1), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return colorized.reshape(-1, 3)


def load_depth_image(env_path: str, frame_idx: int, suffix: str, depth_scale: float, high_res_depth: bool = False) -> np.ndarray:
    depth_dir = os.path.join(env_path, "depth_high_res" if high_res_depth else "depth")
    stem = f"{frame_idx:04d}{suffix}"
    npy_path = os.path.join(depth_dir, stem + ".npy")
    png_path = os.path.join(depth_dir, stem + ".png")

    if os.path.isfile(npy_path):
        image = np.load(npy_path).astype(np.float32)
    elif os.path.isfile(png_path):
        loaded = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
        if loaded is None:
            raise RuntimeError(f"Failed to read depth image: {png_path}")
        image = loaded.astype(np.float32)
    else:
        raise FileNotFoundError(f"Missing depth image for frame {frame_idx}: {npy_path} or {png_path}")

    image = image / float(depth_scale)
    image[~np.isfinite(image)] = 0.0
    return image


def load_rgb_image(rgb_dir: str, frame_idx: int, suffix: str) -> np.ndarray:
    path = os.path.join(rgb_dir, f"{frame_idx:04d}{suffix}.png")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing RGB image for frame {frame_idx}: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def validate_alpha(alpha: float, label: str = "--overlay-alpha") -> float:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"{label} must be between 0.0 and 1.0")
    return alpha


def parse_alpha_values(value: str) -> list:
    try:
        alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--overlay-alpha-values must be comma-separated numbers") from exc

    if len(alphas) != 5:
        raise ValueError("--overlay-alpha-values must contain exactly 5 comma-separated values")
    for alpha in alphas:
        validate_alpha(alpha, label="--overlay-alpha-values")
    return alphas


def alpha_suffix(alpha: float) -> str:
    return f"alpha_{int(round(alpha * 100)):03d}"


def projection_overlay_layer(image_rgb: np.ndarray, projection: ProjectionResult, point_size: int) -> np.ndarray:
    overlay = image_rgb.copy()
    radius = max(1, int(point_size))
    for pixel, color in zip(projection.pixels, projection.colors):
        cv2.circle(
            overlay,
            (int(pixel[0]), int(pixel[1])),
            radius,
            (int(color[0]), int(color[1]), int(color[2])),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return overlay


def blend_projection_layer(image_rgb: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    validate_alpha(alpha)
    return cv2.addWeighted(overlay, alpha, image_rgb, 1.0 - alpha, 0)


def draw_projection(image_rgb: np.ndarray, projection: ProjectionResult, point_size: int, alpha: float) -> np.ndarray:
    overlay = projection_overlay_layer(image_rgb, projection, point_size)
    return blend_projection_layer(image_rgb, overlay, alpha)


def save_projection_images(
    output_dir: str,
    name: str,
    frame_idx: int,
    image_rgb: np.ndarray,
    projection: ProjectionResult,
    point_size: int,
    base_alpha: float,
    alpha_values: list,
) -> None:
    overlay = projection_overlay_layer(image_rgb, projection, point_size)
    base_image = blend_projection_layer(image_rgb, overlay, base_alpha)
    cv2.imwrite(
        os.path.join(output_dir, f"{name}_{frame_idx:04d}.png"),
        cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR),
    )

    for alpha in alpha_values:
        image = blend_projection_layer(image_rgb, overlay, alpha)
        cv2.imwrite(
            os.path.join(output_dir, f"{name}_{frame_idx:04d}_{alpha_suffix(alpha)}.png"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )


def depth_visualization(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    norm = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        near, far = np.percentile(depth[valid], [2, 98])
        if far <= near:
            far = near + 1.0
        norm[valid] = np.clip((depth[valid] - near) / (far - near) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def depth_residual_stats(depth_image: np.ndarray, projection: ProjectionResult, prefix: str = "depth") -> dict:
    if len(projection.pixels) == 0:
        return empty_depth_stats(prefix)
    source_depth = depth_image[projection.pixels[:, 1], projection.pixels[:, 0]]
    valid = np.isfinite(source_depth) & (source_depth > 0)
    if not np.any(valid):
        return empty_depth_stats(prefix)

    residual = np.abs(projection.depths[valid] - source_depth[valid])
    return {
        f"{prefix}_valid_depth_overlap": int(residual.shape[0]),
        f"{prefix}_depth_abs_error_mean": float(np.mean(residual)),
        f"{prefix}_depth_abs_error_median": float(np.median(residual)),
        f"{prefix}_depth_abs_error_p90": float(np.percentile(residual, 90)),
        f"{prefix}_depth_abs_error_max": float(np.max(residual)),
        f"{prefix}_depth_close_025_ratio": float(np.mean(residual < 0.25)),
        f"{prefix}_depth_close_050_ratio": float(np.mean(residual < 0.50)),
        f"{prefix}_depth_close_100_ratio": float(np.mean(residual < 1.00)),
    }


def empty_depth_stats(prefix: str = "depth") -> dict:
    return {
        f"{prefix}_valid_depth_overlap": 0,
        f"{prefix}_depth_abs_error_mean": np.nan,
        f"{prefix}_depth_abs_error_median": np.nan,
        f"{prefix}_depth_abs_error_p90": np.nan,
        f"{prefix}_depth_abs_error_max": np.nan,
        f"{prefix}_depth_close_025_ratio": np.nan,
        f"{prefix}_depth_close_050_ratio": np.nan,
        f"{prefix}_depth_close_100_ratio": np.nan,
    }


def depth_residual_masks(
    depth_image: np.ndarray,
    projection: ProjectionResult,
    residual_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid-overlap, close-depth, and residual arrays for projected pixels."""
    if len(projection.pixels) == 0:
        return (
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=float),
        )
    source_depth = depth_image[projection.pixels[:, 1], projection.pixels[:, 0]]
    valid = np.isfinite(source_depth) & (source_depth > 0)
    residual = np.full(source_depth.shape, np.nan, dtype=float)
    residual[valid] = np.abs(projection.depths[valid] - source_depth[valid])
    close = valid & (residual <= residual_threshold)
    return valid, close, residual


def load_cloud(cloud_path: str, max_points: Optional[int]) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    if not os.path.isfile(cloud_path):
        raise FileNotFoundError(f"Point cloud file does not exist: {cloud_path}")
    pcd = o3d.io.read_point_cloud(cloud_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    total_points = int(points.shape[0])

    if max_points is not None and max_points > 0 and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=int)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]

    return points, colors, total_points


def frame_indices(start_idx: int, end_idx: Optional[int], max_images: Optional[int], stride: int) -> range:
    if stride <= 0:
        raise ValueError("--stride must be greater than zero")
    if end_idx is not None and max_images is not None:
        raise ValueError("Use either --end-idx or --max-images, not both")
    if end_idx is None:
        end_idx = start_idx + (max_images if max_images is not None else 1)
    if end_idx <= start_idx:
        raise ValueError("--end-idx must be greater than --start-idx")
    return range(start_idx, end_idx, stride)


def required_path(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")


def setup_debug_context(args: argparse.Namespace) -> DebugContext:
    cfg = ReconstructionCfg.from_yaml(args.config)
    env_path = cfg.get_data_path()
    cloud_path = args.cloud_path or os.path.join(env_path, "cloud.ply")
    rgb_dir = args.rgb_dir or os.path.join(env_path, "rgb")
    output_dir = args.output_dir or os.path.join(env_path, "debug_projection")
    depth_scale = args.depth_scale if args.depth_scale is not None else cfg.depth_scale
    overlay_alpha = validate_alpha(args.overlay_alpha)
    overlay_alpha_values = parse_alpha_values(args.overlay_alpha_values)

    required_path(env_path, "Environment directory")
    required_path(rgb_dir, "RGB directory")
    required_path(os.path.join(env_path, "intrinsics.txt"), "Intrinsics file")
    rgb_extrinsic_path = os.path.join(env_path, "camera_extrinsic" + cfg.sem_suffix + ".txt")
    depth_extrinsic_path = (
        rgb_extrinsic_path
        if cfg.high_res_depth
        else os.path.join(env_path, "camera_extrinsic" + cfg.depth_suffix + ".txt")
    )
    required_path(depth_extrinsic_path, "Depth extrinsics file")
    required_path(rgb_extrinsic_path, "RGB extrinsics file")

    P = projection_rows(os.path.join(env_path, "intrinsics.txt"))
    if len(P) < 2:
        raise ValueError("Projection debug requires depth and RGB projection rows in intrinsics.txt")
    K_depth = P[0].reshape(3, 4)[:3, :3]
    K_rgb = P[1].reshape(3, 4)[:3, :3]
    if cfg.high_res_depth:
        K_depth = K_rgb

    depth_extrinsics = extrinsic_rows(depth_extrinsic_path)
    rgb_extrinsics = extrinsic_rows(rgb_extrinsic_path)
    points, colors, total_cloud_points = load_cloud(cloud_path, args.max_points)
    frame_ids = list(frame_indices(args.start_idx, args.end_idx, args.max_images, args.stride))

    return DebugContext(
        cfg=cfg,
        env_path=env_path,
        rgb_dir=rgb_dir,
        output_dir=output_dir,
        depth_scale=depth_scale,
        K_depth=K_depth,
        K_rgb=K_rgb,
        depth_extrinsics=depth_extrinsics,
        rgb_extrinsics=rgb_extrinsics,
        points=points,
        colors=colors,
        total_cloud_points=total_cloud_points,
        frame_ids=frame_ids,
        overlay_alpha=overlay_alpha,
        overlay_alpha_values=overlay_alpha_values,
    )


def camera_frame_points_to_world(points_cam: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rot = tf.Rotation.from_quat(pose[3:]).as_matrix()
    return (rot @ points_cam.T).T + pose[:3]


def pixel_rays_camera_frame(K: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    normalized = (np.linalg.inv(K) @ pixels.T).T
    return normalized[:, [2, 0, 1]] * np.array([1.0, -1.0, -1.0])


def backproject_depth_to_world(
    depth_image: np.ndarray,
    pose: np.ndarray,
    K: np.ndarray,
    stride: int,
) -> np.ndarray:
    stride = max(1, int(stride))
    v_coords = np.arange(0, depth_image.shape[0], stride)
    u_coords = np.arange(0, depth_image.shape[1], stride)
    grid_u, grid_v = np.meshgrid(u_coords, v_coords)
    depths = depth_image[grid_v, grid_u].reshape(-1)
    valid = np.isfinite(depths) & (depths > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=float)

    pixels = np.column_stack(
        [
            grid_u.reshape(-1)[valid],
            grid_v.reshape(-1)[valid],
            np.ones(np.count_nonzero(valid), dtype=float),
        ]
    )
    rays = pixel_rays_camera_frame(K, pixels)
    points_cam = rays * depths[valid, np.newaxis]
    return camera_frame_points_to_world(points_cam, pose)


def make_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    uniform_color: Optional[Tuple[float, float, float]] = None,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(points) > 0:
        colors = np.asarray(colors, dtype=float).reshape(-1, 3)
        if np.max(colors) > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    elif uniform_color is not None and len(points) > 0:
        pcd.paint_uniform_color(uniform_color)
    return pcd


def make_camera_frustum(
    pose: np.ndarray,
    K: np.ndarray,
    image_shape: Tuple[int, int],
    scale: float = 1.0,
    color: Tuple[float, float, float] = (1.0, 0.85, 0.0),
) -> o3d.geometry.LineSet:
    height, width = image_shape
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [width - 1.0, 0.0, 1.0],
            [width - 1.0, height - 1.0, 1.0],
            [0.0, height - 1.0, 1.0],
        ]
    )
    rays = pixel_rays_camera_frame(K, corners) * float(scale)
    world_points = np.vstack([pose[:3], camera_frame_points_to_world(rays, pose)])
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return line_set


def make_camera_trajectory(poses: np.ndarray) -> o3d.geometry.LineSet:
    points = np.asarray(poses[:, :3], dtype=float)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    if len(points) > 1:
        lines = np.column_stack([np.arange(len(points) - 1), np.arange(1, len(points))]).astype(np.int32)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(np.tile((0.15, 0.55, 1.0), (len(lines), 1)))
    return line_set


def viewer_frame_geometries(
    context: DebugContext,
    frame_idx: int,
    residual_threshold: float,
    depth_sample_stride: int,
) -> Tuple[dict, dict]:
    if frame_idx >= len(context.depth_extrinsics) or frame_idx >= len(context.rgb_extrinsics):
        raise IndexError(f"Frame {frame_idx} is outside available extrinsics")

    depth_image = load_depth_image(
        context.env_path,
        frame_idx,
        context.cfg.depth_suffix,
        context.depth_scale,
        context.cfg.high_res_depth,
    )
    depth_projection = project_points_to_image(
        context.points,
        context.depth_extrinsics[frame_idx],
        context.K_depth,
        depth_image.shape,
        context.colors,
    )
    depth_zbuffer_projection = zbuffer_projection(depth_projection, depth_image.shape)
    valid, close, residual = depth_residual_masks(depth_image, depth_zbuffer_projection, residual_threshold)

    source_indices = depth_zbuffer_projection.source_indices
    visible_points = context.points[source_indices] if source_indices is not None else np.empty((0, 3), dtype=float)
    visible_colors = (
        context.colors[source_indices] if context.colors is not None and source_indices is not None else None
    )
    matched_points = visible_points[close]
    mismatch_mask = valid & ~close
    mismatch_points = visible_points[mismatch_mask]
    missing_depth_points = visible_points[~valid]
    depth_world_points = backproject_depth_to_world(
        depth_image,
        context.depth_extrinsics[frame_idx],
        context.K_depth,
        depth_sample_stride,
    )
    valid_depths = depth_image[depth_image > 0]
    frustum_scale = float(np.nanpercentile(valid_depths, 25)) if len(valid_depths) > 0 else 1.0
    frustum_scale = max(0.5, min(2.0, frustum_scale))

    geometries = {
        "frustum": make_camera_frustum(
            context.depth_extrinsics[frame_idx],
            context.K_depth,
            depth_image.shape,
            scale=frustum_scale,
        ),
        "visible": make_point_cloud(visible_points, visible_colors, uniform_color=(0.1, 0.7, 1.0)),
        "matched": make_point_cloud(matched_points, uniform_color=(0.0, 0.85, 0.25)),
        "mismatch": make_point_cloud(mismatch_points, uniform_color=(1.0, 0.18, 0.05)),
        "missing_depth": make_point_cloud(missing_depth_points, uniform_color=(1.0, 0.55, 0.05)),
        "depth_cloud": make_point_cloud(depth_world_points, uniform_color=(0.55, 0.55, 0.55)),
    }
    stats = {
        "frame_idx": frame_idx,
        "visible_points": int(len(visible_points)),
        "matched_points": int(len(matched_points)),
        "mismatch_points": int(len(mismatch_points)),
        "missing_depth_points": int(len(missing_depth_points)),
        "depth_cloud_points": int(len(depth_world_points)),
        "median_residual": float(np.nanmedian(residual[valid])) if np.any(valid) else np.nan,
    }
    return geometries, stats


def viewer_display_hint() -> str:
    display = os.environ.get("DISPLAY")
    wayland_display = os.environ.get("WAYLAND_DISPLAY")
    xdg_session = os.environ.get("XDG_SESSION_TYPE")
    return (
        "Open3D could not create a visible window. "
        f"DISPLAY={display!r}, WAYLAND_DISPLAY={wayland_display!r}, XDG_SESSION_TYPE={xdg_session!r}. "
        "If you are running inside Docker, SSH, VS Code Remote, or WSL, make sure X11/Wayland forwarding "
        "or a host X server is available to the container."
    )


def run_viewer(context: DebugContext, args: argparse.Namespace) -> None:
    if not context.frame_ids:
        raise ValueError("No frames selected for the viewer")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    print("opening Open3D viewer window ...")
    if not vis.create_window("vIPlanner projection debug", width=1400, height=900, visible=True):
        raise RuntimeError(viewer_display_hint())
    render = vis.get_render_option()
    if render is None:
        raise RuntimeError(viewer_display_hint())
    render.background_color = np.array([0.02, 0.02, 0.025])
    render.point_size = float(args.viewer_point_size)

    state = {
        "frame_pos": 0,
        "dynamic": {},
        "show_cloud": True,
        "show_visible": True,
        "show_matched": True,
        "show_mismatch": True,
        "show_depth_cloud": False,
    }

    full_cloud = make_point_cloud(context.points, context.colors, uniform_color=(0.6, 0.6, 0.6))
    trajectory = make_camera_trajectory(context.depth_extrinsics[context.frame_ids])
    vis.add_geometry(full_cloud)
    vis.add_geometry(trajectory)

    def remove_dynamic() -> None:
        for geometry in state["dynamic"].values():
            vis.remove_geometry(geometry, reset_bounding_box=False)
        state["dynamic"] = {}

    def add_dynamic(geometries: dict) -> None:
        state["dynamic"]["frustum"] = geometries["frustum"]
        if state["show_visible"]:
            state["dynamic"]["visible"] = geometries["visible"]
        if state["show_matched"]:
            state["dynamic"]["matched"] = geometries["matched"]
        if state["show_mismatch"]:
            state["dynamic"]["mismatch"] = geometries["mismatch"]
            state["dynamic"]["missing_depth"] = geometries["missing_depth"]
        if state["show_depth_cloud"]:
            state["dynamic"]["depth_cloud"] = geometries["depth_cloud"]
        for geometry in state["dynamic"].values():
            vis.add_geometry(geometry, reset_bounding_box=False)

    def refresh(reset_bounding_box: bool = False) -> bool:
        remove_dynamic()
        frame_idx = context.frame_ids[state["frame_pos"]]
        geometries, stats = viewer_frame_geometries(
            context,
            frame_idx,
            args.residual_threshold,
            args.depth_sample_stride,
        )
        add_dynamic(geometries)
        if reset_bounding_box:
            vis.reset_view_point(True)
        print(
            f"viewer frame {stats['frame_idx']:04d}: "
            f"visible={stats['visible_points']} "
            f"matched={stats['matched_points']} "
            f"mismatch={stats['mismatch_points']} "
            f"missing_depth={stats['missing_depth_points']} "
            f"median_error={stats['median_residual']:.3f}"
        )
        return False

    def step(delta: int):
        def callback(_vis):
            state["frame_pos"] = (state["frame_pos"] + delta) % len(context.frame_ids)
            return refresh()

        return callback

    def toggle(name: str):
        def callback(_vis):
            state[name] = not state[name]
            if name == "show_cloud":
                if state[name]:
                    vis.add_geometry(full_cloud, reset_bounding_box=False)
                else:
                    vis.remove_geometry(full_cloud, reset_bounding_box=False)
                return False
            return refresh()

        return callback

    def reset_view(_vis):
        vis.reset_view_point(True)
        return False

    def quit_viewer(_vis):
        _vis.close()
        return False

    print(
        "Open3D viewer controls: N/right next, P/left previous, C cloud, V visible, "
        "M mismatches, D depth cloud, R reset view, Q/Esc quit"
    )
    refresh(reset_bounding_box=True)
    for key, callback in [
        (ord("N"), step(1)),
        (262, step(1)),
        (ord("P"), step(-1)),
        (263, step(-1)),
        (ord("C"), toggle("show_cloud")),
        (ord("V"), toggle("show_visible")),
        (ord("M"), toggle("show_mismatch")),
        (ord("D"), toggle("show_depth_cloud")),
        (ord("R"), reset_view),
        (ord("Q"), quit_viewer),
        (256, quit_viewer),
    ]:
        vis.register_key_callback(key, callback)
    vis.run()
    vis.destroy_window()


def run(args: argparse.Namespace) -> int:
    context = setup_debug_context(args)
    output_dir = context.output_dir
    points = context.points
    colors = context.colors
    total_cloud_points = context.total_cloud_points

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "projection_metrics.csv")
    fieldnames = [
        "frame_idx",
        "total_cloud_points",
        "sampled_cloud_points",
        "in_fov_rgb_points",
        "in_fov_depth_points",
        "raw_valid_depth_overlap",
        "raw_depth_abs_error_mean",
        "raw_depth_abs_error_median",
        "raw_depth_abs_error_p90",
        "raw_depth_abs_error_max",
        "raw_depth_close_025_ratio",
        "raw_depth_close_050_ratio",
        "raw_depth_close_100_ratio",
        "zbuffer_rgb_points",
        "zbuffer_depth_points",
        "zbuffer_valid_depth_overlap",
        "zbuffer_depth_abs_error_mean",
        "zbuffer_depth_abs_error_median",
        "zbuffer_depth_abs_error_p90",
        "zbuffer_depth_abs_error_max",
        "zbuffer_depth_close_025_ratio",
        "zbuffer_depth_close_050_ratio",
        "zbuffer_depth_close_100_ratio",
    ]

    rows = []
    processed_frame_ids = []
    for idx in context.frame_ids:
        try:
            if idx >= len(context.depth_extrinsics) or idx >= len(context.rgb_extrinsics):
                raise IndexError(f"Frame {idx} is outside available extrinsics")
            depth_image = load_depth_image(
                context.env_path,
                idx,
                context.cfg.depth_suffix,
                context.depth_scale,
                context.cfg.high_res_depth,
            )
            rgb_image = load_rgb_image(context.rgb_dir, idx, context.cfg.sem_suffix)

            rgb_projection = project_points_to_image(
                points,
                context.rgb_extrinsics[idx],
                context.K_rgb,
                rgb_image.shape[:2],
                colors,
            )
            depth_projection = project_points_to_image(
                points,
                context.depth_extrinsics[idx],
                context.K_depth,
                depth_image.shape,
                colors,
            )
            rgb_zbuffer_projection = zbuffer_projection(rgb_projection, rgb_image.shape[:2])
            depth_zbuffer_projection = zbuffer_projection(depth_projection, depth_image.shape)

            if not args.no_save_images:
                depth_viz = depth_visualization(depth_image)
                save_projection_images(
                    output_dir,
                    "rgb_overlay",
                    idx,
                    rgb_image,
                    rgb_projection,
                    args.point_size,
                    context.overlay_alpha,
                    context.overlay_alpha_values,
                )
                save_projection_images(
                    output_dir,
                    "depth_overlay",
                    idx,
                    depth_viz,
                    depth_projection,
                    args.point_size,
                    context.overlay_alpha,
                    context.overlay_alpha_values,
                )
                save_projection_images(
                    output_dir,
                    "rgb_zbuffer_overlay",
                    idx,
                    rgb_image,
                    rgb_zbuffer_projection,
                    args.point_size,
                    context.overlay_alpha,
                    context.overlay_alpha_values,
                )
                save_projection_images(
                    output_dir,
                    "depth_zbuffer_overlay",
                    idx,
                    depth_viz,
                    depth_zbuffer_projection,
                    args.point_size,
                    context.overlay_alpha,
                    context.overlay_alpha_values,
                )

            row = {
                "frame_idx": idx,
                "total_cloud_points": total_cloud_points,
                "sampled_cloud_points": int(points.shape[0]),
                "in_fov_rgb_points": int(len(rgb_projection.pixels)),
                "in_fov_depth_points": int(len(depth_projection.pixels)),
                "zbuffer_rgb_points": int(len(rgb_zbuffer_projection.pixels)),
                "zbuffer_depth_points": int(len(depth_zbuffer_projection.pixels)),
            }
            row.update(depth_residual_stats(depth_image, depth_projection, prefix="raw"))
            row.update(depth_residual_stats(depth_image, depth_zbuffer_projection, prefix="zbuffer"))
            rows.append(row)
            processed_frame_ids.append(idx)
            print(
                f"frame {idx:04d}: rgb={row['in_fov_rgb_points']} "
                f"depth={row['in_fov_depth_points']} "
                f"zbuf_depth={row['zbuffer_depth_points']} "
                f"zbuf_overlap={row['zbuffer_valid_depth_overlap']}"
            )
        except (FileNotFoundError, RuntimeError, IndexError) as exc:
            if args.skip_missing:
                print(f"[WARNING] skipping frame {idx:04d}: {exc}")
                continue
            raise

    with open(metrics_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved projection debug outputs to: {output_dir}")
    if args.viewer:
        if args.skip_missing:
            context.frame_ids = processed_frame_ids
        run_viewer(context, args)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Debug Project Cloud To Images",
        description="Project a reconstructed vIPlanner cloud.ply into recorded depth/RGB frames.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML file containing reconstruction config")
    parser.add_argument("--start-idx", type=int, default=0, help="First frame index to project")
    parser.add_argument("--end-idx", type=int, default=None, help="Exclusive end frame index")
    parser.add_argument("--max-images", type=int, default=None, help="Number of frames to process from start index")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--cloud-path", default=None, help="Path to reconstructed cloud.ply")
    parser.add_argument("--rgb-dir", default=None, help="Directory containing RGB images")
    parser.add_argument("--output-dir", default=None, help="Directory for overlays and projection_metrics.csv")
    parser.add_argument("--point-size", type=int, default=1, help="Overlay point radius in pixels")
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Point overlay opacity from 0.0 transparent to 1.0 opaque",
    )
    parser.add_argument(
        "--overlay-alpha-values",
        default=DEFAULT_ALPHA_VALUES,
        help=(
            "Exactly 5 comma-separated opacity values for additional alpha-sweep overlay images "
            f"(default: {DEFAULT_ALPHA_VALUES})"
        ),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Deterministically subsample cloud to this many points for overlays and metrics",
    )
    parser.add_argument("--depth-scale", type=float, default=None, help="Override reconstruction depth scale")
    parser.add_argument("--skip-missing", action="store_true", help="Skip frames with missing inputs")
    parser.add_argument("--viewer", action="store_true", help="Open an interactive Open3D projection debug viewer")
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Skip PNG overlay generation and only write projection_metrics.csv",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=0.25,
        help="Depth residual threshold in meters for green/red viewer classification",
    )
    parser.add_argument(
        "--viewer-point-size",
        type=float,
        default=2.0,
        help="Open3D point size for the interactive viewer",
    )
    parser.add_argument(
        "--depth-sample-stride",
        type=int,
        default=8,
        help="Pixel stride for the optional back-projected depth cloud in the viewer",
    )
    return parser


def main(argv=None) -> int:
    return run(build_argparser().parse_args(argv))


if __name__ == "__main__":
    main()
