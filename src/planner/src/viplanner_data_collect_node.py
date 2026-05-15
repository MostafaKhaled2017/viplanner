#!/usr/bin/env python3

# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import cv_bridge
import numpy as np
import rospkg
import rospy
import scipy.spatial.transform as stf
import tf2_ros
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Header

rospack = rospkg.RosPack()
pack_path = rospack.get_path("viplanner_node")
sys.path.append(pack_path)

from utils.rosutil import ROSArgparse
from utils.data_collection_paths import resolve_output_dir


ROS_TO_VIPLANNER_MAT = stf.Rotation.from_euler("XYZ", [-90, 0, -90], degrees=True).as_matrix()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y", "t")


def _projection_matrix(msg: CameraInfo) -> np.ndarray:
    projection = np.array(msg.P, dtype=np.float64).reshape(3, 4)
    if np.any(projection):
        return projection

    intrinsic = np.array(msg.K, dtype=np.float64).reshape(3, 3)
    projection[:, :3] = intrinsic
    return projection


class VIPlannerDataCollector:
    """Record ROS simulator images and camera poses in vIPlanner dataset format."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.bridge = cv_bridge.CvBridge()
        self.lock = threading.Lock()

        requested_output_dir = Path(os.path.expandvars(os.path.expanduser(cfg.dataset_root))) / cfg.env_name
        self.output_dir = resolve_output_dir(requested_output_dir.parent, requested_output_dir.name)
        self.depth_dir = self.output_dir / "depth"
        self.rgb_dir = self.output_dir / "rgb"
        self.semantics_dir = self.output_dir / "semantics"
        self.depth_extrinsic_path = self.output_dir / f"camera_extrinsic{cfg.depth_suffix}.txt"
        self.rgb_extrinsic_path = self.output_dir / f"camera_extrinsic{cfg.rgb_suffix}.txt"
        self.intrinsics_path = self.output_dir / "intrinsics.txt"

        self.depth_projection: Optional[np.ndarray] = None
        self.rgb_projection: Optional[np.ndarray] = None
        self.intrinsics_written = False

        self.latest_rgb: Optional[Tuple[Header, np.ndarray]] = None
        self.last_saved_position: Optional[np.ndarray] = None
        self.last_saved_quat: Optional[np.ndarray] = None
        self.last_saved_time: Optional[rospy.Time] = None
        self.sample_idx = 0
        self.shutdown_after_max_samples = False

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(float(cfg.tf_buffer_seconds)))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self._validate_config()
        self._prepare_output_dir(requested_output_dir)
        self._init_subscribers()

        rospy.loginfo("vIPlanner data collector writing to %s", self.output_dir)

    @staticmethod
    def _image_shape(image: np.ndarray) -> Tuple[int, int]:
        return int(image.shape[0]), int(image.shape[1])

    def _validate_image_dimensions(self, image: np.ndarray, stream_name: str, expected_height, expected_width) -> None:
        expected_height = int(expected_height)
        expected_width = int(expected_width)
        actual_height, actual_width = self._image_shape(image)
        if actual_height == expected_height and actual_width == expected_width:
            return

        message = (
            f"{stream_name} image size mismatch: expected {expected_width}x{expected_height}, "
            f"received {actual_width}x{actual_height}. Update the simulator output or the collector config."
        )
        rospy.logfatal(message)
        rospy.signal_shutdown(message)
        raise RuntimeError(message)

    def _prepare_output_dir(self, requested_output_dir: Path) -> None:
        if self.output_dir != requested_output_dir:
            rospy.loginfo(
                "Output directory '%s' already exists and is not empty; recording to '%s' instead.",
                requested_output_dir,
                self.output_dir,
            )

        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.semantics_dir.mkdir(parents=True, exist_ok=True)
        self.depth_extrinsic_path.write_text("")
        self.rgb_extrinsic_path.write_text("")

    def _validate_config(self) -> None:
        for name in ("depth_width", "depth_height", "rgb_width", "rgb_height"):
            value = int(getattr(self.cfg, name))
            if value <= 0:
                raise RuntimeError(f"Config parameter '{name}' must be a positive integer, got {value}.")

    def _init_subscribers(self) -> None:
        rospy.Subscriber(
            self.cfg.depth_info_topic,
            CameraInfo,
            callback=self.depth_info_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            self.cfg.rgb_info_topic,
            CameraInfo,
            callback=self.rgb_info_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            self.cfg.depth_topic,
            Image,
            callback=self.depth_callback,
            queue_size=5,
            buff_size=2**24,
        )
        if _as_bool(self.cfg.rgb_compressed):
            rospy.Subscriber(
                self.cfg.rgb_topic,
                CompressedImage,
                callback=self.rgb_compressed_callback,
                queue_size=5,
                buff_size=2**24,
            )
        else:
            rospy.Subscriber(
                self.cfg.rgb_topic,
                Image,
                callback=self.rgb_callback,
                queue_size=5,
                buff_size=2**24,
            )

    def depth_info_callback(self, msg: CameraInfo) -> None:
        with self.lock:
            self.depth_projection = _projection_matrix(msg)
            self._write_intrinsics_if_ready()

    def rgb_info_callback(self, msg: CameraInfo) -> None:
        with self.lock:
            self.rgb_projection = _projection_matrix(msg)
            self._write_intrinsics_if_ready()

    def _write_intrinsics_if_ready(self) -> None:
        if self.intrinsics_written or self.depth_projection is None or self.rgb_projection is None:
            return
        intrinsics = np.stack([self.depth_projection, self.rgb_projection])
        np.savetxt(str(self.intrinsics_path), intrinsics.reshape(2, 12), delimiter=",")
        self.intrinsics_written = True
        rospy.loginfo("Saved vIPlanner intrinsics to %s", self.intrinsics_path)

    def rgb_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except cv_bridge.CvBridgeError as exc:
            rospy.logwarn("Failed to decode RGB image: %s", exc)
            return

        self._validate_image_dimensions(image, "RGB", self.cfg.rgb_height, self.cfg.rgb_width)

        with self.lock:
            self.latest_rgb = (msg.header, image)

    def rgb_compressed_callback(self, msg: CompressedImage) -> None:
        rgb_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
        if image is None:
            rospy.logwarn("Failed to decode compressed RGB image")
            return

        self._validate_image_dimensions(image, "RGB", self.cfg.rgb_height, self.cfg.rgb_width)

        with self.lock:
            self.latest_rgb = (msg.header, image)

    def depth_callback(self, msg: Image) -> None:
        if not self.intrinsics_written:
            rospy.logwarn_throttle(5.0, "Waiting for depth and RGB camera info before recording.")
            return

        with self.lock:
            if self.latest_rgb is None:
                rospy.logwarn_throttle(5.0, "Waiting for synchronized RGB image before recording.")
                return
            rgb_header, rgb_image = self.latest_rgb

        if msg.header.stamp.to_sec() > 0.0 and rgb_header.stamp.to_sec() > 0.0:
            time_delta = abs((msg.header.stamp - rgb_header.stamp).to_sec())
            if time_delta > float(self.cfg.sync_slop_seconds):
                rospy.logwarn_throttle(
                    5.0,
                    "Skipping image pair with sync delta %.3f s > %.3f s",
                    time_delta,
                    float(self.cfg.sync_slop_seconds),
                )
                return

        depth_pose = self._lookup_camera_pose(msg.header, self.cfg.depth_frame_id)
        if depth_pose is None:
            return

        rgb_pose = self._lookup_camera_pose(rgb_header, self.cfg.rgb_frame_id)
        if rgb_pose is None:
            if not _as_bool(self.cfg.rgb_pose_fallback_to_depth):
                return
            rospy.logwarn_throttle(
                10.0,
                "RGB camera pose unavailable; using depth camera pose for RGB extrinsics. "
                "Set rgb_frame_id to a valid TF frame if the RGB camera is not co-located with depth.",
            )
            rgb_pose = depth_pose.copy()

        if not self._passes_sampling_policy(msg.header.stamp, depth_pose):
            return

        try:
            depth_mm = self._depth_to_millimeters(msg)
        except cv_bridge.CvBridgeError as exc:
            rospy.logwarn("Failed to decode depth image: %s", exc)
            return

        self._validate_image_dimensions(depth_mm, "Depth", self.cfg.depth_height, self.cfg.depth_width)

        self._save_sample(msg.header.stamp, depth_mm, rgb_image, depth_pose, rgb_pose)

        if self.shutdown_after_max_samples:
            rospy.signal_shutdown("Reached max_samples")

    def _lookup_camera_pose(self, header, frame_override: str) -> Optional[np.ndarray]:
        frame_id = frame_override if frame_override else header.frame_id
        if not frame_id:
            rospy.logwarn_throttle(5.0, "Cannot record pose for image with empty frame_id.")
            return None

        stamp = rospy.Time(0) if _as_bool(self.cfg.use_latest_tf) else header.stamp
        try:
            transform = self.tf_buffer.lookup_transform(
                self.cfg.world_frame_id,
                frame_id,
                stamp,
                rospy.Duration(float(self.cfg.tf_timeout_seconds)),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                5.0,
                "Pose lookup failed for %s -> %s: %s",
                frame_id,
                self.cfg.world_frame_id,
                exc,
            )
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        quat = np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64)

        if _as_bool(self.cfg.convert_optical_to_viplanner):
            rot_mat = stf.Rotation.from_quat(quat).as_matrix() @ ROS_TO_VIPLANNER_MAT
            quat = stf.Rotation.from_matrix(rot_mat).as_quat()

        return np.array(
            [
                translation.x,
                translation.y,
                translation.z,
                quat[0],
                quat[1],
                quat[2],
                quat[3],
            ],
            dtype=np.float64,
        )

    def _passes_sampling_policy(self, stamp: rospy.Time, pose: np.ndarray) -> bool:
        if self.last_saved_time is not None:
            dt = (stamp - self.last_saved_time).to_sec()
            if stamp.to_sec() > 0.0 and self.last_saved_time.to_sec() > 0.0 and 0.0 <= dt < float(
                self.cfg.sample_min_interval
            ):
                return False

        if self.last_saved_position is None:
            return True

        distance = np.linalg.norm(pose[:3] - self.last_saved_position)
        angle = self._quat_angle(pose[3:], self.last_saved_quat)
        return distance >= float(self.cfg.sample_min_distance) or angle >= math.radians(
            float(self.cfg.sample_min_angle_deg)
        )

    @staticmethod
    def _quat_angle(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
        relative = stf.Rotation.from_quat(quat_a) * stf.Rotation.from_quat(quat_b).inv()
        return relative.magnitude()

    def _depth_to_millimeters(self, msg: Image) -> np.ndarray:
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_encoding = msg.encoding.lower()
        depth_is_uint16 = _as_bool(self.cfg.depth_is_uint16)

        if not depth_is_uint16 and (depth.dtype == np.uint16 or depth_encoding in ("16uc1", "mono16")):
            rospy.logwarn_throttle(
                5.0,
                "Depth config mismatch: received %s/%s depth while depth_is_uint16=false. "
                "Treating the image as uint16 millimeters. Set depth_is_uint16=true for this source.",
                depth_encoding or "<empty>",
                depth.dtype,
            )
            depth_is_uint16 = True

        if depth_is_uint16:
            if depth.dtype != np.uint16:
                depth = np.clip(depth, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            else:
                depth = depth.copy()
            depth[depth > int(float(self.cfg.max_depth) * float(self.cfg.depth_scale))] = 0
            return depth

        depth_m = depth.astype(np.float32, copy=False)
        np.nan_to_num(depth_m, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m[depth_m > float(self.cfg.max_depth)] = 0.0
        return np.clip(depth_m * float(self.cfg.depth_scale), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    def _save_sample(
        self,
        stamp: rospy.Time,
        depth_mm: np.ndarray,
        rgb_image: np.ndarray,
        depth_pose: np.ndarray,
        rgb_pose: np.ndarray,
    ) -> None:
        sample_name = str(self.sample_idx).zfill(4)
        depth_png = self.depth_dir / f"{sample_name}{self.cfg.depth_suffix}.png"
        rgb_png = self.rgb_dir / f"{sample_name}{self.cfg.rgb_suffix}.png"

        if not cv2.imwrite(str(depth_png), depth_mm):
            rospy.logwarn("Failed to write depth image: %s", depth_png)
            return
        if _as_bool(self.cfg.save_depth_npy):
            np.save(str(self.depth_dir / f"{sample_name}{self.cfg.depth_suffix}.npy"), depth_mm)

        if not cv2.imwrite(str(rgb_png), rgb_image):
            rospy.logwarn("Failed to write RGB image: %s", rgb_png)
            return

        with self.depth_extrinsic_path.open("a") as depth_file:
            depth_file.write(",".join(f"{value:.9f}" for value in depth_pose.tolist()) + "\n")
        with self.rgb_extrinsic_path.open("a") as rgb_file:
            rgb_file.write(",".join(f"{value:.9f}" for value in rgb_pose.tolist()) + "\n")

        self.last_saved_position = depth_pose[:3].copy()
        self.last_saved_quat = depth_pose[3:].copy()
        self.last_saved_time = stamp
        self.sample_idx += 1

        rospy.loginfo("Saved vIPlanner dataset sample %s", sample_name)

        max_samples = int(self.cfg.max_samples)
        self.shutdown_after_max_samples = max_samples > 0 and self.sample_idx >= max_samples


if __name__ == "__main__":
    node_name = "viplanner_data_collect_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    parser.add_argument("dataset_root", type=str, default="/tmp/viplanner_data", help="Root directory for datasets")
    parser.add_argument("env_name", type=str, default="sim_env", help="Environment folder name")

    parser.add_argument("depth_topic", type=str, default="/camera/depth/image_raw", help="Depth image topic")
    parser.add_argument("rgb_topic", type=str, default="/camera/color/image_raw/compressed", help="RGB image topic")
    parser.add_argument("depth_info_topic", type=str, default="/camera/depth/camera_info", help="Depth camera info")
    parser.add_argument("rgb_info_topic", type=str, default="/camera/color/camera_info", help="RGB camera info")
    parser.add_argument("rgb_compressed", type=bool, default=True, help="RGB topic uses sensor_msgs/CompressedImage")

    parser.add_argument("world_frame_id", type=str, default="odom", help="World frame used for saved camera poses")
    parser.add_argument("depth_frame_id", type=str, default="", help="Override depth image frame_id if set")
    parser.add_argument("rgb_frame_id", type=str, default="", help="Override RGB image frame_id if set")
    parser.add_argument(
        "rgb_pose_fallback_to_depth",
        type=bool,
        default=True,
        help="Use the depth camera pose for RGB when the RGB frame is not available in TF",
    )
    parser.add_argument("use_latest_tf", type=bool, default=False, help="Use latest TF instead of image timestamp")
    parser.add_argument("tf_buffer_seconds", type=float, default=30.0, help="TF buffer duration")
    parser.add_argument("tf_timeout_seconds", type=float, default=0.2, help="TF lookup timeout")
    parser.add_argument(
        "convert_optical_to_viplanner",
        type=bool,
        default=True,
        help="Convert ROS optical camera orientation to vIPlanner camera convention",
    )

    parser.add_argument("depth_suffix", type=str, default="_cam0", help="Depth camera filename/extrinsic suffix")
    parser.add_argument("rgb_suffix", type=str, default="_cam1", help="RGB camera filename/extrinsic suffix")
    parser.add_argument("depth_width", type=int, default=640, help="Required depth image width in pixels")
    parser.add_argument("depth_height", type=int, default=480, help="Required depth image height in pixels")
    parser.add_argument("rgb_width", type=int, default=640, help="Required RGB image width in pixels")
    parser.add_argument("rgb_height", type=int, default=480, help="Required RGB image height in pixels")
    parser.add_argument("depth_is_uint16", type=bool, default=False, help="Depth image already stores millimeters")
    parser.add_argument("depth_scale", type=float, default=1000.0, help="Meters to integer depth scale")
    parser.add_argument("max_depth", type=float, default=15.0, help="Depth values above this range are set to zero")
    parser.add_argument("save_depth_npy", type=bool, default=True, help="Also save millimeter depth arrays as .npy")

    parser.add_argument("sync_slop_seconds", type=float, default=0.1, help="Maximum depth/RGB timestamp delta")
    parser.add_argument("sample_min_interval", type=float, default=0.2, help="Minimum seconds between saved samples")
    parser.add_argument("sample_min_distance", type=float, default=0.1, help="Minimum translation between samples")
    parser.add_argument("sample_min_angle_deg", type=float, default=5.0, help="Minimum rotation between samples")
    parser.add_argument("max_samples", type=int, default=0, help="Stop after this many samples; 0 records until shutdown")

    args = parser.parse_args()

    try:
        collector = VIPlannerDataCollector(args)
    except RuntimeError as exc:
        rospy.logerr("%s", exc)
        raise SystemExit(1)

    rospy.spin()
