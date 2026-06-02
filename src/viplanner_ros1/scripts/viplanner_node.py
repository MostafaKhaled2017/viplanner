#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from pathlib import Path as FilePath

import cv2
import numpy as np
import rospy
import tf2_ros
import torch
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path as RosPath
from sensor_msgs.msg import CompressedImage, Image, Joy
from std_msgs.msg import Float32, Int16
from tf2_geometry_msgs import do_transform_point

from viplanner_ros1.msg import Fear
from viplanner_ros1.image_utils import (
    depth_msg_to_numpy,
    prepare_depth_image,
    rgb_msg_to_numpy,
    validate_array_dimensions,
    validate_message_dimensions,
)
from viplanner_ros1.inference import VIPlannerInference
from viplanner_ros1.planning_utils import FearState, clip_goal_xy, fear_scalar, is_forward_tracking
from viplanner_ros1.semantic_inference import Mask2FormerPredictor
from viplanner_ros1.debug_utils import array_stats, tensor_to_list, write_json, xyz_summary

ROS_TO_ROBOTICS_MAT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
CAMERA_FLIP_MAT = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
TF_EXCEPTIONS = (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException)


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def get_private_param(name: str, default):
    return rospy.get_param(f"~{name}", default)


class VIPlannerNode:
    def __init__(self):
        self._read_parameters()

        if bool(self.use_sim_time):
            rospy.set_param("/use_sim_time", True)

        if not self.model_save:
            raise ValueError("Parameter 'model_save' must point to a trained VIPlanner model directory.")

        self.planner = VIPlannerInference(
            self.model_save,
            m2f_config_path=self.m2f_config_path,
            m2f_model_path=self.m2f_model_path,
        )
        self.semantic_predictor = None
        if self.planner.requires_semantic_prediction:
            if not self.m2f_config_path or not self.m2f_model_path:
                raise ValueError("Semantic VIPlanner models require 'm2f_config_path' and 'm2f_model_path'.")
            self.semantic_predictor = Mask2FormerPredictor(
                self.m2f_config_path,
                self.m2f_model_path,
                device=self.m2f_device,
                warn_fn=rospy.logwarn,
            )

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.depth_img = None
        self.depth_stamp = None
        self.depth_frame_id = None
        self.rgb_img = None
        self.goal_pose = None
        self.goal_final_robot = None
        self.goal_robot = None
        self.goal_cam = None
        self.cam_offset = None
        self.cam_rot = None
        self.depth_raw_stats = None
        self.depth_prepared_stats = None
        self.debug_tick_count = 0
        self.ready_for_planning = False
        self.is_goal_init = False
        self.is_goal_processed = False
        self.is_smartjoy = False
        self.planner_status = Int16(data=0)
        self.fear_state = FearState(self.buffer_size, self.fear_threshold)

        rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1, buff_size=2**24)
        if self.planner.requires_rgb:
            if self.rgb_compressed:
                rospy.Subscriber(self.rgb_topic, CompressedImage, self.rgb_compressed_callback, queue_size=1, buff_size=2**24)
            else:
                rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.goal_topic, PointStamped, self.goal_callback, queue_size=10)
        rospy.Subscriber("/joy", Joy, self.joy_callback, queue_size=10)

        self.timer_pub = rospy.Publisher("/viplanner/timer", Float32, queue_size=10)
        self.semantic_timer_pub = rospy.Publisher("/viplanner/m2f_timer", Float32, queue_size=10)
        self.status_pub = rospy.Publisher("/viplanner/status", Int16, queue_size=10)
        self.path_pub = rospy.Publisher(self.path_topic, RosPath, queue_size=10)
        self.fear_path_pub = rospy.Publisher(self.path_topic + "_fear", RosPath, queue_size=10)
        self.fear_pub = rospy.Publisher(self.fear_topic, Fear, queue_size=10)
        self.semantic_image_pub = rospy.Publisher("/viplanner/sem_image/compressed", CompressedImage, queue_size=3)

        rospy.loginfo("VIPlanner ROS1 Ready.")

    def _read_parameters(self):
        defaults = {
            "main_freq": 5,
            "verbose": False,
            "use_sim_time": False,
            "model_save": "",
            "depth_topic": "/rgbd_camera/depth/image",
            "depth_width": 640,
            "depth_height": 360,
            "rgb_topic": "/rgbd_camera/color/image",
            "rgb_width": 640,
            "rgb_height": 360,
            "rgb_compressed": False,
            "goal_topic": "/way_point",
            "path_topic": "/viplanner/path",
            "fear_topic": "/viplanner/fear",
            "robot_id": "base_link",
            "world_id": "odom",
            "mount_cam_frame": "",
            "depth_uint_type": False,
            "max_depth": 15.0,
            "image_flip": False,
            "conv_dist": 0.5,
            "m2f_config_path": "",
            "m2f_model_path": "",
            "m2f_device": "cuda:0",
            "is_fear_act": True,
            "fear_threshold": 0.7,
            "buffer_size": 3,
            "angular_thread": 0.3,
            "track_dist": 0.5,
            "joyGoal_scale": 2.5,
            "subgoal_max_distance": 20.0,
            "subgoal_update_distance": 7.0,
            "debug_enabled": False,
            "debug_dump_dir": "logs/viplanner_debug",
            "debug_dump_every_n": 1,
            "debug_save_input_tensors": False,
            "debug_height_warn_threshold": 0.05,
            "use_camera_frame_goal": False,
        }
        for name, default in defaults.items():
            setattr(self, name, get_private_param(name, default))

        for name in (
            "model_save",
            "depth_topic",
            "rgb_topic",
            "goal_topic",
            "path_topic",
            "fear_topic",
            "robot_id",
            "world_id",
            "mount_cam_frame",
            "m2f_config_path",
            "m2f_model_path",
            "m2f_device",
        ):
            setattr(self, name, str(getattr(self, name)))
        self.mount_cam_frame = self.mount_cam_frame or None
        self.main_freq = int(self.main_freq)
        self.verbose = bool(self.verbose)
        self.use_sim_time = bool(self.use_sim_time)
        self.rgb_compressed = bool(self.rgb_compressed)
        self.depth_width = int(self.depth_width)
        self.depth_height = int(self.depth_height)
        self.rgb_width = int(self.rgb_width)
        self.rgb_height = int(self.rgb_height)
        self.depth_uint_type = bool(self.depth_uint_type)
        self.max_depth = float(self.max_depth)
        self.image_flip = bool(self.image_flip)
        self.conv_dist = float(self.conv_dist)
        self.is_fear_act = bool(self.is_fear_act)
        self.fear_threshold = float(self.fear_threshold)
        self.buffer_size = int(self.buffer_size)
        self.angular_thread = float(self.angular_thread)
        self.track_dist = float(self.track_dist)
        self.joyGoal_scale = float(self.joyGoal_scale)
        self.subgoal_max_distance = float(self.subgoal_max_distance)
        self.subgoal_update_distance = float(self.subgoal_update_distance)
        self.debug_enabled = bool(self.debug_enabled)
        self.debug_dump_dir = str(self.debug_dump_dir)
        self.debug_dump_every_n = max(1, int(self.debug_dump_every_n))
        self.debug_save_input_tensors = bool(self.debug_save_input_tensors)
        self.debug_height_warn_threshold = float(self.debug_height_warn_threshold)
        self.use_camera_frame_goal = bool(self.use_camera_frame_goal)

    def spin(self):
        rate = rospy.Rate(max(1, self.main_freq))
        while not rospy.is_shutdown():
            self.tick()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

    def tick(self):
        if not self._has_planning_inputs():
            return

        start = time.time()
        if self.planner.requires_rgb:
            keypoints, traj, fear = self.planner.plan(self.depth_img.copy(), self.rgb_img.copy(), self.goal_cam)
        else:
            keypoints, traj, fear = self.planner.plan_depth(self.depth_img.copy(), self.goal_cam)
        elapsed_ms = (time.time() - start) * 1000.0
        self.timer_pub.publish(Float32(data=float(elapsed_ms)))
        fear_value = fear_scalar(fear)
        fear_msg = Fear()
        fear_msg.header.stamp = self.depth_stamp or rospy.Time.now()
        fear_msg.header.frame_id = self.robot_id
        fear_msg.fear = fear_value
        self.fear_pub.publish(fear_msg)

        traj_cam = traj.detach().cpu().squeeze(0).numpy()
        waypoints = traj_cam.copy()
        if self.use_camera_frame_goal and self.cam_rot is not None and self.cam_offset is not None:
            waypoints = (self.cam_rot @ waypoints.T).T + self.cam_offset

        if self._goal_reached():
            self.ready_for_planning = False
            self.is_goal_init = False
            if self.planner_status.data == 0:
                self.planner_status.data = 1
                self.status_pub.publish(self.planner_status)
            rospy.loginfo("Goal Arrived")

        if self.is_fear_act:
            is_forward = is_forward_tracking(waypoints, self.track_dist, self.angular_thread)
            if self.fear_state.update(fear_value, is_forward):
                rospy.logwarn("Current path prediction is invalid.")
                if self.planner_status.data == 0:
                    self.planner_status.data = -1
                    self.status_pub.publish(self.planner_status)

        self.publish_path(waypoints)
        self._write_debug_dump_if_needed(keypoints, traj, fear, traj_cam, waypoints, elapsed_ms)

    def _has_planning_inputs(self) -> bool:
        return (
            self.ready_for_planning
            and self.is_goal_init
            and self.depth_img is not None
            and self.goal_cam is not None
            and (not self.planner.requires_rgb or self.rgb_img is not None)
        )

    def _goal_reached(self) -> bool:
        if self.goal_final_robot is None or self.is_smartjoy or not self.is_goal_processed:
            return False
        gx = float(self.goal_final_robot[0][0].item())
        gy = float(self.goal_final_robot[0][1].item())
        return math.hypot(gx, gy) < self.conv_dist

    def depth_callback(self, msg: Image):
        validate_message_dimensions(msg, self.depth_width, self.depth_height, "Depth")
        if self.debug_enabled:
            self.depth_raw_stats = array_stats(depth_msg_to_numpy(msg))
        self.depth_img = prepare_depth_image(msg, self.depth_uint_type, self.max_depth, self.image_flip)
        if self.debug_enabled:
            self.depth_prepared_stats = array_stats(self.depth_img)
        self.depth_stamp = msg.header.stamp
        self.depth_frame_id = msg.header.frame_id
        if self.is_goal_init:
            self._update_goal_tensors(msg.header.stamp, msg.header.frame_id)

    def rgb_callback(self, msg: Image):
        validate_message_dimensions(msg, self.rgb_width, self.rgb_height, "RGB")
        try:
            image = rgb_msg_to_numpy(msg)
        except RuntimeError as exc:
            rospy.logerr(str(exc))
            return
        self._store_rgb_image(image, msg.header)

    def rgb_compressed_callback(self, msg: CompressedImage):
        rgb_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
        if image is None:
            rospy.logerr("Failed to decode compressed RGB image.")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        validate_array_dimensions(image, self.rgb_width, self.rgb_height, "RGB")
        self._store_rgb_image(image, msg.header)

    def _store_rgb_image(self, image, header):
        if self.semantic_predictor is not None:
            start = time.time()
            image = self.semantic_predictor.predict(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self.semantic_timer_pub.publish(Float32(data=float((time.time() - start) * 1000.0)))
            self._publish_semantic_image(image, header)
        self.rgb_img = image

    def _publish_semantic_image(self, image, header):
        success, compressed = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            return
        msg = CompressedImage()
        msg.header = header
        msg.format = "jpeg"
        msg.data = compressed.tobytes()
        self.semantic_image_pub.publish(msg)

    def goal_callback(self, msg: PointStamped):
        rospy.loginfo("Received a new goal")
        self.goal_pose = msg
        self.is_smartjoy = False
        self.is_goal_init = True
        self.is_goal_processed = False
        self.fear_state.reset()
        self.planner_status.data = 0
        if self.depth_stamp is not None and self.depth_frame_id is not None:
            self._update_goal_tensors(self.depth_stamp, self.depth_frame_id)

    def joy_callback(self, joy_msg: Joy):
        if len(joy_msg.buttons) > 4 and joy_msg.buttons[4] > 0.9:
            rospy.loginfo("Switch to Smart Joystick mode ...")
            self.is_smartjoy = True
            self.fear_state.reset()

        if not self.is_smartjoy:
            return

        if len(joy_msg.axes) <= 4 or math.hypot(joy_msg.axes[3], joy_msg.axes[4]) < 1e-3:
            self.fear_state.reset()
            self.ready_for_planning = False
            self.is_goal_init = False
            return

        joy_goal = PointStamped()
        joy_goal.header.frame_id = self.robot_id
        joy_goal.header.stamp = rospy.Time.now()
        joy_goal.point.x = float(joy_msg.axes[4] * self.joyGoal_scale)
        joy_goal.point.y = float(joy_msg.axes[3] * self.joyGoal_scale)
        joy_goal.point.z = 0.0
        self.goal_pose = joy_goal
        self.is_goal_init = True
        self.is_goal_processed = False
        if self.depth_stamp is not None and self.depth_frame_id is not None:
            self._update_goal_tensors(self.depth_stamp, self.depth_frame_id)

    def _update_goal_tensors(self, stamp_msg, depth_frame_id):
        try:
            goal_robot = self._transform_goal(self.goal_pose, self.robot_id)
        except TF_EXCEPTIONS as exc:
            rospy.logerr(f"Failed to transform goal into {self.robot_id}: {exc}")
            return

        self.goal_final_robot = torch.tensor(
            [goal_robot.point.x, goal_robot.point.y, goal_robot.point.z],
            dtype=torch.float32,
        )[None, ...]
        if self.debug_enabled and abs(float(goal_robot.point.z)) > self.debug_height_warn_threshold:
            rospy.logwarn(
                "Goal height in robot frame is %.4f m, expected near 0 for same-height validation.",
                float(goal_robot.point.z),
            )

        gx, gy, gz = clip_goal_xy(
            [goal_robot.point.x, goal_robot.point.y, goal_robot.point.z],
            self.subgoal_max_distance,
        )
        self.goal_robot = torch.tensor([gx, gy, gz], dtype=torch.float32)[None, ...]

        if self.use_camera_frame_goal and (self.cam_rot is None or self.cam_offset is None):
            if not self._update_camera_transform(depth_frame_id):
                return

        if self.use_camera_frame_goal:
            goal_cam = self.cam_rot.T @ (np.array([gx, gy, gz], dtype=np.float32) - self.cam_offset).T
            self.goal_cam = torch.tensor(goal_cam, dtype=torch.float32)[None, ...]
        else:
            self.goal_cam = self.goal_robot.clone()
        self.ready_for_planning = True
        self.is_goal_processed = True

    @staticmethod
    def _point_stamped_dict(msg):
        if msg is None:
            return None
        return {
            "frame_id": msg.header.frame_id,
            "stamp": float(msg.header.stamp.to_sec()),
            "point": [float(msg.point.x), float(msg.point.y), float(msg.point.z)],
        }

    @staticmethod
    def _tensor_dict(tensor):
        if tensor is None:
            return None
        return tensor_to_list(tensor.squeeze(0) if tensor.ndim > 1 else tensor)

    def _write_debug_dump_if_needed(self, keypoints, traj, fear, traj_cam, waypoints_robot, elapsed_ms):
        if not self.debug_enabled:
            return
        self.debug_tick_count += 1
        if (self.debug_tick_count - 1) % self.debug_dump_every_n != 0:
            return

        dump_dir = FilePath(self.debug_dump_dir).expanduser()
        dump_name = f"viplanner_debug_{self.debug_tick_count:06d}"
        json_path = dump_dir / f"{dump_name}.json"
        payload = {
            "tick": int(self.debug_tick_count),
            "elapsed_ms": float(elapsed_ms),
            "depth": {
                "frame_id": self.depth_frame_id,
                "stamp": float(self.depth_stamp.to_sec()) if self.depth_stamp is not None else None,
                "raw_stats": self.depth_raw_stats,
                "prepared_stats": self.depth_prepared_stats,
            },
            "goal": {
                "world_or_source": self._point_stamped_dict(self.goal_pose),
                "robot_final": self._tensor_dict(self.goal_final_robot),
                "robot_clipped": self._tensor_dict(self.goal_robot),
                "model_input": self._tensor_dict(self.goal_cam),
                "planner_camera": self._tensor_dict(self.goal_cam) if self.use_camera_frame_goal else None,
                "robot_height_abs": (
                    abs(float(self.goal_robot[0][2].item())) if self.goal_robot is not None else None
                ),
                "use_camera_frame_goal": bool(self.use_camera_frame_goal),
            },
            "camera_transform": {
                "offset_robot_frame": self.cam_offset.tolist() if self.cam_offset is not None else None,
                "rotation_robotics": self.cam_rot.tolist() if self.cam_rot is not None else None,
                "depth_frame_id": self.depth_frame_id,
                "mount_cam_frame": self.mount_cam_frame,
            },
            "model": {
                "raw_keypoints": tensor_to_list(keypoints.squeeze(0)),
                "raw_keypoints_xyz": xyz_summary(keypoints),
                "fear": tensor_to_list(fear.squeeze()),
                "trajectory_camera_xyz": xyz_summary(traj),
                "published_path_robot_xyz": xyz_summary(waypoints_robot),
            },
            "path": {
                "published_point_count": int(np.asarray(waypoints_robot).reshape(-1, 3).shape[0]),
            },
        }

        if self.debug_save_input_tensors:
            npz_path = dump_dir / f"{dump_name}.npz"
            dump_dir.mkdir(parents=True, exist_ok=True)
            depth_tensor = self.planner.img_converter(self.depth_img.copy()).detach().cpu().numpy()
            npz_payload = {
                "depth_image": self.depth_img.copy(),
                "depth_tensor": depth_tensor,
                "goal_model_input": self.goal_cam.detach().cpu().numpy() if self.goal_cam is not None else np.array([]),
                "keypoints": keypoints.detach().cpu().numpy(),
                "trajectory_camera": traj.detach().cpu().numpy(),
                "trajectory_robot": np.asarray(waypoints_robot),
                "fear": fear.detach().cpu().numpy(),
            }
            if self.planner.requires_rgb and self.rgb_img is not None:
                npz_payload["rgb_or_sem_image"] = self.rgb_img.copy()
                npz_payload["rgb_or_sem_tensor"] = self.planner.sem_rgb_converter(self.rgb_img.copy()).detach().cpu().numpy()
            np.savez_compressed(npz_path, **npz_payload)
            payload["input_tensor_npz"] = str(npz_path)

        write_json(json_path, payload)

    def _transform_goal(self, point_stamped, target_frame):
        if point_stamped.header.frame_id == target_frame:
            return point_stamped
        tfm = self._lookup_transform(target_frame, point_stamped.header.frame_id)
        return do_transform_point(point_stamped, tfm)

    def _update_camera_transform(self, depth_frame_id) -> bool:
        source_frame = self.mount_cam_frame or depth_frame_id
        try:
            transform = self._lookup_transform(self.robot_id, source_frame)
        except TF_EXCEPTIONS as exc:
            rospy.logerr(f"Failed to transform camera frame {source_frame} into {self.robot_id}: {exc}")
            return False

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        self.cam_offset = np.array([translation.x, translation.y, translation.z], dtype=np.float32)
        self.cam_rot = quaternion_to_matrix(rotation.x, rotation.y, rotation.z, rotation.w) @ ROS_TO_ROBOTICS_MAT
        if not self.image_flip:
            self.cam_rot = self.cam_rot @ CAMERA_FLIP_MAT
        return True

    def _lookup_transform(self, target_frame, source_frame):
        return self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(1.0),
        )

    def publish_path(self, waypoints):
        path = RosPath()
        fear_path = RosPath()
        path.header.frame_id = self.robot_id
        fear_path.header.frame_id = self.robot_id
        path.header.stamp = self.depth_stamp or rospy.Time.now()
        fear_path.header.stamp = path.header.stamp

        if self.is_goal_init:
            for point in np.squeeze(waypoints):
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = float(point[0])
                pose.pose.position.y = float(point[1])
                pose.pose.position.z = float(point[2])
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)

        if self.fear_state.active:
            fear_path.poses = list(path.poses)
            path.poses = path.poses[:1]

        self.fear_path_pub.publish(fear_path)
        self.path_pub.publish(path)


def main():
    rospy.init_node("viplanner_node", anonymous=False)
    node = VIPlannerNode()
    node.spin()


if __name__ == "__main__":
    main()
