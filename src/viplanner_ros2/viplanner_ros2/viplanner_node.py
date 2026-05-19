from __future__ import annotations

import math
import time

import cv2
import numpy as np
import torch
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, Joy
from std_msgs.msg import Float32, Int16
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener

import rclpy

from .image_utils import prepare_depth_image, rgb_msg_to_numpy
from .inference import VIPlannerInference
from .planning_utils import FearState, clip_goal_xy, fear_scalar, is_forward_tracking
from .semantic_inference import Mask2FormerPredictor

ROS_TO_ROBOTICS_MAT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
CAMERA_FLIP_MAT = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)


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


class VIPlannerNode(Node):
    def __init__(self):
        super().__init__("viplanner_node")
        self._declare_parameters()
        self._read_parameters()

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
                warn_fn=self.get_logger().warn,
            )

        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
        self.ready_for_planning = False
        self.is_goal_init = False
        self.is_goal_processed = False
        self.is_smartjoy = False
        self.planner_status = Int16(data=0)
        self.fear_state = FearState(self.buffer_size, self.fear_threshold)

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, qos)
        if self.planner.requires_rgb:
            if self.rgb_compressed:
                self.create_subscription(CompressedImage, self.rgb_topic, self.rgb_compressed_callback, qos)
            else:
                self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos)
        self.create_subscription(PointStamped, self.goal_topic, self.goal_callback, qos)
        self.create_subscription(Joy, "/joy", self.joy_callback, qos)

        self.timer_pub = self.create_publisher(Float32, "/viplanner/timer", qos)
        self.semantic_timer_pub = self.create_publisher(Float32, "/viplanner/m2f_timer", qos)
        self.status_pub = self.create_publisher(Int16, "/viplanner/status", qos)
        self.path_pub = self.create_publisher(Path, self.path_topic, qos)
        self.fear_path_pub = self.create_publisher(Path, self.path_topic + "_fear", qos)
        self.semantic_image_pub = self.create_publisher(CompressedImage, "/viplanner/sem_image/compressed", qos)

        self.create_timer(1.0 / max(1, self.main_freq), self.tick)
        self.get_logger().info("VIPlanner ROS2 Ready.")

    def _declare_parameters(self):
        defaults = {
            "main_freq": 5,
            "verbose": False,
            "model_save": "",
            "depth_topic": "/rgbd_camera/depth/image",
            "rgb_topic": "/rgbd_camera/color/image",
            "rgb_compressed": False,
            "goal_topic": "/way_point",
            "path_topic": "/viplanner/path",
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
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self):
        for name in (
            "model_save",
            "depth_topic",
            "rgb_topic",
            "goal_topic",
            "path_topic",
            "robot_id",
            "world_id",
            "mount_cam_frame",
            "m2f_config_path",
            "m2f_model_path",
            "m2f_device",
        ):
            setattr(self, name, str(self.get_parameter(name).value))
        self.mount_cam_frame = self.mount_cam_frame or None
        self.main_freq = int(self.get_parameter("main_freq").value)
        self.verbose = bool(self.get_parameter("verbose").value)
        self.rgb_compressed = bool(self.get_parameter("rgb_compressed").value)
        self.depth_uint_type = bool(self.get_parameter("depth_uint_type").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.image_flip = bool(self.get_parameter("image_flip").value)
        self.conv_dist = float(self.get_parameter("conv_dist").value)
        self.is_fear_act = bool(self.get_parameter("is_fear_act").value)
        self.fear_threshold = float(self.get_parameter("fear_threshold").value)
        self.buffer_size = int(self.get_parameter("buffer_size").value)
        self.angular_thread = float(self.get_parameter("angular_thread").value)
        self.track_dist = float(self.get_parameter("track_dist").value)
        self.joyGoal_scale = float(self.get_parameter("joyGoal_scale").value)
        self.subgoal_max_distance = float(self.get_parameter("subgoal_max_distance").value)
        self.subgoal_update_distance = float(self.get_parameter("subgoal_update_distance").value)

    def tick(self):
        if not self._has_planning_inputs():
            return

        start = time.time()
        if self.planner.requires_rgb:
            _, traj, fear = self.planner.plan(self.depth_img.copy(), self.rgb_img.copy(), self.goal_cam)
        else:
            _, traj, fear = self.planner.plan_depth(self.depth_img.copy(), self.goal_cam)
        elapsed_ms = (time.time() - start) * 1000.0
        self.timer_pub.publish(Float32(data=float(elapsed_ms)))

        waypoints = traj.detach().cpu().squeeze(0).numpy()
        if self.cam_rot is not None and self.cam_offset is not None:
            waypoints = (self.cam_rot @ waypoints.T).T + self.cam_offset

        if self._goal_reached():
            self.ready_for_planning = False
            self.is_goal_init = False
            if self.planner_status.data == 0:
                self.planner_status.data = 1
                self.status_pub.publish(self.planner_status)
            self.get_logger().info("Goal Arrived")

        if self.is_fear_act:
            fear_value = fear_scalar(fear)
            is_forward = is_forward_tracking(waypoints, self.track_dist, self.angular_thread)
            if self.fear_state.update(fear_value, is_forward):
                self.get_logger().warn("Current path prediction is invalid.")
                if self.planner_status.data == 0:
                    self.planner_status.data = -1
                    self.status_pub.publish(self.planner_status)

        self.publish_path(waypoints)

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
        self.depth_img = prepare_depth_image(msg, self.depth_uint_type, self.max_depth, self.image_flip)
        self.depth_stamp = msg.header.stamp
        self.depth_frame_id = msg.header.frame_id
        if self.is_goal_init:
            self._update_goal_tensors(msg.header.stamp, msg.header.frame_id)

    def rgb_callback(self, msg: Image):
        try:
            image = rgb_msg_to_numpy(msg)
        except RuntimeError as exc:
            self.get_logger().error(str(exc))
            return
        self._store_rgb_image(image, msg.header)

    def rgb_compressed_callback(self, msg: CompressedImage):
        rgb_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().error("Failed to decode compressed RGB image.")
            return
        self._store_rgb_image(image, msg.header)

    def _store_rgb_image(self, image, header):
        if self.semantic_predictor is not None:
            start = time.time()
            image = self.semantic_predictor.predict(image)
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
        self.get_logger().info("Received a new goal")
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
            self.get_logger().info("Switch to Smart Joystick mode ...")
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
        joy_goal.header.stamp = self.get_clock().now().to_msg()
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
            goal_robot = self._transform_goal(self.goal_pose, self.robot_id, stamp_msg)
        except TransformException as exc:
            self.get_logger().error(f"Failed to transform goal into {self.robot_id}: {exc}")
            return

        self.goal_final_robot = torch.tensor(
            [goal_robot.point.x, goal_robot.point.y, goal_robot.point.z],
            dtype=torch.float32,
        )[None, ...]

        gx, gy, gz = clip_goal_xy(
            [goal_robot.point.x, goal_robot.point.y, goal_robot.point.z],
            self.subgoal_max_distance,
        )
        self.goal_robot = torch.tensor([gx, gy, gz], dtype=torch.float32)[None, ...]

        if self.cam_rot is None or self.cam_offset is None:
            if not self._update_camera_transform(depth_frame_id, stamp_msg):
                return

        goal_cam = self.cam_rot.T @ (np.array([gx, gy, gz], dtype=np.float32) - self.cam_offset).T
        self.goal_cam = torch.tensor(goal_cam, dtype=torch.float32)[None, ...]
        self.ready_for_planning = True
        self.is_goal_processed = True

    def _transform_goal(self, point_stamped, target_frame, stamp_msg):
        if point_stamped.header.frame_id == target_frame:
            return point_stamped
        tfm = self.tf_buffer.lookup_transform(
            target_frame,
            point_stamped.header.frame_id,
            rclpy.time.Time.from_msg(stamp_msg),
            timeout=Duration(seconds=1.0),
        )
        return do_transform_point(point_stamped, tfm)

    def _update_camera_transform(self, depth_frame_id, stamp_msg) -> bool:
        source_frame = self.mount_cam_frame or depth_frame_id
        try:
            transform = self.tf_buffer.lookup_transform(
                self.robot_id,
                source_frame,
                rclpy.time.Time.from_msg(stamp_msg),
                timeout=Duration(seconds=1.0),
            )
        except TransformException as exc:
            self.get_logger().error(f"Failed to transform camera frame {source_frame} into {self.robot_id}: {exc}")
            return False

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        self.cam_offset = np.array([translation.x, translation.y, translation.z], dtype=np.float32)
        self.cam_rot = quaternion_to_matrix(rotation.x, rotation.y, rotation.z, rotation.w) @ ROS_TO_ROBOTICS_MAT
        if not self.image_flip:
            self.cam_rot = self.cam_rot @ CAMERA_FLIP_MAT
        return True

    def publish_path(self, waypoints):
        path = Path()
        fear_path = Path()
        path.header.frame_id = self.robot_id
        fear_path.header.frame_id = self.robot_id
        path.header.stamp = self.depth_stamp or self.get_clock().now().to_msg()
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
    rclpy.init()
    node = VIPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
