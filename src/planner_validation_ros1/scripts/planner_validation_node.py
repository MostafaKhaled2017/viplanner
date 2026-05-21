#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import rospy
import tf2_ros
from geometry_msgs.msg import PointStamped, Pose, Quaternion
from std_msgs.msg import Bool, Int16

from planner_validation_ros1.validation_core import (
    append_result_row,
    distance3,
    has_reached,
    init_results_csv,
    load_scenarios_txt,
    resolve_package_path,
    yaw_from_start_to_goal,
    yaw_to_quaternion_zw,
)

TF_EXCEPTIONS = (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException)


def get_private_param(name: str, default):
    return rospy.get_param(f"~{name}", default)


def resolve_package_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [script_path.parents[1]]
    if len(script_path.parents) > 2:
        candidates.append(script_path.parents[2])

    for candidate in candidates:
        if (candidate / "package.xml").is_file() and candidate.name == "planner_validation_ros1":
            return candidate

    for base_dir in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep):
        if not base_dir:
            continue
        candidate = Path(base_dir) / "planner_validation_ros1"
        if (candidate / "package.xml").is_file():
            return candidate.resolve()

    return script_path.parents[1]


class PlannerValidationNode:
    def __init__(self):
        self.package_root_dir = resolve_package_root()
        self._read_parameters()

        os.makedirs(self.log_dir, exist_ok=True)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.start_pose_pub = rospy.Publisher(self.start_pose_topic, Pose, queue_size=1, latch=True)
        self.goal_pub = rospy.Publisher(self.goal_topic, PointStamped, queue_size=1, latch=True)

        rospy.Subscriber(self.planner_status_topic, Int16, self._planner_status_callback, queue_size=10)
        rospy.Subscriber(self.collision_topic, Bool, self._collision_callback, queue_size=10)

        self._planner_status = 0
        self._trial_collided = False
        self._start_pose_repub_period_sec = 3.0
        self._last_start_pose_pub_wall = 0.0
        self._trial_start_time: Optional[float] = None
        self._trial_start_robot_pos = None
        self._validation_phase = "goto_start"
        self._scenarios = load_scenarios_txt(self.validation_paths_file, warn_fn=rospy.logwarn)
        self._sc_idx = 0
        self._completed = False

        if not self._scenarios:
            raise RuntimeError(f"No scenarios found in {self.validation_paths_file}")

        self._csv_path = init_results_csv(self.log_dir, include_debug_columns=self.debug_csv_columns)

        rospy.loginfo(f"[validation] Loaded {len(self._scenarios)} scenarios from {self.validation_paths_file}")
        rospy.loginfo(f"[validation] Results will be written to {self._csv_path}")

    def _read_parameters(self):
        self.main_freq = int(get_private_param("main_freq", 5))
        self.validation_paths_file = resolve_package_path(
            str(get_private_param("validation_paths_file", "scenarios/validation_forest.txt")),
            self.package_root_dir,
        )
        self.log_dir = resolve_package_path(str(get_private_param("log_dir", "logs")), self.package_root_dir)
        self.planner_id = str(get_private_param("planner_id", "planner"))
        self.run_id = str(get_private_param("run_id", ""))

        self.goal_topic = str(get_private_param("goal_topic", "/way_point"))
        self.start_pose_topic = str(get_private_param("start_pose_topic", "/drone_1/start_pose"))
        self.planner_status_topic = str(get_private_param("planner_status_topic", "/ip_planner_status"))
        self.collision_topic = str(get_private_param("collision_topic", "/drone_1/collision"))

        self.frame_id = str(get_private_param("robot_id", "d1_base_link"))
        self.world_id = str(get_private_param("world_id", "world"))
        self.start_yaw_offset_rad = float(get_private_param("start_yaw_offset_rad", 0.0))
        self.trial_timeout_sec = float(get_private_param("trial_timeout_sec", 30.0))
        self.start_arrival_thresh = float(get_private_param("start_arrival_thresh", 0.5))
        self.goal_arrival_thresh = float(get_private_param("goal_arrival_thresh", 0.5))
        self.debug_csv_columns = bool(get_private_param("debug_csv_columns", False))

    def spin(self):
        rate = rospy.Rate(max(1, self.main_freq))
        while not rospy.is_shutdown():
            self.tick()
            rate.sleep()

    def _planner_status_callback(self, msg):
        self._planner_status = int(msg.data)

    def _collision_callback(self, msg):
        if msg.data:
            self._trial_collided = True

    def _publish_start_pose(self, start_xyz, goal_xyz):
        sx, sy, sz = start_xyz
        yaw = yaw_from_start_to_goal(start_xyz, goal_xyz, self.start_yaw_offset_rad)
        qz, qw = yaw_to_quaternion_zw(yaw)

        pose = Pose()
        pose.position.x = sx
        pose.position.y = sy
        pose.position.z = sz
        pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)

        self.start_pose_pub.publish(pose)
        rospy.loginfo(f"[validation] Published start pose at ({sx:.2f},{sy:.2f},{sz:.2f}), yaw={yaw:.2f} rad")

    def _publish_goal_once(self, goal_xyz):
        gx, gy, gz = goal_xyz
        msg = PointStamped()
        msg.header.frame_id = self.world_id
        msg.header.stamp = rospy.Time.now()
        msg.point.x = gx
        msg.point.y = gy
        msg.point.z = gz
        self.goal_pub.publish(msg)
        rospy.loginfo(f"[validation] Published goal to {self.goal_topic}: ({gx:.2f},{gy:.2f},{gz:.2f})")

    def _robot_position_world(self):
        try:
            tfm = self.tf_buffer.lookup_transform(self.world_id, self.frame_id, rospy.Time(0), rospy.Duration(0.2))
            t = tfm.transform.translation
            return float(t.x), float(t.y), float(t.z)
        except TF_EXCEPTIONS as exc:
            rospy.logwarn(f"[validation] TF lookup failed ({self.world_id}->{self.frame_id}): {exc}")
            return None

    def _write_result_row(self, scenario, reached_goal, distance_to_goal, elapsed_sec, robot_end_pos):
        debug_values = None
        if self.debug_csv_columns:
            debug_values = {
                "start_z": float(scenario.start[2]),
                "goal_z": float(scenario.goal[2]),
                "robot_start_z": float(self._trial_start_robot_pos[2]) if self._trial_start_robot_pos else "",
                "robot_end_z": float(robot_end_pos[2]) if robot_end_pos else "",
                "final_distance": float(distance_to_goal),
                "planner_status": int(self._planner_status),
                "outcome": "reached" if reached_goal else "timeout",
            }
        append_result_row(
            self._csv_path,
            pair_id=scenario.pair_id,
            planner_id=self.planner_id,
            collided=self._trial_collided,
            reached_goal=reached_goal,
            distance_to_goal=distance_to_goal,
            elapsed_sec=elapsed_sec,
            debug_values=debug_values,
        )

    def tick(self):
        if self._completed:
            return

        if self._sc_idx >= len(self._scenarios):
            rospy.loginfo("[validation] Completed all scenarios.")
            self._completed = True
            return

        scenario = self._scenarios[self._sc_idx]
        pair_id = scenario.pair_id
        start_xyz = scenario.start
        goal_xyz = scenario.goal

        if self._validation_phase == "goto_start":
            self._publish_start_pose(start_xyz, goal_xyz)
            self._last_start_pose_pub_wall = time.time()
            self._trial_collided = False
            self._planner_status = 0
            self._trial_start_time = None
            self._trial_start_robot_pos = None
            self._validation_phase = "wait_start"
            rospy.loginfo(f"[validation] Scenario {pair_id}: heading to start...")
            return

        if self._validation_phase == "wait_start":
            now = time.time()
            if now - self._last_start_pose_pub_wall >= self._start_pose_repub_period_sec:
                self._publish_start_pose(start_xyz, goal_xyz)
                self._last_start_pose_pub_wall = now

            robot_pos = self._robot_position_world()
            if robot_pos is None:
                return

            if has_reached(robot_pos, start_xyz, self.start_arrival_thresh):
                self._planner_status = 0
                self._publish_goal_once(goal_xyz)
                self._trial_start_robot_pos = robot_pos
                self._trial_start_time = time.time()
                self._validation_phase = "run_trial"
                rospy.loginfo(
                    f"[validation] Scenario {pair_id}: start reached, running trial "
                    f"(timeout={self.trial_timeout_sec}s)"
                )
            return

        if self._validation_phase != "run_trial":
            self._validation_phase = "goto_start"
            return

        robot_pos = self._robot_position_world()
        if robot_pos is None:
            return

        elapsed = time.time() - (self._trial_start_time or time.time())
        reached_now = (self._planner_status == 1) or has_reached(robot_pos, goal_xyz, self.goal_arrival_thresh)
        timeout_now = elapsed >= self.trial_timeout_sec

        if not (reached_now or timeout_now):
            return

        dist_to_goal = distance3(robot_pos, goal_xyz)
        self._write_result_row(scenario, reached_now, dist_to_goal, elapsed, robot_pos)

        if reached_now:
            rospy.loginfo(
                f"[validation] Scenario {pair_id}: REACHED "
                f"(dist={dist_to_goal:.2f} m, t={elapsed:.1f}s, collided={self._trial_collided})"
            )
        else:
            rospy.logwarn(
                f"[validation] Scenario {pair_id}: TIMEOUT "
                f"(dist={dist_to_goal:.2f} m, t={elapsed:.1f}s, collided={self._trial_collided})"
            )

        self._sc_idx += 1
        self._validation_phase = "goto_start"
        self._trial_collided = False
        self._planner_status = 0


def main():
    rospy.init_node("planner_validation_node")
    node = PlannerValidationNode()
    node.spin()


if __name__ == "__main__":
    main()
