# ROS Topics

## Planner Inputs

From `src/planner/config/anymal_mount.yaml`:

- `/depth_cam_mounted_front/depth/image_rect_raw`: Depth image input.
- `/depth_cam_mounted_front/depth/camera_info`: Depth camera info.
- `/depth_cam_mounted_front/color/image_raw/compressed`: RGB image input.
- `/depth_cam_mounted_front/color/camera_info`: RGB camera info.
- `/mp_waypoint`: Goal input.
- `/joy`: Joystick input.

From `src/viplanner_ros1/config/viplanner.yaml`:

- `/drone_1/depth`: Depth image input.
- `/drone_1/cam/compressed`: RGB image input.
- `/way_point`: Goal input.
- `/joy`: Joystick input.

## Planner Outputs

From planner code and configs:

- `/viplanner/path`: Default path topic in `src/planner/config/anymal_mount.yaml`.
- `/drone_1/path_follow`: Path topic in `src/viplanner_ros1/config/viplanner.yaml`.
- `<path_topic>_viz`: Visualization path from `src/planner/src/viplanner_node.py`.
- `<path_topic>_fear`: Fear path.
- `/viplanner/fear`: Fear value in `viplanner_ros1`.
- `/viplanner/status`: Planner status.
- `/viplanner/timer`: Planner inference timing.
- `/viplanner/m2f_timer`: Semantic inference timing.
- `/viplanner/sem_image/compressed`: Compressed semantic visualization.
- `viplanner/visualization/crop_goal`: Crop goal marker input/output topic from planner node.
- `viplanner/visualization/odom_circle`: RViz marker.
- `viplanner/visualization/goal_line`: RViz marker.

## Data Collection Topics

From `src/planner/config/data_collect_sim.yaml`:

- `/drone_1/depth`
- `/drone_1/cam/compressed`
- `/drone_1/depth/camera_info`
- `/drone_1/cam/camera_info`
- `/tf`
- `/tf_static`

## Validation Topics

From `src/planner_validation_ros1/config/planner_validation.yaml`:

- `/way_point`: Goal topic published by validator.
- `/drone_1/start_pose`: Start pose published by validator.
- `/viplanner/status`: Planner status subscribed by validator.
- `/drone_1/collision`: Collision indicator subscribed by validator.

## Path Follower Topics

From `src/pathFollower/launch/path_follower.launch` and source:

- `/state_estimator/pose_in_odom`: Default odometry/pose input.
- `/viplanner/path`: Path input.
- `/joy`: Joystick input.
- `/speed`: Speed input.
- `/stop`: Stop input.
- `/local_guidance_path_follower/twist`: Default twist command output.

## Visualization Topics

From `src/visualizer/config/*.yaml`:

- `/viplanner/viz_path_depth`: Depth visualization output.
- `/viplanner/viz_path_rgb`: RGB visualization output.
- `/viplanner/path`: Path input.
- `/mp_waypoint`: Goal input.
- Camera image and camera-info topics vary by ANYmal C/D depth/RGB config.

## Waypoint RViz Plugin Topics

From `src/waypoint_rviz_plugin/src/waypoint_tool.cpp`:

- `/mp_waypoint`: Published waypoint.
- `/joy`: Published joystick message.
- `/state_estimator/pose_in_odom`: Default subscribed odometry topic unless overridden by plugin property.
