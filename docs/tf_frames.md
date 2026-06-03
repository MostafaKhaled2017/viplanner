# TF Frames

## Planner Frames

From `src/planner/config/anymal_mount.yaml`:

- `robot_id: base_inverted`
- `world_id: odom`
- `mount_cam_frame: wide_angle_camera_rear_camera_parent`

From `src/viplanner_ros1/config/viplanner.yaml`:

- `robot_id: d1_base_link`
- `world_id: world`
- `mount_cam_frame: ""`

## Data Collection Frames

From `src/planner/config/data_collect_sim.yaml`:

- `world_frame_id: world`
- `depth_frame_id: ""`
- `rgb_frame_id: ""`
- `rgb_pose_fallback_to_depth: true`
- `use_latest_tf: false`
- `convert_optical_to_viplanner: false`

Empty depth/RGB frame values mean the collector uses image header frame IDs.

## Validation Frames

From `src/planner_validation_ros1/config/planner_validation.yaml`:

- `robot_id: d1_base_link`
- `world_id: world`

## Visualization Frames

From `src/visualizer/config/*.yaml`:

- `robot_frame: base`
- `odom_frame: odom`

## Unknown

- Complete robot TF tree definitions are not present in this repository.
- Simulator-specific frame publishers are external to the inspected packages.
