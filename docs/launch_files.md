# Launch Files

## Planner

`src/planner/launch/viplanner.launch`

- Args: `config_viplanner`, `use_path_follow`, `enable_visualization`, `enable_smart_joystick`.
- Optionally includes `path_follower`, `ps3joy`, and `viplanner_viz`.
- Starts `viplanner_node.py` with config from `src/planner/config/<config_viplanner>.yaml`.

## Data Collection

`src/planner/launch/data_collect_sim.launch`

- Arg: `config`.
- Starts `viplanner_data_collect_node.py` with config from `src/planner/config/<config>.yaml`.

## Rosbag Recording

`src/planner/launch/rosbag_rec_anymal_mount.launch`

- Starts `rosbag record` for VIPlanner, camera, joystick, odometry, IMU, TF, and point-cloud topics.

## Standalone ROS 1 Planner

`src/viplanner_ros1/launch/viplanner.launch`

- Arg: `config_file`.
- Starts `viplanner_ros1/scripts/viplanner_node.py`.

## Planner Validation

`src/planner_validation_ros1/launch/planner_validation.launch`

- Arg: `config_file`.
- Starts `planner_validation_node.py`.

## Path Follower

`src/pathFollower/launch/path_follower.launch`

- Starts `path_follower_node`.
- Provides topic, frame, speed, tracking, stop, and autonomy parameters.

## Visualizer

`src/visualizer/launch/viplannerViz.launch`

- Args: `config_depth`, `config_rgb`.
- Starts two `viplanner_viz_node` instances, one for depth and one for RGB visualization.
