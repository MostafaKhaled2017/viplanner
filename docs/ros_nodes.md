# ROS Nodes

## `viplanner_node` Package (`src/planner`)

- `viplanner_node.py`: Planner inference node for generating waypoints from depth and semantic/RGB inputs.
- `viplanner_data_collect_node.py`: Dataset collection node for ROS simulators publishing depth, RGB, camera info, and TF.
- `m2f_inference.py`: Semantic inference helper script.
- `print_image_values.py`: Debug helper for image values.

Launch files:

- `src/planner/launch/viplanner.launch`
- `src/planner/launch/data_collect_sim.launch`
- `src/planner/launch/rosbag_rec_anymal_mount.launch`

## `viplanner_ros1`

- `viplanner_node.py`: Standalone ROS 1 Noetic evaluation node for trained VIPlanner checkpoints.

Launch file:

- `src/viplanner_ros1/launch/viplanner.launch`

## `planner_validation_ros1`

- `planner_validation_node.py`: Scenario-based validation node.
- `viplanner_model_sweep.py`: Runs validation over model directories.

Launch file:

- `src/planner_validation_ros1/launch/planner_validation.launch`

## `path_follower`

- `path_follower_node`: C++ executable built from `src/pathFollower/src/pathFollower.cpp`.

Launch file:

- `src/pathFollower/launch/path_follower.launch`

## `viplanner_viz`

- `viplanner_viz_node`: C++ executable built from `src/visualizer/src/viplannerViz.cpp`.
- `viplanner_viz_node_open3d`: Open3D variant source exists in `src/visualizer/src/viplannerViz_open3d.cpp`; build status depends on package CMake configuration.

Launch file:

- `src/visualizer/launch/viplannerViz.launch`

## `waypoint_rviz_plugin`

- RViz waypoint tool plugin publishing goals and joystick messages.

Plugin descriptor:

- `src/waypoint_rviz_plugin/plugin_description.xml`
