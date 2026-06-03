# Architecture

## Top-Level Structure

- `viplanner/`: Python package with model, training, cost-map, reconstruction, semantics, and utility code.
- `src/`: ROS 1 catkin packages.
- `omniverse/`: Omniverse and Isaac Sim extension and standalone scripts.
- `sim/RosProBridge/`: ROS bridge helper project.
- `tests/`: Python tests for config, training, reconstruction, cost maps, package metadata, and debug utilities.
- `assets/`: Static media used by top-level documentation.
- `bin/`: Docker and ROS entrypoint scripts.

## Python Package

The `viplanner` package contains:

- `plannernet/`: Planner neural network modules.
- `traj_cost_opt/`: Trajectory cost and optimization code.
- `cost_maps/`: Semantic and TSDF cost-map builders and point-cloud export helpers.
- `utils/`: Dataset, trainer, torch, semantic inference, evaluation, and debug utilities.
- `config/`: Python config classes and YAML configs for training, cost maps, and sweeps.

## ROS 1 Packages

- `viplanner_pkgs`: Metapackage for the ROS packages.
- `viplanner_node`: ROS wrapper and data-collection package under `src/planner`.
- `viplanner_ros1`: Standalone ROS 1 evaluation node for VIPlanner checkpoints.
- `planner_validation_ros1`: Scenario-based planner validation node and model sweep utility.
- `path_follower`: C++ path follower for executing planner paths.
- `viplanner_viz`: C++ visualization nodes for depth and RGB path overlays.
- `waypoint_rviz_plugin`: RViz waypoint tool.
- `joy` and `ps3joy`: Joystick packages under `src/joystick_drivers`.

## Data Flow

Typical training flow:

1. Generate or collect depth/RGB/semantic datasets.
2. Reconstruct a point cloud with `viplanner/depth_reconstruct.py`.
3. Build a cost map with `viplanner/cost_builder.py`.
4. Train with `viplanner/train.py`.
5. Validate or run inference through ROS 1 launch files.

Typical ROS inference flow:

1. A ROS camera publishes depth, camera info, and optionally RGB.
2. A goal is published as `geometry_msgs/PointStamped`.
3. The planner node loads a `model.pt` and `model.yaml` directory.
4. The node publishes a `nav_msgs/Path`, status, timing, fear output, and optional semantic visualization.
5. `path_follower` can consume `/viplanner/path` and publish twist commands.

## External Systems

- ROS Noetic and catkin are used for runtime packages.
- NVIDIA Isaac Sim / IsaacLab is referenced for simulation and data generation.
- Mask2Former, MMDetection, Detectron2, and OpenMMLab dependencies are used for semantic inference paths when enabled.
