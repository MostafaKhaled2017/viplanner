# viplanner_ros1

Standalone ROS1 Noetic package for evaluating VIPlanner models.

## What The Node Does

The `viplanner_node.py` node:

- loads a trained VIPlanner model directory containing `model.pt` and `model.yaml`
- reads the saved training config to choose depth-only, RGB, or semantic input mode
- subscribes to depth images, goals, joystick input, and RGB images when required
- runs VIPlanner inference and trajectory interpolation
- publishes a local path, fear path, fear value, planner status, and inference timing

## Build

From a ROS1 Noetic catkin workspace containing this repository:

```bash
catkin_make --pkg viplanner_ros1
source devel/setup.bash
```

The package depends on standard ROS1 Python packages plus Python runtime dependencies used by VIPlanner inference, including `torch`, `torchvision`, `numpy`, `Pillow`, `opencv-python`, and `PyYAML`.

Semantic mode additionally requires an installed external semantic inference backend:

- `mmdet` for MMDetection-style config files
- `detectron2` and an installed `mask2former` package for Detectron2 YAML configs

## Configure

Default parameters are in `config/viplanner.yaml`. They intentionally match the ROS2 package names where ROS1 syntax allows it.

Important parameters:

- `model_save`: trained VIPlanner model directory
- `depth_topic`: depth image topic
- `depth_width`, `depth_height`: required depth image dimensions; the node exits on mismatch
- `rgb_topic`: RGB image topic, used only for RGB or semantic models
- `rgb_width`, `rgb_height`: required RGB image dimensions; the node exits on mismatch
- `rgb_compressed`: set to `true` when `rgb_topic` publishes `sensor_msgs/CompressedImage`
- `goal_topic`: `geometry_msgs/PointStamped` goal topic
- `path_topic`: output `nav_msgs/Path` topic
- `fear_topic`: output `viplanner_ros1/Fear` topic for the raw model fear value
- `robot_id`: robot/base frame
- `world_id`: world/odometry frame
- `mount_cam_frame`: optional mounted camera frame override
- `depth_uint_type`: set to `true` for millimeter `uint16` depth images
- `max_depth`: depth values above this distance are zeroed
- `image_flip`: rotate input depth images by 180 degrees
- `conv_dist`: goal-arrival threshold in robot-frame XY distance
- `m2f_config_path`, `m2f_model_path`, `m2f_device`: semantic model settings for `sem: true` VIPlanner checkpoints
- `is_fear_act`, `fear_threshold`, `buffer_size`, `angular_thread`, `track_dist`: fear-path behavior
- `joyGoal_scale`: scale for smart joystick goals
- `subgoal_max_distance`: maximum robot-frame XY distance passed to the network

## Launch

Launch with the packaged config:

```bash
roslaunch viplanner_ros1 viplanner.launch
```

Launch with a custom config:

```bash
roslaunch viplanner_ros1 viplanner.launch config_file:=/path/to/viplanner.yaml
```

Run directly:

```bash
rosrun viplanner_ros1 viplanner_node.py _model_save:=/path/to/experiment/models/2026-05-19_12-00-00
```

## Topics

Inputs:

- depth image: `depth_topic`
- RGB image: `rgb_topic`, only when the saved model config has `rgb: true` or `sem: true`
- goal: `goal_topic`
- joystick: `/joy`
- TF: goal, robot, world, and camera frames

Outputs:

- path: `path_topic`
- fear path: `path_topic + "_fear"`
- fear value: `fear_topic` (`viplanner_ros1/Fear`)
- planner status: `/viplanner/status`
- planner inference time in milliseconds: `/viplanner/timer`
- semantic inference time in milliseconds: `/viplanner/m2f_timer`
- compressed semantic visualization: `/viplanner/sem_image/compressed`

Planner status values:

- `0`: active or no terminal state
- `1`: goal reached
- `-1`: fear reaction active

## Notes

- ONNX checkpoints are not supported; use the PyTorch `model.pt` saved by `viplanner/train.py`.
- Semantic dependencies are optional until a semantic checkpoint is used.
- Goal and camera transforms use the latest available TF transform, matching the ROS2 package behavior.
