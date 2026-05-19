# viplanner_ros2

Standalone ROS2 package for evaluating VIPlanner models.


## What The Node Does

The `viplanner_node`:

- loads a trained VIPlanner model directory containing `model.pt` and `model.yaml`
- reads the saved training config to choose depth-only, RGB, or semantic input mode
- subscribes to depth images, goals, joystick input, and RGB images when required
- runs VIPlanner inference and trajectory interpolation
- publishes a local path, fear path, planner status, and inference timing

## Model Directory

Set `model_save` to the directory produced by training:

```text
<experiment_root>/models/<run_name>
├── model.pt
└── model.yaml
```

Both files are required. The YAML file is used to reconstruct the model architecture, so keep it with the checkpoint.

## Build

From a ROS2 workspace containing this repository:

```bash
colcon build --packages-select viplanner_ros2
source install/setup.bash
```

The package depends on standard ROS2 Python packages plus Python runtime dependencies used by VIPlanner inference, including `torch`, `torchvision`, `numpy`, `Pillow`, `opencv-python`, and `PyYAML`.

Semantic mode additionally requires an installed external semantic inference backend:

- `mmdet` for MMDetection-style config files
- `detectron2` and an installed `mask2former` package for Detectron2 YAML configs

## Configure

Default parameters are in `config/viplanner.yaml`.

Important parameters:

- `model_save`: trained VIPlanner model directory
- `depth_topic`: depth image topic
- `rgb_topic`: RGB image topic, used only for RGB or semantic models
- `rgb_compressed`: set to `true` when `rgb_topic` publishes `sensor_msgs/CompressedImage`
- `goal_topic`: `geometry_msgs/PointStamped` goal topic
- `path_topic`: output `nav_msgs/Path` topic
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
ros2 launch viplanner_ros2 viplanner.launch.py
```

Run directly with a config file:

```bash
ros2 run viplanner_ros2 viplanner_node --ros-args --params-file src/viplanner_ros2/config/viplanner.yaml
```

Override the model directory when running the node directly:

```bash
ros2 run viplanner_ros2 viplanner_node --ros-args \
  --params-file src/viplanner_ros2/config/viplanner.yaml \
  -p model_save:=/path/to/experiment/models/2026-05-19_12-00-00
```

For launch-based workflows, edit `config/viplanner.yaml` or use a project-specific launch file that passes parameter overrides to `viplanner_node`.

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
- planner status: `/viplanner/status`
- planner inference time in milliseconds: `/viplanner/timer`
- semantic inference time in milliseconds: `/viplanner/m2f_timer`
- compressed semantic visualization: `/viplanner/sem_image/compressed`

Planner status values:

- `0`: active or no terminal state
- `1`: goal reached
- `-1`: fear reaction active

## Input Modes

The node selects its input mode from `model.yaml`:

- `sem: false`, `rgb: false`: depth-only planning
- `rgb: true`: depth plus RGB image planning
- `sem: true`: depth plus semantic image planning; semantic labels are generated from RGB input with Mask2Former

RGB and semantic images are expected to be aligned with the depth stream for the current package version. The package does not perform RGB-depth image warping.

## Notes

- ONNX checkpoints are not supported; use the PyTorch `model.pt` saved by `viplanner/train.py`.
- Semantic dependencies are optional until a semantic checkpoint is used.
- For reproducible evaluation, use the same camera orientation, depth scaling, and image size assumptions used during training.
