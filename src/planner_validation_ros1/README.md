# planner_validation_ros1

ROS1 Noetic package for scenario-based planner evaluation.

## What The Node Does

The `planner_validation_node.py` node:

- reads validation scenarios from a text file
- publishes a latched start pose for each scenario
- publishes a latched scenario goal
- monitors robot pose through TF
- monitors planner status and collision state
- writes per-scenario results to a CSV file

## Build

From a ROS1 Noetic catkin workspace containing this repository:

```bash
catkin build planner_validation_ros1 --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash
```

## Configure

Default parameters are in `config/planner_validation.yaml`.

Important parameters:

- `validation_paths_file`: scenario list path, package-relative unless absolute
- `log_dir`: CSV output directory, package-relative unless absolute
- `planner_id`: planner name stored in result rows
- `run_id`: optional run label retained for compatibility
- `goal_topic`: `geometry_msgs/PointStamped` scenario goal topic
- `start_pose_topic`: `geometry_msgs/Pose` scenario start pose topic
- `planner_status_topic`: `std_msgs/Int16` planner status topic
- `collision_topic`: `std_msgs/Bool` collision indicator topic
- `robot_id`: robot TF frame
- `world_id`: world TF frame
- `trial_timeout_sec`: per-scenario timeout
- `start_arrival_thresh`: threshold for considering the robot at the start
- `goal_arrival_thresh`: threshold for considering the robot at the goal
- `shutdown_on_complete`: when `true`, shut down the node after all scenarios finish

## Scenario File Format

Each non-comment line must contain 7 whitespace-separated columns:

```text
pair_id start_x start_y start_z goal_x goal_y goal_z
```

Malformed rows with the wrong number of columns are skipped with a warning.

## Launch

Launch with the packaged config:

```bash
roslaunch planner_validation_ros1 planner_validation.launch
```

Launch with a custom config:

```bash
roslaunch planner_validation_ros1 planner_validation.launch config_file:=/path/to/planner_validation.yaml
```

Run directly:

```bash
rosrun planner_validation_ros1 planner_validation_node.py _validation_paths_file:=scenarios/validation_forest.txt
```

## Planner Interface Contract

The planner under test is expected to use:

- goal input topic: `goal_topic`
- planner status topic: `planner_status_topic`
- collision topic: `collision_topic`
- TF from `world_id` to `robot_id`

Planner status `1` means the goal was reached. Any other value means not reached yet. The validator also checks geometric goal arrival using `goal_arrival_thresh`.

## VIPlanner Model Sweep

The package includes an external sweep runner for validating every immediate child model directory under a parent directory. A valid model directory must contain both `model.pt` and `model.yaml`.

The simulator, bridge, world/map, and path follower should already be running. The runner manages only `viplanner_ros1` and `planner_validation_ros1` launches:

```bash
rosrun planner_validation_ros1 viplanner_model_sweep.py \
  --models-parent /path/to/models_parent \
  --output-dir /path/to/validation_sweeps \
  --viplanner-config /path/to/viplanner.yaml \
  --validation-config /path/to/planner_validation.yaml
```

For each model, the runner writes temporary per-run configs, sets VIPlanner `model_save`, sets validation `planner_id` and `log_dir`, and forces `planner_status_topic` to `/viplanner/status`.

Sweep outputs are written under a timestamped directory:

- `sweep_trials.csv`: combined per-scenario rows with `model_name` and `model_path`
- `sweep_summary.csv`: per-model success, collision, timing, distance, and failure status
