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
