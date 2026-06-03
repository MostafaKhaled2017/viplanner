# Important Files

## Top-Level Files

- `README.md`: Main project overview, install notes, inference/training outline, and citation.
- `TRAINING.md`: Training, reconstruction, cost-map, TensorBoard, and resume instructions.
- `configs.md`: Existing configuration reference.
- `commands.bash`: Local command scratch file with bridge, build, reconstruction, validation, training, and semantics commands.
- `pyproject.toml`: Python package metadata and dependencies.
- `Dockerfile`: Jetson/ROS Noetic-oriented Docker setup.
- `.devcontainer/devcontainer.json`: Dev container configuration.
- `.pre-commit-config.yaml`: Formatting and lint hook configuration.

## Python Package

- `viplanner/cost_builder.py`: Cost-map build CLI.
- `viplanner/depth_reconstruct.py`: Dataset depth reconstruction CLI.
- `viplanner/train.py`: Training CLI.
- `viplanner/tune_train.py`: Sweep runner.
- `viplanner/generate_semantics.py`: RGB-to-semantics generation CLI.
- `viplanner/config/costmap.yaml`: Shared reconstruction and cost-map YAML.
- `viplanner/config/train.yaml`: Training YAML.
- `viplanner/config/costmap_cfg.py`: Cost-map and reconstruction config classes.
- `viplanner/config/learning_cfg.py`: Training config classes.
- `viplanner/plannernet/PlannerNet.py`: Main planner network implementation.
- `viplanner/utils/dataset.py`: Dataset utilities.
- `viplanner/utils/trainer.py`: Training loop utilities.

## ROS Packages

- `src/planner/package.xml`: `viplanner_node` package metadata.
- `src/planner/src/viplanner_node.py`: Legacy ROS wrapper node.
- `src/planner/src/viplanner_data_collect_node.py`: ROS data collection node.
- `src/viplanner_ros1/scripts/viplanner_node.py`: Standalone ROS 1 evaluation node.
- `src/viplanner_ros1/config/viplanner.yaml`: Standalone ROS 1 planner config.
- `src/planner_validation_ros1/scripts/planner_validation_node.py`: Scenario validation node.
- `src/planner_validation_ros1/scripts/viplanner_model_sweep.py`: Model sweep runner.
- `src/pathFollower/src/pathFollower.cpp`: C++ path follower.
- `src/visualizer/src/viplannerViz.cpp`: C++ visualization node.
- `src/waypoint_rviz_plugin/src/waypoint_tool.cpp`: RViz waypoint tool.

## Tests

- `tests/test_cost_builder.py`
- `tests/test_reconstruction_cfg.py`
- `tests/test_sem_cost_map.py`
- `tests/test_viplanner_ros1_package.py`
- `tests/test_planner_validation_ros1_package.py`
- Additional tests cover training config, datasets, debug utilities, point-cloud comparison, and tuning.
