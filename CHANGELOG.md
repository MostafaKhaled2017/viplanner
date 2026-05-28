## 2026-05-28 07:26 - Add configurable VIPlanner encoder freezing

**Change size**
- `M`

**Files changed**
- `viplanner/config/learning_cfg.py`
- `viplanner/config/train.yaml`
- `viplanner/utils/trainer.py`
- `src/viplanner_ros1/src/viplanner_ros1/learning_cfg.py`
- `tests/test_training_freeze_layers.py`
- `README.md`
- `TRAINING.md`
- `CHANGELOG.md`

**What changed**
- Added `freeze_layers` to the training config with a default of `0`, preserving full-model training by default.
- Added trainer logic that validates `freeze_layers` in `[0, 5]` and freezes progressively earlier `PlannerNet` encoder stages while leaving the decoder trainable.
- Updated optimizer construction to include only trainable parameters after freezing.
- Added ROS1 training-config compatibility so saved model configs containing `freeze_layers` can still be loaded by inference.
- Added focused unit tests for config parsing, single- and dual-stream encoder freezing, invalid values, and optimizer parameter filtering.
- Documented `freeze_layers` in the training docs and top-level training overview.

**Context**
- Fine-tuning VIPlanner on small new-simulator datasets needs a config-driven way to freeze stable visual encoder stages without editing training code.
- The feature is limited to `PlannerNet` encoders; pretrained RGB/Mask2Former encoder freezing remains controlled by `pre_train_freeze`.

**Validation**
- `pytest -q tests/test_training_freeze_layers.py`
- `pytest -q tests/test_training_freeze_layers.py tests/test_training_run_directory.py tests/test_training_early_stopping.py tests/test_viplanner_ros1_package.py`
- `git diff --check -- CHANGELOG.md README.md TRAINING.md viplanner/config/learning_cfg.py viplanner/config/train.yaml viplanner/utils/trainer.py src/viplanner_ros1/src/viplanner_ros1/learning_cfg.py tests/test_training_freeze_layers.py`

**Notes**
- `freeze_layers: 4` freezes `conv1` through `layer3`; `freeze_layers: 5` freezes the full `PlannerNet` encoder.
- The focused test runs passed with existing environment warnings about CUDA availability and pytest cache permissions.

## 2026-05-21 11:36 - Add external VIPlanner model sweep runner

**Change size**
- `M`

**Files changed**
- `src/planner_validation_ros1/CMakeLists.txt`
- `src/planner_validation_ros1/README.md`
- `src/planner_validation_ros1/config/planner_validation.yaml`
- `src/planner_validation_ros1/package.xml`
- `src/planner_validation_ros1/scripts/planner_validation_node.py`
- `src/planner_validation_ros1/scripts/viplanner_model_sweep.py`
- `src/planner_validation_ros1/src/planner_validation_ros1/validation_core.py`
- `src/viplanner_ros1/scripts/viplanner_node.py`
- `tests/test_planner_validation_ros1_package.py`
- `commands.bash`
- `CHANGELOG.md`

**What changed**
- Added an external ROS1 sweep runner that discovers immediate child VIPlanner model directories containing `model.pt` and `model.yaml`, launches `viplanner_ros1` and `planner_validation_ros1` once per model, and writes combined trial and summary CSV reports.
- Added reusable validation helpers for model discovery, sweep config overrides, validation CSV merging, and per-model summary metrics.
- Added `shutdown_on_complete`, defaulting to `false`, so sweep-generated validation configs can make the validator exit after all scenarios finish.
- Installed the sweep runner through catkin, declared the YAML runtime dependency, documented the sweep workflow, and added a command example.
- Added focused unit tests for model discovery, sweep config generation, CSV merge behavior, summary metrics, summary output columns, and runner installation.
- Handled normal ROS shutdown interrupts in the validation and VIPlanner spin loops so completed sweep runs do not print Python tracebacks during intentional shutdown.

**Context**
- Batch validation was needed for comparing all VIPlanner checkpoint subdirectories under one parent without adding a runtime model-reload API or moving scenario orchestration into the planner node.
- The runner assumes the simulator, bridge, world/map, and path follower are already running and manages only the VIPlanner and validation launches.

**Validation**
- `PYTHONPYCACHEPREFIX=/tmp/viplanner_pycache python3 -m py_compile src/planner_validation_ros1/scripts/viplanner_model_sweep.py src/planner_validation_ros1/scripts/planner_validation_node.py src/planner_validation_ros1/src/planner_validation_ros1/validation_core.py tests/test_planner_validation_ros1_package.py`
- `python3 -m unittest tests.test_planner_validation_ros1_package`
- `python3 src/planner_validation_ros1/scripts/viplanner_model_sweep.py --help`
- `PYTHONPYCACHEPREFIX=/tmp/viplanner_pycache python3 -m py_compile src/planner_validation_ros1/scripts/planner_validation_node.py src/viplanner_ros1/scripts/viplanner_node.py src/planner_validation_ros1/scripts/viplanner_model_sweep.py`
- `git diff --check -- src/planner_validation_ros1 tests/test_planner_validation_ros1_package.py commands.bash`
- `catkin build planner_validation_ros1 viplanner_ros1 --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5`
- `python3 -m unittest tests.test_planner_validation_ros1_package tests.test_viplanner_ros1_package` was attempted and failed only because `src/viplanner_ros1/config/viplanner.yaml` currently has `debug_enabled: true` while the existing VIPlanner test expects the repository default `false`.

**Notes**
- Sweep output is written under a timestamped directory containing per-model generated configs, validation logs, `sweep_trials.csv`, and `sweep_summary.csv`.
- The catkin build succeeded with CMake/catkin/googletest deprecation and developer warnings from the Noetic environment.

## 2026-05-21 10:15 - Add VIPlanner training and ROS1 debug dumps

**Change size**
- `M`

**Files changed**
- `viplanner/debug_training_pipeline.py`
- `viplanner/utils/debug_summary.py`
- `src/viplanner_ros1/config/viplanner.yaml`
- `src/viplanner_ros1/scripts/viplanner_node.py`
- `src/viplanner_ros1/src/viplanner_ros1/debug_utils.py`
- `src/planner_validation_ros1/config/planner_validation.yaml`
- `src/planner_validation_ros1/scripts/planner_validation_node.py`
- `src/planner_validation_ros1/src/planner_validation_ros1/validation_core.py`
- `tests/test_debug_pipeline.py`
- `tests/test_viplanner_ros1_package.py`
- `tests/test_planner_validation_ros1_package.py`
- `CHANGELOG.md`

**What changed**
- Added an opt-in `viplanner.debug_training_pipeline` script that rebuilds the configured training/test data path, runs a trained checkpoint, and writes JSON/optional NPZ dumps containing model inputs, odometry, goals, raw keypoints, generated waypoints, fear output, z summaries, and per-sample loss components.
- Added shared training debug summary helpers for tensor statistics, xyz summaries, JSON-safe conversion, and augmented-sample loss-frame mirroring.
- Fixed the xyz summary helper to move torch tensors to CPU before NumPy conversion, allowing the debug script to summarize CUDA model outputs.
- Added ROS1 VIPlanner debug parameters, defaulted off, for runtime JSON/optional NPZ dumps of depth statistics, goal transforms, camera transform, raw model outputs, generated trajectories, published path z ranges, and body-frame goal height.
- Fixed the ROS1 debug dump writer to use separate aliases for filesystem paths and `nav_msgs/Path`, preventing debug mode from constructing a ROS path message when opening the dump directory.
- Updated the ROS1 path publishers to use the same `RosPath` alias so planner startup no longer references the removed bare `Path` name.
- Added `use_camera_frame_goal`, defaulting to `false`, so the ROS1 planner feeds body-frame goals to the model and publishes body-frame waypoints directly by default. This matches the depth-only training/debug data and avoids rotating forward distance into the model z axis.
- Added optional planner-validation CSV debug columns for start/goal z, robot start/end z, final distance, planner status, and reached/timeout outcome.
- Added focused unit tests for debug helpers, ROS1 debug config defaults, goal z preservation, and optional validation CSV debug columns.

**Context**
- Training and ROS1 validation needed inspectable artifacts to determine whether bad vertical path behavior comes from dataset labels, preprocessing, model outputs, trajectory interpolation, loss terms, or TF/body-frame height handling.
- Debug output is opt-in so normal training, inference, and validation behavior remains unchanged unless debug parameters or script flags are enabled.

**Validation**
- `PYTHONPYCACHEPREFIX=/tmp/viplanner_pycache python3 -m py_compile viplanner/debug_training_pipeline.py viplanner/utils/debug_summary.py src/viplanner_ros1/scripts/viplanner_node.py src/viplanner_ros1/src/viplanner_ros1/debug_utils.py src/planner_validation_ros1/scripts/planner_validation_node.py src/planner_validation_ros1/src/planner_validation_ros1/validation_core.py tests/test_debug_pipeline.py tests/test_viplanner_ros1_package.py tests/test_planner_validation_ros1_package.py`
- `python3 -m unittest tests.test_debug_pipeline tests.test_viplanner_ros1_package tests.test_planner_validation_ros1_package`
- `python3 -m unittest tests.test_viplanner_ros1_package`
- `python3 -m unittest tests.test_planner_validation_ros1_package`
- `PYTHONPYCACHEPREFIX=/tmp/viplanner_pycache python3 -m viplanner.debug_training_pipeline --model-dir src/viplanner_ros1/models/2026-05-19_07-36-47 --split test --num-samples 1 --output-dir /tmp/viplanner_debug_training`
- `PYTHONPYCACHEPREFIX=/tmp/viplanner_pycache python3 -m viplanner.debug_training_pipeline --model-dir src/viplanner_ros1/models/2026-05-19_07-36-47 --split test --num-samples 1 --output-dir /tmp/viplanner_debug_training_fix`
- `PYTHONPYCACHEPREFIX=/tmp/viplanner_pycache python3 -m py_compile src/viplanner_ros1/scripts/viplanner_node.py`
- `python3 -m unittest tests.test_viplanner_ros1_package` was rerun after enabling local runtime debug and failed only because `src/viplanner_ros1/config/viplanner.yaml` currently has `debug_enabled: true` instead of the repository default `false`.
- `PYTHONPYCACHEPREFIX=/tmp/viplanner_pycache python3 -m py_compile src/viplanner_ros1/scripts/viplanner_node.py`

**Notes**
- The real-model smoke dump used one sample to verify the exact pipeline quickly; larger sweeps can use `--num-samples 32` or more.
- The smoke dump ran on CPU because CUDA was unavailable in the current environment.
- The first smoke-dump sample had `goal_z_abs: 0.0`; raw keypoint z values were near zero for that sample.

## 2026-05-21 08:35 - Add ROS1 planner validation package

**Change size**
- `M`

**Files changed**
- `src/planner_validation_ros1/CMakeLists.txt`
- `src/planner_validation_ros1/README.md`
- `src/planner_validation_ros1/config/planner_validation.yaml`
- `src/planner_validation_ros1/launch/planner_validation.launch`
- `src/planner_validation_ros1/package.xml`
- `src/planner_validation_ros1/scenarios/validation_forest.txt`
- `src/planner_validation_ros1/scenarios/validation_forest_5.txt`
- `src/planner_validation_ros1/scripts/planner_validation_node.py`
- `src/planner_validation_ros1/setup.py`
- `src/planner_validation_ros1/src/planner_validation_ros1/__init__.py`
- `src/planner_validation_ros1/src/planner_validation_ros1/validation_core.py`
- `tests/test_planner_validation_ros1_package.py`
- `CHANGELOG.md`

**What changed**
- Added a new `planner_validation_ros1` catkin package that ports the reference planner validation node from ROS2 `rclpy` to ROS1 `rospy`.
- Added a ROS1 launch file, flat rosparam YAML, package metadata, catkin Python setup, copied scenario files, and package README documentation.
- Preserved the validation state machine for start-pose publishing, goal publishing, TF-based arrival checks, planner status monitoring, collision tracking, timeout handling, and CSV result output.
- Added package-local testable helpers for scenario parsing, distance checks, start yaw conversion, quaternion conversion, relative path resolution, and CSV writing.
- Added focused unit tests for scenario parsing, helper math, ROS1 config shape, launch target metadata, and absence of ROS2-only imports.

**Context**
- A ROS1 version of `ref/planner_validation` was needed under the workspace `src` directory without changing the reference package or the existing planner packages.
- The package name was set to `planner_validation_ros1` to make the ROS version explicit.

**Validation**
- `python3 -m unittest tests.test_planner_validation_ros1_package`
- `python3 -m py_compile src/planner_validation_ros1/scripts/planner_validation_node.py src/planner_validation_ros1/src/planner_validation_ros1/__init__.py src/planner_validation_ros1/src/planner_validation_ros1/validation_core.py tests/test_planner_validation_ros1_package.py`
- `git diff --check -- src/planner_validation_ros1 tests/test_planner_validation_ros1_package.py`
- `catkin build planner_validation_ros1 --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5`

**Notes**
- Historical CSV logs and ROS2 resource files from `ref/planner_validation` were intentionally not copied.
- The catkin build succeeded with CMake/catkin/googletest deprecation and developer warnings from the Noetic environment.

## 2026-05-21 08:16 - Add standalone ROS1 Noetic VIPlanner package

**Change size**
- `L`

**Files changed**
- `src/viplanner_ros1/CMakeLists.txt`
- `src/viplanner_ros1/README.md`
- `src/viplanner_ros1/config/viplanner.yaml`
- `src/viplanner_ros1/launch/viplanner.launch`
- `src/viplanner_ros1/package.xml`
- `src/viplanner_ros1/scripts/viplanner_node.py`
- `src/viplanner_ros1/setup.py`
- `src/viplanner_ros1/src/viplanner_ros1/__init__.py`
- `src/viplanner_ros1/src/viplanner_ros1/autoencoder.py`
- `src/viplanner_ros1/src/viplanner_ros1/image_utils.py`
- `src/viplanner_ros1/src/viplanner_ros1/inference.py`
- `src/viplanner_ros1/src/viplanner_ros1/learning_cfg.py`
- `src/viplanner_ros1/src/viplanner_ros1/planner_net.py`
- `src/viplanner_ros1/src/viplanner_ros1/planning_utils.py`
- `src/viplanner_ros1/src/viplanner_ros1/rgb_encoder.py`
- `src/viplanner_ros1/src/viplanner_ros1/semantic_inference.py`
- `src/viplanner_ros1/src/viplanner_ros1/semantic_meta.py`
- `src/viplanner_ros1/src/viplanner_ros1/traj_opt.py`
- `tests/test_viplanner_ros1_package.py`
- `CHANGELOG.md`

**What changed**
- Added a new `viplanner_ros1` catkin package for standalone ROS1 Noetic evaluation of VIPlanner checkpoints.
- Copied the package-local ROS2 inference, model, image, planning, trajectory, and semantic runtime modules into the ROS1 package with ROS1 package imports.
- Added a `rospy` node wrapper that preserves the ROS2 package behavior for depth-only, RGB, semantic, goal, joystick, latest-TF, fear-path, timing, status, and semantic visualization flows.
- Added ROS1 rosparam YAML, roslaunch entrypoint, package metadata, catkin Python setup, and package README documentation.
- Added focused unit tests for model directory validation, checkpoint extraction, config parsing, image conversion, dimension validation, fear/goal helpers, and import isolation.

**Context**
- A ROS1 Noetic version of `src/viplanner_ros2` was needed without replacing the existing ROS2 package or the older `src/planner` ROS1 package.
- The target behavior is parity with the standalone ROS2 package, not expansion of legacy ROS1-only visualization or RGB-depth warping behavior.

**Validation**
- `python3 -m unittest tests.test_viplanner_ros1_package`
- `python3 -m py_compile src/viplanner_ros1/scripts/viplanner_node.py src/viplanner_ros1/src/viplanner_ros1/__init__.py src/viplanner_ros1/src/viplanner_ros1/autoencoder.py src/viplanner_ros1/src/viplanner_ros1/image_utils.py src/viplanner_ros1/src/viplanner_ros1/inference.py src/viplanner_ros1/src/viplanner_ros1/learning_cfg.py src/viplanner_ros1/src/viplanner_ros1/planner_net.py src/viplanner_ros1/src/viplanner_ros1/planning_utils.py src/viplanner_ros1/src/viplanner_ros1/rgb_encoder.py src/viplanner_ros1/src/viplanner_ros1/semantic_inference.py src/viplanner_ros1/src/viplanner_ros1/semantic_meta.py src/viplanner_ros1/src/viplanner_ros1/traj_opt.py tests/test_viplanner_ros1_package.py`
- `git diff --check -- src/viplanner_ros1 tests/test_viplanner_ros1_package.py CHANGELOG.md`
- `catkin_make --pkg viplanner_ros1` was attempted but this workspace was previously built by `catkin build`, so `catkin_make` refused to use the existing build space.
- `catkin build viplanner_ros1` was attempted and reached the package configure step, but the environment's newer CMake rejected the system `/usr/src/googletest` CMake minimum.
- `catkin build viplanner_ros1 --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5` succeeded.

**Notes**
- Legacy-only ROS1 features from `src/planner`, including `_viz` world-frame paths, crop-goal markers, camera-info subscribers, and RGB-depth warping, are intentionally out of scope.
- The successful catkin build emitted CMake deprecation and developer warnings from Noetic/catkin/googletest compatibility with the installed CMake version.

## 2026-05-20 07:14 - Align ROS2 VIPlanner TF lookup timing with reference package

**Change size**
- `S`

**Files changed**
- `src/viplanner_ros2/README.md`
- `src/viplanner_ros2/config/viplanner.yaml`
- `src/viplanner_ros2/viplanner_ros2/planning_utils.py`
- `src/viplanner_ros2/viplanner_ros2/viplanner_node.py`
- `tests/test_viplanner_ros2_package.py`
- `CHANGELOG.md`

**What changed**
- Enabled `use_sim_time` in the ROS2 VIPlanner package config so the node uses the simulator clock when `/clock` is available.
- Passed the node clock into the TF buffer when supported so TF cache behavior follows the node's configured time source.
- Switched goal and camera-frame transforms to use the latest available TF transform, matching `ref/iplanner` and avoiding repeated future-extrapolation warnings when image stamps run slightly ahead of buffered TF data.
- Added small ROS timestamp helper functions and focused unit coverage for their comparison behavior.
- Documented the latest-transform policy in the ROS2 package README.

**Context**
- Depth image timestamps and `world -> d1_base_link` TF timestamps can arrive in different time states during simulation, causing exact-time goal transforms to fail with future extrapolation before inference.
- A guarded fallback allowed inference to continue, but it still produced throttled warnings whenever the stamped lookup missed the newest TF sample.
- The reference iPlanner ROS2 package uses latest available transforms for these operations, which is more tolerant of simulator and callback scheduling jitter while keeping the path in the robot/body frame.

**Validation**
- `python3 -m unittest tests.test_viplanner_ros2_package`
- `python3 -m py_compile src/viplanner_ros2/viplanner_ros2/viplanner_node.py src/viplanner_ros2/viplanner_ros2/planning_utils.py tests/test_viplanner_ros2_package.py`
- `git diff --check -- src/viplanner_ros2/viplanner_ros2/viplanner_node.py src/viplanner_ros2/README.md CHANGELOG.md`

**Notes**
- TF lookup failures other than unavailable transforms still surface as errors from the underlying TF buffer.
- ROS2 distributions whose Python TF buffer does not accept a node argument fall back to the previous buffer construction path.

## 2026-05-19 21:25 - Add YAML-driven training tuning launcher

**Change size**
- `M`

**Files changed**
- `viplanner/tune_train.py`
- `viplanner/config/sweep_depth_geom.yaml`
- `tests/test_tune_train.py`
- `CHANGELOG.md`

**What changed**
- Added a YAML-driven training sweep launcher that expands parameter grids, writes per-trial training configs, runs `viplanner/train.py`, and summarizes results.
- Added GPU-pool scheduling with user-provided GPU IDs, a maximum parallel run limit, and one active training process per GPU.
- Added deterministic `max_trials` subsampling based on the sweep seed.
- Added summary output in both YAML and CSV formats, including trial parameters, generated config path, model path, return code, validation loss, and test loss.
- Added support for dry runs that generate trial configs and summaries without starting training.
- Added a sample depth/geometric-cost sweep YAML covering learning rate, batch size, loss weights, and goal-distance sampling.
- Added focused unit tests for sweep loading, grid expansion, deterministic trial limiting, config generation, GPU scheduling, and result parsing.

**Context**
- Training quality tuning needs to run many parameter combinations without manually editing config files and launching each training run.
- The sweep space should be defined in YAML while keeping the training script itself unchanged.

**Validation**
- `python3 -m unittest tests.test_tune_train`
- `python3 -m py_compile viplanner/tune_train.py tests/test_tune_train.py`
- `python3 viplanner/tune_train.py --help`
- Dry-run validation with temporary base and sweep YAML files using `--gpus 0,1 --max-parallel 2 --max-trials 3 --dry-run`.
- Parsed `viplanner/config/sweep_depth_geom.yaml` and verified that it expands to 32 trials.

**Notes**
- No Optuna, Ray Tune, or W&B Sweeps dependency was added.
- Grid expansion is the only supported sweep format in this version.

## 2026-05-19 21:07 - Add standalone ROS2 VIPlanner evaluation package

**Change size**
- `L`

**Files changed**
- `src/viplanner_ros2/package.xml`
- `src/viplanner_ros2/README.md`
- `src/viplanner_ros2/setup.py`
- `src/viplanner_ros2/setup.cfg`
- `src/viplanner_ros2/resource/viplanner_ros2`
- `src/viplanner_ros2/config/viplanner.yaml`
- `src/viplanner_ros2/launch/viplanner.launch.py`
- `src/viplanner_ros2/viplanner_ros2/__init__.py`
- `src/viplanner_ros2/viplanner_ros2/learning_cfg.py`
- `src/viplanner_ros2/viplanner_ros2/planner_net.py`
- `src/viplanner_ros2/viplanner_ros2/rgb_encoder.py`
- `src/viplanner_ros2/viplanner_ros2/autoencoder.py`
- `src/viplanner_ros2/viplanner_ros2/traj_opt.py`
- `src/viplanner_ros2/viplanner_ros2/image_utils.py`
- `src/viplanner_ros2/viplanner_ros2/planning_utils.py`
- `src/viplanner_ros2/viplanner_ros2/inference.py`
- `src/viplanner_ros2/viplanner_ros2/semantic_meta.py`
- `src/viplanner_ros2/viplanner_ros2/semantic_inference.py`
- `src/viplanner_ros2/viplanner_ros2/viplanner_node.py`
- `tests/test_viplanner_ros2_package.py`
- `CHANGELOG.md`

**What changed**
- Added a standalone ROS2 `ament_python` package for evaluating VIPlanner checkpoints from a trained model directory containing `model.pt` and `model.yaml`.
- Vendored the minimal model config, network, trajectory generation, image conversion, semantic mapping, and inference runtime needed by the ROS2 package.
- Added a ROS2 node that follows the iPlanner-style runtime loop while adapting goal projection, model loading, and input selection to VIPlanner depth, RGB, and semantic model configurations.
- Added launch and YAML parameter files for the new `viplanner_ros2` package.
- Added package-local README documentation covering model directory layout, build and launch usage, parameters, topics, input modes, and semantic dependency notes.
- Fixed ROS2 RGB preprocessing to keep RGB channel order and preserve float normalization before tensor conversion, matching the training data path.
- Added required depth and RGB image dimension parameters, with runtime failures when simulator image dimensions do not match the configured sizes.
- Added focused tests for model directory validation, config parsing, checkpoint extraction, depth conversion, fear/path helper logic, and import isolation.

**Context**
- The ROS2 evaluation package needs to behave like `ref/iplanner` while remaining independent from other repository packages.
- The package must not import local `viplanner`, `src/planner`, or `ref/iplanner` modules so it can be copied or built independently.

**Validation**
- `python -m unittest tests.test_viplanner_ros2_package`
- `python -m py_compile src/viplanner_ros2/viplanner_ros2/learning_cfg.py src/viplanner_ros2/viplanner_ros2/planner_net.py src/viplanner_ros2/viplanner_ros2/rgb_encoder.py src/viplanner_ros2/viplanner_ros2/autoencoder.py src/viplanner_ros2/viplanner_ros2/traj_opt.py src/viplanner_ros2/viplanner_ros2/image_utils.py src/viplanner_ros2/viplanner_ros2/planning_utils.py src/viplanner_ros2/viplanner_ros2/inference.py src/viplanner_ros2/viplanner_ros2/semantic_meta.py src/viplanner_ros2/viplanner_ros2/semantic_inference.py tests/test_viplanner_ros2_package.py`
- `python -m py_compile src/viplanner_ros2/viplanner_ros2/viplanner_node.py`
- `python -m pytest tests`
- `git diff --check -- src/viplanner_ros2 tests/test_viplanner_ros2_package.py CHANGELOG.md`
- `git diff --check -- src/viplanner_ros2/README.md CHANGELOG.md`
- `colcon build --packages-select viplanner_ros2` was not run because `colcon` is not installed in the current environment.

**Notes**
- ONNX support is intentionally out of scope for this package version.
- Semantic inference depends on external `mmdet` or `detectron2`/`mask2former` installations only when a semantic model is used.

## 2026-05-19 06:40 - Replace training wandb logging with TensorBoard

**Change size**
- `M`

**Files changed**
- `viplanner/config/learning_cfg.py`
- `viplanner/config/train.yaml`
- `viplanner/utils/trainer.py`
- `viplanner/traj_cost_opt/traj_cost.py`
- `TRAINING.md`
- `tests/test_training_early_stopping.py`
- `tests/test_training_run_directory.py`
- `pyproject.toml`
- `.devcontainer/Dockerfile`
- `README.md`
- `CHANGELOG.md`

**What changed**
- Replaced the training-path `wandb` dependency and metric calls with `torch.utils.tensorboard.SummaryWriter`.
- Added per-run TensorBoard event directories under `logs/YYYY-MM-DD_HH-MM-SS`.
- Routed trainer epoch losses, plus averaged trajectory cost loss components, to TensorBoard scalar logs.
- Grouped TensorBoard scalar tags under `train/...` and `val/...`, and flushed the writer after each epoch.
- Made train and validation scalar steps monotonic across environments to avoid repeated TensorBoard x-axis steps.
- Changed TensorBoard logging to write training and validation metrics only at epoch end, with total loss logged before component losses.
- Added pre-BCELoss validation for non-finite fear predictions, final loss validation before backward, and clamped valid probabilities to avoid CUDA device-side asserts.
- Removed unused `wb_project`, `wb_entity`, and `wb_api_key` training config fields while preserving compatibility with YAML files that still contain them.
- Updated the default training YAML, training dependency extra, documentation, and focused tests for TensorBoard logging.
- Added TensorBoard troubleshooting documentation for protobuf 5.x `MessageToJson` failures.
- Installed the training extra in the dev container and repinned `protobuf<5` after third-party package installs.

**Context**
- Training metrics should be logged without requiring a Weights & Biases account or API key.
- TensorBoard keeps metrics local to the existing timestamped experiment directory layout.

**Validation**
- `python -m unittest tests.test_training_early_stopping tests.test_training_run_directory`
- `python -m py_compile viplanner/train.py viplanner/config/learning_cfg.py viplanner/utils/trainer.py viplanner/traj_cost_opt/traj_cost.py`
- `python viplanner/train.py --help`
- `git diff --check -- viplanner/config/learning_cfg.py viplanner/config/train.yaml viplanner/utils/trainer.py viplanner/traj_cost_opt/traj_cost.py viplanner/train.py TRAINING.md README.md .devcontainer/Dockerfile tests/test_training_early_stopping.py tests/test_training_run_directory.py pyproject.toml CHANGELOG.md`
- `python3 -m pip show tensorboard protobuf`

**Notes**
- TensorBoard graph tracing is intentionally not added because the model has multiple input signatures depending on training configuration.
- The training extra pins `protobuf<5` because TensorBoard 2.14.x's HParams plugin is incompatible with protobuf 5.x.

## 2026-05-19 06:31 - Add independent early stopping patience

**Change size**
- `M`

**Files changed**
- `viplanner/config/learning_cfg.py`
- `viplanner/config/train.yaml`
- `viplanner/utils/trainer.py`
- `tests/test_training_early_stopping.py`
- `CHANGELOG.md`

**What changed**
- Added `early_stop_patience` to the training configuration with a default value of `10`.
- Added `early_stop_patience: 10` to the default training YAML.
- Decoupled early stopping from the ReduceLROnPlateau scheduler by tracking consecutive validation epochs without improvement in the trainer loop.
- Kept the existing `patience` field dedicated to learning-rate scheduling.
- Added focused tests for the new configuration field and lightweight early-stopping loop behavior.

**Context**
- The existing `patience` parameter controlled the learning-rate scheduler, while early stopping was implicitly tied to scheduler behavior and did not stop training reliably.
- Training now has an explicit stop threshold that is independent from learning-rate reductions.

**Validation**
- `python -m py_compile viplanner/config/learning_cfg.py viplanner/utils/trainer.py tests/test_training_early_stopping.py`
- `python -c "from viplanner.config import TrainCfg; cfg = TrainCfg.from_yaml('viplanner/config/train.yaml'); print(cfg.early_stop_patience, cfg.patience)"`
- `python -m unittest tests.test_training_early_stopping` (2 passed, 1 skipped because importing `Trainer` is currently blocked by an existing `IndentationError` in `viplanner/traj_cost_opt/traj_cost.py`)
- `python -m unittest tests.test_training_run_directory` failed before running tests because importing `Trainer` is currently blocked by the same `IndentationError` in `viplanner/traj_cost_opt/traj_cost.py`
- `git diff --check -- viplanner/config/learning_cfg.py viplanner/config/train.yaml viplanner/utils/trainer.py tests/test_training_early_stopping.py CHANGELOG.md`

**Notes**
- Full trainer-loop validation should be rerun after the existing `viplanner/traj_cost_opt/traj_cost.py` indentation issue is resolved.

## 2026-05-19 05:59 - Use timestamped training run directories

**Change size**
- `M`

**Files changed**
- `viplanner/config/learning_cfg.py`
- `viplanner/utils/trainer.py`
- `viplanner/train.py`
- `TRAINING.md`
- `tests/test_training_run_directory.py`
- `CHANGELOG.md`

**What changed**
- Replaced training-configuration-derived model directory names with local timestamp run directory names.
- Added collision suffix handling for timestamped model directories.
- Added `model_dir_name` and `resume_model_path` training config fields.
- Wrote `model.yaml` at trainer initialization with the effective training config, then updated it with validation and test losses at the end.
- Added `--resume-from` support for checkpoint files or model directories while continuing to save outputs in a new timestamped run directory.
- Added focused tests for timestamp naming, collision handling, config snapshots, resume overrides, and checkpoint path resolution.
- Updated training documentation for the timestamped model directory layout and resume usage.

**Context**
- Model run directories previously depended on training-related details such as environment, epoch count, input domain, cost map, and optimizer.
- Timestamped directories avoid accidental coupling between run identity and training configuration while keeping `model.pt` and `model.yaml` as stable artifacts.
- Saving `model.yaml` at the start preserves the effective configuration even if training, testing, or visualization fails later.

**Validation**
- `python -m unittest tests.test_training_run_directory`
- `python -m py_compile viplanner/train.py viplanner/config/learning_cfg.py viplanner/utils/trainer.py`
- `python viplanner/train.py --help`
- `python -m unittest`
- `python -m unittest discover tests`
- `git diff --check -- viplanner/config/learning_cfg.py viplanner/utils/trainer.py viplanner/train.py TRAINING.md tests/test_training_run_directory.py CHANGELOG.md`

**Notes**
- Existing YAML configs without `model_dir_name` or `resume_model_path` still load with defaults.
- `EXPERIMENT_DIRECTORY` continues to control the experiment root for data, models, and logs.

## 2026-05-19 05:44 - Add test visualization CLI switch

**Change size**
- `S`

**Files changed**
- `viplanner/train.py`
- `viplanner/utils/trainer.py`
- `CHANGELOG.md`

**What changed**
- Added mutually exclusive `--show-test-visualizations` and `--no-test-visualizations` command-line flags to `viplanner/train.py`.
- Passed the selected visualization mode into `Trainer.test()`.
- Kept the previous `EXPERIMENT_DIRECTORY`-based visualization behavior when neither CLI flag is provided.
- Updated the test phase log message to report testing instead of training.

**Context**
- Test visualizations can trigger native Open3D rendering paths after training, which may fail in headless or display-mismatched environments.
- A direct CLI switch makes visualization behavior explicit without requiring users to set `EXPERIMENT_DIRECTORY` solely to suppress rendering.

**Validation**
- `python -m py_compile viplanner/train.py viplanner/utils/trainer.py`
- `python viplanner/train.py --help`

**Notes**
- The default behavior remains unchanged: visualizations are shown only when `EXPERIMENT_DIRECTORY` is unset unless a CLI flag overrides it.

## 2026-05-18 05:24 - Integrate environment install into devcontainer image

**Change size**
- `M`

**Files changed**
- `.devcontainer/Dockerfile`
- `CHANGELOG.md`
- `pyproject.toml`

**What changed**
- Added CUDA toolkit repository setup and installation to the devcontainer image build.
- Added Python package installation steps from `install_environment.bash`, including editable ViPlanner extras, Detectron2, Mask2Former requirements, and Mask2Former custom op compilation.
- Added environment variables and shell initialization for CUDA and ROS setup.
- Changed the devcontainer workspace path from `/workspaces/iPlanner_ws` to `/workspaces/viplanner` so image build commands match the repository path.
- Split pip installation into separate Docker layers and used explicit CUDA PyTorch wheel versions with PyPI still available for transitive dependencies.
- Installed `PyYAML==6.0.3` with `--ignore-installed` before editable package installation to avoid uninstalling Ubuntu/ROS Noetic's distutils-installed `PyYAML 5.3.1`.
- Switched the devcontainer CUDA toolkit package to `cuda-toolkit-11-7`, set `CUDA_HOME` to `/usr/local/cuda-11.7`, and enabled `FORCE_CUDA` plus a fixed `TORCH_CUDA_ARCH_LIST` for Mask2Former custom op compilation during Docker builds.
- Corrected the package license metadata to reference the existing `LICENSE` file.

**Context**
- The environment setup script was separate from the devcontainer image, requiring manual execution after container creation.
- Folding the setup into the Dockerfile makes the development container more reproducible and closer to the expected runtime environment.
- The first integrated Docker build failed during the combined pip installation layer, where `--index-url` restricted dependency lookup to the PyTorch wheel index and the editable package metadata referenced a missing license file.
- A later devcontainer build failed during `python3 -m pip install -e ".[standard]"` because pip tried to replace the base image's distutils-installed `PyYAML 5.3.1`.
- The Mask2Former custom op build then failed because Docker build did not expose a CUDA runtime to `torch.cuda.is_available()`, while the installed PyTorch and MMCV wheels target CUDA 11.7.

**Validation**
- `git diff --check -- .devcontainer/Dockerfile CHANGELOG.md pyproject.toml`
- Reviewed `.devcontainer/Dockerfile`, `CHANGELOG.md`, and `pyproject.toml` after editing.
- Full Docker image build was not run because it requires large network downloads and CUDA/Python dependency installation.

**Notes**
- `git submodule update --init` during Docker build expects the build context to include the repository metadata.
- `/workspaces/viplanner/devel/setup.bash` is sourced only if it exists, because the catkin workspace may not be built when the image is created.

## 2026-05-17 08:54 - Fix depth-only training cost map selection

**Files changed**
- `CHANGELOG.md`
- `viplanner/config/train.yaml`
- `viplanner/cost_maps/cost_to_pcd.py`
- `viplanner/utils/dataset.py`
- `tests/test_cost_to_pcd.py`
- `tests/test_dataset_graph.py`

**What changed**
- Set the default training config for the current forest dataset to use `file_path: "src/planner"` so the configured environments resolve under `src/planner/data`.
- Changed depth-only training to load `cost_map_geom` instead of `cost_map_sem`.
- Added explicit missing-file validation when loading cost maps, including a depth-only training hint.
- Added a focused unit test for the missing cost-map error message.
- Treated graph interpolation samples outside the occupancy map as collisions instead of indexing out of bounds.
- Added a focused unit test for out-of-bounds occupancy samples during graph collision checks.
- Recorded the change and validation in `CHANGELOG.md`.

**Context**
- Depth-only training still needs a cost map for odometry filtering, graph construction, and trajectory loss.
- The training config had `sem: false` and `rgb: false`, but still requested `cost_map_sem`, while the generated forest maps only include `cost_map_geom`.
- The placeholder `file_path` also resolved to a literal `${USER_PATH_TO_MODEL_DATA}` directory unless overridden by `EXPERIMENT_DIRECTORY`.
- A follow-up training run reached graph construction and found odom neighbor interpolation points just outside the geometric map bounds.

**Validation**
- `python -m unittest tests.test_cost_to_pcd`
- `python -m py_compile viplanner/cost_maps/cost_to_pcd.py`
- `python -c "from viplanner.config import TrainCfg; cfg=TrainCfg.from_yaml('viplanner/config/train.yaml'); print(cfg.data_dir); print(cfg.cost_map_name)"`
- `python -c "from viplanner.cost_maps.cost_to_pcd import CostMapPCD; m = CostMapPCD.ReadTSDFMap('src/planner/data/forest_s1_high', 'cost_map_geom', gpu_id=None); print(m.num_x, m.num_y)"`
- `python -m unittest tests.test_dataset_graph`
- `python -m py_compile viplanner/utils/dataset.py`
- `python -c "from viplanner.config import TrainCfg; from viplanner.utils.trainer import Trainer; cfg=TrainCfg.from_yaml('viplanner/config/train.yaml'); cfg.env_list=['forest_s1_high']; cfg.test_env_id=-1; cfg.gpu_id=None; trainer=Trainer(cfg); trainer._load_data(train=True); print(len(trainer.data_generators), trainer.data_generators[0].graph.number_of_edges()); [generator.cleanup() for generator in trainer.data_generators]"`

**Notes**
- If training is launched from a different working directory, use `EXPERIMENT_DIRECTORY` or update `file_path` to the correct root containing `data`, `models`, and `logs`.
