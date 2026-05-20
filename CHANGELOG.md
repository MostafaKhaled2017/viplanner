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
