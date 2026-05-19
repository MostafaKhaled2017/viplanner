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

**Context**
- Training metrics should be logged without requiring a Weights & Biases account or API key.
- TensorBoard keeps metrics local to the existing timestamped experiment directory layout.

**Validation**
- `python -m unittest tests.test_training_early_stopping tests.test_training_run_directory`
- `python -m py_compile viplanner/train.py viplanner/config/learning_cfg.py viplanner/utils/trainer.py viplanner/traj_cost_opt/traj_cost.py`
- `python viplanner/train.py --help`
- `git diff --check -- viplanner/config/learning_cfg.py viplanner/config/train.yaml viplanner/utils/trainer.py viplanner/traj_cost_opt/traj_cost.py viplanner/train.py TRAINING.md tests/test_training_early_stopping.py tests/test_training_run_directory.py pyproject.toml CHANGELOG.md`

**Notes**
- TensorBoard graph tracing is intentionally not added because the model has multiple input signatures depending on training configuration.
- The training extra pins `protobuf<5` because TensorBoard 2.14.0's HParams plugin is incompatible with protobuf 5.x.

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
