# Current State

## Documentation Snapshot

Created on 2026-06-03.

## Verified Repository Shape

- Mixed Python, ML/CV, ROS 1, and Omniverse repository.
- Python package metadata is defined in `pyproject.toml`.
- ROS packages are present under `src/`.
- Tests are present under `tests/`.
- Existing generated/build directories are present: `build/`, `devel/`, and `logs/`.
- Model/checkpoint files are present under `viplanner/checkpoint/` and `src/planner/models/`.
- Dataset directories referenced by configs may be environment-specific and were not fully inspected.

## Active Config Focus

The latest training fix treats `resume_model_path` checkpoints outside the current run directory as warm-start checkpoints. The model weights are loaded, but the new run resets `best_loss` so early stopping and model selection compare against validation losses from the current run instead of the source checkpoint's stored loss.

The latest training-data fix isolates generated warped semantic images and depth-edge images under per-generator subdirectories. This prevents parallel sweep trials from deleting another active trial's generated `img_warp` inputs during cleanup.

The active editor selection referenced hyperparameter sweep training with:

```bash
python3 viplanner/tune_train.py --sweep-config viplanner/config/sweep.yaml --gpus 1,2,3 --max-parallel 6
```

`viplanner/config/sweep.yaml` uses `data_cfg.max_depth` for depth-range tuning because `max_depth` belongs to `DataCfg`, not top-level `TrainCfg`.
The sweep resume checkpoint path is repository-relative: `viplanner/checkpoint`.

Recent cost-map work referenced:

```bash
python viplanner/cost_builder.py --config viplanner/config/costmap.yaml
```

`viplanner/config/costmap.yaml` currently includes both a `reconstruction:` section and a `config:` section for cost-map generation.
The reconstruction section uses `env_list`, and reconstruction/cost-map scripts process each listed environment independently.
For multi-environment cost-map builds, final Open3D cost-map visualization is disabled by default to avoid blocking the batch after the first environment.

## Working Tree Note

At the time this documentation was created, the repository already had uncommitted changes unrelated to this documentation task. Those changes were not reverted or modified.

## Validation Status

Targeted validation for the multi-environment reconstruction and cost-map config update passed:

```bash
pytest tests/test_reconstruction_cfg.py tests/test_robot_height_cfg.py tests/test_cost_builder.py
pytest tests/test_debug_project_cloud_to_images.py
pytest tests/test_reconstruction_cfg.py tests/test_robot_height_cfg.py tests/test_cost_builder.py tests/test_debug_project_cloud_to_images.py
python -m py_compile viplanner/cost_builder.py
pytest tests/test_cost_builder.py
```

All commands reported passing tests. Pytest emitted cache-write warnings because `.pytest_cache` could not be written in the workspace.

Targeted validation for the sweep config key update passed:

```bash
python3 viplanner/tune_train.py --sweep-config viplanner/config/sweep.yaml --gpus 1 --max-parallel 1 --max-trials 1 --dry-run
python3 -c "from viplanner.config import TrainCfg; cfg=TrainCfg.from_yaml('logs/tuning/depth_geom_moderate/configs/trial_0000.yaml'); print(cfg.data_cfg.max_depth); print(hasattr(cfg, 'max_depth')); print(cfg.resume_model_path)"
```

Targeted validation for generated training-data cleanup passed:

```bash
pytest tests/test_dataset_generated_dirs.py
```

Targeted validation for warm-start checkpoint early stopping passed:

```bash
PYTHONPYCACHEPREFIX=/tmp/viplanner-pycache python3 -m py_compile viplanner/utils/trainer.py tests/test_training_early_stopping.py
pytest tests/test_training_early_stopping.py
```
