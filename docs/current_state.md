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

The active editor selection referenced:

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
