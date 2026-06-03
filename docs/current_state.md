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

## Working Tree Note

At the time this documentation was created, the repository already had uncommitted changes unrelated to this documentation task. Those changes were not reverted or modified.

## Validation Status

Documentation files were added. No build, training, simulation, or test command was run for this documentation-only change.
