# Decisions

## Existing Project Decisions

- Use ROS Noetic / ROS 1 for runtime packages.
- Use catkin packages under `src/`.
- Keep the Python package installable through `pyproject.toml`.
- Use YAML-driven configs for reconstruction, cost-map building, training, planner inference, and validation.
- Support semantic, RGB, and depth/geometric planner paths.
- Store trained model artifacts as `model.pt` plus `model.yaml`.
- Use `docs/` as the project wiki and engineering memory.

## Documentation Decisions

- Keep initial docs factual and repository-grounded.
- Use `Unknown` or scoped notes where repository files do not establish a fact.
- Record documentation-only validation separately from runtime validation.
