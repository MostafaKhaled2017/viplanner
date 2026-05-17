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
