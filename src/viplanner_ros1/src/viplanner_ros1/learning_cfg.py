from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


class Loader(yaml.SafeLoader):
    pass


@dataclass
class DataCfg:
    real_world_data: bool = False
    carla: bool = False
    max_depth: float = 15.0
    max_goal_distance: float = 15.0
    min_goal_distance: float = 0.5
    distance_scheme: dict = field(default_factory=lambda: {1: 0.2, 3: 0.35, 5: 0.25, 7.5: 0.15, 10: 0.05})
    obs_cost_height: float = 1.5
    fov_scale: float = 1.0
    depth_scale: float = 1000.0
    ratio: float = 0.9
    max_train_pairs: Optional[int] = None
    pairs_per_image: int = 4
    ratio_fov_samples: float = 1.0
    ratio_front_samples: float = 0.0
    ratio_back_samples: float = 0.0
    noise_edges: bool = False
    edge_threshold: int = 100
    extend_kernel_size: Tuple[int, int] = field(default_factory=lambda: [5, 5])
    depth_salt_pepper: Optional[float] = None
    depth_gaussian: Optional[float] = None
    depth_random_polygons_nb: Optional[int] = None
    depth_random_polygon_size: int = 10
    sem_rgb_pepper: Optional[float] = None
    sem_rgb_black_img: Optional[float] = None
    sem_rgb_random_polygons_nb: Optional[int] = None
    sem_rgb_random_polygon_size: int = 20


def _construct_datacfg(loader, node):
    add_dicts = {}
    for node_entry in list(node.value):
        if isinstance(node_entry[1], yaml.MappingNode):
            add_dicts[node_entry[0].value] = loader.construct_mapping(node_entry[1])
            node.value.remove(node_entry)
    return DataCfg(**loader.construct_mapping(node), **add_dicts)


Loader.add_constructor(
    "tag:yaml.org,2002:python/object:viplanner.config.learning_cfg.DataCfg",
    _construct_datacfg,
)


def _to_yaml_safe(value):
    if isinstance(value, dict):
        return {key: _to_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_yaml_safe(item) for item in value]
    return value


@dataclass
class TrainCfg:
    sem: bool = True
    rgb: bool = False
    file_name: Optional[str] = None
    seed: int = 0
    gpu_id: int = 0
    file_path: str = "${USER_PATH_TO_MODEL_DATA}"
    cost_map_name: str = "cost_map_sem"
    env_list: List[str] = field(default_factory=list)
    test_env_id: int = 0
    data_cfg: Union[DataCfg, List[DataCfg]] = field(default_factory=DataCfg)
    multi_epoch_dataloader: bool = False
    num_workers: int = 4
    load_in_ram: bool = False
    fear_ahead_dist: float = 2.5
    w_obs: float = 0.25
    w_height: float = 1.0
    w_motion: float = 1.5
    w_goal: float = 4.0
    obstacle_thread: float = 1.2
    img_input_size: Tuple[int, int] = field(default_factory=lambda: [360, 640])
    in_channel: int = 16
    knodes: int = 5
    pre_train_sem: bool = True
    pre_train_cfg: Optional[str] = "m2f_model/coco/panoptic/maskformer2_R50_bs16_50ep.yaml"
    pre_train_weights: Optional[str] = "m2f_model/coco/panoptic/model_final_94dc52.pkl"
    pre_train_freeze: bool = True
    decoder_small: bool = False
    freeze_layers: int = 0
    resume: bool = False
    resume_model_path: Optional[str] = None
    model_dir_name: Optional[str] = None
    epochs: int = 100
    batch_size: int = 64
    checkpoint_interval: int = 10
    hierarchical: bool = False
    hierarchical_step: int = 50
    hierarchical_front_step_ratio: float = 0.02
    hierarchical_back_step_ratio: float = 0.01
    lr: float = 2e-3
    factor: float = 0.5
    min_lr: float = 1e-5
    patience: int = 3
    early_stop_patience: int = 10
    optimizer: str = "sgd"
    momentum: float = 0.1
    w_decay: float = 1e-4
    camera_tilt: float = 0.15
    n_visualize: int = 15

    def ensure_model_dir_name(self) -> str:
        if getattr(self, "_model_dir_name_reserved", False):
            return self.model_dir_name

        base_name = self.model_dir_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        candidate = base_name
        suffix = 1
        while os.path.exists(os.path.join(self.all_model_dir, candidate)):
            candidate = f"{base_name}_{suffix:02d}"
            suffix += 1

        self.model_dir_name = candidate
        self._model_dir_name_reserved = True
        return self.model_dir_name

    def to_dict(self) -> Dict[str, Any]:
        return _to_yaml_safe(asdict(self))

    @property
    def all_model_dir(self):
        return os.path.join(os.getenv("EXPERIMENT_DIRECTORY", self.file_path), "models")

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path) as file:
            cfg_dict = yaml.load(file, Loader=Loader)

        if not isinstance(cfg_dict, dict) or "config" not in cfg_dict:
            raise ValueError(f"Training config must contain a top-level 'config' mapping: {yaml_path}")

        config = dict(cfg_dict["config"])
        for legacy_key in ("wb_project", "wb_entity", "wb_api_key"):
            config.pop(legacy_key, None)

        data_cfg = config.get("data_cfg")
        if isinstance(data_cfg, dict):
            config["data_cfg"] = DataCfg(**data_cfg)
        elif isinstance(data_cfg, list):
            config["data_cfg"] = [entry if isinstance(entry, DataCfg) else DataCfg(**entry) for entry in data_cfg]

        if config.get("file_name") is None:
            config["file_name"] = Path(yaml_path).stem

        return cls(**config)
