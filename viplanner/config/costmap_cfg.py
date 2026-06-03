# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import os
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import yaml


ROOT_PATH_PLACEHOLDER = "<path-to-data>/<env-name>"


class Loader(yaml.SafeLoader):
    pass


@dataclass
class RobotHeightInfo:
    """Robot height inferred from camera altitude samples."""

    altitude: float
    margin: float
    threshold: float
    z_min: float
    z_max: float
    z_range: float
    robot_height: float
    sample_indices: np.ndarray
    extrinsic_path: str


def _extrinsic_path_for_height(cfg: "ReconstructionCfg") -> str:
    suffix = cfg.sem_suffix if cfg.high_res_depth else cfg.depth_suffix
    return os.path.join(cfg.get_data_path(), "camera_extrinsic" + suffix + ".txt")


def _extrinsic_rows(values: np.ndarray, extrinsic_path: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1 and values.size == 7:
        return values.reshape(1, 7)
    if values.ndim == 2 and values.shape[1] == 7:
        return values
    raise ValueError(
        "Expected camera extrinsics with rows of 7 values "
        f"(x, y, z, qx, qy, qz, qw): {extrinsic_path}"
    )


def evenly_spaced_sample_indices(num_frames: int, sample_count: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("Cannot sample robot height from an empty extrinsic file.")
    if sample_count <= 0:
        raise ValueError("robot_height_sample_count must be greater than zero.")
    if sample_count >= num_frames:
        return np.arange(num_frames, dtype=int)
    return np.floor(np.arange(sample_count) * num_frames / sample_count).astype(int)


def compute_robot_height_from_dataset(cfg: "ReconstructionCfg") -> RobotHeightInfo:
    extrinsic_path = _extrinsic_path_for_height(cfg)
    if not os.path.exists(extrinsic_path):
        raise FileNotFoundError(f"Camera extrinsic file does not exist: {extrinsic_path}")
    if os.path.getsize(extrinsic_path) == 0:
        raise ValueError(f"Camera extrinsic file is empty: {extrinsic_path}")

    extrinsics = _extrinsic_rows(np.loadtxt(extrinsic_path, delimiter=","), extrinsic_path)
    sample_indices = evenly_spaced_sample_indices(
        extrinsics.shape[0],
        cfg.robot_height_sample_count,
    )
    sampled_z = extrinsics[sample_indices, 2]

    if not np.all(np.isfinite(sampled_z)):
        raise ValueError(f"Camera extrinsic z values must be finite: {extrinsic_path}")

    z_min = float(np.min(sampled_z))
    z_max = float(np.max(sampled_z))
    z_range = z_max - z_min
    if z_range > cfg.robot_height_variation_threshold:
        raise ValueError(
            "Dataset camera altitude varies by "
            f"{z_range:.6f} m, which exceeds robot_height_variation_threshold "
            f"{cfg.robot_height_variation_threshold:.6f} m."
        )

    altitude = float(np.mean(sampled_z))
    robot_height = altitude + cfg.robot_height_margin
    return RobotHeightInfo(
        altitude=altitude,
        margin=cfg.robot_height_margin,
        threshold=cfg.robot_height_variation_threshold,
        z_min=z_min,
        z_max=z_max,
        z_range=z_range,
        robot_height=robot_height,
        sample_indices=sample_indices,
        extrinsic_path=extrinsic_path,
    )


def robot_height_info_message(info: RobotHeightInfo) -> str:
    return (
        f"Using robot height: {info.robot_height:.3f} m "
        f"(dataset altitude {info.altitude:.3f} m + margin {info.margin:.3f} m; "
        f"sampled {len(info.sample_indices)} frame(s), z range {info.z_range:.6f} m)"
    )


def _normalized_config_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def _is_unset_root_path(root_path: Optional[str]) -> bool:
    if root_path is None:
        return True
    return str(root_path).strip() in {"", ROOT_PATH_PLACEHOLDER}


def _resolve_costmap_root_path(cfg: "GeneralCostMapConfig", reconstruction_cfg: "ReconstructionCfg") -> None:
    expected_path = _normalized_config_path(reconstruction_cfg.get_data_path())

    if _is_unset_root_path(cfg.root_path):
        cfg.root_path = expected_path
        return

    configured_path = _normalized_config_path(str(cfg.root_path))
    if configured_path != expected_path:
        raise ValueError(
            "Cost map root_path does not match reconstruction environment path. "
            f"Expected '{expected_path}' from reconstruction.data_dir/reconstruction.env, "
            f"but config.general.root_path is '{configured_path}'. "
            "Set config.general.root_path to null or update it to the same environment."
        )

    cfg.root_path = configured_path


def construct_GeneralCostMapConfig(loader, node):
    return GeneralCostMapConfig(**loader.construct_mapping(node))


Loader.add_constructor(
    "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.GeneralCostMapConfig",
    construct_GeneralCostMapConfig,
)


def construct_ReconstructionCfg(loader, node):
    return ReconstructionCfg(**loader.construct_mapping(node))


Loader.add_constructor(
    "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.ReconstructionCfg",
    construct_ReconstructionCfg,
)


def construct_SemCostMapConfig(loader, node):
    return SemCostMapConfig(**loader.construct_mapping(node))


Loader.add_constructor(
    "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.SemCostMapConfig",
    construct_SemCostMapConfig,
)


def construct_TsdfCostMapConfig(loader, node):
    return TsdfCostMapConfig(**loader.construct_mapping(node))


Loader.add_constructor(
    "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.TsdfCostMapConfig",
    construct_TsdfCostMapConfig,
)


@dataclass
class ReconstructionCfg:
    """
    Arguments for 3D reconstruction using depth maps
    """

    # directory where the environment with the depth (and semantic) images is located
    data_dir: str = "${USER_PATH_TO_DATA}"  # e.g. "<path-to-repo>/omniverse/extension/omni.viplanner/data/warehouse"
    # environment name
    env: str = "warehouse_new"  # has to be adjusted
    # image suffix
    depth_suffix: str = ""
    sem_suffix: str = ""
    # higher resolution depth images available for reconstruction  (meaning that the depth images are also taked by the semantic camera)
    high_res_depth: bool = False

    # reconstruction parameters
    voxel_size: float = 0.05  # [m] 0.05 for matterport 0.1 for carla
    start_idx: int = 0  # start index for reconstruction
    max_images: Optional[int] = 1000  # maximum number of images to reconstruct, if None, all images are used
    depth_scale: float = 1000  # depth scale factor
    # robot height inferred from recorded camera altitude
    robot_height_margin: float = 0.3
    robot_height_variation_threshold: float = 0.01
    robot_height_sample_count: int = 50
    # semantic reconstruction
    semantics: bool = True
    # semantic classes to drop during reconstruction because they represent unlabeled pixels
    semantic_ignore_classes: Optional[list] = None

    # speed vs. memory trade-off parameters
    point_cloud_batch_size: int = (
        200  # 3d points of nbr images added to point cloud at once (higher values use more memory but faster)
    )

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path) as f:
            cfg_dict = yaml.load(f, Loader=Loader)

        if not isinstance(cfg_dict, dict):
            raise ValueError(f"Reconstruction config file must contain a mapping: {yaml_path}")

        if "reconstruction" in cfg_dict:
            config = cfg_dict["reconstruction"]
        elif "config" in cfg_dict:
            raise ValueError(
                "Reconstruction config missing top-level 'reconstruction' section in shared config file: "
                f"{yaml_path}"
            )
        else:
            config = cfg_dict

        if isinstance(config, cls):
            return config
        if not isinstance(config, dict):
            raise ValueError(f"Reconstruction config section must be a mapping: {yaml_path}")

        valid_fields = {field.name for field in fields(cls)}
        unknown_fields = sorted(set(config.keys()) - valid_fields)
        if unknown_fields:
            raise ValueError(
                "Unknown reconstruction config field(s): " + ", ".join(unknown_fields)
            )

        return cls(**config)

    """ Internal functions """

    def get_data_path(self) -> str:
        return os.path.join(self.data_dir, self.env)

    def get_out_path(self) -> str:
        return os.path.join(self.out_dir, self.env)


@dataclass
class SemCostMapConfig:
    """Configuration for the semantic cost map"""

    # point-cloud filter parameters
    ground_height: Optional[float] = -0.5  # None for matterport  -0.5 for carla  -1.0 for nomoko
    nb_neighbors: int = 100
    std_ratio: float = 2.0  # keep high, otherwise ground will be removed
    downsample: bool = False
    # smoothing
    nb_neigh: int = 15
    change_decimal: int = 3
    conv_crit: float = (
        0.45  # ration of points that have to change by at least the #change_decimal decimal value to converge
    )
    nb_tasks: Optional[int] = 10  # number of tasks for parallel processing, if None, all available cores are used
    sigma_smooth: float = 2.5
    max_iterations: int = 1
    # obstacle threshold  (multiplied with highest loss value defined for a semantic class)
    obstacle_threshold: float = 0.5  # 0.5/ 0.6 for matterport, 0.8 for carla
    # negative reward for space with smallest cost (introduces a gradient in area with smallest loss value, steering towards center)
    # NOTE: at the end cost map is elevated by that amount to ensure that the smallest cost is 0
    negative_reward: float = 0.5
    # loss values rounded up to decimal #round_decimal_traversable equal to 0.0 are selected and the traversable gradient is determined based on them
    round_decimal_traversable: int = 2
    # compute height map
    compute_height_map: bool = False  # false for matterport, true for carla and nomoko
    # optional per-class semantic loss overrides, keyed by VIPlanner semantic class name
    class_loss_overrides: Optional[dict] = None


@dataclass
class TsdfCostMapConfig:
    """Configuration for the tsdf cost map"""

    # offset of the point cloud
    offset_z: float = 0.0
    # filter parameters
    ground_height: float = 0.35
    nb_neighbors: int = 50
    std_ratio: float = 0.2
    filter_outliers: bool = True
    # dilation parameters
    sigma_expand: float = 2.0
    obstacle_threshold: float = 0.01
    free_space_threshold: float = 0.5


@dataclass
class GeneralCostMapConfig:
    """General Cost Map Configuration"""

    # path to point cloud
    root_path: Optional[str] = ROOT_PATH_PLACEHOLDER
    ply_file: str = "cloud.ply"
    # resolution of the cost map
    resolution: float = 0.04  # [m]  (0.04 for matterport, 0.1 for carla)
    # map parameters
    clear_dist: float = 1.0  # cost map expansion over the point cloud space (prevent paths to go out of the map)
    # smoothing parameters
    sigma_smooth: float = 3.0
    # cost map expansion
    x_min: Optional[float] = None
    # [m] if None, the minimum of the point cloud is used None (carla town01:  -8.05   matterport: None)
    y_min: Optional[float] = None
    # [m] if None, the minimum of the point cloud is used None (carla town01:  -8.05   matterport: None)
    x_max: Optional[float] = None
    # [m] if None, the maximum of the point cloud is used None (carla town01:  346.22  matterport: None)
    y_max: Optional[float] = None
    # [m] if None, the maximum of the point cloud is used None (carla town01:  336.65  matterport: None)


@dataclass
class CostMapConfig:
    """General Cost Map Configuration"""

    # cost map domains
    semantics: bool = True
    geometry: bool = False

    # name
    map_name: str = "cost_map_sem"

    # general cost map configuration
    general: GeneralCostMapConfig = GeneralCostMapConfig()

    # individual cost map configurations
    sem_cost_map: SemCostMapConfig = SemCostMapConfig()
    tsdf_cost_map: TsdfCostMapConfig = TsdfCostMapConfig()

    # visualize cost map
    visualize: bool = True

    # FILLED BY CODE -> DO NOT CHANGE ###
    x_start: float = None
    y_start: float = None

    @classmethod
    def from_yaml(cls, yaml_path: str, reconstruction_cfg: Optional[ReconstructionCfg] = None):
        with open(yaml_path) as f:
            cfg_dict = yaml.load(f, Loader=Loader)

        if not isinstance(cfg_dict, dict):
            raise ValueError(f"Cost map config file must contain a mapping: {yaml_path}")

        has_reconstruction_section = "reconstruction" in cfg_dict
        if reconstruction_cfg is None and has_reconstruction_section:
            reconstruction_cfg = ReconstructionCfg.from_yaml(yaml_path)

        config = dict(cfg_dict["config"]) if "config" in cfg_dict else dict(cfg_dict)

        general = config.get("general")
        if isinstance(general, dict):
            config["general"] = GeneralCostMapConfig(**general)

        sem_cost_map = config.get("sem_cost_map")
        if isinstance(sem_cost_map, dict):
            config["sem_cost_map"] = SemCostMapConfig(**sem_cost_map)

        tsdf_cost_map = config.get("tsdf_cost_map")
        if isinstance(tsdf_cost_map, dict):
            config["tsdf_cost_map"] = TsdfCostMapConfig(**tsdf_cost_map)

        cfg = cls(**config)
        if reconstruction_cfg is not None:
            _resolve_costmap_root_path(cfg.general, reconstruction_cfg)

        return cfg


# EoF
