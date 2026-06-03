# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from typing import Optional

# imperative-cost-map
from viplanner.config import (
    CostMapConfig,
    ReconstructionCfg,
    compute_robot_height_from_dataset,
    robot_height_info_message,
)
from viplanner.cost_maps import CostMapPCD, SemCostMap, TsdfCostMap


def main(cfg: CostMapConfig, robot_height: float, final_viz: bool = True):
    assert any([cfg.semantics, cfg.geometry]), "no cost map type selected"

    # create semantic cost map
    if cfg.semantics:
        print("============ Creating Semantic Map from cloud ===============")
        sem_cost_map = SemCostMap(
            cfg.general,
            cfg.sem_cost_map,
            robot_height=robot_height,
            visualize=cfg.visualize,
        )
        sem_cost_map.pcd_init()
        data, coord = sem_cost_map.create_costmap()
    # create tsdf cost map
    elif cfg.geometry:
        print("============== Creating tsdf Map from cloud =================")
        tsdf_cost_map = TsdfCostMap(cfg.general, cfg.tsdf_cost_map, robot_height=robot_height)
        tsdf_cost_map.ReadPointFromFile()
        data, coord = tsdf_cost_map.CreateTSDFMap()
        (tsdf_cost_map.VizCloud(tsdf_cost_map.obs_pcd) if cfg.visualize else None)
    else:
        raise ValueError("no cost map type selected")

    # set coords in costmap config
    cfg.x_start, cfg.y_start = coord

    # construct final cost map as pcd and save parameters
    print("======== Generate and Save costmap as Point-Cloud ===========")
    cost_mapper = CostMapPCD(
        cfg=cfg,
        tsdf_array=data[0],
        viz_points=data[1],
        ground_array=data[2],
        load_from_file=False,
    )
    cost_mapper.SaveTSDFMap()
    if final_viz:
        cost_mapper.ShowTSDFMap(cost_map=True)
    return


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="Build Costmap", description="Build a cost map from a point cloud")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the cost map yaml config file",
    )
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument(
        "--final-viz",
        dest="final_viz",
        action="store_true",
        help="Show the final cost map visualization for each processed environment.",
    )
    viz_group.add_argument(
        "--no-final-viz",
        dest="final_viz",
        action="store_false",
        help="Do not show the final cost map visualization.",
    )
    parser.set_defaults(final_viz=None)
    return parser


def run_from_config(config_path: str, final_viz: Optional[bool] = None) -> None:
    reconstruction_cfg = ReconstructionCfg.from_yaml(config_path)
    show_final_viz = final_viz if final_viz is not None else len(reconstruction_cfg.env_list) == 1
    if final_viz is None and len(reconstruction_cfg.env_list) > 1:
        print("[INFO] Final cost map visualization disabled for multi-environment batch processing.")
    for env_name in reconstruction_cfg.env_list:
        env_reconstruction_cfg = reconstruction_cfg.for_env(env_name)
        print(f"============ Processing environment: {env_name} ============")
        robot_height_info = compute_robot_height_from_dataset(env_reconstruction_cfg)
        print(robot_height_info_message(robot_height_info))

        cfg = CostMapConfig.from_yaml(config_path, reconstruction_cfg=env_reconstruction_cfg)
        main(cfg, robot_height=robot_height_info.robot_height, final_viz=show_final_viz)


if __name__ == "__main__":
    args = build_argparser().parse_args()

    run_from_config(args.config, final_viz=args.final_viz)

# EoF
