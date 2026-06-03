# Overview

## Repository Type

Mixed project:

- Python package
- Machine learning and computer vision project
- ROS 1 / catkin workspace packages
- NVIDIA Isaac Sim / Omniverse extension code

## Purpose

VIPlanner is a visual semantic local planner for legged robot navigation. The repository contains Python training, cost-map, and inference utilities, ROS Noetic packages for runtime integration and validation, and Omniverse/Isaac Sim assets for simulation workflows.

## Main Capabilities

- Build environment point clouds from depth and semantic image datasets.
- Build semantic or geometric cost maps.
- Train VIPlanner neural models.
- Run ROS 1 planner inference nodes.
- Run scenario-based planner validation.
- Visualize planned paths in ROS.
- Collect simulation datasets from ROS-published camera and TF streams.

## Primary Entry Points

- `viplanner/depth_reconstruct.py`
- `viplanner/cost_builder.py`
- `viplanner/train.py`
- `viplanner/tune_train.py`
- `viplanner/generate_semantics.py`
- `src/viplanner_ros1/scripts/viplanner_node.py`
- `src/planner/src/viplanner_node.py`
- `src/planner/src/viplanner_data_collect_node.py`
- `src/planner_validation_ros1/scripts/planner_validation_node.py`

## Status

The project is described in `README.md` as research code for ROS Noetic on Ubuntu 20.04. It includes active local additions and edits in this checkout at the time this documentation was created.
