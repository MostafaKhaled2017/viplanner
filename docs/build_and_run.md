# Build and Run

## Python Commands

Build or install:

```bash
python3 -m pip install .
python3 -m pip install -e .[standard,inference]
python3 -m pip install -e .[standard,training]
```

Depth reconstruction:

```bash
python viplanner/depth_reconstruct.py --config viplanner/config/costmap.yaml
```

Cost-map building:

```bash
python viplanner/cost_builder.py --config viplanner/config/costmap.yaml
```

Training:

```bash
python viplanner/train.py --config viplanner/config/train.yaml
```

Training resume example from `commands.bash`:

```bash
python viplanner/train.py --config viplanner/config/train.yaml --resume-from /workspaces/viplanner/viplanner/checkpoint --no-test-visualizations
```

TensorBoard:

```bash
tensorboard --logdir src/planner/logs --host 0.0.0.0 --port 6006
```

Hyperparameter sweep:

```bash
python3 viplanner/tune_train.py --sweep-config viplanner/config/sweep_depth_geom.yaml --gpus 1,2 --max-parallel 4
```

Generate semantics from RGB:

```bash
python3 viplanner/generate_semantics.py --input /workspaces/viplanner/src/planner/data/forest_s1_high --config /workspaces/viplanner/viplanner/third_party/mask2former/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml --checkpoint /workspaces/viplanner/src/planner/models/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth --device cuda:0
```

## ROS Build Commands

Commands present in `commands.bash`:

```bash
catkin build viplanner_pkgs --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5
catkin build planner_validation_ros1 --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5
catkin build viplanner_ros1 --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

`src/viplanner_ros1/README.md` also documents:

```bash
catkin_make --pkg viplanner_ros1
source devel/setup.bash
```

## ROS Run Commands

Data collection:

```bash
roslaunch viplanner_node data_collect_sim.launch
```

Planner:

```bash
roslaunch viplanner_ros1 viplanner.launch
```

Planner validation:

```bash
roslaunch planner_validation_ros1 planner_validation.launch
```

Model sweep validation:

```bash
rosrun planner_validation_ros1 viplanner_model_sweep.py --models-parent /workspaces/viplanner/src/planner/models --output-dir /workspaces/viplanner/src/planner_validation_ros1/logs --viplanner-config /workspaces/viplanner/src/viplanner_ros1/config/viplanner.yaml --validation-config /workspaces/viplanner/src/planner_validation_ros1/config/planner_validation.yaml
```

Bridge:

```bash
python sim/RosProBridge/pro_bridge/bridge.py sim/RosProBridge/config/recv.json
```

## Validation Commands

Repository tests are in `tests/`. A typical narrow validation command is:

```bash
pytest tests
```

This documentation bootstrap did not run the full test suite because it changed documentation only.
