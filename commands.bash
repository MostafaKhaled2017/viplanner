# Run the bridge
python sim/RosProBridge/pro_bridge/bridge.py sim/RosProBridge/config/recv.json

# Building the ros packages
catkin build viplanner_pkgs

# Data collection
roslaunch viplanner_node data_collect_sim.launch

# Build environment pointcloud
python viplanner/depth_reconstruct.py --config viplanner/config/costmap.yaml

# Generating PCL & Building cost map
# Need also to adjust the config/costmap.yaml file
# Important params: map_name, robot_height, semantics, geometry
python viplanner/cost_builder.py --config viplanner/config/costmap.yaml

# Run training
# Adjust configs in viplanner/config/train.yaml
python viplanner/train.py --config viplanner/config/train.yaml --no-test-visualizations

# Running tensorboard
tensorboard --logdir src/planner/logs --host 0.0.0.0 --port 6006

python3 viplanner/tune_train.py \
  --sweep-config viplanner/config/sweep_depth_geom.yaml \
  --gpus 1,2 \
  --max-parallel 4

# Generate Sementics from RGB
python3 viplanner/generate_semantics.py \
      --input /workspaces/viplanner/src/planner/data/forest_s1 \
      --config /workspaces/viplanner/viplanner/third_party/mask2former/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
      --checkpoint /workspaces/viplanner/src/planner/models/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth \
      --device cuda:0

# Debug depth image
python3 src/planner/src/print_image_values.py src/planner/data/forest_s1/depth/0000_cam0.png
