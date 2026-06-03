# Dependencies

## Python Runtime

From `pyproject.toml`:

- `torch`
- `torchvision`
- `PyYAML>=5.4.1,<7`
- `tqdm`
- `matplotlib`
- `networkx`
- `scipy`
- `open3d==0.17.0`
- `opencv-python-headless`

## Python Optional Extras

`inference`:

- `mmcv==2.0.0`
- `mmengine`
- `mmdet`

`standard`:

- `pypose`

`training`:

- `protobuf<5`
- `tensorboard`

`jetson`:

- `torch==1.11`

## ROS Dependencies

Repository package manifests reference:

- `catkin`
- `rospy`
- `roscpp`
- `std_msgs`
- `sensor_msgs`
- `geometry_msgs`
- `nav_msgs`
- `tf`
- `tf2`
- `tf2_ros`
- `tf2_geometry_msgs`
- `image_transport`
- `cv_bridge`
- `message_generation`
- `message_runtime`
- `message_filters`
- `pcl_ros`
- `rosbag`
- `rviz`
- `qtbase5-dev`
- `joy`
- `ps3joy`

## Semantic Inference Dependencies

The README and package docs reference:

- Mask2Former
- Detectron2
- MMDetection
- MMCV / OpenMMLab wheels
- COCO panoptic API

## System / Container Dependencies

The Dockerfile installs ROS Noetic, CUDA toolkit, cuDNN, OpenCV, build tools, Python development packages, OpenMPI, BLAS/LAPACK, HDF5, protobuf compiler, LLVM, and related libraries.

## Dependency Notes

- `README.md` documents a PyYAML/Open3D issue in Ubuntu/ROS Noetic environments and recommends `python3 -m pip install --ignore-installed PyYAML==6.0.3`.
- `README.md` documents a TensorBoard/protobuf compatibility issue and recommends the training extra or `protobuf<5`.
- `README.md` documents an MMCV CUDA 11.7 / torch 2.0.x wheel command.
