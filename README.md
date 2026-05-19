# ViPlanner: Visual Semantic Imperative Learning for Local Navigation

<p align="center">
  <a href="https://leggedrobotics.github.io/viplanner.github.io/">Project Page</a> •
  <a href="https://arxiv.org/abs/2310.00982">arXiv</a> •
  <a href="https://youtu.be/8KO4NoDw6CM">Video</a> •
  <a href="#citing-viplanner">BibTeX</a>

  Click on the image for the demo video!
  [![Demo Video](./assets/crosswalk.jpg)](https://youtu.be/8KO4NoDw6CM)

</p>

ViPlanner is a robust learning-based local path planner based on semantic and depth images.
Fully trained in simulation, the planner can be applied in dynamic indoor as well as outdoor environments.
We provide it as an extension for [NVIDIA Isaac-Sim](https://developer.nvidia.com/isaac-sim) within the [IsaacLab](https://isaac-sim.github.io/IsaacLab/) project (details [here](./omniverse/README.md)).
Furthermore, a ready-to-use [ROS Noetic](http://wiki.ros.org/noetic) package is available within this repo for direct integration on any robot (tested and developed on ANYmal C and D).

**Keywords:** Visual Navigation, Local Planning, Imperative Learning


## Install

- Install `pyproject.toml` with pip by running:
  ```bash
  python3 -m pip install .
  ```
  or
  ```bash
  python3 -m pip install -e .[standard]
  ```
  if you want to edit the code. To apply the planner in the ROS-Node, install it with the inference setting:
  ```bash
  python3 -m pip install -e .[standard,inference]
  ```
  To enable TensorBoard logging during training, install:
  ```bash
  python3 -m pip install -e .[standard,training]
  ```
  Make sure the CUDA toolkit is of the same version as used to compile torch. We assume 11.7. If you are using a different version, adjust the string for the mmcv install as given . If the toolkit is not found, set the `CUDA_HOME` environment variable, as follows:
  ```
  export CUDA_HOME=/usr/local/cuda
  ```
  On the Jetson, please use
  ```bash
  python3 -m pip install -e .[inference,jetson]
  ```
  as `mmdet` requires torch.distributed which is only build until version 1.11 and not compatible with pypose. See the [Dockerfile](./Dockerfile) for a workaround.

**Known Issue**
- Ubuntu/ROS Noetic environments may provide `PyYAML 5.3.1` through distutils. Since `open3d==0.17.0` requires `PyYAML>=5.4.1`, install PyYAML without uninstalling the system package before installing VIPlanner:
  ```bash
  python3 -m pip install --ignore-installed PyYAML==6.0.3
  python3 -m pip install .
  ```
- mmcv build wheel does not finish:
  - fix by installing with defined CUDA version, as detailed [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip). For CUDA Version 11.7 and torch==2.0.x use
  ```
  python3 -m pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
  ```
- `pip install -r third_party/mask2former/requirements.txt` fails with `Invalid version: '1.1-linux32'`:
  - use `python3 -m pip` instead of `/usr/bin/pip`, since Ubuntu/ROS images often ship an older system `pip` resolver that crashes on malformed package metadata from PyPI.
- TensorBoard fails with `MessageToJson() got an unexpected keyword argument 'including_default_value_fields'`:
  - this is a TensorBoard 2.14.x and protobuf 5.x compatibility issue. Install the training extra or downgrade protobuf below 5 in the same Python environment used to launch TensorBoard:
  ```bash
  python3 -m pip install -e .[training]
  python3 -m pip install "protobuf<5"
  python3 -m pip show tensorboard protobuf
  ```

**Extension**

This work includes the switch from semantic to direct RGB input for the training pipeline to facilitate further research. For RGB input, an option exists to employ a backbone with mask2former pre-trained weights. For this option, include the GitHub submodule, install the requirements included there, and build the necessary Cuda operators. These steps are not necessary for the published planner!

```bash
python3 -m pip install git+https://github.com/facebookresearch/detectron2.git
git submodule update --init
python3 -m pip install -r third_party/mask2former/requirements.txt
cd third_party/mask2former/mask2former/modeling/pixel_decoder/ops \
sh make.sh
```

**Remark**

Note that for an editable installation of packages without setup.py, PEP660 has to be fulfilled. This requires the following versions (as described [here](https://stackoverflow.com/questions/69711606/how-to-install-a-package-using-pip-in-editable-mode-with-pyproject-toml) in detail)
- [pip >= 21.3](https://pip.pypa.io/en/stable/news/#v21-3)
	```
  python3 -m pip install --upgrade pip
  ```
- [setuptools >= 64.0.0](https://github.com/pypa/setuptools/blob/main/CHANGES.rst#v6400)
	```
  python3 -m pip install --upgrade setuptools
  ```


## Inference and Model Demo

1. Real-World <br>

	ROS-Node is provided to run the planner on the LeggedRobot ANYmal; for details, please see [ROS-Node-README](ros/README.md).

2. NVIDIA Isaac-Sim <br>

	The planner can be executed within Nvidia Isaac Sim. It is implemented as part of the [IsaacLab Framework](https://isaac-sim.github.io/IsaacLab/) with an own extension. For details, please see [Omniverse Extension](./omniverse/README.md). This includes a **planner demo** in different environments with the trained model.


## Training

Here is an overview of the steps involved in training the policy.
For more detailed instructions, please refer to [TRAINING.md](TRAINING.md).

0. Training Data Generation <br>
Training data is generated from the [Matterport 3D](https://github.com/niessner/Matterport), [Carla](https://carla.org/) and [NVIDIA Warehouse](https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_static_assets.html) using IsaacLab. For detailed instructions on how to install the extension and run the data collection script, please see [here](omniverse/README.md)

    For ROS-based simulators that publish depth, RGB, camera info, and TF, data can also be recorded directly with the ROS collector:

    ```bash
    cd <catkin_ws>
    catkin build viplanner_pkgs
    source devel/setup.bash
    ```

    ```bash
    roslaunch viplanner_node data_collect_sim.launch
    ```

    The collector is configured in [`src/planner/config/data_collect_sim.yaml`](src/planner/config/data_collect_sim.yaml). It writes a vIPlanner-compatible environment directory:

    ```graphql
    env_name
    ├── intrinsics.txt
    ├── camera_extrinsic_cam0.txt
    ├── camera_extrinsic_cam1.txt
    ├── depth
    │   ├── 0000_cam0.png
    │   ├── 0000_cam0.npy
    ├── rgb
    │   ├── 0000_cam1.png
    └── semantics
        ├── 0000_cam1.png                    # generated after recording
    ```

    If the simulator only provides RGB and not semantic labels, generate vIPlanner-color semantic images after recording:

    ```bash
    python3 viplanner/generate_semantics.py \
      --input <dataset_root>/<env_name> \
      --config <mask2former_config.py> \
      --checkpoint <mask2former_checkpoint.pth> \
      --device cuda:0
    ```

    The most important collector parameters are:

    - `dataset_root` and `env_name`: Output location. Training expects environments under `TrainCfg.file_path/data` or `EXPERIMENT_DIRECTORY/data`. If `env_name` already exists and contains data, the collector records into `env_name_v2`, `env_name_v3`, and so on.
    - `depth_topic`, `rgb_topic`, `depth_info_topic`, `rgb_info_topic`: Simulator camera topics.
    - `rgb_compressed`: Set to `true` for `sensor_msgs/CompressedImage`, `false` for raw `sensor_msgs/Image`.
    - `world_frame_id`, `depth_frame_id`, `rgb_frame_id`: TF frames used to save camera poses. Leave camera frame overrides empty to use each image header frame.
    - `rgb_pose_fallback_to_depth`: Use the depth camera pose for RGB if the RGB image frame is not published in TF. Prefer setting `rgb_frame_id` to a valid TF frame when RGB and depth are not co-located.
    - `convert_optical_to_viplanner`: Keep `true` for normal ROS optical camera frames so saved poses match vIPlanner reconstruction.
    - `depth_width`, `depth_height`, `rgb_width`, `rgb_height`: Required recorded image sizes. The collector exits immediately if any incoming frame does not match.
    - `depth_is_uint16`, `depth_scale`, `max_depth`: Depth unit handling. Use `depth_is_uint16: false` when depth arrives in meters.
    - `sample_min_interval`, `sample_min_distance`, `sample_min_angle_deg`: Controls how densely frames are saved while the robot moves.
    - `sync_slop_seconds`: Maximum allowed timestamp difference between depth and RGB frames. Increase this if the simulator publishes RGB and depth at slightly offset times.
    A single "Waiting for depth and RGB camera info" warning at startup is expected if images arrive before both `CameraInfo` topics. Repeated pose lookup warnings mean the image `frame_id` is not available in TF; set `depth_frame_id` or `rgb_frame_id` to a valid camera frame, or use `rgb_pose_fallback_to_depth` only when that approximation is acceptable.

1. Build Cost-Map <br>
The first step in training the policy is to build a cost-map from the available depth and semantic data. A cost-map is a representation of the environment where each cell is assigned a cost value indicating its traversability. The cost-map guides the optimization, therefore, it is required to be differentiable. Cost-maps are built using the [cost-builder](viplanner/cost_builder.py) with configs [here](viplanner/config/costmap_cfg.py), given a pointcloud of the environment with semantic information (either from simulation or real-world information). The point-cloud of the simulated environments can be generated with the [reconstruction-script](viplanner/depth_reconstruct.py), which reads the top-level `reconstruction:` section from [viplanner/config/costmap.yaml](viplanner/config/costmap.yaml).

2. Training <br>
Once the cost-map is constructed, the next step is to train the policy. The policy is a machine learning model that learns to make decisions based on depth and semantic measurements. An example training script can be found [here](viplanner/train.py) with configs [here](viplanner/config/learning_cfg.py)

3. Evaluation <br>
Performance assessment can be performed on simulation and real-world data. The policy will be evaluated regarding multiple metrics such as distance to the goal, average and maximum cost, and path length. In order to let the policy be executed on anymal in simulation, please refer to [Omniverse Extension](./omniverse/README.md)


### Model Download
The latest model is available to download: [[checkpoint](https://drive.google.com/file/d/1PY7XBkyIGESjdh1cMSiJgwwaIT0WaxIc/view?usp=sharing)] [[config](https://drive.google.com/file/d/1r1yhNQAJnjpn9-xpAQWGaQedwma5zokr/view?usp=sharing)]

## <a name="CitingViPlanner"></a>Citing ViPlanner
```
@inproceedings{roth2024viplanner,
  title={Viplanner: Visual semantic imperative learning for local navigation},
  author={Roth, Pascal and Nubert, Julian and Yang, Fan and Mittal, Mayank and Hutter, Marco},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={5243--5249},
  year={2024},
  organization={IEEE}
}
```

### License

This code belongs to Robotic Systems Lab, ETH Zurich.
All right reserved

**Authors: [Pascal Roth](https://github.com/pascal-roth), [Julian Nubert](https://juliannubert.com/), [Fan Yang](https://github.com/MichaelFYang), [Mayank Mittal](https://mayankm96.github.io/), and [Marco Hutter](https://rsl.ethz.ch/the-lab/people/person-detail.MTIxOTEx.TGlzdC8yNDQxLC0xNDI1MTk1NzM1.html)<br />
Maintainer: Pascal Roth, rothpa@ethz.ch**

The ViPlanner package has been tested under ROS Noetic on Ubuntu 20.04.
This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.
