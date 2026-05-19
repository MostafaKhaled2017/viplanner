FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-11.7
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV ROS_VERSION=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-tk \
    python3-rospy \
    unzip \
    wget \
    gnupg \
    ca-certificates \
    libusb-dev \
    build-essential \
    cmake \
    curl \
    libeigen3-dev \
    libboost-all-dev \
    libopencv-dev \
    python3-catkin-tools \
    python3-osrf-pycommon \
    libpcl-dev \
    libceres-dev \
    geographiclib-tools \
    ros-${ROS_DISTRO}-libg2o && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-11-7 && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspaces/viplanner
COPY . /workspaces/viplanner

RUN python3 -m pip install --upgrade "pip<25" "setuptools<75" wheel

RUN python3 -m pip install --ignore-installed PyYAML==6.0.3

RUN python3 -m pip install pyzmq

RUN python3 -m pip install \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

RUN python3 -m pip install -e ".[standard,training]"

RUN python3 -m pip uninstall -y mmcv mmcv-full mmdet mmengine && \
    python3 -m pip install mmcv==2.0.0 \
      -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html && \
    python3 -m pip install "mmengine>=0.7.1,<1.0.0" "mmdet==3.3.0"

RUN python3 -m pip install git+https://github.com/facebookresearch/detectron2.git && \
    git submodule update --init && \
    python3 -m pip install -r viplanner/third_party/mask2former/requirements.txt

RUN python3 -m pip install "protobuf<5"

WORKDIR /workspaces/viplanner/viplanner/third_party/mask2former/mask2former/modeling/pixel_decoder/ops
RUN sh make.sh

WORKDIR /workspaces/viplanner

RUN printf '%s\n' \
    'source /opt/ros/noetic/setup.bash' \
    '[ -f /workspaces/viplanner/devel/setup.bash ] && source /workspaces/viplanner/devel/setup.bash' \
    >> /etc/bash.bashrc
