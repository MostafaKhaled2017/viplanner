FROM osrf/ros:noetic-desktop-full

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-tk \
    unzip \
    wget \
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

# Create conda environment and install additional Python dependencies
WORKDIR /workspaces/iPlanner_ws

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
