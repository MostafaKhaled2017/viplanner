

# Install pip
apt update
apt install -y python3-pip

# Installing CUDA
apt-get update
apt-get install -y wget gnupg ca-certificates

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb

apt-get update
apt-get install -y cuda-toolkit-12-1

# Update bashrc file
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH="$CUDA_HOME/bin:$PATH"' >> ~/.bashrc
echo 'export ROS_VERSION=1' >> ~/.bashrc
echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
echo 'source /workspaces/viplanner/devel/setup.bash' >> ~/.bashrc

# Installing rospy and zmq
sudo apt install python3-rospy
pip install zmq

# Installing repo
python3 -m pip install -e .
python3 -m pip install -e .[standard,inference]


# Extension installing
python3 -m pip install git+https://github.com/facebookresearch/detectron2.git
git submodule update --init
python3 -m pip install -r viplanner/third_party/mask2former/requirements.txt
cd viplanner/third_party/mask2former/mask2former/modeling/pixel_decoder/ops && sh make.sh


# Adjusting versions for mmdetection and mmcv
python3 -m pip uninstall -y mmcv mmcv-full mmdet mmengine

python3 -m pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu117

python3 -m pip install mmcv==2.0.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

python3 -m pip install "mmengine>=0.7.1,<1.0.0" "mmdet==3.3.0"
