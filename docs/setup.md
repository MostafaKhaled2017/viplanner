# Setup

## Python Installation

Install the Python package from the repository root:

```bash
python3 -m pip install .
```

Editable install for local development:

```bash
python3 -m pip install -e .[standard]
```

Inference install:

```bash
python3 -m pip install -e .[standard,inference]
```

Training install with TensorBoard support:

```bash
python3 -m pip install -e .[standard,training]
```

Jetson install path from `README.md`:

```bash
python3 -m pip install -e .[inference,jetson]
```

## ROS Environment

The repository targets ROS Noetic. The top-level `Dockerfile` and `.devcontainer/` define ROS Noetic development environments.

Common ROS setup after building:

```bash
source devel/setup.bash
```

## Submodules

The repository has a `.gitmodules` file and references a Mask2Former third-party submodule. For RGB or semantic extensions, `README.md` instructs:

```bash
git submodule update --init
python3 -m pip install -r third_party/mask2former/requirements.txt
```

The local path used in tests and configs is `viplanner/third_party/mask2former`.

## CUDA Notes

`README.md` assumes CUDA 11.7 for the documented MMCV wheel command. If the CUDA toolkit is not found, set:

```bash
export CUDA_HOME=/usr/local/cuda
```

## Development Container

`.devcontainer/devcontainer.json` defines a ROS Noetic dev container with GPU, privileged mode, host networking, X11 mount, and root user.

## Unknowns

- Exact host OS and GPU driver requirements outside the documented Docker/devcontainer setup are not fully specified in repository files.
- Dataset availability is environment-specific.
