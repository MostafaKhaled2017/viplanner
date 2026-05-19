import os
from glob import glob

from setuptools import find_packages, setup

package_name = "viplanner_ros2"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [os.path.join("resource", package_name)]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Pascal Roth",
    maintainer_email="rothpa@ethz.ch",
    description="Standalone ROS2 evaluation node for VIPlanner checkpoints.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "viplanner_node = viplanner_ros2.viplanner_node:main",
        ],
    },
)
