from setuptools import find_packages, setup

setup(
    name="planner_validation_ros1",
    version="0.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
