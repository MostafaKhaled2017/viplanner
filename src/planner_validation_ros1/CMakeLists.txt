cmake_minimum_required(VERSION 3.5)
project(planner_validation_ros1)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  tf2_ros
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
  rospy
  std_msgs
  geometry_msgs
  tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/planner_validation_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch scenarios
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
