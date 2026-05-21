cmake_minimum_required(VERSION 3.5)
project(viplanner_ros1)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
)

catkin_install_python(PROGRAMS
  scripts/viplanner_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
