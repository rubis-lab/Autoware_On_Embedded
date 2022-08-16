cmake_minimum_required(VERSION 2.8.3)
project(carla_fake_control)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs    
  carla_msgs
  autoware_msgs
)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  src/carla_fake_control/carla_fake_control_node.py  
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
