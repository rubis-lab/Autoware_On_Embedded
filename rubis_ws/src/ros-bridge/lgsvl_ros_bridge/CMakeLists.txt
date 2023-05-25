cmake_minimum_required(VERSION 3.0.2)
project(lgsvl_ros_bridge)

catkin_python_setup()

catkin_package()

find_package(catkin REQUIRED COMPONENTS
  rosapi
  rosauth
  lgsvl_rosbridge_library
  rosbridge_msgs
  roscpp
  rospy
  rostest
  rubis_msgs
  std_msgs
)

catkin_install_python(PROGRAMS
  scripts/rosbridge_websocket.py
  scripts/rosbridge_websocket
  scripts/rosbridge_tcp.py
  scripts/rosbridge_tcp
  scripts/rosbridge_udp
  scripts/rosbridge_udp.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

install(
  DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

if (CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest(test/websocket/test_smoke.test)
endif()
