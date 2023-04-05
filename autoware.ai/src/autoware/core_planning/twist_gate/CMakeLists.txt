cmake_minimum_required(VERSION 2.8.3)
project(twist_gate)

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
  add_definitions(-D__aarch64__)
endif()


find_package(
  catkin REQUIRED COMPONENTS
    autoware_config_msgs
    autoware_msgs
    geometry_msgs
    ros_observer
    roscpp
    roslint
    std_msgs
    tablet_socket_msgs
    rubis_lib
    rubis_msgs
)

catkin_package(
  INCLUDE_DIRS include
)

SET(CMAKE_CXX_FLAGS "-O2 -g -Wall ${CMAKE_CXX_FLAGS}")
set(ROSLINT_CPP_OPTS "--filter=-build/c++11")

roslint_cpp()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(twist_gate src/twist_gate_node.cpp src/twist_gate.cpp)
target_link_libraries(twist_gate ${catkin_LIBRARIES})
add_dependencies(twist_gate ${catkin_EXPORTED_TARGETS})

install(
  TARGETS twist_gate
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
