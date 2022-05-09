cmake_minimum_required(VERSION 2.8.3)
project(can_translate)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  can_msgs
  can_data_msgs
  std_msgs
  kvaser_interface
  roslint
)

catkin_package(
  CATKIN_DEPENDS roscpp can_msgs kvaser_interface can_data_msgs std_msgs
  INCLUDE_DIRS include
)

include_directories(
  include ${catkin_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
  src/can_translate.cpp
)

add_dependencies(can_translate 
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(can_translate
  ${catkin_LIBRARIES}
)