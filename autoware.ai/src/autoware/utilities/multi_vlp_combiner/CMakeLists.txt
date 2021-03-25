cmake_minimum_required(VERSION 3.0.2)
project(multi_vlp_combiner)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  pcl_ros
)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(multi_vlp_combiner_lib SHARED
  src/combiner.cpp
)

target_link_libraries(multi_vlp_combiner_lib
  ${catkin_LIBRARIES}
)

add_executable(multi_vlp_combiner src/combiner.cpp)

target_link_libraries(multi_vlp_combiner
  multi_vlp_combiner_lib
)

install(
  TARGETS
    multi_vlp_combiner
    multi_vlp_combiner_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)