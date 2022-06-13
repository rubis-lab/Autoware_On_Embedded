cmake_minimum_required(VERSION 3.0.2)
project(gnss_module)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  inertiallabs_msgs
  geometry_msgs
  rubis_msgs
  message_filters
  tf
)

find_package(Eigen3 REQUIRED)
set(EIGEN_PACKAGE EIGEN3)
if(NOT EIGEN3_FOUND)
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})
  set(EIGEN_PACKAGE Eigen)
endif()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gnss_converter
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(gnss_module
  include/LLH2UTM.h
  include/LKF.h
  include/quaternion_euler.h
  src/gnss_module_node.cpp
  src/gnss_module.cpp
  src/LKF.cpp
  src/LLH2UTM.cpp
)
add_dependencies(gnss_module ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(gnss_module
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

# install(
#   TARGETS
#     gnss_module
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# install(DIRECTORY include/
#   DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
#   PATTERN ".svn" EXCLUDE
# )
# install(DIRECTORY launch/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
#   PATTERN ".svn" EXCLUDE
# )

# install(DIRECTORY cfg/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cfg
# )