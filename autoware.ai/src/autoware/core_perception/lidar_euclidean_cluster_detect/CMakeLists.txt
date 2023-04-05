cmake_minimum_required(VERSION 2.8.3)
project(lidar_euclidean_cluster_detect)

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
  add_definitions(-D__aarch64__)
endif()



find_package(catkin REQUIRED COMPONENTS
  autoware_msgs
  geometry_msgs
  grid_map_cv
  grid_map_msgs
  grid_map_ros
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  tf
  vector_map_server
  rubis_lib
  rubis_msgs
)

find_package(OpenMP)
find_package(OpenCV REQUIRED)
find_package(Eigen3 QUIET)

set(CMAKE_CXX_FLAGS "-O2 -Wall ${CMAKE_CXX_FLAGS}")

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    vector_map_server
    grid_map_ros
    grid_map_cv
    grid_map_msgs
    rubis_lib
)

# Resolve system dependency on yaml-cpp, which apparently does not
# provide a CMake find_package() module.
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
find_path(YAML_CPP_INCLUDE_DIR NAMES yaml_cpp.h PATHS ${YAML_CPP_INCLUDE_DIRS})
find_library(YAML_CPP_LIBRARY NAMES YAML_CPP PATHS ${YAML_CPP_LIBRARY_DIRS})
link_directories(${YAML_CPP_LIBRARY_DIRS})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
link_directories(${OpenCV_LIBRARY_DIRS})

#Euclidean Cluster
add_executable(lidar_euclidean_cluster_detect
  nodes/lidar_euclidean_cluster_detect/lidar_euclidean_cluster_detect.cpp
  nodes/lidar_euclidean_cluster_detect/cluster.cpp
)

target_link_libraries(lidar_euclidean_cluster_detect
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

add_dependencies(lidar_euclidean_cluster_detect
  ${catkin_EXPORTED_TARGETS}
)

install(
  TARGETS
    lidar_euclidean_cluster_detect
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  PATTERN ".svn" EXCLUDE
)
