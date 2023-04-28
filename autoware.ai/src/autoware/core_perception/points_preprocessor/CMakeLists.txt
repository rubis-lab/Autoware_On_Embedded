cmake_minimum_required(VERSION 2.8.3)
project(points_preprocessor)

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
  add_definitions(-D__aarch64__)
endif()



find_package(catkin REQUIRED COMPONENTS
  autoware_msgs
  autoware_config_msgs
  cv_bridge
  pcl_conversions
  pcl_ros
  roscpp
  roslint
  sensor_msgs
  std_msgs
  tf
  tf2
  tf2_eigen
  tf2_ros
  velodyne_pointcloud
  rubis_lib
  rubis_msgs
)

catkin_package()

set(ROSLINT_CPP_OPTS "--filter=-build/c++14")
roslint_cpp(
  nodes/ray_ground_filter/ray_ground_filter.cpp
  nodes/ray_ground_filter/ray_ground_filter_main.cpp
  include/points_preprocessor/ray_ground_filter/ray_ground_filter.h
)

find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(OpenMP)

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
)

SET(CMAKE_CXX_FLAGS "-O2 -g -Wall ${CMAKE_CXX_FLAGS}")

link_directories(${PCL_LIBRARY_DIRS})

# Ray Ground Filter
add_library(ray_ground_filter_lib SHARED
  nodes/ray_ground_filter/ray_ground_filter.cpp
)

if(OPENMP_FOUND)
  set_target_properties(ray_ground_filter_lib PROPERTIES
    COMPILE_FLAGS ${OpenMP_CXX_FLAGS}
    LINK_FLAGS ${OpenMP_CXX_FLAGS}
  )
endif()

target_include_directories(ray_ground_filter_lib PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(ray_ground_filter_lib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_dependencies(ray_ground_filter_lib ${catkin_EXPORTED_TARGETS})

add_executable(ray_ground_filter
  nodes/ray_ground_filter/ray_ground_filter_main.cpp
)
target_link_libraries(ray_ground_filter ray_ground_filter_lib)
add_dependencies(ray_ground_filter ${catkin_EXPORTED_TARGETS})

#Compare Map Filter
add_executable(compare_map_filter
  nodes/compare_map_filter/compare_map_filter.cpp
)

target_include_directories(compare_map_filter PRIVATE
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(compare_map_filter
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
add_dependencies(compare_map_filter ${catkin_EXPORTED_TARGETS})


install(
  TARGETS
    ray_ground_filter_lib
    ray_ground_filter
    compare_map_filter
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
  PATTERN ".svn" EXCLUDE
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)
