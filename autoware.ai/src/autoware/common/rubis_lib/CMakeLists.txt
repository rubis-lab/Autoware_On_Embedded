cmake_minimum_required(VERSION 2.8.3)
project(rubis_lib)



find_package(PCL REQUIRED)

if(NOT (PCL_VERSION VERSION_LESS "1.7.2"))
  SET(PCL_OPENMP_PACKAGES pcl_omp_registration)
endif()

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  sensor_msgs
  rubis_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rubis_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(rubis_lib SHARED
  src/sched_c.c
  src/sched_profiling_c.c
  src/sched.cpp
  src/sched_profiling.cpp
)

target_link_libraries(rubis_lib
  ${catkin_LIBRARIES}
)

target_include_directories(rubis_lib PRIVATE
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

install(TARGETS rubis_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
)
