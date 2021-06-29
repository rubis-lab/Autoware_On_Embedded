cmake_minimum_required(VERSION 2.8.3)
project(rubis_sched)

find_package(autoware_build_flags REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rubis_sched
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(rubis_sched SHARED
  src/sched.cpp
  src/sched_profiling.cpp
)

target_link_libraries(rubis_sched
  ${catkin_LIBRARIES}
)

install(TARGETS rubis_sched
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)
