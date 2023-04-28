cmake_minimum_required(VERSION 3.0.2)
project(ins_twist_generator)

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
  can_data_msgs
  rubis_msgs
  message_filters
  tf
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gnss_converter
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(ins_twist_generator
  src/ins_twist_generator.cpp
)
add_dependencies(ins_twist_generator ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(ins_twist_generator
  ${catkin_LIBRARIES}
)

add_executable(dynamic_ins_twist_generator
  src/dynamic_ins_twist_generator.cpp
)
add_dependencies(dynamic_ins_twist_generator ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(dynamic_ins_twist_generator
  ${catkin_LIBRARIES}
)

install(
  TARGETS
    ins_twist_generator
    dynamic_ins_twist_generator
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)
