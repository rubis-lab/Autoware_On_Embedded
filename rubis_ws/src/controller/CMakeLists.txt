cmake_minimum_required(VERSION 3.0.2)
project(controller)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  autoware_msgs
  geometry_msgs
  can_data_msgs
  std_msgs
  nav_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES controller
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
  include ${catkin_INCLUDE_DIRS}
# include
# ${catkin_INCLUDE_DIRS}
)


add_executable(controller
  include/controller/controller.h
  src/controller.cpp
)
add_dependencies(controller ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(controller
  ${catkin_LIBRARIES}
)

install(
  TARGETS
    controller
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

install(DIRECTORY cfg/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cfg
)