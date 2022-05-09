cmake_minimum_required(VERSION 3.0.2)
project(rubis_logger)

find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  autoware_msgs
  geometry_msgs
  can_data_msgs
  std_msgs
  nav_msgs
  rubis_logger_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rubis_logger
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
  include ${catkin_INCLUDE_DIRS}
# include
# ${catkin_INCLUDE_DIRS}
)

add_executable(rubis_logger
  include/rubis_logger/rubis_logger.h
  src/rubis_logger.cpp
)
add_dependencies(rubis_logger ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(rubis_logger
  ${catkin_LIBRARIES}
)