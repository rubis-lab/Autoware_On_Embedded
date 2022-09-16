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
  # INCLUDE_DIRS include
#  LIBRARIES rubis_logger
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(logger_topic
  include/rubis_logger/logger_topic.h
  src/logger_topic.cpp
)

add_executable(logger_brake_dist
  include/rubis_logger/logger_brake_dist.h
  src/logger_brake_dist.cpp
)

target_link_libraries(logger_topic
  ${catkin_LIBRARIES}
)
target_link_libraries(logger_brake_dist
  ${catkin_LIBRARIES}
)
add_dependencies(logger_topic
  ${catkin_EXPORTED_TARGETS}
)
add_dependencies(logger_brake_dist
  ${catkin_EXPORTED_TARGETS}
)

# add_dependencies(rubis_logger ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(rubis_logger
#   ${catkin_LIBRARIES}
# )