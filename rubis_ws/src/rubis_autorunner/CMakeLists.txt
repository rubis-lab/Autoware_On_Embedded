cmake_minimum_required(VERSION 3.0.2)
project(rubis_autorunner)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(autoware_msgs REQUIRED)
find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  autoware_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES minicar_autorunner
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
  include ${catkin_INCLUDE_DIRS}
  ${autoware_msgs_INCLUDE_DIRS}
)

add_library(ros_autorunner_lib
  include/ros_autorunner_lib/ros_autorunner_def.h
  include/ros_autorunner_lib/ros_autorunner.h
  src/ros_autorunner_lib/ros_autorunner.cpp
)

add_executable(lgsvl_triple_lidar_autorunner
  include/lgsvl_triple_lidar_autorunner/lgsvl_triple_lidar_autorunner.h
  src/lgsvl_triple_lidar_autorunner/lgsvl_triple_lidar_autorunner_node.cpp
  src/lgsvl_triple_lidar_autorunner/lgsvl_triple_lidar_autorunner.cpp
)

add_executable(desktop_lane_keeping
  include/desktop_lane_keeping/desktop_lane_keeping.h
  src/desktop_lane_keeping/desktop_lane_keeping_node.cpp
  src/desktop_lane_keeping/desktop_lane_keeping.cpp
)

add_executable(minicar_lane_keeping
  include/minicar_lane_keeping/minicar_lane_keeping.h
  src/minicar_lane_keeping/minicar_lane_keeping_node.cpp
  src/minicar_lane_keeping/minicar_lane_keeping.cpp
)
  
add_executable(cubetown_lkas_autorunner
  include/cubetown_autorunner/cubetown_autorunner.h
  src/cubetown_autorunner/cubetown_lkas_autorunner_node.cpp
  src/cubetown_autorunner/cubetown_lkas_autorunner.cpp
)

add_executable(cubetown_full_autorunner
  include/cubetown_autorunner/cubetown_autorunner.h
  src/cubetown_autorunner/cubetown_full_autorunner_node.cpp
  src/cubetown_autorunner/cubetown_full_autorunner.cpp
)

add_executable(tutorial_autorunner
  include/tutorial_autorunner/tutorial_autorunner.h
  src/tutorial_autorunner/tutorial_autorunner_node.cpp
  src/tutorial_autorunner/tutorial_autorunner.cpp
)

add_executable(rubis_testbed_autorunner
  include/rubis_testbed_autorunner/rubis_testbed_autorunner.h
  src/rubis_testbed_autorunner/rubis_testbed_autorunner_node.cpp
  src/rubis_testbed_autorunner/rubis_testbed_autorunner.cpp
)

add_dependencies(lgsvl_triple_lidar_autorunner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(lgsvl_triple_lidar_autorunner
  ${catkin_LIBRARIES}
  ros_autorunner_lib
)

add_dependencies(desktop_lane_keeping ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(desktop_lane_keeping
  ${catkin_LIBRARIES}
  ros_autorunner_lib
)

add_dependencies(minicar_lane_keeping ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(minicar_lane_keeping
  ${catkin_LIBRARIES}
  ros_autorunner_lib
)

add_dependencies(cubetown_lkas_autorunner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(cubetown_lkas_autorunner
  ${catkin_LIBRARIES}
  ros_autorunner_lib
)

add_dependencies(cubetown_full_autorunner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(cubetown_full_autorunner
  ${catkin_LIBRARIES}
  ros_autorunner_lib
)

add_dependencies(tutorial_autorunner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(tutorial_autorunner
  ${catkin_LIBRARIES}
  ros_autorunner_lib
)

add_dependencies(rubis_testbed_autorunner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(rubis_testbed_autorunner
  ${catkin_LIBRARIES}
  ros_autorunner_lib
)