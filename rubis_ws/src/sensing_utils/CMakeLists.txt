cmake_minimum_required(VERSION 3.0.2)
project(sensing_utils)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  inertiallabs_msgs
  tf
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES sensing_utils
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
  include ${catkin_INCLUDE_DIRS}
)

add_executable(odom_converter
  src/odom_converter.cpp
)
add_dependencies(odom_converter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(odom_converter
  ${catkin_LIBRARIES}
)

add_executable(quaternion_to_rpy
  src/quaternion_to_rpy.cpp
)
add_dependencies(quaternion_to_rpy ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(quaternion_to_rpy
  ${catkin_LIBRARIES}
)

add_executable(ins_sync_test
  src/ins_sync_test.cpp
)
add_dependencies(ins_sync_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(ins_sync_test
  ${catkin_LIBRARIES}
)

add_executable(yaw_offset_calculator
  src/yaw_offset_calculator.cpp
)
add_dependencies(yaw_offset_calculator ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(yaw_offset_calculator
  ${catkin_LIBRARIES}
)

install(
  TARGETS
    odom_converter
    quaternion_to_rpy
    ins_sync_test
    yaw_offset_calculator
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)
