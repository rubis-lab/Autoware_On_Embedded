cmake_minimum_required(VERSION 2.8.3)
project(rubis_pkg)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  jsk_recognition_msgs
  autoware_msgs
  geometry_msgs
  nmea_msgs
  std_msgs
  tf
  gnss
  rubis_lib
  rubis_msgs
  visualization_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rubis_pkg
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include ${catkin_INCLUDE_DIRS}
# include
# ${catkin_INCLUDE_DIRS}
)

add_executable(lidar_republisher
  src/lidar_republisher.cpp
)
add_dependencies(lidar_republisher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(lidar_republisher
  ${catkin_LIBRARIES}
)

add_executable(camera_republisher
  src/camera_republisher.cpp
)
add_dependencies(camera_republisher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(camera_republisher
  ${catkin_LIBRARIES}
)

add_executable(gnss_republisher
  src/gnss_republisher.cpp
)
add_dependencies(gnss_republisher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(gnss_republisher
  ${catkin_LIBRARIES}
)

add_executable(gnss_localizer
  src/gnss_localizer_node.cpp
  src/gnss_localizer.cpp
)
add_dependencies(gnss_localizer ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(gnss_localizer
  ${catkin_LIBRARIES}
)

add_executable(fake_current_pose
  src/fake_current_pose.cpp
)
add_dependencies(fake_current_pose ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(fake_current_pose
  ${catkin_LIBRARIES}
)

add_executable(fake_traffic_signal_generator
  src/fake_traffic_signal_generator.cpp
)
add_dependencies(fake_traffic_signal_generator ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(fake_traffic_signal_generator
  ${catkin_LIBRARIES}
)

add_executable(rubis_pose_relay
  src/rubis_pose_relay.cpp
)
add_dependencies(rubis_pose_relay ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(rubis_pose_relay
  ${catkin_LIBRARIES}
)

install(
  TARGETS
    lidar_republisher
    camera_republisher
    gnss_republisher
    gnss_localizer
    fake_current_pose
    fake_traffic_signal_generator
    rubis_pose_relay
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
