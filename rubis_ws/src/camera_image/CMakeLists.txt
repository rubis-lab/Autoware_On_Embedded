cmake_minimum_required(VERSION 2.8.3)
project(camera_image)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED)

# add by Park
find_package(OpenCV REQUIRED)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_generation
  image_transport
  cv_bridge
  rubis_lib
)

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
#  INCLUDE_DIRS include
#  LIBRARIES camera_image
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
  include ${catkin_INCLUDE_DIRS}
  #add by Park
  include ${OpenCV_INCLUDE_DIRS}
# include
# ${catkin_INCLUDE_DIRS}
)


add_executable(camera_image
  src/camera_image.cpp
)
add_dependencies(camera_image ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(camera_image
  ${catkin_LIBRARIES}
  # add by Park
  ${OpenCV_LIBS}
)

add_executable(camera_image_sub
  src/camera_image_sub.cpp
)
add_dependencies(camera_image_sub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(camera_image_sub
  ${catkin_LIBRARIES}
  #add by Park
  ${OpenCV_LIBS}
)

install(
  TARGETS
    camera_image
    camera_image_sub
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)

# install(DIRECTORY cfg/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cfg
# )
