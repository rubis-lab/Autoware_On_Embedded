cmake_minimum_required(VERSION 3.0.2)
project(rubis_autorunner)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(autoware_msgs REQUIRED)
find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  autoware_msgs
  visualization_msgs
  nmea_msgs
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


add_executable(rubis_testbed_autorunner
  include/rubis_testbed_autorunner/rubis_testbed_autorunner.h
  src/rubis_testbed_autorunner/rubis_testbed_autorunner_node.cpp
  src/rubis_testbed_autorunner/rubis_testbed_autorunner.cpp
)

add_executable(carla_lkas_autorunner
  include/carla_autorunner/carla_autorunner.h
  src/carla_autorunner/carla_lkas_autorunner_node.cpp
  src/carla_autorunner/carla_lkas_autorunner.cpp
)

add_executable(carla_full_autorunner
  include/carla_autorunner/carla_autorunner.h
  src/carla_autorunner/carla_full_autorunner_node.cpp
  src/carla_autorunner/carla_full_autorunner.cpp
)

add_executable(exynos_carla_lkas_autorunner
  include/carla_autorunner/carla_autorunner.h
  src/carla_autorunner/exynos_carla_lkas_autorunner_node.cpp
  src/carla_autorunner/exynos_carla_lkas_autorunner.cpp
)

add_executable(exynos_carla_full_autorunner
  include/carla_autorunner/carla_autorunner.h
  src/carla_autorunner/exynos_carla_full_autorunner_node.cpp
  src/carla_autorunner/exynos_carla_full_autorunner.cpp
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

add_dependencies(rubis_testbed_autorunner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(rubis_testbed_autorunner
  ${catkin_LIBRARIES}
  ros_autorunner_lib
)

add_dependencies(carla_lkas_autorunner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(carla_lkas_autorunner
  ${catkin_LIBRARIES}
  ros_autorunner_lib
)

add_dependencies(carla_full_autorunner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(carla_full_autorunner
  ${catkin_LIBRARIES}
  ros_autorunner_lib
)

add_dependencies(exynos_carla_lkas_autorunner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(exynos_carla_lkas_autorunner
  ${catkin_LIBRARIES}
  ros_autorunner_lib
)

add_dependencies(exynos_carla_full_autorunner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(exynos_carla_full_autorunner
  ${catkin_LIBRARIES}
  ros_autorunner_lib
)

install(
  TARGETS
    ros_autorunner_lib
    carla_lkas_autorunner
    carla_full_autorunner
    exynos_carla_lkas_autorunner
    exynos_carla_full_autorunner
    cubetown_lkas_autorunner
    cubetown_full_autorunner
    rubis_testbed_autorunner
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

install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
)
