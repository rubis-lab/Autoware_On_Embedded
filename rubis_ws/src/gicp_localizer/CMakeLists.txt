cmake_minimum_required(VERSION 3.0.2)
project(gicp_localizer)

set(CMAKE_CXX_STANDARD 14)

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
else()
    add_definitions(-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
    set(CMAKE_C_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")
    set(CMAKE_CXX_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")
endif()

set(CMAKE_BUILD_TYPE "Release")

find_package(catkin REQUIRED COMPONENTS
  diagnostic_msgs
  geometry_msgs
  message_generation
  nav_msgs
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

generate_messages(
        DEPENDENCIES
        std_msgs
)

find_package(PCL REQUIRED QUIET)
find_package(Eigen3 REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

catkin_package(
        INCLUDE_DIRS include
        CATKIN_DEPENDS
        roscpp
        tf2
        tf2_ros
        tf2_geometry_msgs
        geometry_msgs
        sensor_msgs
        nav_msgs
        diagnostic_msgs
        pcl_conversions
        pcl_ros
        message_generation
)

if(catkin_FOUND)
    catkin_package(
        INCLUDE_DIRS include
        LIBRARIES fast_gicp
    )
endif()

include_directories(include/gicp_localizer ${catkin_INCLUDE_DIRS})

###########
## Build ##
###########

add_library(fast_gicp SHARED
    src/fast_gicp/gicp/lsq_registration.cpp
    src/fast_gicp/gicp/fast_gicp.cpp
    src/fast_gicp/gicp/fast_gicp_st.cpp
    src/fast_gicp/gicp/fast_vgicp.cpp
)
target_link_libraries(
    fast_gicp ${PCL_LIBRARIES}
)
target_include_directories(
    fast_gicp PUBLIC
    include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)


SET(CMAKE_CXX_FLAGS "-O2 -g -Wall ${CMAKE_CXX_FLAGS}")

add_executable(voxel_grid_filter src/voxel_grid_filter.cpp)

add_dependencies(voxel_grid_filter ${catkin_EXPORTED_TARGETS})

target_link_libraries(voxel_grid_filter ${catkin_LIBRARIES})

add_executable(map_loader src/mapLoader.cpp)
target_link_libraries(map_loader ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(gicp_localizer_node src/gicp.cpp)
target_link_libraries(gicp_localizer_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} fast_gicp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_gicp_localizer.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
