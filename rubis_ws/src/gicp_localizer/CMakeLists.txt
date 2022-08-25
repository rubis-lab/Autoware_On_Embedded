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

add_executable(gicp_localizer_node src/gicp.cpp)
target_link_libraries(gicp_localizer_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} fast_gicp)