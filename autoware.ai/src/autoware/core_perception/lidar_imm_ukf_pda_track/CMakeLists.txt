cmake_minimum_required(VERSION 2.8.3)
project(imm_ukf_pda_track)

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
  add_definitions(-D__aarch64__)
endif()



find_package(catkin REQUIRED COMPONENTS
  amathutils_lib
  autoware_msgs
  geometry_msgs
  pcl_ros
  roscpp
  roslint
  tf
  vector_map
  rubis_lib
  rubis_msgs
)


set(CMAKE_CXX_FLAGS "-O2 -Wall ${CMAKE_CXX_FLAGS}")

catkin_package(
  CATKIN_DEPENDS
    vector_map
    amathutils_lib
)

set(ROSLINT_CPP_OPTS "--filter=-build/c++14")

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

#imm_ukf_pda
add_executable(imm_ukf_pda
  nodes/imm_ukf_pda/imm_ukf_pda_main.cpp
  nodes/imm_ukf_pda/imm_ukf_pda.cpp
  nodes/imm_ukf_pda/ukf.cpp
)
target_link_libraries(imm_ukf_pda
  ${catkin_LIBRARIES}
)
add_dependencies(imm_ukf_pda
  ${catkin_EXPORTED_TARGETS}
)

install(
  TARGETS
    imm_ukf_pda
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)