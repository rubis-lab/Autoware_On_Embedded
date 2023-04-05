cmake_minimum_required(VERSION 2.8.3)
project(gnss)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    roslint
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES gnss
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(gnss
  src/geo_pos_conv.cpp
)

target_link_libraries(gnss
  ${catkin_LIBRARIES}
)

file(GLOB_RECURSE ROSLINT_FILES
  LIST_DIRECTORIES false
  *.cpp *.h *.hpp
)

list(APPEND ROSLINT_CPP_OPTS
  "--extensions=cc,h,hpp,cpp,cu,cuh"
  "--filter=-build/c++14"
)

roslint_cpp(${ROSLINT_FILES})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS gnss
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)