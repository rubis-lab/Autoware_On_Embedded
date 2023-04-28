cmake_minimum_required(VERSION 2.8.3)
project(inertiallabs_ins)

find_package(catkin REQUIRED COMPONENTS
tf
rospy
roscpp
geometry_msgs
sensor_msgs
std_msgs
dynamic_reconfigure
actionlib_msgs
inertiallabs_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

find_package(OpenCV)

set(DEPRECATION_FLAG "-Wsizeof-array-argument")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra ${DEPRECATION_FLAG}")

catkin_package(
#  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy sensor_msgs geometry_msgs tf dynamic_reconfigure  actionlib_msgs inertiallabs_msgs
  DEPENDS system_lib
)

include_directories(${Opencv_INCLUDE_DIRS})
include_directories(${system_lib_INCLUDE_DIRS})

include_directories(src/inertiallabs_sdk/)

if(WIN32)
	set(IL_PLATFORM "windows")
else()
	set(IL_PLATFORM "linux")
endif()

add_executable(il_ins src/il_ins.cpp
                      src/inertiallabs_sdk/ILDriver.cpp
                      src/inertiallabs_sdk/UDDParser.cpp
                      src/inertiallabs_sdk/platforms/${IL_PLATFORM}/SerialPort.cpp
                      src/inertiallabs_sdk/platforms/${IL_PLATFORM}/NetClient.cpp
)
add_dependencies(il_ins inertiallabs_msgs_generate_messages_cpp)
if(WIN32)
	target_link_libraries(il_ins ${catkin_LIBRARIES} )
else()
	target_link_libraries(il_ins ${catkin_LIBRARIES} pthread )
endif()

install(
  TARGETS
    il_ins
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)
