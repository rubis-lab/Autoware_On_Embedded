cmake_minimum_required(VERSION 2.8.3)
project(rubis_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  jsk_recognition_msgs
  message_generation
  sensor_msgs
  std_msgs
  autoware_msgs
)

add_message_files(
  DIRECTORY msg
  FILES
    PointCloud2.msg
    PoseStamped.msg
    TwistStamped.msg
    VehicleCmd.msg
    InsStat.msg
)

generate_messages(
  DEPENDENCIES
    autoware_msgs
    geometry_msgs
    jsk_recognition_msgs
    sensor_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    autoware_msgs
    geometry_msgs
    jsk_recognition_msgs
    message_runtime
    sensor_msgs
    std_msgs
)
