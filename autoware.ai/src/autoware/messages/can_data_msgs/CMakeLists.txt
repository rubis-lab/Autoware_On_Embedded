cmake_minimum_required(VERSION 2.8.3)
project(can_data_msgs)

find_package(catkin REQUIRED
  COMPONENTS
    message_generation
    std_msgs
)

add_message_files(
  DIRECTORY msg
  FILES
    Car_ctrl_output.msg
    Car_ctrl_input.msg
)

generate_messages(
  DEPENDENCIES
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs
)