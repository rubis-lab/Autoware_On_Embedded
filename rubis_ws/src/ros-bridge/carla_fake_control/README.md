# Carla Ackermann Control

ROS Node to forward OpenPlanner control messages to [CarlaEgoVehicleControl](carla_ros_bridge/msg/CarlaEgoVehicleControl.msg).

# Topics

Received /op_controller_cmd (autoware_msgs::VehicleCmd)
Publish /carla/<ROLE NAME>/vehicle_control_cmd (carla_msgs::CarlaEgoVehicleControl)

The role name is specified within the configuration.