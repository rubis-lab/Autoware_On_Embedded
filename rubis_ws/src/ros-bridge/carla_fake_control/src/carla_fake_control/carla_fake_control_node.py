#!/usr/bin/env python

#
# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Modified by Hatem Darweesh
"""
Forward command from Autoware VehicleCmd to CRLA ROS bridge
"""
import sys
import datetime
import numpy
import rospy

from autoware_msgs.msg import VehicleCmd
from carla_msgs.msg import CarlaEgoVehicleControl  # pylint: disable=no-name-in-module,import-error

pub = None
steerAngle = 0

def op_callback(data):
    """
    forward method
    """
    #print("Received Data in fake controller %s ", data.accel_cmd.accel)
    cmd_msg = CarlaEgoVehicleControl()
    # cmd_msg.steer = data.steer_cmd.steer/100.0 # will use pure_pursuit controller
    cmd_msg.steer = steerAngle
    acceleration = data.accel_cmd.accel/100.0
    if acceleration < 0:
        cmd_msg.reverse = 1
    else:
        cmd_msg.reverse = 0

    cmd_msg.throttle = abs(acceleration)
    cmd_msg.brake = data.brake_cmd.brake/100.
    cmd_msg.gear = int(data.gear_cmd.gear)
    
    # print(data.gear_cmd)
    # print(data.gear_cmd.gear)
    #data.lamp_cmd.l 
    #data.lamp_cmd.r
    #data.mode 
    #data.twist_cmd
    #data.ctrl_cmd             
    pub.publish(cmd_msg)

def pp_callback(data):
    global steerAngle
    """
    forward method
    """
    
    steerAngle = -data.ctrl_cmd.steering_angle*180/3.141/70.0
    


def main():
    """
    Main method
    """
    global pub
    rospy.init_node('carla_fake_control', anonymous=True)

    role_name = rospy.get_param("/role_name", "ego_vehicle")

    # to send command to carla
    pub = rospy.Publisher("/carla/" + role_name + "/vehicle_control_cmd", CarlaEgoVehicleControl, queue_size=1)

    # Autoware drive commands
    rospy.Subscriber("/op_controller_cmd", VehicleCmd, op_callback)
    rospy.Subscriber("/vehicle_cmd", VehicleCmd, pp_callback)

    rospy.spin()


if __name__ == "__main__":
    main()
