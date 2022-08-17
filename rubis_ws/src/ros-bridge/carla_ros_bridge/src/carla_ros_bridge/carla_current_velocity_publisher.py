#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped

pub_ = rospy.Publisher('carla_current_velocity', TwistStamped, queue_size=10)
msg_ = TwistStamped()

def speed_callback(data):
    msg_.twist.linear.x = data.data

def imu_callback(data):
    msg_.header = data.header
    msg_.twist.angular = data.angular_velocity
    pub_.publish(msg_)

def listener():
    rospy.init_node('carla_current_velocity_publisher', anonymous=True)
    rospy.Subscriber('/carla/ego_vehicle/speedometer', Float32, speed_callback)
    rospy.Subscriber('/carla/ego_vehicle/imu', Imu, imu_callback)
    rospy.spin()

if __name__ == "__main__":
    listener()