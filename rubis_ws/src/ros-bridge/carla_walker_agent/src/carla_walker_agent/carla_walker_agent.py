#!/usr/bin/env python
#
# Copyright (c) 2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
Agent for Walker
"""

import math

import ros_compatibility as roscomp
from ros_compatibility.exceptions import ROSInterruptException
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy

from carla_msgs.msg import CarlaWalkerControl
from geometry_msgs.msg import Pose, Vector3, PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64
import rospy


class CarlaWalkerAgent(CompatibleNode):
    """
    walker agent
    """
    # minimum distance to target waypoint before switching to next
    MIN_DISTANCE = 0.5
    walker_moved=False
    def __init__(self):
        """
        Constructor
        """
        super(CarlaWalkerAgent, self).__init__('carla_walker_agent')

        role_name = self.get_param("role_name", "ego_vehicle")
        self._target_speed = self.get_param("target_speed", 2.0)
        self._fisrt_waypoint_x = self.get_param("fisrt_waypoint_x", 205.4)
        self._fisrt_waypoint_y = self.get_param("fisrt_waypoint_y", 311.1)
        self._second_waypoint_x = self.get_param("second_waypoint_x", 190.4)
        self._second_waypoint_y = self.get_param("second_waypoint_y", 311.1)
        self._third_waypoint_x = self.get_param("third_waypoint_x", 190.4)
        self._third_waypoint_y = self.get_param("third_waypoint_y", 311.1)

        self._route_assigned = False
        self._waypoints = []
        self._current_pose = Pose()
        self._ego_current_pose = Pose()

        # wait for ros bridge to create relevant topics
        try:
            self.wait_for_message("/carla/{}/odometry".format(role_name), Odometry, qos_profile=10)
        except ROSInterruptException as e:
            if not roscomp.ok:
                raise e

        self._odometry_subscriber = self.new_subscription(
            Odometry,
            "/carla/{}/odometry".format(role_name),
            self.odometry_updated,
            qos_profile=10)

        self._ego_odometry_subscriber = self.new_subscription(
            Odometry,
            "/carla/{}/odometry".format("ego_vehicle"),
            self.ego_odometry_updated,
            qos_profile=10)

        self.control_publisher = self.new_publisher(
            CarlaWalkerControl,
            "/carla/{}/walker_control_cmd".format(role_name),
            qos_profile=1)

        self._route_subscriber = self.new_subscription(
            Path,
            "/carla/{}/waypoints".format(role_name),
            self.path_updated,
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self._target_speed_subscriber = self.new_subscription(
            Float64, 
            "/carla/{}/target_speed".format(role_name),
            self.target_speed_updated,
            qos_profile=10)

    def _on_shutdown(self):
        """
        callback on shutdown
        """
        self.loginfo("Shutting down, stopping walker...")
        self.control_publisher.publish(CarlaWalkerControl())  # stop

    def target_speed_updated(self, target_speed):
        """
        callback on new target speed
        """
        self.loginfo("New target speed received: {}".format(target_speed.data))
        self._target_speed = target_speed.data

    def path_updated(self, path):
        """
        callback on new route
        """
        self.loginfo("New plan with {} waypoints received. Assigning plan...".format(
            len(path.poses)))
        self.control_publisher.publish(CarlaWalkerControl())  # stop
        self._waypoints = []
        for elem in path.poses:
            self._waypoints.append(elem.pose)

    def odometry_updated(self, odo):
        """
        callback on new odometry
        """
        self._current_pose = odo.pose.pose

    def ego_odometry_updated(self, odo):
        """
        callback on new odometry
        """
        self._ego_current_pose = odo.pose.pose

    def run_step(self):
        dist=math.sqrt((self._ego_current_pose.position.x-self._current_pose.position.x)**2+(self._ego_current_pose.position.y-self._current_pose.position.y)**2)
        if 30<dist<40 and self.walker_moved==False:
            waypoint=Path()
            pose1=PoseStamped()
            pose1.pose=self._current_pose
            pose2=PoseStamped()
            pose2.pose=self._current_pose
            pose3=PoseStamped()
            pose3.pose=self._current_pose
            pose1.pose.position.x=self._fisrt_waypoint_x
            pose1.pose.position.y=self._fisrt_waypoint_y
            pose2.pose.position.x=self._second_waypoint_x
            pose2.pose.position.y=self._second_waypoint_y
            pose3.pose.position.x=self._third_waypoint_x
            pose3.pose.position.y=self._third_waypoint_y
            waypoint.poses.append(pose1)
            waypoint.poses.append(pose2)
            waypoint.poses.append(pose3)
            self.path_updated(waypoint)
            self.walker_moved=True

        if self._waypoints:
            control = CarlaWalkerControl()
            direction = Vector3()
            direction.x = self._waypoints[0].position.x - self._current_pose.position.x
            direction.y = self._waypoints[0].position.y - self._current_pose.position.y
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            rospy.logwarn(len(self._waypoints))
            if direction_norm > CarlaWalkerAgent.MIN_DISTANCE:
                control.speed = self._target_speed
                control.direction.x = direction.x / direction_norm
                control.direction.y = direction.y / direction_norm
            else:
                self._waypoints = self._waypoints[1:]
                
                if len(self._waypoints)==2:
                    self._target_speed=0.1
                else:
                    self._target_speed=5.0
                if self._waypoints:
                    self.loginfo("next waypoint: {} {}".format(
                        self._waypoints[0].position.x, self._waypoints[0].position.y))
                else:
                    self.loginfo("Route finished.")
            self.control_publisher.publish(control)


def main(args=None):
    """

    main function

    :return:
    """
    roscomp.init("carla_walker_agent", args)
    controller = None

    try:
        controller = CarlaWalkerAgent()

        roscomp.on_shutdown(controller._on_shutdown)

        update_timer = controller.new_timer(
            0.05, lambda timer_event=None: controller.run_step())

        controller.spin()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()
        print("Done")


if __name__ == "__main__":
    main()
