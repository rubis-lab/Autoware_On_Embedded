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
from transforms3d.euler import euler2quat

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
    walker_waypoint_ready=False
    def __init__(self):
        """
        Constructor
        """
        super(CarlaWalkerAgent, self).__init__('carla_walker_agent')

        role_name = self.get_param("role_name", "walker")
        self._walker_target_speed = float(self.get_param("walker_target_speed", 2.0))
        # self._fisrt_waypoint_x = self.get_param("fisrt_waypoint_x", 205.4)
        # self._fisrt_waypoint_y = self.get_param("fisrt_waypoint_y", 311.1)
        # self._second_waypoint_x = self.get_param("second_waypoint_x", 190.4)
        # self._second_waypoint_y = self.get_param("second_waypoint_y", 311.1)
        # self._third_waypoint_x = self.get_param("third_waypoint_x", 190.4)
        # self._third_waypoint_y = self.get_param("third_waypoint_y", 311.1)
        self._waypoint_params = []
        self._waypoints = []
        
        waypoint_iter=1
        while waypoint_iter is not 0:
            waypoint_param=self.get_param("walker_point"+str(waypoint_iter))
            if waypoint_param:
                self._waypoint_params.append(waypoint_param)
                waypoint_iter += 1
            else:
                waypoint_iter = 0

        
        rospy.logwarn(self._waypoint_params)
        
        #spawn_point_param = self.get_param("spawn_point_" + vehicle["id"], None)
        self._route_assigned = False
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
    
    def create_waypoint(self, x, y, z, roll, pitch, yaw):
        waypoint = Pose()
        waypoint.position.x = x
        waypoint.position.y = y
        waypoint.position.z = z
        quat = euler2quat(math.radians(roll), math.radians(pitch), math.radians(yaw))

        waypoint.orientation.w = quat[0]
        waypoint.orientation.x = quat[1]
        waypoint.orientation.y = quat[2]
        waypoint.orientation.z = quat[3]
        return waypoint
    
    def check_waypoint_param(self, waypoint_parameter):
        components = waypoint_parameter.split(',')
        if len(components) != 6:
            self.logwarn("Invalid spawnpoint '{}'".format(waypoint_parameter))
            return None
        waypoint = self.create_waypoint(
            float(components[0]),
            float(components[1]),
            float(components[2]),
            float(components[3]),
            float(components[4]),
            float(components[5])
        )
        return waypoint

    def set_waypoints(self):
        for _waypoint_param in self._waypoint_params:
            self._waypoints.append(self.check_waypoint_param(_waypoint_param))
        self.walker_waypoint_ready=True
        # rospy.logwarn(self._waypoints)

    def run_step(self):
        dist=math.sqrt((self._ego_current_pose.position.x-self._current_pose.position.x)**2+(self._ego_current_pose.position.y-self._current_pose.position.y)**2)
        # rospy.logwarn('dist: '+str(dist))
        if 30<dist<40 and self.walker_waypoint_ready==False:
            self.set_waypoints()
            # rospy.logwarn('dist: '+str(dist))
            # waypoints = Path()
            # for waypoint in self._waypoints:
            #     waypoints.append(waypoint)
            # rospy.logwarn(waypoints)
            # waypoint=Path()
            # pose1=PoseStamped()
            # pose1.pose=self._current_pose
            # pose2=PoseStamped()
            # pose2.pose=self._current_pose
            # pose3=PoseStamped()
            # pose3.pose=self._current_pose
            # pose1.pose.position.x=self._fisrt_waypoint_x
            # pose1.pose.position.y=self._fisrt_waypoint_y
            # pose2.pose.position.x=self._second_waypoint_x
            # pose2.pose.position.y=self._second_waypoint_y
            # pose3.pose.position.x=self._third_waypoint_x
            # pose3.pose.position.y=self._third_waypoint_y
            # waypoint.poses.append(pose1)
            # waypoint.poses.append(pose2)
            # waypoint.poses.append(pose3)
            # self.path_updated(waypoint)

        if self._waypoints:
            control = CarlaWalkerControl()
            direction = Vector3()
            direction.x = self._waypoints[0].position.x - self._current_pose.position.x
            direction.y = self._waypoints[0].position.y - self._current_pose.position.y
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            walker_target_speed = self._walker_target_speed
            if len(self._waypoints)==2:
                walker_target_speed = self._walker_target_speed/5
            # self.loginfo("next waypoint: {} {}".format(
            #             self._waypoints[0].position.x, self._waypoints[0].position.y))
            if direction_norm > CarlaWalkerAgent.MIN_DISTANCE:
                control.speed = walker_target_speed
                control.direction.x = direction.x / direction_norm
                control.direction.y = direction.y / direction_norm
            else:
                self._waypoints = self._waypoints[1:]
                # rospy.logwarn('pass waypoint')
                if self._waypoints:
                    self.loginfo("next waypoint: {} {}".format(
                        self._waypoints[0].position.x, self._waypoints[0].position.y))
                else:
                    self.loginfo("Route finished.")
                    self.walker_waypoint_ready=True
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
