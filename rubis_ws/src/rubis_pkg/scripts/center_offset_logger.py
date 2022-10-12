import rospy
import argparse
from autoware_msgs.msg import LaneArray, VehicleCmd
from geometry_msgs.msg import PoseStamped
import math
import time
import csv
import datetime as dt

POSE_TOPIC = "/ndt_pose"

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def dis(wp1, wp2):
    return math.sqrt((wp1[0] - wp2[0]) * (wp1[0] - wp2[0]) + (wp1[1] - wp2[1]) * (wp1[1] - wp2[1]))

def find_closest_point(map_wp_list, wp, yaw_deg):
    min_distance = 500
    min_wp = [0, 0]

    for map_wp in map_wp_list:
        if dis(map_wp, wp) < min_distance:
            min_distance = dis(map_wp, wp)
            min_wp = map_wp

    if 45 <= yaw_deg and yaw_deg < 135:
        if wp[0] < min_wp[0]:
            min_distance *= -1
    elif 135 <= yaw_deg and yaw_deg < 225:
        if wp[1] < min_wp[1]:
            min_distance *= -1
    elif 225 <= yaw_deg and yaw_deg < 315:
        if wp[0] > min_wp[0]:
            min_distance *= -1
    elif wp[1] > min_wp[1]:
        min_distance *= 1
    
    return min_wp, min_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='output log file name')
    args = parser.parse_args()

    rospy.init_node('center_offset_logger')

    map_wp_list = []

    gnss_x = 0
    gnss_y = 0

    lane_msg = rospy.wait_for_message('/lane_waypoints_array', LaneArray, timeout=None)

    print('[System] Vector Map Loaded')

    for wp in lane_msg.lanes[0].waypoints:
        map_wp_list.append([wp.pose.pose.position.x, wp.pose.pose.position.y])

    # Wait until car starts
    rospy.wait_for_message('/vehicle_cmd', VehicleCmd, timeout=None)

    print('[System] Car moved and start logging')

    ts = dt.datetime.now()
    file_name = './data/' + str(ts.year % 100) + str(ts.month).zfill(2) + str(ts.day).zfill(2) + '_' + args.output_file + '.csv'

    with open(file_name, "w") as f:
        wr = csv.writer(f)
        wr.writerow(['ts', 'center_offset'])
        prev_dis = 0
        while not rospy.is_shutdown():
            gnss_msg = rospy.wait_for_message(POSE_TOPIC, PoseStamped, timeout=None)
            gnss_x = round(gnss_msg.pose.position.x, 3)
            gnss_y = round(gnss_msg.pose.position.y, 3)
            ori_x = gnss_msg.pose.orientation.x
            ori_y = gnss_msg.pose.orientation.y
            ori_z = gnss_msg.pose.orientation.z
            ori_w = gnss_msg.pose.orientation.w
            r, p, y = euler_from_quaternion(ori_x, ori_y, ori_z, ori_w)

            yaw_deg = (y * 180 / math.pi + 1800) % 360
            
            min_wp, min_dis = find_closest_point(map_wp_list, [gnss_x, gnss_y], yaw_deg)

            if abs(prev_dis) > 1.5 and min_dis * prev_dis < 0:
                min_dis *= 1
            prev_dis = min_dis

            wr.writerow([time.clock_gettime(time.CLOCK_MONOTONIC), str(min_dis)])