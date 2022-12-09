import yaml
import matplotlib.pyplot as plt
import rospy
import argparse
from autoware_msgs.msg import LaneArray, NDTStat, VehicleCmd
from geometry_msgs.msg import PoseStamped
from rubis_msgs.msg import PoseStamped as RubisPoseStamped
from visualization_msgs.msg import MarkerArray
import math
import time
import csv
from os import getenv, makedirs

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

def center_off(output_file):
    if output_file.split('.')[-1] != 'csv':
        print('Output file should be csv file!')
        exit(1)

    rospy.init_node('driving_progress_logger')

    map_wp_list = []

    pose_x = 0
    pose_y = 0

    lane_msg = rospy.wait_for_message('/lane_waypoints_array', LaneArray, timeout=None)

    for wp in lane_msg.lanes[0].waypoints:
        map_wp_list.append([wp.pose.pose.position.x, wp.pose.pose.position.y])

    ndt_log_dir = './data/' + output_file.split('.')[0]
    makedirs(ndt_log_dir, exist_ok=True)

    # Wait until car starts
    rospy.wait_for_message('/vehicle_cmd', VehicleCmd, timeout=None)
    output_file = output_file.split('.')[0] + "_center_offset.csv"
    print(ndt_log_dir + "/" + output_file)
    with open(ndt_log_dir + "/" + output_file, "w") as f:
        wr = csv.writer(f)
        wr.writerow(['ts', 'state', 'center_offset', 'res_t', 'instance'])
        prev_dis = 0
        while not rospy.is_shutdown():
            gnss_msg = rospy.wait_for_message('/gnss_pose', PoseStamped, timeout=None)
            state_msg = rospy.wait_for_message('/behavior_state', MarkerArray, timeout=None)
            ndt_stat_msg = rospy.wait_for_message('/ndt_stat', NDTStat, timeout=None)
            rubis_ndt_pose_msg = rospy.wait_for_message('/rubis_ndt_pose', RubisPoseStamped, timeout=None)

            instance=rubis_ndt_pose_msg.instance

            pose_x = round(gnss_msg.pose.position.x, 3)
            pose_y = round(gnss_msg.pose.position.y, 3)
            ori_x = gnss_msg.pose.orientation.x
            ori_y = gnss_msg.pose.orientation.y
            ori_z = gnss_msg.pose.orientation.z
            ori_w = gnss_msg.pose.orientation.w
            r, p, y = euler_from_quaternion(ori_x, ori_y, ori_z, ori_w)

            yaw_deg = (y * 180 / math.pi + 1800) % 360
            
            min_wp, min_dis = find_closest_point(map_wp_list, [pose_x, pose_y], yaw_deg)

            if abs(prev_dis) > 1.5 and min_dis * prev_dis < 0:
                min_dis *= 1
            prev_dis = min_dis

            if state_msg.markers[0].text == '(0)LKAS':
                state_text = 'Backup'
            else:
                state_text = 'Normal'

            wr.writerow([time.clock_gettime(time.CLOCK_MONOTONIC), state_text, str(min_dis), str(ndt_stat_msg.exe_time), instance])     
def calculate_e2e_response_time(): #e2e reponse time
    # Load yamls
    with open('autoware_analyzer.yaml') as f:
        data = yaml.load(f, yaml.FullLoader)
        node_paths = data['node_paths']
        e2e_response_time_path = data['e2e_response_time_path']

    first_node_path = list(node_paths.items())[0][1]
    last_node_path = list(node_paths.items())[-1][1]

    e2e_response_time_data = [] # start, end, instance

    # Update last node
    with open(last_node_path) as f:
        reader = csv.reader(f)
        prev_instance = -1
        for i, row in enumerate(reader):
            if i == 0: continue

            end = float(row[3])
            activation = int(row[5])
            instance = int(row[4])
            
            # TODO: BUG - instance id is replicated!
            if instance == prev_instance: continue

            if activation == 0: continue # Skip actvation is 0
            line = [-1, end, instance] # start, -1, instance
            e2e_response_time_data.append(line)

            prev_instance = instance

    # Update first node
    with open(first_node_path) as f:
        reader = csv.reader(f)
        prev_instance = -1

        it = iter(e2e_response_time_data)
        target = it.__next__()
        last_instance = e2e_response_time_data[-1][2]
        for i, row in enumerate(reader):
            if i == 0: continue
            start = float(row[2])
            instance = int(row[4])
            actvation = int(row[5])
            
            if instance > last_instance: break

            target_instance = target[2]
            if target_instance != instance: continue            
            target[0] = start
            response_time = target[1] - target[0]
            target.append(response_time)
            try: target = it.__next__()
            except StopIteration as e: break
    
    with open(e2e_response_time_path, 'w') as f:
        writer = csv.writer(f)
        for line in e2e_response_time_data:
            writer.writerow(line)

    return 0
        
def plot_e2e_and_center_offset_by_instance():

    with open('autoware_analyzer.yaml') as f:
        data = yaml.load(f, yaml.FullLoader)
        center_offset_path = data['center_offset_path']
        e2e_response_time_path = data['e2e_response_time_path']
        center_offset_ylim = data['center_offset_ylim']
        e2e_response_ylim = data['e2e_response_time_ylim']
        plot_path = data['plot_path']

    center_offset_instance_list = []
    center_offset_list = []
    
    with open(center_offset_path) as f:
        reader = csv.reader(f)
        
        for i, line in enumerate(reader):
            #print(line)
            if i == 0: continue
            instance = float(line[4])
            center_offset = abs(float(line[2]))
            center_offset_instance_list.append(instance)
            center_offset_list.append(center_offset)

    center_offset_avg = sum(center_offset_list)/len(center_offset_list)
    center_offset_worst = sorted(center_offset_list)[-1]

    e2e_response_instance_list = []
    e2e_response_list = []

    e2e_response_start = -1.0
    with open(e2e_response_time_path) as f:
        reader = csv.reader(f)
        for i,  line in enumerate(reader):
            if i==0: continue
            # Assign response time to end time
            instance = float(line[2])
            e2e_response = float(line[3])*1000
            e2e_response_instance_list.append(instance)
            e2e_response_list.append(e2e_response)
    
    e2e_response_avg = sum(e2e_response_list)/len(e2e_response_list)
    e2e_response_worst = sorted(e2e_response_list)[-1]

    ax1 = plt.subplot()
    ax1.set_ylabel('E2E response time (ms)')
    ax1.set_xlabel('Instance')    
    lns1= ax1.plot(e2e_response_instance_list, e2e_response_list, '-r', label='E2E reponse time')
    ax1.plot(e2e_response_instance_list, [e2e_response_avg for i in range(len(e2e_response_instance_list))], ':r')
    ax1.set_ylim(e2e_response_ylim)

    ax2 = ax1.twinx()    
    ax2.set_ylabel('Center offset (m)')
    lns2 = ax2.plot(center_offset_instance_list, center_offset_list, '-b', label='center_offset')
    ax2.plot(center_offset_instance_list, [center_offset_avg for i in range(len(center_offset_instance_list))], ':b')
    ax2.set_ylim(center_offset_ylim)

    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')

    print('avg_center_offset:' + str(center_offset_avg))
    print('worst_center_offset:' + str(center_offset_worst))
    print('avg_e2e_response_time:' + str(e2e_response_avg))
    print('worst_e2e_response_time:' + str(e2e_response_worst))
    # plt.show()
    plt.savefig(plot_path)
    plt.close()