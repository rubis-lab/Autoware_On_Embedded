#!/usr/bin/env python3
#
# Copyright (c) 2019 LG Electronics, Inc.
#
# This software contains code licensed as described in LICENSE.
#

import os
import lgsvl
import json
import time
import subprocess
import rospy
import copy
from pathlib import Path
from autoware_msgs.msg import RUBISTrafficSignalArray, RUBISTrafficSignal
from visualization_msgs.msg import Marker, MarkerArray


## main ##
pub = rospy.Publisher('v2x_traffic_signal', RUBISTrafficSignalArray, queue_size=50)
stop_line_rviz_pub = rospy.Publisher('stop_line_marker', MarkerArray, queue_size=10)
rospy.init_node('traffic_signal_pub', anonymous=True)
rate = rospy.Rate(10)
spin_rate = float(10)
test = 1


dict_path = os.path.join(str(Path.home()), "autoware.ai/autoware_files/lgsvl/scripts/testbed_scenario")
file_path = os.path.join(dict_path, "traffic_signal_policy.json")
subprocess.Popen(["./testbed_scenario.py"],cwd=dict_path)
with open(file_path, "r") as read_json:
  light_list = json.load(read_json)

stop_line_param = rospy.get_param("/lgsvl_traffic_signal_test/stop_line_list")

topic_list = []
topic_typelist = []

signal_array_msg = RUBISTrafficSignalArray()
stop_line_marker_array = MarkerArray()

### Make V2X Topic msg
for light in light_list:
  sync_time = 5 + float(light['time'])
  topic = {
    'id': light['id'],
    'type': light['type'],
    'time': sync_time
  }
  topic_list.append(topic)
  topic_typelist.append(light['type_list'])

  signal_msg = RUBISTrafficSignal()
  signal_msg.id = light['id']
  signal_msg.type = light['type']
  signal_msg.time = sync_time
  signal_array_msg.signals.append(signal_msg)


### Make Stop Line Marker Topic msg
for sl in stop_line_param:
  marker = Marker()
  marker.header.frame_id = "world"

  marker.id = sl['id']
  marker.type = 2 # sphere
  marker.ns = str(sl['id'])

  marker.pose.position.x = sl["pose"]["x"]
  marker.pose.position.y = sl["pose"]["y"]
  marker.pose.position.z = sl["pose"]["z"]
  marker.pose.orientation.w = 1

  marker.scale.x = 3
  marker.scale.y = 3
  marker.scale.z = 3

  marker.color.r = 0.0
  marker.color.g = 1.0
  marker.color.b = 0.0
  marker.color.a = 1.0

  marker.lifetime = rospy.Duration()

  text_marker = Marker()
  text_marker.type = 9
  text_marker.header.frame_id = "world"
  text_marker.ns = "text"
  text_marker.id = -sl['id']
  text_marker.text = "StopLine " + str(sl['id'])
  text_marker.pose = copy.deepcopy(marker.pose)
  text_marker.pose.position.z += 5
  text_marker.scale.z = 2
  
  text_marker.color.r = 1.0
  text_marker.color.g = 1.0
  text_marker.color.b = 1.0
  text_marker.color.a = 1.0
  
  stop_line_marker_array.markers.append(marker)
  stop_line_marker_array.markers.append(text_marker)


###
# 0 : Red
# 1 : Yellow
# 2 : Green

while not rospy.is_shutdown():
  for (i, topic) in enumerate(topic_list):
    if topic['time'] < 0.05:
      if topic['type'] == 0: # Red -> Green
        topic['type'] = 2
        topic['time'] = float(topic_typelist[topic['id']]['green'])
      elif topic['type'] == 1: # Yellow -> Red
        topic['type'] = 0
        topic['time'] = float(topic_typelist[topic['id']]['red'])
      elif topic['type'] == 2: # Green -> Yellow
        topic['type'] = 1
        topic['time'] = float(topic_typelist[topic['id']]['yellow'])
    topic['time'] = topic['time'] - (1/spin_rate)

    signal_array_msg.signals[i].type = topic['type']
    signal_array_msg.signals[i].time = topic['time']
  
  for i in range(len(stop_line_param)):
    tl_idx = 0
    for (idx, tl) in enumerate(topic_list):
      if tl['id'] == stop_line_param[i]['tl_id']:
        tl_idx = idx
        
    signal_type = topic_list[tl_idx]['type']
    
    if(signal_type == 0): # Red
      stop_line_marker_array.markers[2*i].color.r = 1.0
      stop_line_marker_array.markers[2*i].color.g = 0.0
    elif(signal_type == 1): # Yellow
      stop_line_marker_array.markers[2*i].color.r = 1.0
      stop_line_marker_array.markers[2*i].color.g = 1.0
    else: # Green
      stop_line_marker_array.markers[2*i].color.r = 0.0
      stop_line_marker_array.markers[2*i].color.g = 1.0

  data = json.dumps(topic_list, indent=4)
  stop_line_rviz_pub.publish(stop_line_marker_array)
  pub.publish(signal_array_msg)
  rate.sleep()