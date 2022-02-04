#!/usr/bin/env python3
#
# Copyright (c) 2019 LG Electronics, Inc.
#
# This software contains code licensed as described in LICENSE.
#

import os
import lgsvl
import random
import time
from pathlib import Path
import json

sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)

layer_mask = 0
layer_mask |= 1 << 0  # 0 is the layer for the road (default)

if sim.current_scene == "Testbed":
  sim.reset()
else:
  sim.load("Testbed")

spawns = sim.get_spawn()
forward = lgsvl.utils.transform_to_forward(spawns[0])
right = lgsvl.utils.transform_to_right(spawns[0])
sx = spawns[0].position.x
sz = spawns[0].position.z

spawns = sim.get_spawn()

state = lgsvl.AgentState()
state.transform.position = spawns[0].position
#state.transform.position = spawns[0].position + 300 * forward
state.transform.rotation = spawns[0].rotation

#ego = sim.add_agent("Lexus2016RXHybrid (Autoware)", lgsvl.AgentType.EGO, state)
#ego = sim.add_agent("DoubleLiDAR (Autoware)", lgsvl.AgentType.EGO, state)
ego = sim.add_agent("SingleLiDAR (Autoware)", lgsvl.AgentType.EGO, state)
ego.connect_bridge(os.environ.get("BRIDGE_HOST", "127.0.0.1"), 9090)



light_list = []
## Get a list of controllable objects
set_control = "red=10;green=10;yellow=2;loop"
signal = sim.get_controllables("signal")
signal[0].control(set_control)
signal[1].control(set_control)
signal[2].control(set_control)
signal[3].control(set_control)
controllables = sim.get_controllables("signal")

# print("\n# List of controllable objects in {} scene:".format(scene_name))
for idx, c in enumerate(controllables):
  light = {
    'id': 0,
    'type_list': {
      'red': 0,
      'yellow': 0,
      'green': 0
    },
    'type': 0,
    'time': 0
  }
  policy_time = []

  temp = c.__repr__()
  _temp = str.split(temp, ',')
  s_temp = str.split(_temp[14], ':')
  light['id'] = idx
  if idx == 1:
    control_policy = str.split(set_control, ';')
  else:
    control_policy = str.split(s_temp[1], ';')
    control_policy[0] = control_policy[0][2:]
  control_policy.pop()

  for color_list in control_policy:
    policy_time.append(str.split(color_list, '='))

  for color in policy_time:
    light['type_list'][color[0]] = float(color[1])
  light['time'] = int(light['type_list']['red'])
  light_list.append(light)


dict_path = os.getcwd()
file_path = os.path.join(dict_path, "traffic_signal_policy.json")
config_file = open(file_path, 'w')
json.dump(light_list, config_file, indent=4)
config_file.close()


# #------- Fast Pedestrian -------#
# fp_waypoints = []
# speed = 7

# #set start waypoint
# start = spawns[0].position + 81 * forward + 44 * right

# #you can change trigger_distance what you want
# fp_wp1 = lgsvl.WalkWaypoint(position=lgsvl.Vector(start.x, start.y, start.z), speed=speed, idle=5.0,
#                             trigger_distance=60, trigger=None)
# fp_waypoints.append(fp_wp1)


# second = spawns[0].position + 81 * forward + 20 * right

# fp_wp2 = lgsvl.WalkWaypoint(position=lgsvl.Vector(second.x, second.y, second.z), speed=speed, idle=8.0,
#                             trigger_distance=0, trigger=None)
# fp_waypoints.append(fp_wp2)

# third = spawns[0].position + 110 * forward + 8 * right
# fp_wp3 = lgsvl.WalkWaypoint(position=lgsvl.Vector(third.x, third.y, third.z), speed=speed, idle=0,
#                             trigger_distance=0, trigger=None)
# fp_waypoints.append(fp_wp3)

# end = spawns[0].position + 110 * forward - 3 * right
# fp_wp4 = lgsvl.WalkWaypoint(position=lgsvl.Vector(end.x, end.y, end.z), speed=speed, idle=0,
#                             trigger_distance=0, trigger=None)
# fp_waypoints.append(fp_wp4)

# #set position of fast pedestrian
# fp_state = lgsvl.AgentState()
# fp_state.transform.position = spawns[0].position + 81 * forward + 45 * right
# fp_state.transform.rotation = spawns[0].rotation

# fast_pedestrian = sim.add_agent("Bob", lgsvl.AgentType.PEDESTRIAN, fp_state)
# fast_pedestrian.follow(fp_waypoints, False)


#------- Stand vehicle -------#
#set stand vehicle's initial position
sv_state = lgsvl.AgentState()
sv_state.transform.position = spawns[0].position + 60 * forward
sv_state.transform.rotation = spawns[0].rotation

stand_vehicle = sim.add_agent("Sedan", lgsvl.AgentType.NPC, sv_state)


#------- Narrow path -------#
#set np vehicle1's initial position
np1_state = lgsvl.AgentState()
np1_state.transform.position = spawns[0].position + 270 * forward + 3.5 * right
np1_state.transform.rotation = spawns[0].rotation

np_vehicle1 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, np1_state)

np2_state = lgsvl.AgentState()
np2_state.transform.position = spawns[0].position + 270 * forward - 5 * right
np2_state.transform.rotation = lgsvl.Vector(0, -180, 0)

np_vehicle2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, np2_state)


#------- Construction section -------#

#set traffic cone1
cone1_state = lgsvl.ObjectState()
cone1_state.transform.position = spawns[0].position + 384 * forward - right
cone1_state.transform.rotation = lgsvl.Vector(0, 0, 0)

cone1 = sim.controllable_add("TrafficCone", cone1_state)

#set traffic cone2
cone2_state = lgsvl.ObjectState()
cone2_state.transform.position = spawns[0].position + 384 * forward
cone2_state.transform.rotation = lgsvl.Vector(0, 0, 0)

cone2 = sim.controllable_add("TrafficCone", cone2_state)

#set traffic cone3
cone3_state = lgsvl.ObjectState()
cone3_state.transform.position = spawns[0].position + 384 * forward + right
cone3_state.transform.rotation = lgsvl.Vector(0, 0, 0)

cone3 = sim.controllable_add("TrafficCone", cone3_state)

#set worker
cunscar1_state = lgsvl.ObjectState()
cunscar1_state.transform.position = spawns[0].position + 390 * forward
cunscar1_state.transform.rotation = lgsvl.Vector(0, 0, 0)
worker1 = sim.add_agent("BoxTruck", lgsvl.AgentType.NPC, cunscar1_state)


#------- Move pedestrian -------#

#set position of move pedestrian
mp_state = lgsvl.AgentState()
mp_state.transform.position = spawns[0].position + 440 * forward + 9 * right
mp_state.transform.rotation = spawns[0].rotation

mp_waypoints = []

#set start waypoint of cross walk
mp_start = spawns[0].position + 440 * forward + 8 * right

#you can change trigger_distance what you want
mp_wp1 = lgsvl.WalkWaypoint(position=lgsvl.Vector(mp_start.x, mp_start.y, mp_start.z), speed=3, idle=0,
                            trigger_distance=50, trigger=None)
mp_waypoints.append(mp_wp1)

#set end waypoint of cross walk
mp_mid = mp_start - 3 * right
mp_wp2 = lgsvl.WalkWaypoint(position=lgsvl.Vector(mp_mid.x, mp_mid.y, mp_mid.z), speed=2, idle=7,
                            trigger_distance=0, trigger=None)
mp_waypoints.append(mp_wp2)

mp_end = mp_mid - 18 * right
mp_wp3 = lgsvl.WalkWaypoint(position=lgsvl.Vector(mp_end.x, mp_end.y, mp_end.z), speed=2, idle=0,
                            trigger_distance=0, trigger=None)
mp_waypoints.append(mp_wp3)



move_pedestrian = sim.add_agent("Bob", lgsvl.AgentType.PEDESTRIAN, mp_state)
move_pedestrian.follow(mp_waypoints, False)


#------- Congestion section -------#
#set cs_vehicle1's initial position
cs1_state = lgsvl.AgentState()
cs1_state.transform.position = spawns[0].position + 485 * forward - 40 * right
cs1_state.transform.rotation = lgsvl.Vector(0, -90, 0)

cs_angle = cs1_state.transform.rotation
cs_speed = 5

#set cs_vehicle1's start waypoint of congestion section
cs1_waypoints = []
cs1_start = cs1_state.transform.position - 2 * right
cs1_wp1 = lgsvl.DriveWaypoint(position=lgsvl.Vector(cs1_start.x, cs1_start.y, cs1_start.z), speed=cs_speed,
                              angle=cs_angle, idle=0, trigger_distance=30, trigger=None)
cs1_waypoints.append(cs1_wp1)

#set cs_vehicle1's end waypoint of congestion section
cs1_end = cs1_start - 100 * right
cs1_wp2 = lgsvl.DriveWaypoint(position=lgsvl.Vector(cs1_end.x, cs1_end.y, cs1_end.z), speed=cs_speed,
                              angle=cs_angle, idle=100, trigger_distance=0, trigger=None)
cs1_waypoints.append(cs1_wp2)

cs1_dump = cs1_end - 5000 * forward - 5000 * right
cs1_wp3 = lgsvl.DriveWaypoint(position=lgsvl.Vector(cs1_dump.x, cs1_dump.y, cs1_dump.z), speed=1,
                              angle=cs_angle, idle=0, trigger_distance=0, trigger=None)
cs1_waypoints.append(cs1_wp3)

cs_vehicle1 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, cs1_state)
cs_vehicle1.follow(cs1_waypoints)

#set cs_vehicle2's initial position
cs2_state = lgsvl.AgentState()
cs2_state.transform.position = spawns[0].position + 481 * forward - 65 * right
cs2_state.transform.rotation = lgsvl.Vector(0, -90, 0)

cs2_waypoints = []

#set cs_vehicle2's start waypoint of congestion section
cs2_start = cs2_state.transform.position - 2 * right
cs2_wp1 = lgsvl.DriveWaypoint(position=lgsvl.Vector(cs2_start.x, cs2_start.y, cs2_start.z),
                              speed=cs_speed, angle=cs_angle, idle=0, trigger_distance=55, trigger=None)
cs2_waypoints.append(cs2_wp1)

#set cs_vehicle2's end waypoint of congestion section
cs2_end = cs2_start - 100 * right
cs2_wp2 = lgsvl.DriveWaypoint(position=lgsvl.Vector(cs2_end.x, cs2_end.y, cs2_end.z), speed=cs_speed,
                              angle=cs_angle, idle=100, trigger_distance=0, trigger=None)
cs2_waypoints.append(cs2_wp2)

cs_vehicle2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, cs2_state)
cs_vehicle2.follow(cs2_waypoints)

cs2_dump = cs2_end - 5000 * forward - 5000 * right
cs2_wp3 = lgsvl.DriveWaypoint(position=lgsvl.Vector(cs2_dump.x, cs2_dump.y, cs2_dump.z), speed=1,
                              angle=cs_angle, idle=0, trigger_distance=0, trigger=None)
cs2_waypoints.append(cs2_wp3)

#------- Cut in scenario 1 -------#

#set vehicle 1 in Cut in scenario
ci1_state = lgsvl.AgentState()
ci1_state.transform.position = spawns[0].position + 483 * forward - 350 * right
ci1_state.transform.rotation = lgsvl.Vector(0, -90, 0)

ci_speed = 6
ci_angle = ci1_state.transform.rotation

ci1_waypoints = []

#set ci_vehicle1's waypoints of Cut in scenario
ci1_start = ci1_state.transform.position - 2 * forward
ci1_wp1 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci1_start.x, ci1_start.y, ci1_start.z), speed=ci_speed,
                              angle=ci_angle, idle=0, trigger_distance=30, trigger=None)
ci1_waypoints.append(ci1_wp1)

ci1_way1 = ci1_start - 5 * right
ci1_wp2 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci1_way1.x, ci1_way1.y, ci1_way1.z), speed=ci_speed, angle=ci_angle,
                              idle=0, trigger_distance=0, trigger=None)
ci1_waypoints.append(ci1_wp2)

ci1_way2 = ci1_way1 + 5 * forward - 10 * right
ci1_wp3 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci1_way2.x, ci1_way2.y, ci1_way2.z), speed=ci_speed, angle=ci_angle,
                              idle=0, trigger_distance=0, trigger=None)
ci1_waypoints.append(ci1_wp3)

ci1_end = ci1_way2 - 30 * right
ci1_wp4 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci1_end.x, ci1_end.y, ci1_end.z), speed=ci_speed,
                              angle=ci_angle, idle=100, trigger_distance=0, trigger=None)
ci1_waypoints.append(ci1_wp4)

ci1_dump = ci1_end + 1000 * right
ci1_wp5 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci1_dump.x, ci1_dump.y, ci1_dump.z), speed=ci_speed,
                              angle=ci_angle, idle=0, trigger_distance=0, trigger=None)
ci1_waypoints.append(ci1_wp5)

ci_vehicle1 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, ci1_state)
ci_vehicle1.follow(ci1_waypoints)

#set vehicle 2 in Cut in scenario
ci2_state = lgsvl.AgentState()
ci2_state.transform.position = spawns[0].position + 486 * forward - 450 * right
ci2_state.transform.rotation = lgsvl.Vector(0, -180, 0)


ci2_waypoints = []

#set ci_vehicle2's waypoints of Cut in scenario
ci2_start = ci2_state.transform.position - 2 * right
ci2_wp1 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci2_start.x, ci2_start.y, ci2_start.z), speed=ci_speed,
                              angle=ci_angle, idle=0, trigger_distance=30, trigger=None)
ci2_waypoints.append(ci2_wp1)

ci2_way1 = ci2_start - 5 * right
ci2_wp2 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci2_way1.x, ci2_way1.y, ci2_way1.z), speed=ci_speed,
                              angle=ci_angle, idle=0, trigger_distance=0, trigger=None)
ci2_waypoints.append(ci2_wp2)

ci2_way2 = ci2_way1 - 5 * forward - 10 * right
ci2_wp3 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci2_way2.x, ci2_way2.y, ci2_way2.z), speed=ci_speed,
                              angle=ci_angle, idle=0, trigger_distance=0, trigger=None)
ci2_waypoints.append(ci2_wp3)

ci2_end = ci2_way2 - 30 * right
ci2_wp4 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci2_end.x, ci2_end.y, ci2_end.z), speed=ci_speed,
                              angle=ci_angle, idle=10, trigger_distance=0, trigger=None)
ci2_waypoints.append(ci2_wp4)

ci2_dump = ci2_end + 1000 * right
ci2_wp5 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci2_dump.x, ci2_dump.y, ci2_dump.z), speed=ci_speed,
                              angle=ci_angle, idle=0, trigger_distance=0, trigger=None)
ci2_waypoints.append(ci2_wp5)

ci_vehicle2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, ci2_state)
ci_vehicle2.follow(ci2_waypoints)

### Left Turn

#set all of npc_vehicle speed on scenes
npc_speed = 6

npc2_s1_state = lgsvl.AgentState()
npc2_s1_state.transform.position = spawns[0].position + 470 * forward - 640 * right
npc2_s1_state.transform.rotation = lgsvl.Vector(0, -270, 0)

npc2_s1_waypoints = []
npc2_s1_angle = npc2_s1_state.transform.rotation
#set npc_s1_vehicle2's waypoint1
npc2_s1_start = npc2_s1_state.transform.position + 2 * right
npc2_s1_wp1 = lgsvl.DriveWaypoint(position=lgsvl.Vector(npc2_s1_start.x, npc2_s1_start.y, npc2_s1_start.z),
                                  speed=npc_speed, angle=npc2_s1_angle, idle=4, trigger_distance=75, trigger=None)
npc2_s1_waypoints.append(npc2_s1_wp1)

#set npc_s1_vehicle2's waypoint2
npc2_s1_way1 = npc2_s1_start + 100 * right
npc2_s1_wp2 = lgsvl.DriveWaypoint(position=lgsvl.Vector(npc2_s1_way1.x, npc2_s1_way1.y, npc2_s1_way1.z),
                                  speed=1, angle=npc2_s1_angle, idle=0, trigger_distance=0, trigger=None)
npc2_s1_waypoints.append(npc2_s1_wp2)

#set npc_s1_vehicle2's waypoint3
npc2_s1_way2 = npc2_s1_way1 + 5000 * right
npc2_s1_wp3 = lgsvl.DriveWaypoint(position=lgsvl.Vector(npc2_s1_way2.x, npc2_s1_way2.y, npc2_s1_way2.z),
                                  speed=npc_speed, angle=npc2_s1_angle, idle=0, trigger_distance=0, trigger=None)
npc2_s1_waypoints.append(npc2_s1_wp3)

#set npc_s1_vehicle2
npc_s1_vehicle2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, npc2_s1_state)
npc_s1_vehicle2.follow(npc2_s1_waypoints)



sim.run()
