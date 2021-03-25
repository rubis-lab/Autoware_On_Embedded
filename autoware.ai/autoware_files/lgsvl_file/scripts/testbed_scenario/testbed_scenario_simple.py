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
state.transform.rotation = spawns[0].rotation

#ego = sim.add_agent("Lexus2016RXHybrid (Autoware)", lgsvl.AgentType.EGO, state)
#ego = sim.add_agent("DoubleLiDAR (Autoware)", lgsvl.AgentType.EGO, state)
ego = sim.add_agent("TripleLiDAR (Autoware)", lgsvl.AgentType.EGO, state)
ego.connect_bridge(os.environ.get("BRIDGE_HOST", "127.0.0.1"), 9090)



light_list = []
## Get a list of controllable objects
set_control = "red=15;green=5;yellow=2;loop"
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
#sv_state = lgsvl.AgentState()
#sv_state.transform.position = spawns[0].position + 60 * forward
#sv_state.transform.rotation = spawns[0].rotation

#stand_vehicle = sim.add_agent("Sedan", lgsvl.AgentType.NPC, sv_state)


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
                            trigger_distance=30, trigger=None)
mp_waypoints.append(mp_wp1)

#set end waypoint of cross walk
mp_mid = mp_start - 3 * right
mp_wp2 = lgsvl.WalkWaypoint(position=lgsvl.Vector(mp_mid.x, mp_mid.y, mp_mid.z), speed=2, idle=3,
                            trigger_distance=0, trigger=None)
mp_waypoints.append(mp_wp2)

mp_end = mp_mid - 18 * right
mp_wp3 = lgsvl.WalkWaypoint(position=lgsvl.Vector(mp_end.x, mp_end.y, mp_end.z), speed=2, idle=0,
                            trigger_distance=0, trigger=None)
mp_waypoints.append(mp_wp3)



move_pedestrian = sim.add_agent("Bob", lgsvl.AgentType.PEDESTRIAN, mp_state)
move_pedestrian.follow(mp_waypoints, False)


sim.run()
