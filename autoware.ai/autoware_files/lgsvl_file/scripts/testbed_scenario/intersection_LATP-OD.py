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

### Initial Poistion ###
state.transform.position = spawns[0].position + 300 * forward
state.transform.rotation = spawns[0].rotation

### Intersection Position ###
#state.transform.position = spawns[0].position + 481 * forward - 550 * right
#state.transform.rotation = lgsvl.Vector(0, -90, 0)

#ego = sim.add_agent("Lexus2016RXHybrid (Autoware)", lgsvl.AgentType.EGO, state)
#ego = sim.add_agent("DoubleLiDAR (Autoware)", lgsvl.AgentType.EGO, state)
ego = sim.add_agent("TripleLiDAR (Autoware)", lgsvl.AgentType.EGO, state)
ego.connect_bridge(os.environ.get("BRIDGE_HOST", "127.0.0.1"), 9090)


#set all of npc_vehicle speed on scenes
npc_speed = 6

#------- NPC car scene 1(정면) -------#


### 멀리서 출발 ###
# npc1_s1_state = lgsvl.AgentState()
# npc1_s1_state.transform.position = spawns[0].position + 470 * forward - 690 * right
# npc1_s1_state.transform.rotation = lgsvl.Vector(0, -270, 0)

# npc_s1_angle = npc1_s1_state.transform.rotation

# npc1_s1_waypoints = []

# #set npc_s1_vehicle1's waypoint1
# npc1_s1_start = npc1_s1_state.transform.position + 2 * right
# npc1_wp1 = lgsvl.DriveWaypoint(position=lgsvl.Vector(npc1_s1_start.x, npc1_s1_start.y, npc1_s1_start.z),
#                                speed=npc_speed, angle=npc_s1_angle, idle=3, trigger_distance=150, trigger=None)
# npc1_s1_waypoints.append(npc1_wp1)

# #set npc_s1_vehicle1's waypoint2
# npc1_s1_way1 = npc1_s1_start + 100 * right
# npc1_s1_wp2 = lgsvl.DriveWaypoint(position=lgsvl.Vector(npc1_s1_way1.x, npc1_s1_way1.y, npc1_s1_way1.z),
#                                   speed=npc_speed, angle=npc_s1_angle, idle=0, trigger_distance=0, trigger=None)
# npc1_s1_waypoints.append(npc1_s1_wp2)


# #set npc_s1_vehicle1's waypoint3
# npc1_s1_way2 = npc1_s1_way1 + 5000 * right
# npc1_s1_wp3 = lgsvl.DriveWaypoint(position=lgsvl.Vector(npc1_s1_way2.x, npc1_s1_way2.y, npc1_s1_way2.z),
#                                   speed=npc_speed, angle=npc_s1_angle, idle=0, trigger_distance=0, trigger=None)
# npc1_s1_waypoints.append(npc1_s1_wp3)

# #set npc_s1_vehicle1
# npc_s1_vehicle1 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, npc1_s1_state)
# npc_s1_vehicle1.follow(npc1_s1_waypoints)


### 정지 후 출발 ###
npc2_s1_state = lgsvl.AgentState()
npc2_s1_state.transform.position = spawns[0].position + 470 * forward - 660 * right
npc2_s1_state.transform.rotation = lgsvl.Vector(0, -270, 0)

npc2_s1_waypoints = []
npc2_s1_angle = npc2_s1_state.transform.rotation
#set npc_s1_vehicle2's waypoint1
npc2_s1_start = npc2_s1_state.transform.position + 2 * right
npc2_s1_wp1 = lgsvl.DriveWaypoint(position=lgsvl.Vector(npc2_s1_start.x, npc2_s1_start.y, npc2_s1_start.z),
                                  speed=npc_speed, angle=npc2_s1_angle, idle=2, trigger_distance=95, trigger=None)
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
