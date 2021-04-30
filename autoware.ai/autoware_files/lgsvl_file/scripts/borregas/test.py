#!/usr/bin/env python3
#
# Copyright (c) 2019 LG Electronics, Inc.
#
# This software contains code licensed as described in LICENSE.
#

import os
import lgsvl

sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)

if sim.current_scene == "BorregasAve":
  sim.reset()
else:
  sim.load("BorregasAve")

spawns = sim.get_spawn()
forward = lgsvl.utils.transform_to_forward(spawns[0])
right = lgsvl.utils.transform_to_right(spawns[0])

sx = spawns[0].position.x
sz = spawns[0].position.z
spawns = sim.get_spawn()

#set ego vehicle's initial position
state = lgsvl.AgentState()
state.transform = spawns[0]
state.transform.rotation = spawns[0].rotation

ego = sim.add_agent("Lexus2016RXHybrid", lgsvl.AgentType.EGO, state)

ego.connect_bridge(os.environ.get("BRIDGE_HOST", "127.0.0.1"), 9090)

#set npc vehicle1's initial position
state = lgsvl.AgentState()
state.transform.position = spawns[0].position + 30 * forward + 2 * right
state.transform.rotation = spawns[0].rotation
npc1 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)

#set npc vehicle2's initial position
state = lgsvl.AgentState()
state.transform.position = spawns[0].position + 70 * forward - 3 * right
state.transform.rotation = lgsvl.Vector(0, -75.823371887207, 0)
npc2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)

#set npc vehicle3's initial position
angle = spawns[0].rotation
state = lgsvl.AgentState()
state.transform.position = spawns[0].position + 100 * forward + 3 * right
state.transform.rotation = lgsvl.Vector(0, -75.823371887207, 0)
npc3 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)

#set npc vehicle3's initial position
angle = spawns[0].rotation
state = lgsvl.AgentState()
state.transform.position = spawns[0].position + 150 * forward - 2 * right
state.transform.rotation = lgsvl.Vector(0, -75.823371887207, 0)
npc4 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)


for sensor in ego.get_sensors():
  if sensor.name == "Lidar":
    sensor.save("lidar.pcd")

sim.run()