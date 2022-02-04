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
state.transform.position = spawns[0].position + 5 * right
state.transform.rotation = spawns[0].rotation

# ego = sim.add_agent("DoubleLiDAR (Autoware)", lgsvl.AgentType.EGO, state)
ego = sim.add_agent("SingleLiDAR (Autoware)", lgsvl.AgentType.EGO, state)
ego.connect_bridge(os.environ.get("BRIDGE_HOST", "127.0.0.1"), 9090)

set_control = "red=7;yellow=3;green=7;loop"
signal = sim.get_controllables("signal")
signal[1].control(set_control)

sim.run()
