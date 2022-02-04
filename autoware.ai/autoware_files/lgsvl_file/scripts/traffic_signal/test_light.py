#!/usr/bin/env python3
#
# Copyright (c) 2019 LG Electronics, Inc.
#
# This software contains code licensed as described in LICENSE.
#

import os
import lgsvl
import json
from pathlib import Path

## main ##

sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)

scene_name = "BorregasAve"
sim.load(scene_name)

spawns = sim.get_spawn()

state = lgsvl.AgentState()
forward = lgsvl.utils.transform_to_forward(spawns[0])
state.transform = spawns[0]
state.transform.position = spawns[0].position
ego = sim.add_agent("DoubleLiDAR (Autoware)", lgsvl.AgentType.EGO, state)

ego.connect_bridge(os.environ.get("BRIDGE_HOST", "127.0.0.1"), 9090)

light_list = []
## Get a list of controllable objects
set_control = "red=7;green=7;yellow=3;loop"
signal = sim.get_controllable(lgsvl.Vector(15.5465927124023, 4.72256088256836, -23.8751735687256), "signal")
signal.control(set_control)
controllables = sim.get_controllables("signal")

print("\n# List of controllable objects in {} scene:".format(scene_name))
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


dict_path = os.path.join(str(Path.home()), "RUBIS-SelfDriving/autoware_files/lgsvl/scripts/traffic_signal")
file_path = os.path.join(dict_path, "traffic_signal_policy.json")
config_file = open(file_path, 'w')
json.dump(light_list, config_file, indent=4)
config_file.close()

sim.run()
