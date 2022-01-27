#!/usr/bin/python3
import lgsvl
from lgsvl.geometry import Transform
import tqdm
import os
import random
import yaml


class Exp(object):
	def __init__(self):
		# load cfg file
		cfg_file = __file__.replace('.py', '.yaml')
		with open(cfg_file, 'r') as f:
			self.cfg = yaml.load(f, Loader=yaml.FullLoader)
		
		self.sim = lgsvl.Simulator(
		    address=self.cfg['simulator']['address'],
		    port=self.cfg['simulator']['port'])
		# self.sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST","127.0.0.1"), 8181)

		# reset scene
		target_scene = self.cfg['simulator']['scene']
		if self.sim.current_scene == target_scene:
			self.sim.reset()
		else:
			# self.sim.load(scene="9a734c9d-adfb-4efa-9a61-5ec8aa347873",seed=0)
			self.sim.load(target_scene)
		# calc position
		spawns = self.sim.get_spawn()
		# calc offset
		self.origin = Transform(
			spawns[0].position,
			spawns[0].rotation)
		self.origin.position.x += self.cfg['origin']['offset']['x']
		self.origin.position.y += self.cfg['origin']['offset']['y']
		self.origin.position.z += self.cfg['origin']['offset']['z']
		self.origin.rotation.y += self.cfg['origin']['offset']['r']

		self.u_forward = lgsvl.utils.transform_to_forward(self.origin)
		self.u_right = lgsvl.utils.transform_to_right(self.origin)

		# tracking info
		self.collisions = []

	def create_ego(self, sim):
		# ego (main car)
		ego_state = lgsvl.AgentState()
		ego_state.transform = \
			Transform(self.origin.position, self.origin.rotation)
		ego = sim.add_agent(self.cfg['ego']['asset-id'],
			lgsvl.AgentType.EGO, ego_state)
		
		ego.connect_bridge(
		    self.cfg['lgsvl_bridge']['address'],
		    self.cfg['lgsvl_bridge']['port'])		

		def ego_collision(agent1, agent2, contact):
			self.collisions.append([agent1, agent2, contact])
			return

		ego.on_collision(ego_collision)
		return

	def create_stand_vehicle_scene(self, sim):
		#set stand vehicle's initial position
		sv_state = lgsvl.AgentState()
		sv_state.transform.position = self.origin.position + 80 * self.u_forward
		sv_state.transform.rotation = self.origin.rotation

		stand_vehicle = sim.add_agent("Sedan", lgsvl.AgentType.NPC, sv_state)
		return
	
	def create_fast_pedestrian_scene(self, sim):
		fp_waypoints = []
		speed = 7

		#set start waypoint
		start = self.origin.position + 81 * self.u_forward + 44 * self.u_right

		#you can change trigger_distance what you want
		fp_wp1 = lgsvl.WalkWaypoint(position=lgsvl.Vector(start.x, start.y, start.z), speed=speed, idle=5.0,
									trigger_distance=60, trigger=None)
		fp_waypoints.append(fp_wp1)


		second = self.origin.position + 81 * self.u_forward + 20 * self.u_right

		fp_wp2 = lgsvl.WalkWaypoint(position=lgsvl.Vector(second.x, second.y, second.z), speed=speed, idle=8.0,
									trigger_distance=0, trigger=None)
		fp_waypoints.append(fp_wp2)

		third = self.origin.position + 110 * self.u_forward + 8 * self.u_right
		fp_wp3 = lgsvl.WalkWaypoint(position=lgsvl.Vector(third.x, third.y, third.z), speed=speed, idle=0,
									trigger_distance=0, trigger=None)
		fp_waypoints.append(fp_wp3)

		end = self.origin.position + 110 * self.u_forward - 3 * self.u_right
		fp_wp4 = lgsvl.WalkWaypoint(position=lgsvl.Vector(end.x, end.y, end.z), speed=speed, idle=0,
									trigger_distance=0, trigger=None)
		fp_waypoints.append(fp_wp4)

		#set position of fast pedestrian
		fp_state = lgsvl.AgentState()
		fp_state.transform.position = self.origin.position + 81 * self.u_forward + 45 * self.u_right
		fp_state.transform.rotation = self.origin.rotation

		fast_pedestrian = sim.add_agent("Bob", lgsvl.AgentType.PEDESTRIAN, fp_state)
		fast_pedestrian.follow(fp_waypoints, False)
		return

	def create_narrow_path_scene(self, sim):
		#set np vehicle1's initial position
		np1_state = lgsvl.AgentState()
		np1_state.transform.position = self.origin.position + 270 * self.u_forward + 3.5 * self.u_right
		np1_state.transform.rotation = self.origin.rotation

		np_vehicle1 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, np1_state)

		np2_state = lgsvl.AgentState()
		np2_state.transform.position = self.origin.position + 270 * self.u_forward - 5 * self.u_right
		np2_state.transform.rotation = lgsvl.Vector(0, -180, 0)

		np_vehicle2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, np2_state)
		return

	def create_construction_section_scene(self, sim):
		#set traffic cone1
		cone1_state = lgsvl.ObjectState()
		cone1_state.transform.position = self.origin.position + 383 * self.u_forward - self.u_right
		cone1_state.transform.rotation = lgsvl.Vector(0, 0, 0)

		cone1 = sim.controllable_add("TrafficCone", cone1_state)

		#set traffic cone2
		cone2_state = lgsvl.ObjectState()
		cone2_state.transform.position = self.origin.position + 383 * self.u_forward
		cone2_state.transform.rotation = lgsvl.Vector(0, 0, 0)

		cone2 = sim.controllable_add("TrafficCone", cone2_state)

		#set traffic cone3
		cone3_state = lgsvl.ObjectState()
		cone3_state.transform.position = self.origin.position + 383 * self.u_forward + self.u_right
		cone3_state.transform.rotation = lgsvl.Vector(0, 0, 0)

		cone3 = sim.controllable_add("TrafficCone", cone3_state)

		#set worker
		cunscar1_state = lgsvl.ObjectState()
		cunscar1_state.transform.position = self.origin.position + 390 * self.u_forward
		cunscar1_state.transform.rotation = lgsvl.Vector(0, 0, 0)
		worker1 = sim.add_agent("BoxTruck", lgsvl.AgentType.NPC, cunscar1_state)
		return
	
	def create_move_pedestrian_scene(self, sim):
		#set position of move pedestrian
		mp_state = lgsvl.AgentState()
		mp_state.transform.position = self.origin.position + 440 * self.u_forward + 9 * self.u_right
		mp_state.transform.rotation = self.origin.rotation

		mp_waypoints = []

		#set start waypoint of cross walk
		mp_start = self.origin.position + 440 * self.u_forward + 8 * self.u_right

		#you can change trigger_distance what you want
		mp_wp1 = lgsvl.WalkWaypoint(position=lgsvl.Vector(mp_start.x, mp_start.y, mp_start.z), speed=3, idle=0,
									trigger_distance=30, trigger=None)
		mp_waypoints.append(mp_wp1)

		#set end waypoint of cross walk
		mp_mid = mp_start - 3 * self.u_right
		mp_wp2 = lgsvl.WalkWaypoint(position=lgsvl.Vector(mp_mid.x, mp_mid.y, mp_mid.z), speed=2, idle=3,
									trigger_distance=0, trigger=None)
		mp_waypoints.append(mp_wp2)

		mp_end = mp_mid - 18 * self.u_right
		mp_wp3 = lgsvl.WalkWaypoint(position=lgsvl.Vector(mp_end.x, mp_end.y, mp_end.z), speed=2, idle=0,
									trigger_distance=0, trigger=None)
		mp_waypoints.append(mp_wp3)

		move_pedestrian = sim.add_agent("Bob", lgsvl.AgentType.PEDESTRIAN, mp_state)
		move_pedestrian.follow(mp_waypoints, False)
		return

	def create_congestion_section_scene(self, sim):
		#set cs_vehicle1's initial position
		cs1_state = lgsvl.AgentState()
		cs1_state.transform.position = self.origin.position + 485 * self.u_forward - 30 * self.u_right
		cs1_state.transform.rotation = lgsvl.Vector(0, -90, 0)

		cs_angle = cs1_state.transform.rotation
		cs_speed = 5

		#set cs_vehicle1's start waypoint of congestion section
		cs1_waypoints = []
		cs1_start = cs1_state.transform.position - 2 * self.u_right
		cs1_wp1 = lgsvl.DriveWaypoint(position=lgsvl.Vector(cs1_start.x, cs1_start.y, cs1_start.z), speed=cs_speed,
									angle=cs_angle, idle=0, trigger_distance=30, trigger=None)
		cs1_waypoints.append(cs1_wp1)

		#set cs_vehicle1's end waypoint of congestion section
		cs1_end = cs1_start - 100 * self.u_right
		cs1_wp2 = lgsvl.DriveWaypoint(position=lgsvl.Vector(cs1_end.x, cs1_end.y, cs1_end.z), speed=cs_speed,
									angle=cs_angle, idle=100, trigger_distance=0, trigger=None)
		cs1_waypoints.append(cs1_wp2)

		cs1_dump = cs1_end - 5000 * self.u_forward - 5000 * self.u_right
		cs1_wp3 = lgsvl.DriveWaypoint(position=lgsvl.Vector(cs1_dump.x, cs1_dump.y, cs1_dump.z), speed=1,
									angle=cs_angle, idle=0, trigger_distance=0, trigger=None)
		cs1_waypoints.append(cs1_wp3)

		cs_vehicle1 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, cs1_state)
		cs_vehicle1.follow(cs1_waypoints)

		#set cs_vehicle2's initial position
		cs2_state = lgsvl.AgentState()
		cs2_state.transform.position = self.origin.position + 481 * self.u_forward - 55 * self.u_right
		cs2_state.transform.rotation = lgsvl.Vector(0, -90, 0)

		cs2_waypoints = []

		#set cs_vehicle2's start waypoint of congestion section
		cs2_start = cs2_state.transform.position - 2 * self.u_right
		cs2_wp1 = lgsvl.DriveWaypoint(position=lgsvl.Vector(cs2_start.x, cs2_start.y, cs2_start.z),
									speed=cs_speed, angle=cs_angle, idle=0, trigger_distance=55, trigger=None)
		cs2_waypoints.append(cs2_wp1)

		#set cs_vehicle2's end waypoint of congestion section
		cs2_end = cs2_start - 100 * self.u_right
		cs2_wp2 = lgsvl.DriveWaypoint(position=lgsvl.Vector(cs2_end.x, cs2_end.y, cs2_end.z), speed=cs_speed,
									angle=cs_angle, idle=100, trigger_distance=0, trigger=None)
		cs2_waypoints.append(cs2_wp2)

		cs_vehicle2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, cs2_state)
		cs_vehicle2.follow(cs2_waypoints)

		cs2_dump = cs2_end - 5000 * self.u_forward - 5000 * self.u_right
		cs2_wp3 = lgsvl.DriveWaypoint(position=lgsvl.Vector(cs2_dump.x, cs2_dump.y, cs2_dump.z), speed=1,
									angle=cs_angle, idle=0, trigger_distance=0, trigger=None)
		cs2_waypoints.append(cs2_wp3)
		return

	def create_cut_in_scene(self, sim):
		#set vehicle 1 in Cut in scenario
		ci1_state = lgsvl.AgentState()
		ci1_state.transform.position = self.origin.position + 483 * self.u_forward - 250 * self.u_right
		ci1_state.transform.rotation = lgsvl.Vector(0, -90, 0)

		ci_speed = 6
		ci_angle = ci1_state.transform.rotation

		ci1_waypoints = []

		#set ci_vehicle1's waypoints of Cut in scenario
		ci1_start = ci1_state.transform.position - 2 * self.u_forward
		ci1_wp1 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci1_start.x, ci1_start.y, ci1_start.z), speed=ci_speed,
									angle=ci_angle, idle=0, trigger_distance=30, trigger=None)
		ci1_waypoints.append(ci1_wp1)

		ci1_way1 = ci1_start - 5 * self.u_right
		ci1_wp2 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci1_way1.x, ci1_way1.y, ci1_way1.z), speed=ci_speed, angle=ci_angle,
									idle=0, trigger_distance=0, trigger=None)
		ci1_waypoints.append(ci1_wp2)

		ci1_way2 = ci1_way1 + 5 * self.u_forward - 10 * self.u_right
		ci1_wp3 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci1_way2.x, ci1_way2.y, ci1_way2.z), speed=ci_speed, angle=ci_angle,
									idle=0, trigger_distance=0, trigger=None)
		ci1_waypoints.append(ci1_wp3)

		ci1_end = ci1_way2 - 30 * self.u_right
		ci1_wp4 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci1_end.x, ci1_end.y, ci1_end.z), speed=ci_speed,
									angle=ci_angle, idle=100, trigger_distance=0, trigger=None)
		ci1_waypoints.append(ci1_wp4)

		ci1_dump = ci1_end + 1000 * self.u_right
		ci1_wp5 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci1_dump.x, ci1_dump.y, ci1_dump.z), speed=ci_speed,
									angle=ci_angle, idle=0, trigger_distance=0, trigger=None)
		ci1_waypoints.append(ci1_wp5)

		ci_vehicle1 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, ci1_state)
		ci_vehicle1.follow(ci1_waypoints)

		#set vehicle 2 in Cut in scenario
		ci2_state = lgsvl.AgentState()
		ci2_state.transform.position = self.origin.position + 486 * self.u_forward - 350 * self.u_right
		ci2_state.transform.rotation = lgsvl.Vector(0, -180, 0)


		ci2_waypoints = []

		#set ci_vehicle2's waypoints of Cut in scenario
		ci2_start = ci2_state.transform.position - 2 * self.u_right
		ci2_wp1 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci2_start.x, ci2_start.y, ci2_start.z), speed=ci_speed,
									angle=ci_angle, idle=0, trigger_distance=30, trigger=None)
		ci2_waypoints.append(ci2_wp1)

		ci2_way1 = ci2_start - 5 * self.u_right
		ci2_wp2 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci2_way1.x, ci2_way1.y, ci2_way1.z), speed=ci_speed,
									angle=ci_angle, idle=0, trigger_distance=0, trigger=None)
		ci2_waypoints.append(ci2_wp2)

		ci2_way2 = ci2_way1 - 5 * self.u_forward - 10 * self.u_right
		ci2_wp3 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci2_way2.x, ci2_way2.y, ci2_way2.z), speed=ci_speed,
									angle=ci_angle, idle=0, trigger_distance=0, trigger=None)
		ci2_waypoints.append(ci2_wp3)

		ci2_end = ci2_way2 - 30 * self.u_right
		ci2_wp4 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci2_end.x, ci2_end.y, ci2_end.z), speed=ci_speed,
									angle=ci_angle, idle=10, trigger_distance=0, trigger=None)
		ci2_waypoints.append(ci2_wp4)

		ci2_dump = ci2_end + 1000 * self.u_right
		ci2_wp5 = lgsvl.DriveWaypoint(position=lgsvl.Vector(ci2_dump.x, ci2_dump.y, ci2_dump.z), speed=ci_speed,
									angle=ci_angle, idle=0, trigger_distance=0, trigger=None)
		ci2_waypoints.append(ci2_wp5)

		ci_vehicle2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, ci2_state)
		ci_vehicle2.follow(ci2_waypoints)
		return

	def setup_sim(self):
		self.create_ego(self.sim)
		if(self.cfg['scenarios']['scenario0']['flag']):
			self.create_stand_vehicle_scene(self.sim)

		if(self.cfg['scenarios']['scenario1']['flag']):
			self.create_fast_pedestrian_scene(self.sim)

		if(self.cfg['scenarios']['scenario2']['flag']):
			self.create_narrow_path_scene(self.sim)

		if(self.cfg['scenarios']['scenario3']['flag']):
			self.create_construction_section_scene(self.sim)

		if(self.cfg['scenarios']['scenario4']['flag']):
			self.create_move_pedestrian_scene(self.sim)

		if(self.cfg['scenarios']['scenario5']['flag']):
			self.create_congestion_section_scene(self.sim)

		if(self.cfg['scenarios']['scenario6']['flag']):
			self.create_cut_in_scene(self.sim)
		return

	def run(self):
		for exp_iter in range(self.cfg['exp']['iteration']):
			self.sim.reset()			
			self.setup_sim()
			# collisions = []
			print('starting exp #{}'.format(exp_iter))

			if(self.cfg['simulator']['timeout'] == 0):
				print(f'1')
				self.sim.run()
				print(f'2')
			else:
				for _ in tqdm.tqdm(range(self.cfg['simulator']['timeout'])):
					self.sim.run(1)

		print('self.collisions: {}'.format(len(self.collisions)))
		print('success')


if __name__ == '__main__':
	e = Exp()
	e.run()
	exit()
