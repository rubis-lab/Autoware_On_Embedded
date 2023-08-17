/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "op_total_planner_core.h"
#include "op_planner/MappingHelpers.h"
#include "op_ros_helpers/op_ROSHelpers.h"
#include <rubis_lib/sched.hpp>

#define SPIN_PROFILING

namespace TotalPlannerNS
{



TotalPlanner::TotalPlanner()
{
  bInitPos = false;
  bNewCurrentPos = false;
  bVehicleStatus = false;
  bWayGlobalPath = false;

  ros::NodeHandle _nh;
  UpdatePlanningParams(_nh);

  /* Trjaecotry Generation*/
  tf::StampedTransform transform;
  PlannerHNS::ROSHelpers::GetTransformFromTF("map", "world", transform);
  m_OriginPos.position.x  = transform.getOrigin().x();
  m_OriginPos.position.y  = transform.getOrigin().y();
  m_OriginPos.position.z  = transform.getOrigin().z();

  pub_LocalTrajectoriesWithPoseTwist = nh.advertise<rubis_msgs::LaneArrayWithPoseTwist>("local_trajectories_with_pose_twist", 1);
  pub_LocalTrajectoriesRviz = nh.advertise<visualization_msgs::MarkerArray>("local_trajectories_gen_rviz", 1);

  sub_initialpose = nh.subscribe("/initialpose", 1, &TotalPlanner::callbackGetInitPose, this);

  int bVelSource = 1;
  _nh.getParam("/op_total_planner/velocitySource", bVelSource);

  sub_GlobalPlannerPaths = nh.subscribe("/lane_waypoints_array", 1, &TotalPlanner::callbackGetGlobalPlannerPath, this);
  sub_pose_twist = nh.subscribe("/rubis_current_pose_twist", 1, &TotalPlanner::callbackGetCurrentPoseTwist, this); // Def: 10

  /* Trajector Evaluation */
  bWayGlobalPathToUse = false;
  m_bUseMoveingObjectsPrediction = false;
  m_noVehicleCnt = 0;
  is_objects_updated_ = false;

  pub_CollisionPointsRviz = nh.advertise<visualization_msgs::MarkerArray>("dynamic_collision_points_rviz", 1);      
  pub_SafetyBorderRviz = nh.advertise<visualization_msgs::Marker>("safety_border", 1); 
  pub_LocalWeightedTrajectoriesRviz = nh.advertise<visualization_msgs::MarkerArray>("local_trajectories_eval_rviz", 1);

  sub_predicted_objects = nh.subscribe("/rubis_predicted_objects", 1, &TotalPlanner::callbackGetPredictedObjects, this);  

  PlannerHNS::ROSHelpers::InitCollisionPointsMarkers(50, m_CollisionsDummy);

  /* Behavior Selection */
  bWayGlobalPathLogs = false;
  bNewLightStatus = false;
  bNewLightSignal = false;
  bBestCost = false;
  bMap = false;
  bRollOuts = false;
  UtilityHNS::UtilityH::GetTickCount(planningTimer);
  distance_to_pdestrian_ = 1000.0;
  m_sprintSwitch = false;

    // RUBIS driving parameter
  nh.getParam("/op_total_planner/distanceToPedestrianThreshold", m_distanceToPedestrianThreshold);
  nh.param("/op_total_planner/turnThreshold", m_turnThreshold, 20.0);

  pub_LocalPathWithPosePub = nh.advertise<rubis_msgs::LaneWithPoseTwist>("final_waypoints_with_pose_twist", 1,true);
  pub_LocalBasePath = nh.advertise<autoware_msgs::Lane>("base_waypoints", 1,true);
  pub_LocalPath = nh.advertise<autoware_msgs::Lane>("final_waypoints", 1,true);
  pub_ClosestIndex = nh.advertise<std_msgs::Int32>("closest_waypoint", 1,true);
  pub_SimuBoxPose    = nh.advertise<geometry_msgs::PoseArray>("sim_box_pose_ego", 1);
  pub_BehaviorState = nh.advertise<geometry_msgs::TwistStamped>("current_behavior", 1);
  pub_BehaviorStateRviz = nh.advertise<visualization_msgs::MarkerArray>("behavior_state", 1);
  pub_SelectedPathRviz = nh.advertise<visualization_msgs::MarkerArray>("local_selected_trajectory_rviz", 1);
  pub_EmergencyStop = nh.advertise<std_msgs::Bool>("emergency_stop", 1);
  pub_turnAngle = nh.advertise<std_msgs::Float64>("turn_angle", 1);
  pub_turnMarker = nh.advertise<visualization_msgs::MarkerArray>("turn_marker", 1);
  
  //Mapping Section
  sub_lanes = nh.subscribe("/vector_map_info/lane", 1, &TotalPlanner::callbackGetVMLanes,  this);
  sub_points = nh.subscribe("/vector_map_info/point", 1, &TotalPlanner::callbackGetVMPoints,  this);
  sub_dt_lanes = nh.subscribe("/vector_map_info/dtlane", 1, &TotalPlanner::callbackGetVMdtLanes,  this);
  sub_intersect = nh.subscribe("/vector_map_info/cross_road", 1, &TotalPlanner::callbackGetVMIntersections,  this);
  sup_area = nh.subscribe("/vector_map_info/area", 1, &TotalPlanner::callbackGetVMAreas,  this);
  sub_lines = nh.subscribe("/vector_map_info/line", 1, &TotalPlanner::callbackGetVMLines,  this);
  sub_stop_line = nh.subscribe("/vector_map_info/stop_line", 1, &TotalPlanner::callbackGetVMStopLines,  this);
  sub_signals = nh.subscribe("/vector_map_info/signal", 1, &TotalPlanner::callbackGetVMSignal,  this);
  sub_vectors = nh.subscribe("/vector_map_info/vector", 1, &TotalPlanner::callbackGetVMVectors,  this);
  sub_curbs = nh.subscribe("/vector_map_info/curb", 1, &TotalPlanner::callbackGetVMCurbs,  this);
  sub_edges = nh.subscribe("/vector_map_info/road_edge", 1, &TotalPlanner::callbackGetVMRoadEdges,  this);
  sub_way_areas = nh.subscribe("/vector_map_info/way_area", 1, &TotalPlanner::callbackGetVMWayAreas,  this);
  sub_cross_walk = nh.subscribe("/vector_map_info/cross_walk", 1, &TotalPlanner::callbackGetVMCrossWalks,  this);
  sub_nodes = nh.subscribe("/vector_map_info/node", 1, &TotalPlanner::callbackGetVMNodes,  this);

  while(1){
    if(UpdateTf() == true)
      break;
  }
}

TotalPlanner::~TotalPlanner()
{
  UtilityHNS::DataRW::WriteLogData(UtilityHNS::UtilityH::GetHomeDirectory()+UtilityHNS::DataRW::LoggingMainfolderName+UtilityHNS::DataRW::StatesLogFolderName, "MainLog",
      "time,dt, Behavior_i, Behavior_str, RollOuts_n, Blocked_i, Central_i, Selected_i, StopSign_id, Light_id, Stop_Dist, Follow_Dist, Follow_Vel,"
      "Target_Vel, PID_Vel, T_cmd_Vel, C_cmd_Vel, Vel, Steer, X, Y, Z, Theta,"
      , m_LogData);
}

/* @ Trajecoty Geneator*/
void TotalPlanner::UpdatePlanningParams(ros::NodeHandle& _nh)
{
  /* Common params */
  _nh.getParam("/op_common_params/enableSwerving", m_PlanningParams.enableSwerving);
  if(m_PlanningParams.enableSwerving)
    m_PlanningParams.enableFollowing = true;
  else
    _nh.getParam("/op_common_params/enableFollowing", m_PlanningParams.enableFollowing);

  _nh.getParam("/op_common_params/enableTrafficLightBehavior", m_PlanningParams.enableTrafficLightBehavior);
  _nh.getParam("/op_common_params/enableStopSignBehavior", m_PlanningParams.enableStopSignBehavior);

  _nh.getParam("/op_common_params/maxVelocity", m_PlanningParams.maxSpeed);
  _nh.getParam("/op_common_params/minVelocity", m_PlanningParams.minSpeed);
  _nh.getParam("/op_common_params/maxLocalPlanDistance", m_PlanningParams.microPlanDistance);

  _nh.getParam("/op_common_params/pathDensity", m_PlanningParams.pathDensity);
  _nh.getParam("/op_common_params/rollOutDensity", m_PlanningParams.rollOutDensity);
  if(m_PlanningParams.enableSwerving)
    _nh.getParam("/op_common_params/rollOutsNumber", m_PlanningParams.rollOutNumber);
  else
    m_PlanningParams.rollOutNumber = 0;

  _nh.getParam("/op_common_params/horizonDistance", m_PlanningParams.horizonDistance);
  _nh.getParam("/op_common_params/minFollowingDistance", m_PlanningParams.minFollowingDistance);
  _nh.getParam("/op_common_params/minDistanceToAvoid", m_PlanningParams.minDistanceToAvoid);
  _nh.getParam("/op_common_params/maxDistanceToAvoid", m_PlanningParams.maxDistanceToAvoid);
  _nh.getParam("/op_common_params/speedProfileFactor", m_PlanningParams.speedProfileFactor);

  _nh.getParam("/op_common_params/smoothingDataWeight", m_PlanningParams.smoothingDataWeight);
  _nh.getParam("/op_common_params/smoothingSmoothWeight", m_PlanningParams.smoothingSmoothWeight);

  _nh.getParam("/op_common_params/horizontalSafetyDistance", m_PlanningParams.horizontalSafetyDistancel);
  _nh.getParam("/op_common_params/verticalSafetyDistance", m_PlanningParams.verticalSafetyDistance);

  _nh.getParam("/op_common_params/enableLaneChange", m_PlanningParams.enableLaneChange);

  _nh.getParam("/op_common_params/width", m_CarInfo.width);
  _nh.getParam("/op_common_params/length", m_CarInfo.length);
  _nh.getParam("/op_common_params/wheelBaseLength", m_CarInfo.wheel_base);
  _nh.getParam("/op_common_params/turningRadius", m_CarInfo.turning_radius);
  _nh.getParam("/op_common_params/maxSteerAngle", m_CarInfo.max_steer_angle);
  _nh.getParam("/op_common_params/maxAcceleration", m_CarInfo.max_acceleration);
  _nh.getParam("/op_common_params/maxDeceleration", m_CarInfo.max_deceleration);

  /* Trajectory generation params */
  _nh.getParam("/op_total_planner/samplingTipMargin", m_PlanningParams.carTipMargin);
  _nh.getParam("/op_total_planner/samplingOutMargin", m_PlanningParams.rollInMargin);
  _nh.getParam("/op_total_planner/samplingSpeedFactor", m_PlanningParams.rollInSpeedFactor);
  _nh.getParam("/op_total_planner/enableHeadingSmoothing", m_PlanningParams.enableHeadingSmoothing);

  m_CarInfo.max_speed_forward = m_PlanningParams.maxSpeed;
  m_CarInfo.min_speed_forward = m_PlanningParams.minSpeed;

  /* Trajectory evaluation params */
  _nh.getParam("/op_total_planner/enablePrediction", m_bUseMoveingObjectsPrediction);
  _nh.param("/op_total_planner/PedestrianRightThreshold", m_PedestrianRightThreshold, 7.0);
  _nh.param("/op_total_planner/PedestrianLeftThreshold", m_PedestrianLeftThreshold, 2.0);
  _nh.param("/op_total_planner/PedestrianImageDetectionRange", m_PedestrianImageDetectionRange, 0.7);
  _nh.param("/op_total_planner/PedestrianStopImgHeightThreshold", m_pedestrian_stop_img_height_threshold, 120);
  _nh.param("/op_total_planner/ImageWidth", m_ImageWidth, 1920);
  _nh.param("/op_total_planner/ImageHeight", m_ImageHeight, 1080);
  _nh.param("/op_total_planner/VehicleImageDetectionRange", m_VehicleImageDetectionRange, 0.3);
  _nh.param("/op_total_planner/VehicleImageWidthThreshold", m_VehicleImageWidthThreshold, 0.05);
  _nh.param("/op_total_planner/SprintDecisionTime", m_SprintDecisionTime, 5.0);
  
  m_VehicleImageWidthThreshold = m_VehicleImageWidthThreshold * m_ImageWidth;
  m_PedestrianRightThreshold *= -1;

  /* Behavior selection params */
  _nh.getParam("/op_common_params/stopLineMargin", m_PlanningParams.stopLineMargin);
  _nh.getParam("/op_common_params/stopLineDetectionDistance", m_PlanningParams.stopLineDetectionDistance);

  PlannerHNS::ControllerParams controlParams;
  controlParams.Steering_Gain = PlannerHNS::PID_CONST(0.07, 0.02, 0.01);
  controlParams.Velocity_Gain = PlannerHNS::PID_CONST(0.1, 0.005, 0.1);
  nh.getParam("/op_common_params/steeringDelay", controlParams.SteeringDelay);
  nh.getParam("/op_common_params/minPursuiteDistance", controlParams.minPursuiteDistance );
  nh.getParam("/op_common_params/additionalBrakingDistance", m_PlanningParams.additionalBrakingDistance );
  nh.getParam("/op_common_params/giveUpDistance", m_PlanningParams.giveUpDistance );
  nh.getParam("/op_common_params/enableSlowDownOnCurve", m_PlanningParams.enableSlowDownOnCurve );
  nh.getParam("/op_common_params/curveVelocityRatio", m_PlanningParams.curveVelocityRatio );

  int iSource = 0;
  _nh.getParam("/op_common_params/mapSource" , iSource);
  if(iSource == 0) m_MapType = PlannerHNS::MAP_AUTOWARE;
  else if (iSource == 1) m_MapType = PlannerHNS::MAP_FOLDER;
  else if(iSource == 2) m_MapType = PlannerHNS::MAP_KML_FILE;

  _nh.getParam("/op_common_params/mapFileName" , m_MapPath);
  _nh.getParam("/op_total_planner/evidence_tust_number", m_PlanningParams.nReliableCount);

  //std::cout << "nReliableCount: " << m_PlanningParams.nReliableCount << std::endl;
  
  _nh.param("/op_behavior_selector/sprintSpeed", m_sprintSpeed, 13.5);
  _nh.param("/op_behavior_selector/obstacleWaitingTimeinIntersection", m_obstacleWaitingTimeinIntersection, 1.0);

  m_BehaviorGenerator.Init(controlParams, m_PlanningParams, m_CarInfo, m_sprintSpeed);  
  m_BehaviorGenerator.m_pCurrentBehaviorState->m_Behavior = PlannerHNS::INITIAL_STATE;
  m_BehaviorGenerator.m_obstacleWaitingTimeinIntersection = m_obstacleWaitingTimeinIntersection;
}

void TotalPlanner::callbackGetInitPose(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg)
{
  if(!bInitPos)
  {
    m_InitPos = PlannerHNS::WayPoint(msg->pose.pose.position.x+m_OriginPos.position.x,
        msg->pose.pose.position.y+m_OriginPos.position.y,
        msg->pose.pose.position.z+m_OriginPos.position.z,
        tf::getYaw(msg->pose.pose.orientation));
    m_CurrentPos = m_InitPos;
    bInitPos = true;
  }
}

void TotalPlanner::callbackGetCurrentPoseTwist(const rubis_msgs::PoseTwistStampedPtr& msg){
  rubis::start_task_profiling();

  // Callback
  rubis::instance_ = msg->instance;
  
  m_CurrentPos = PlannerHNS::WayPoint(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z, tf::getYaw(msg->pose.pose.orientation));
  m_InitPos = m_CurrentPos;
  bNewCurrentPos = true;
  bInitPos = true;

  m_VehicleStatus.speed = msg->twist.twist.linear.x;
  m_CurrentPos.v = m_VehicleStatus.speed;
  if(fabs(msg->twist.twist.linear.x) > 0.25)
    m_VehicleStatus.steer = atan(m_CarInfo.wheel_base * msg->twist.twist.angular.z/msg->twist.twist.linear.x);
  UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
  bVehicleStatus = true;

  current_pose_ = msg->pose;
  current_twist_ = msg->twist;

  rubis_msgs::LaneArrayWithPoseTwist local_lanes = trajectoryGeneration();
  if(local_lanes.instance < 0) return;
  rubis_msgs::LaneArrayWithPoseTwist weighted_local_lanes = trajectoryEvaluation(local_lanes);
  if(weighted_local_lanes.instance < 0) return;
  behaviorSelection(weighted_local_lanes);

  rubis::stop_task_profiling(rubis::instance_, 0);

  return;
}

rubis_msgs::LaneArrayWithPoseTwist TotalPlanner::trajectoryGeneration(){
  rubis_msgs::LaneArrayWithPoseTwist local_lanes;
  local_lanes.instance = -1;

  if(bInitPos && m_GlobalPaths.size()>0)
  {
    m_GlobalPathSections.clear();

    for(unsigned int i = 0; i < m_GlobalPaths.size(); i++)
    {
      t_centerTrajectorySmoothed.clear();
      PlannerHNS::PlanningHelpers::ExtractPartFromPointToDistanceDirectionFast(m_GlobalPaths.at(i), m_CurrentPos, m_PlanningParams.horizonDistance ,
          m_PlanningParams.pathDensity ,t_centerTrajectorySmoothed);

      m_GlobalPathSections.push_back(t_centerTrajectorySmoothed);
    }

    std::vector<PlannerHNS::WayPoint> sampledPoints_debug;
    m_Planner.GenerateRunoffTrajectory(m_GlobalPathSections, m_CurrentPos,
              m_PlanningParams.enableLaneChange,
              m_VehicleStatus.speed,
              m_PlanningParams.microPlanDistance,
              m_PlanningParams.maxSpeed,
              m_PlanningParams.minSpeed,
              m_PlanningParams.carTipMargin,
              m_PlanningParams.rollInMargin,
              m_PlanningParams.rollInSpeedFactor,
              m_PlanningParams.pathDensity,
              m_PlanningParams.rollOutDensity,
              m_PlanningParams.rollOutNumber,
              m_PlanningParams.smoothingDataWeight,
              m_PlanningParams.smoothingSmoothWeight,
              m_PlanningParams.smoothingToleranceError,
              m_PlanningParams.speedProfileFactor,
              m_PlanningParams.enableHeadingSmoothing,
              -1 , -1,
              m_RollOuts, sampledPoints_debug);
    
    for(unsigned int i=0; i < m_RollOuts.size(); i++)
    {
      for(unsigned int j=0; j < m_RollOuts.at(i).size(); j++)
      {
        autoware_msgs::Lane lane;
        PlannerHNS::PlanningHelpers::PredictConstantTimeCostForTrajectory(m_RollOuts.at(i).at(j), m_CurrentPos, m_PlanningParams.minSpeed, m_PlanningParams.microPlanDistance);
        PlannerHNS::ROSHelpers::ConvertFromLocalLaneToAutowareLane(m_RollOuts.at(i).at(j), lane);
        lane.closest_object_distance = 0;
        lane.closest_object_velocity = 0;
        lane.cost = 0;
        lane.is_blocked = false;
        lane.lane_index = i;
        local_lanes.lane_array.lanes.push_back(lane);
      }
    }

    local_lanes.instance = rubis::instance_;
    local_lanes.pose = current_pose_;
    local_lanes.twist = current_twist_;

    pub_LocalTrajectoriesWithPoseTwist.publish(local_lanes);
    // pub_LocalTrajectories.publish(local_lanes.lane_array);
    
  }
  else{
    sub_GlobalPlannerPaths = nh.subscribe("/lane_waypoints_array",   1,    &TotalPlanner::callbackGetGlobalPlannerPath,   this);

    visualization_msgs::MarkerArray all_rollOuts;
    PlannerHNS::ROSHelpers::TrajectoriesToMarkers(m_RollOuts, all_rollOuts);
    pub_LocalTrajectoriesRviz.publish(all_rollOuts);
  }

  return local_lanes;
}

void TotalPlanner::callbackGetGlobalPlannerPath(const autoware_msgs::LaneArrayConstPtr& msg)
{
  if(msg->lanes.size() > 0)
  {
    bool bOldGlobalPath = m_GlobalPaths.size() == msg->lanes.size();
    
    m_GlobalPaths.clear();    

    for(unsigned int i = 0 ; i < msg->lanes.size(); i++)
    {
      PlannerHNS::ROSHelpers::ConvertFromAutowareLaneToLocalLane(msg->lanes.at(i), m_temp_path);      

      PlannerHNS::Lane* pPrevValid = 0;

      for(unsigned int j = 0 ; j < m_temp_path.size(); j++)
      {        
        PlannerHNS::Lane* pLane = 0;
        pLane = PlannerHNS::MappingHelpers::GetLaneById(m_temp_path.at(j).laneId, m_Map);       
        if(!pLane)
        {
          pLane = PlannerHNS::MappingHelpers::GetClosestLaneFromMapDirectionBased(m_temp_path.at(j), m_Map, 1);
          if(!pLane && !pPrevValid)
          {
            ROS_ERROR("Map inconsistency between Global Path and Local Planer Map, Can't identify current lane.");
            return;
          }

          if(!pLane)
            m_temp_path.at(j).pLane = pPrevValid;
          else
          {
            m_temp_path.at(j).pLane = pLane;
            pPrevValid = pLane ;
          }

          m_temp_path.at(j).laneId = m_temp_path.at(j).pLane->id;
        }
        else{          
          m_temp_path.at(j).pLane = pLane;
        }

        //std::cout << "StopLineInGlobalPath: " << m_temp_path.at(j).stopLineID << std::endl;
      }

      PlannerHNS::PlanningHelpers::CalcAngleAndCost(m_temp_path);      
      m_GlobalPaths.push_back(m_temp_path);

      if(bOldGlobalPath)
      {
        bOldGlobalPath = PlannerHNS::PlanningHelpers::CompareTrajectories(m_temp_path, m_GlobalPaths.at(i));
      }
    }

    if(!bOldGlobalPath)
    {
      bWayGlobalPath = true;
      bWayGlobalPathLogs = true;
      for(unsigned int i = 0; i < m_GlobalPaths.size(); i++)
      {
        PlannerHNS::PlanningHelpers::FixPathDensity(m_GlobalPaths.at(i), m_PlanningParams.pathDensity);
        PlannerHNS::PlanningHelpers::SmoothPath(m_GlobalPaths.at(i), 0.35, 0.4, 0.05);

        PlannerHNS::PlanningHelpers::GenerateRecommendedSpeed(m_GlobalPaths.at(i), m_CarInfo.max_speed_forward, m_PlanningParams.speedProfileFactor);
        m_GlobalPaths.at(i).at(m_GlobalPaths.at(i).size()-1).v = 0;
      }
      std::cout << "Received New Global Path Generator ! " << std::endl;
    }
    else
    {
      std::cout<<" global path clear in global path callback" << std::endl;
      m_GlobalPaths.clear();
    }
  }  
}

void TotalPlanner::MainLoop()
{
  ros::NodeHandle private_nh("~");
  m_BehaviorGenerator.m_turnThreshold = m_turnThreshold;

  // Scheduling & Profiling Setup
  std::string node_name = ros::this_node::getName();
  std::string task_response_time_filename;
  private_nh.param<std::string>(node_name+"/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/op_total_planner.csv");

  struct rubis::sched_attr attr;
  std::string policy;
  int priority, exec_time ,deadline, period;
    
  private_nh.param(node_name+"/task_scheduling_configs/policy", policy, std::string("NONE"));    
  private_nh.param(node_name+"/task_scheduling_configs/priority", priority, 99);
  private_nh.param(node_name+"/task_scheduling_configs/exec_time", exec_time, 0);
  private_nh.param(node_name+"/task_scheduling_configs/deadline", deadline, 0);
  private_nh.param(node_name+"/task_scheduling_configs/period", period, 0);
  attr = rubis::create_sched_attr(priority, exec_time, deadline, period);    
  rubis::init_task_scheduling(policy, attr);

  rubis::init_task_profiling(task_response_time_filename);

  PlannerHNS::WayPoint prevState, state_change;

  // Add Crossing Info from yaml file
  XmlRpc::XmlRpcValue intersection_xml;  
  nh.getParam("/op_trajectory_evaluator/intersection_list", intersection_xml);
  PlannerHNS::MappingHelpers::ConstructIntersection_RUBIS(intersection_list_, intersection_xml);

  ros::spin();

  return;
}

/* @ Trajectory Evaluator*/
rubis_msgs::LaneArrayWithPoseTwist TotalPlanner::trajectoryEvaluation(rubis_msgs::LaneArrayWithPoseTwist& local_lanes){
  rubis_msgs::LaneArrayWithPoseTwist weighted_local_lanes;
  weighted_local_lanes.instance = -1;

  // if(local_lanes.instance < 0) return weighted_local_lanes;
  
  UpdateMyParams();
  UpdateTf();

  // callback for objects
  if(is_objects_updated_){
    _callbackGetPredictedObjects(object_msg_);
    is_objects_updated_ = false;
  }

  static double prev_x = 0.0, prev_y = 0.0, prev_speed = 0.0;
  
  // callback for current pose
  if(prev_x != local_lanes.pose.pose.position.x || prev_y != local_lanes.pose.pose.position.y){
    m_CurrentPos = PlannerHNS::WayPoint(local_lanes.pose.pose.position.x, local_lanes.pose.pose.position.y, local_lanes.pose.pose.position.z, tf::getYaw(local_lanes.pose.pose.orientation));
    bNewCurrentPos = true;
    prev_x = local_lanes.pose.pose.position.x;
    prev_y = local_lanes.pose.pose.position.y;
  }

  // callback for vehicle status
  if(prev_speed != local_lanes.twist.twist.linear.x){
    prev_speed = local_lanes.twist.twist.linear.x;
  }

  // callback for local planner path
  if(local_lanes.lane_array.lanes.size() > 0 && local_lanes.instance >= 0)
  {
    m_GeneratedRollOuts.clear();
    int globalPathId_roll_outs = -1;

    for(unsigned int i = 0 ; i < local_lanes.lane_array.lanes.size(); i++)
    {
      std::vector<PlannerHNS::WayPoint> path;
      PlannerHNS::ROSHelpers::ConvertFromAutowareLaneToLocalLane(local_lanes.lane_array.lanes.at(i), path);
      m_GeneratedRollOuts.push_back(path);
      if(path.size() > 0)
        globalPathId_roll_outs = path.at(0).gid;
    }

    if(bWayGlobalPath && m_GlobalPaths.size() > 0 && m_GlobalPaths.at(0).size() > 0)
    {
      int globalPathId = m_GlobalPaths.at(0).at(0).gid;
      std::cout << "Before Synchronization At Trajectory Evaluator: GlobalID: " <<  globalPathId << ", LocalID: " << globalPathId_roll_outs << std::endl;

      if(globalPathId_roll_outs == globalPathId)
      {        
        // bWayGlobalPath = false;
        m_GlobalPathsToUse = m_GlobalPaths;
        std::cout << "Synchronization At Trajectory Evaluator: GlobalID: " <<  globalPathId << ", LocalID: " << globalPathId_roll_outs << std::endl;
      }
    }    

    bRollOuts = true;

    // Evaluate trajectories
    PlannerHNS::TrajectoryCost tc;  
    if(bNewCurrentPos && m_GlobalPaths.size()>0)
    {
      m_GlobalPathSections.clear();

      for(unsigned int i = 0; i < m_GlobalPathsToUse.size(); i++)
      {
        t_centerTrajectorySmoothed.clear();
        PlannerHNS::PlanningHelpers::ExtractPartFromPointToDistanceDirectionFast(m_GlobalPathsToUse.at(i), m_CurrentPos, m_PlanningParams.horizonDistance , m_PlanningParams.pathDensity ,t_centerTrajectorySmoothed);
        m_GlobalPathSections.push_back(t_centerTrajectorySmoothed);
      }

      if(m_GlobalPathSections.size()>0)
      {
        if(m_bUseMoveingObjectsPrediction)
          tc = m_TrajectoryCostsCalculator.DoOneStepDynamic(m_GeneratedRollOuts, m_GlobalPathSections.at(0), m_CurrentPos,m_PlanningParams,  m_CarInfo,m_VehicleStatus, m_PredictedObjects, m_CurrentBehavior.iTrajectory);
        else
          tc = m_TrajectoryCostsCalculator.DoOneStepStatic(m_GeneratedRollOuts, m_GlobalPathSections.at(0), m_CurrentPos,  m_PlanningParams,  m_CarInfo,m_VehicleStatus, m_PredictedObjects, m_CurrentBehavior.state);

        autoware_msgs::Lane l;
        l.closest_object_distance = tc.closest_obj_distance;
        l.closest_object_velocity = tc.closest_obj_velocity;
        l.cost = tc.cost;
        l.is_blocked = tc.bBlocked;
        l.lane_index = tc.index;
        trajectory_cost_ = l;

        // hjw added : Check if ego is on intersection and obstacles are in risky area 
        int intersectionID = -1;
        double closestIntersectionDistance = -1;
        bool isInsideIntersection = false;
        bool riskyLeftTurn = false;
        bool riskyRightTurn = false;

        PlannerHNS::PlanningHelpers::GetIntersectionCondition(m_CurrentPos, intersection_list_, m_PredictedObjects, intersectionID, closestIntersectionDistance, isInsideIntersection, riskyLeftTurn, riskyRightTurn);
        
        m_BehaviorGenerator.m_isInsideIntersection = isInsideIntersection;
        m_BehaviorGenerator.m_closestIntersectionDistance = closestIntersectionDistance;
        m_BehaviorGenerator.m_riskyLeft = riskyLeftTurn;
        m_BehaviorGenerator.m_riskyRight = riskyRightTurn;

      }
      
      if(m_TrajectoryCostsCalculator.m_TrajectoryCosts.size() == m_GeneratedRollOuts.size())
      {
        for(unsigned int i=0; i < m_GeneratedRollOuts.size(); i++)
        {
          autoware_msgs::Lane lane;
          PlannerHNS::ROSHelpers::ConvertFromLocalLaneToAutowareLane(m_GeneratedRollOuts.at(i), lane);
          lane.closest_object_distance = m_TrajectoryCostsCalculator.m_TrajectoryCosts.at(i).closest_obj_distance;
          lane.closest_object_velocity = m_TrajectoryCostsCalculator.m_TrajectoryCosts.at(i).closest_obj_velocity;
          lane.cost = m_TrajectoryCostsCalculator.m_TrajectoryCosts.at(i).cost;
          lane.is_blocked = m_TrajectoryCostsCalculator.m_TrajectoryCosts.at(i).bBlocked;
          lane.lane_index = i;
          weighted_local_lanes.lane_array.lanes.push_back(lane);
        }
      
        weighted_local_lanes.instance = rubis::instance_;
        weighted_local_lanes.obj_instance = rubis::obj_instance_;
        weighted_local_lanes.pose = local_lanes.pose;
        weighted_local_lanes.twist = local_lanes.twist;
      }
      else
      {
        ROS_ERROR("m_TrajectoryCosts.size() Not Equal m_GeneratedRollOuts.size()");
      }

      if(m_TrajectoryCostsCalculator.m_TrajectoryCosts.size()>0)
      {
        visualization_msgs::MarkerArray all_rollOuts;
        PlannerHNS::ROSHelpers::TrajectoriesToColoredMarkers(m_GeneratedRollOuts, m_TrajectoryCostsCalculator.m_TrajectoryCosts, m_CurrentBehavior.iTrajectory, all_rollOuts);
        pub_LocalWeightedTrajectoriesRviz.publish(all_rollOuts);

        PlannerHNS::ROSHelpers::ConvertCollisionPointsMarkers(m_TrajectoryCostsCalculator.m_CollisionPoints, m_CollisionsActual, m_CollisionsDummy);
        pub_CollisionPointsRviz.publish(m_CollisionsActual);

        //Visualize Safety Box
        visualization_msgs::Marker safety_box;
        PlannerHNS::ROSHelpers::ConvertFromPlannerHRectangleToAutowareRviz(m_TrajectoryCostsCalculator.m_SafetyBorder.points, safety_box);
        pub_SafetyBorderRviz.publish(safety_box);
      }
    }
    
  }
  else
    sub_GlobalPlannerPaths = nh.subscribe("/lane_waypoints_array",   1,    &TotalPlanner::callbackGetGlobalPlannerPath,   this);

  return weighted_local_lanes;
}

void TotalPlanner::callbackGetPredictedObjects(const rubis_msgs::DetectedObjectArrayConstPtr& msg)
{  
  object_msg_ = msg->object_array;
  rubis::obj_instance_ = msg->obj_instance;
  is_objects_updated_ = true;
  // _callbackGetPredictedObjects(object_msg_);
}

void TotalPlanner::_callbackGetPredictedObjects(const autoware_msgs::DetectedObjectArray& objects_msg){
  m_PredictedObjects.clear();
  // ROS_WARN("callbackGetPredictedObjects Called");
  bPredictedObjects = true;
  double distance_to_pedestrian = 1000;
  int image_person_detection_range_left = m_ImageWidth/2 - m_ImageWidth*m_PedestrianImageDetectionRange/2;
  int image_person_detection_range_right = m_ImageWidth/2 + m_ImageWidth*m_PedestrianImageDetectionRange/2;
  int image_vehicle_detection_range_left = m_ImageWidth/2 - m_ImageWidth*m_VehicleImageDetectionRange/2;
  int image_vehicle_detection_range_right = m_ImageWidth/2 + m_ImageWidth*m_VehicleImageDetectionRange/2;
  int vehicle_cnt = 0;

  PlannerHNS::DetectedObject obj;  
  for(unsigned int i = 0 ; i <objects_msg.objects.size(); i++)
  {    
    if(objects_msg.objects.at(i).pose.position.y < -20 || objects_msg.objects.at(i).pose.position.y > 20)
      continue;    
      
    if(objects_msg.objects.at(i).pose.position.z > 1 || objects_msg.objects.at(i).pose.position.z < -1.5)
      continue;

    autoware_msgs::DetectedObject msg_obj = objects_msg.objects.at(i);     

    if(msg_obj.label == "car" || msg_obj.label == "truck" || msg_obj.label == "bus"){
      vehicle_cnt += 1;
    }

    PlannerHNS::ROSHelpers::ConvertFromAutowareDetectedObjectToOpenPlannerDetectedObject(objects_msg.objects.at(i), obj);
    geometry_msgs::PoseStamped pose_in_map;
    pose_in_map.header = msg_obj.header;
    pose_in_map.pose = msg_obj.pose;
    while(1){
      try{
        m_vtom_listener.transformPose("/map", pose_in_map, pose_in_map);
        break;
      }
      catch(tf::TransformException& ex)
      {
        // ROS_ERROR("Cannot transform object pose: %s", ex.what());
        continue;
      }
    }
    // msg_obj.header.frame_id = "map";
    obj.center.pos.x = pose_in_map.pose.position.x;
    obj.center.pos.y = pose_in_map.pose.position.y;
    obj.center.pos.z = pose_in_map.pose.position.z;

    // transform contour into map frame
    for(unsigned int j = 0; j < msg_obj.convex_hull.polygon.points.size(); j++){
      geometry_msgs::PoseStamped contour_point_in_map;
      contour_point_in_map.header = msg_obj.header;
      contour_point_in_map.pose.position.x = msg_obj.convex_hull.polygon.points.at(j).x;
      contour_point_in_map.pose.position.y = msg_obj.convex_hull.polygon.points.at(j).y;
      contour_point_in_map.pose.position.z = msg_obj.convex_hull.polygon.points.at(j).z;

      // For resolve TF malform, set orientation w to 1
      contour_point_in_map.pose.orientation.w = 1;

      for(int i = 0; i < 1000; i++){
        try{
          m_vtom_listener.transformPose("/map", contour_point_in_map, contour_point_in_map);
          break;
        }
        catch(tf::TransformException& ex){
          // ROS_ERROR("Cannot transform contour pose: %s", ex.what());
          continue;
        }
      }      

      obj.contour.at(j).x = contour_point_in_map.pose.position.x;
      obj.contour.at(j).y = contour_point_in_map.pose.position.y;
      obj.contour.at(j).z = contour_point_in_map.pose.position.z;
    }

    msg_obj.header.frame_id = "map";

    m_PredictedObjects.push_back(obj);

    int image_obj_center_x = msg_obj.x+msg_obj.width/2;
    int image_obj_center_y = msg_obj.y+msg_obj.height/2;
    if (msg_obj.label == "person"){// If person is detected only in image
      ROS_WARN("==========================================");
      ROS_WARN("person detected!");
      ROS_WARN("==========================================");
      if(image_obj_center_x >= image_person_detection_range_left && image_obj_center_x <= image_person_detection_range_right){ 
        double temp_x_distance = 1000;
        if(msg_obj.height >= m_pedestrian_stop_img_height_threshold) temp_x_distance = 10;
        if(abs(temp_x_distance) < abs(distance_to_pedestrian)) distance_to_pedestrian = temp_x_distance;
      }
      ROS_WARN("==========================================");
      ROS_WARN("distance_to_pedestrian: %lf", distance_to_pedestrian);
      ROS_WARN("==========================================");
    }
  }

  // Publish Sprint Switch
  if(vehicle_cnt != 0){
    m_noVehicleCnt = 0;
    m_sprintSwitch = false;
  }
  else{ // No vehicle is exist in front of the car
    if(m_noVehicleCnt < m_SprintDecisionTime*10) {
      m_noVehicleCnt +=1;
      m_sprintSwitch = false;
    }
    else if (m_noVehicleCnt >= 5) m_sprintSwitch = true;
  }  

  distance_to_pedestrian_ = distance_to_pedestrian;
}

/* @ Behavior Selector */
void TotalPlanner::updatePedestrianAppearence(){
  if(distance_to_pdestrian_ < m_distanceToPedestrianThreshold){
    m_PlanningParams.pedestrianAppearence = true;
  }
  else
  {
    m_PlanningParams.pedestrianAppearence = false;
  }
  m_BehaviorGenerator.UpdatePedestrianAppearence(m_PlanningParams.pedestrianAppearence);
}

void TotalPlanner::updateTrajectoryCost()
{
  if(m_BehaviorGenerator.m_pCurrentBehaviorState->m_Behavior == PlannerHNS::INTERSECTION_STATE){
    bBestCost = true;
    m_TrajectoryBestCost.closest_obj_distance = trajectory_cost_.closest_object_distance;
    m_TrajectoryBestCost.closest_obj_velocity = trajectory_cost_.closest_object_velocity;
    return;
  }
  bBestCost = true;
  m_TrajectoryBestCost.bBlocked = trajectory_cost_.is_blocked;
  m_TrajectoryBestCost.index = trajectory_cost_.lane_index;
  m_TrajectoryBestCost.cost = trajectory_cost_.cost;
  m_TrajectoryBestCost.closest_obj_distance = trajectory_cost_.closest_object_distance;
  m_TrajectoryBestCost.closest_obj_velocity = trajectory_cost_.closest_object_velocity;
}

void TotalPlanner::behaviorSelection(rubis_msgs::LaneArrayWithPoseTwist& weighted_local_lanes)
{  
  // Pedestrian & trajectory cost
  updatePedestrianAppearence();
  updateTrajectoryCost();

  // Callback for local planner path
  if(weighted_local_lanes.lane_array.lanes.size() > 0)
  {
    m_BehaviorRollOuts.clear();
    int globalPathId_roll_outs = -1;

    for(unsigned int i = 0 ; i < weighted_local_lanes.lane_array.lanes.size(); i++)
    {
      std::vector<PlannerHNS::WayPoint> path;
      PlannerHNS::ROSHelpers::ConvertFromAutowareLaneToLocalLane(weighted_local_lanes.lane_array.lanes.at(i), path);
      m_BehaviorRollOuts.push_back(path);

      if(path.size() > 0)
        globalPathId_roll_outs = path.at(0).gid;
    }    

    if(bWayGlobalPath && m_GlobalPaths.size() > 0)
    {
      if(m_GlobalPaths.at(0).size() > 0)
      {
        int globalPathId = m_GlobalPaths.at(0).at(0).gid;
        std::cout << "Before Synchronization At Behavior Selector: GlobalID: " <<  globalPathId << ", LocalID: " << globalPathId_roll_outs << std::endl;

        if(globalPathId_roll_outs == globalPathId)
        {          
          bWayGlobalPath = false;
          m_GlobalPathsToUse = m_GlobalPaths;
          m_BehaviorGenerator.SetNewGlobalPath(m_GlobalPathsToUse);
          std::cout << "Synchronization At Behavior Selector: GlobalID: " <<  globalPathId << ", LocalID: " << globalPathId_roll_outs << std::endl;          
        }
      }
    }

    m_BehaviorGenerator.m_RollOuts = m_BehaviorRollOuts;
    bRollOuts = true;
  }

  // Main Loop
  // Check Pedestrian is Appeared
  double dt  = UtilityHNS::UtilityH::GetTimeDiffNow(planningTimer);
  UtilityHNS::UtilityH::GetTickCount(planningTimer);

  if(m_MapType == PlannerHNS::MAP_KML_FILE && !bMap)
  {
    bMap = true;
    PlannerHNS::MappingHelpers::LoadKML(m_MapPath, m_Map);
  }
  else if (m_MapType == PlannerHNS::MAP_FOLDER && !bMap)
  {
    bMap = true;
    PlannerHNS::MappingHelpers::ConstructRoadNetworkFromDataFiles(m_MapPath, m_Map, true);

  }
  else if (m_MapType == PlannerHNS::MAP_AUTOWARE && !bMap)
  {
    std::vector<UtilityHNS::AisanDataConnFileReader::DataConn> conn_data;;

    if(m_MapRaw.GetVersion()==2)
    {
      PlannerHNS::MappingHelpers::ConstructRoadNetworkFromROSMessageV2(m_MapRaw.pLanes->m_data_list, m_MapRaw.pPoints->m_data_list,
          m_MapRaw.pCenterLines->m_data_list, m_MapRaw.pIntersections->m_data_list,m_MapRaw.pAreas->m_data_list,
          m_MapRaw.pLines->m_data_list, m_MapRaw.pStopLines->m_data_list,  m_MapRaw.pSignals->m_data_list,
          m_MapRaw.pVectors->m_data_list, m_MapRaw.pCurbs->m_data_list, m_MapRaw.pRoadedges->m_data_list, m_MapRaw.pWayAreas->m_data_list,
          m_MapRaw.pCrossWalks->m_data_list, m_MapRaw.pNodes->m_data_list, conn_data,
          m_MapRaw.pLanes, m_MapRaw.pPoints, m_MapRaw.pNodes, m_MapRaw.pLines, PlannerHNS::GPSPoint(), m_Map, true, m_PlanningParams.enableLaneChange, false);

      try{
        // Add Traffic Signal Info from yaml file
        XmlRpc::XmlRpcValue traffic_light_list;
        nh.getParam("/op_behavior_selector/traffic_light_list", traffic_light_list);

        // Add Stop Line Info from yaml file
        XmlRpc::XmlRpcValue stop_line_list;
        nh.getParam("/op_behavior_selector/stop_line_list", stop_line_list);

        // Add Crossing Info from yaml file
        // XmlRpc::XmlRpcValue intersection_list;
        // nh.getParam("/op_behavior_selector/intersection_list", intersection_list);

        PlannerHNS::MappingHelpers::ConstructRoadNetwork_RUBIS(m_Map, traffic_light_list, stop_line_list);
      }
      catch(XmlRpc::XmlRpcException& e){
        ROS_ERROR("[XmlRpc Error] %s", e.getMessage().c_str());
        exit(1);
      }

      m_BehaviorGenerator.m_Map = m_Map;

      if(m_Map.roadSegments.size() > 0)
      {
        bMap = true;
        std::cout << " ******* Map V2 Is Loaded successfully from the Behavior Selector !! " << std::endl;
      }
    }
    else if(m_MapRaw.GetVersion()==1)
    {
      PlannerHNS::MappingHelpers::ConstructRoadNetworkFromROSMessage(m_MapRaw.pLanes->m_data_list, m_MapRaw.pPoints->m_data_list,
          m_MapRaw.pCenterLines->m_data_list, m_MapRaw.pIntersections->m_data_list,m_MapRaw.pAreas->m_data_list,
          m_MapRaw.pLines->m_data_list, m_MapRaw.pStopLines->m_data_list,  m_MapRaw.pSignals->m_data_list,
          m_MapRaw.pVectors->m_data_list, m_MapRaw.pCurbs->m_data_list, m_MapRaw.pRoadedges->m_data_list, m_MapRaw.pWayAreas->m_data_list,
          m_MapRaw.pCrossWalks->m_data_list, m_MapRaw.pNodes->m_data_list, conn_data,  PlannerHNS::GPSPoint(), m_Map, true, m_PlanningParams.enableLaneChange, false);

      if(m_Map.roadSegments.size() > 0)
      {
        bMap = true;
        std::cout << " ******* Map V1 Is Loaded successfully from the Behavior Selector !! " << std::endl;
      }
    }
  }

  if(bNewCurrentPos && m_GlobalPaths.size()>0 && weighted_local_lanes.instance >= 0)
  {
    if(bNewLightSignal)
    {
      m_PrevTrafficLight = m_CurrTrafficLight;
      bNewLightSignal = false;
    }

    if(bNewLightStatus)
    {
      bNewLightStatus = false;
      for(unsigned int itls = 0 ; itls < m_PrevTrafficLight.size() ; itls++)
        m_PrevTrafficLight.at(itls).lightState = m_CurrLightStatus;
    }
    
    m_BehaviorGenerator.m_sprintSwitch = m_sprintSwitch;
    m_CurrentBehavior = m_BehaviorGenerator.DoOneStep(dt, m_CurrentPos, m_VehicleStatus, 1, m_CurrTrafficLight, m_TrajectoryBestCost, 0);    

    CalculateTurnAngle(m_BehaviorGenerator.m_turnWaypoint);
    m_BehaviorGenerator.m_turnAngle = m_turnAngle;

    std_msgs::Float64 turn_angle_msg;
    turn_angle_msg.data = m_turnAngle;
    pub_turnAngle.publish(turn_angle_msg);

    emergency_stop_msg.data = false;
    if(m_CurrentBehavior.maxVelocity == -1)//Emergency Stop!
      emergency_stop_msg.data = true;
    pub_EmergencyStop.publish(emergency_stop_msg);

    SendLocalPlanningTopics(weighted_local_lanes);
    VisualizeLocalPlanner();
    LogLocalPlanningInfo(dt);

    // Publish turn_marker
    visualization_msgs::MarkerArray turn_marker;
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.type = 2;
    marker.pose.position.x = m_BehaviorGenerator.m_turnWaypoint.pos.x;
    marker.pose.position.y = m_BehaviorGenerator.m_turnWaypoint.pos.y;
    marker.pose.position.z = m_BehaviorGenerator.m_turnWaypoint.pos.z;
    marker.scale.x = 3;
    marker.scale.y = 3;
    marker.scale.z = 3;
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;
    marker.header.stamp = ros::Time::now();
    marker.header.frame_id = "map";
    turn_marker.markers.push_back(marker);

    pub_turnMarker.publish(turn_marker);
  }
  else{
    sub_GlobalPlannerPaths = nh.subscribe("/lane_waypoints_array",   1,    &TotalPlanner::callbackGetGlobalPlannerPath,   this);
  }
}

/*
void TotalPlanner::callbackGetV2XTrafficLightSignals(const autoware_msgs::RUBISTrafficSignalArray& msg)
{
  bNewLightSignal = true;
  std::vector<PlannerHNS::TrafficLight> simulatedLights;
  for(unsigned int i = 0 ; i < msg.signals.size() ; i++)
  {
    PlannerHNS::TrafficLight tl;
    tl.id = msg.signals.at(i).id;
    tl.remainTime = msg.signals.at(i).time;

    for(unsigned int k = 0; k < m_Map.trafficLights.size(); k++)
    {
      if(m_Map.trafficLights.at(k).id == tl.id)
      {
        tl.pos = m_Map.trafficLights.at(k).pos;
        tl.routine = m_Map.trafficLights.at(k).routine;
        break;
      }
    }

    if(msg.signals.at(i).type == 0)
    {
      tl.lightState = PlannerHNS::RED_LIGHT;
    }
    else if(msg.signals.at(i).type == 1)
    {
      tl.lightState = PlannerHNS::YELLOW_LIGHT;
    }
    else
    {
      tl.lightState = PlannerHNS::GREEN_LIGHT;
    }

    simulatedLights.push_back(tl);
  }

  m_CurrTrafficLight = simulatedLights;
}
*/

void TotalPlanner::VisualizeLocalPlanner()
{
  visualization_msgs::Marker behavior_rviz;
  int iDirection = 0;
  if(m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->iCurrSafeTrajectory > m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->iCentralTrajectory)
    iDirection = 1;
  else if(m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->iCurrSafeTrajectory < m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->iCentralTrajectory)
    iDirection = -1;
  PlannerHNS::ROSHelpers::VisualizeBehaviorState(m_CurrentPos, m_CurrentBehavior, !m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->bTrafficIsRed , iDirection, behavior_rviz, "beh_state");
  //pub_BehaviorStateRviz.publish(behavior_rviz);

  visualization_msgs::MarkerArray markerArray;

  //PlannerHNS::ROSHelpers::GetIndicatorArrows(m_CurrentPos, m_CarInfo.width, m_CarInfo.length, m_CurrentBehavior.indicator, 0, markerArray);

  markerArray.markers.push_back(behavior_rviz);

  pub_BehaviorStateRviz.publish(markerArray);

  //To Test Synchronization Problem
//  visualization_msgs::MarkerArray selected_path;
//  std::vector<std::vector<std::vector<PlannerHNS::WayPoint> > > paths;
//  paths.push_back(std::vector<std::vector<PlannerHNS::WayPoint> >());
//  paths.at(0).push_back(m_BehaviorGenerator.m_Path);
//  paths.push_back(m_GlobalPathsToUse);
//  paths.push_back(m_RollOuts);
//  PlannerHNS::ROSHelpers::TrajectoriesToMarkers(paths, selected_path);
//  pub_SelectedPathRviz.publish(selected_path);
}

void TotalPlanner::SendLocalPlanningTopics(const rubis_msgs::LaneArrayWithPoseTwist& msg)
{
  //Send Behavior State
  geometry_msgs::Twist t;
  geometry_msgs::TwistStamped behavior;
  t.linear.x = m_CurrentBehavior.bNewPlan;
  t.linear.y = m_CurrentBehavior.followDistance;
  t.linear.z = m_CurrentBehavior.followVelocity;
  t.angular.x = (int)m_CurrentBehavior.indicator;
  t.angular.y = (int)m_CurrentBehavior.state;
  t.angular.z = m_CurrentBehavior.iTrajectory;
  behavior.twist = t;
  behavior.header.stamp = ros::Time::now();
  pub_BehaviorState.publish(behavior);

  //Send Ego Vehicle Simulation Pose Data
  geometry_msgs::PoseArray sim_data;
  geometry_msgs::Pose p_id, p_pose, p_box;

  sim_data.header.frame_id = "map";
  sim_data.header.stamp = ros::Time();
  p_id.position.x = 0;
  p_pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0, 0, UtilityHNS::UtilityH::SplitPositiveAngle(m_BehaviorGenerator.state.pos.a));

  PlannerHNS::WayPoint pose_center = PlannerHNS::PlanningHelpers::GetRealCenter(m_BehaviorGenerator.state, m_CarInfo.wheel_base);

  p_pose.position.x = pose_center.pos.x;
  p_pose.position.y = pose_center.pos.y;
  p_pose.position.z = pose_center.pos.z;
  p_box.position.x = m_BehaviorGenerator.m_CarInfo.width;
  p_box.position.y = m_BehaviorGenerator.m_CarInfo.length;
  p_box.position.z = 2.2;
  sim_data.poses.push_back(p_id);
  sim_data.poses.push_back(p_pose);
  sim_data.poses.push_back(p_box);
  pub_SimuBoxPose.publish(sim_data);

  //Send Trajectory Data to path following nodes
  std_msgs::Int32 closest_waypoint;
  PlannerHNS::RelativeInfo info;
  PlannerHNS::PlanningHelpers::GetRelativeInfo(m_BehaviorGenerator.m_Path, m_BehaviorGenerator.state, info);
  PlannerHNS::ROSHelpers::ConvertFromLocalLaneToAutowareLane(m_BehaviorGenerator.m_Path, m_CurrentTrajectoryToSend, info.iBack);
  //std::cout << "Path Size: " << m_BehaviorGenerator.m_Path.size() << ", Send Size: " << m_CurrentTrajectoryToSend << std::endl;  

  closest_waypoint.data = 1;
  pub_ClosestIndex.publish(closest_waypoint);
  pub_LocalBasePath.publish(m_CurrentTrajectoryToSend);

  rubis_msgs::LaneWithPoseTwist final_waypoints_with_pose_twist_msg;
  final_waypoints_with_pose_twist_msg.instance = rubis::instance_;
  final_waypoints_with_pose_twist_msg.obj_instance = rubis::obj_instance_;
  final_waypoints_with_pose_twist_msg.lane = m_CurrentTrajectoryToSend;  
  final_waypoints_with_pose_twist_msg.pose = msg.pose;
  final_waypoints_with_pose_twist_msg.twist = msg.twist;

  pub_LocalPathWithPosePub.publish(final_waypoints_with_pose_twist_msg);
  pub_LocalPath.publish(m_CurrentTrajectoryToSend);

  
}

void TotalPlanner::LogLocalPlanningInfo(double dt)
{
  timespec log_t;
  UtilityHNS::UtilityH::GetTickCount(log_t);
  std::ostringstream dataLine;
  dataLine << UtilityHNS::UtilityH::GetLongTime(log_t) <<"," << dt << "," << m_CurrentBehavior.state << ","<< PlannerHNS::ROSHelpers::GetBehaviorNameFromCode(m_CurrentBehavior.state) << "," <<
      m_BehaviorGenerator.m_pCurrentBehaviorState->m_pParams->rollOutNumber << "," <<
      m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->bFullyBlock << "," <<
      m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->iCentralTrajectory << "," <<
      m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->iCurrSafeTrajectory << "," <<
      m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->currentStopSignID << "," <<
      m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->currentTrafficLightID << "," <<
      m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->minStoppingDistance << "," <<
      m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->distanceToNext << "," <<
      m_BehaviorGenerator.m_pCurrentBehaviorState->GetCalcParams()->velocityOfNext << "," <<
      m_CurrentBehavior.maxVelocity << "," <<
      m_VehicleStatus.speed << "," <<
      m_VehicleStatus.steer << "," <<
      m_BehaviorGenerator.state.pos.x << "," << m_BehaviorGenerator.state.pos.y << "," << m_BehaviorGenerator.state.pos.z << "," << UtilityHNS::UtilityH::SplitPositiveAngle(m_BehaviorGenerator.state.pos.a)+M_PI << ",";
  m_LogData.push_back(dataLine.str());


  if(bWayGlobalPathLogs)
  {
    for(unsigned int i=0; i < m_GlobalPaths.size(); i++)
    {
      std::ostringstream str_out;
      str_out << UtilityHNS::UtilityH::GetHomeDirectory();
      str_out << UtilityHNS::DataRW::LoggingMainfolderName;
      str_out << UtilityHNS::DataRW::PathLogFolderName;
      str_out << "Global_Vel";
      str_out << i;
      str_out << "_";
      PlannerHNS::PlanningHelpers::WritePathToFile(str_out.str(), m_GlobalPaths.at(i));
    }
    bWayGlobalPathLogs = false;
  }
}

void TotalPlanner::CalculateTurnAngle(PlannerHNS::WayPoint turn_point){
  geometry_msgs::PoseStamped turn_pose;

  if(GetBaseMapTF()){
    // std::cout<<"BEFORE:"<<turn_point.pos.x<<" "<<turn_point.pos.y<<" "<<turn_point.rot.x<<" "<<turn_point.rot.y<<" "<<turn_point.rot.z<<std::endl;
    turn_pose.pose.position.x = turn_point.pos.x;
    turn_pose.pose.position.y = turn_point.pos.y;
    turn_pose.pose.position.z = turn_point.pos.z;
    turn_pose.pose.orientation.x = turn_point.rot.x;
    turn_pose.pose.orientation.y = turn_point.rot.y;
    turn_pose.pose.orientation.z = turn_point.rot.z;
    turn_pose.pose.orientation.w = turn_point.rot.w;
    TransformPose(turn_pose, turn_pose, m_map_base_transform);
    // std::cout<<"AFTER:"<<turn_pose.pose.position.x<<" "<<turn_pose.pose.position.y<<" "<<turn_pose.pose.orientation.x<<" "<<turn_pose.pose.orientation.y<<" "<<turn_pose.pose.orientation.z<<std::endl;

    double hypot_length = hypot(turn_pose.pose.position.x, turn_pose.pose.position.y);

    if(hypot_length <= 0)
      m_turnAngle = 0;
    else
      m_turnAngle = acos(abs(turn_pose.pose.position.x)/hypot_length)*180.0/PI;
    if(turn_pose.pose.position.y < 0)
      m_turnAngle = -1 * m_turnAngle;
  }

  return;
}

bool TotalPlanner::GetBaseMapTF(){
  
  try{
    m_map_base_listener.waitForTransform("base_link", "map", ros::Time(0), ros::Duration(0.001));
    m_map_base_listener.lookupTransform("base_link", "map", ros::Time(0), m_map_base_transform);
    return true;
  }
  catch(tf::TransformException& ex)
  {
    return false;
  }
  
}

void TotalPlanner::TransformPose(const geometry_msgs::PoseStamped &in_pose, geometry_msgs::PoseStamped& out_pose, const tf::StampedTransform &in_transform)
{

  tf::Vector3 in_pos(in_pose.pose.position.x,
                     in_pose.pose.position.y,
                     in_pose.pose.position.z);
  tf::Quaternion in_quat(in_pose.pose.orientation.x,
                         in_pose.pose.orientation.y,
                         in_pose.pose.orientation.w,
                         in_pose.pose.orientation.z);

  tf::Vector3 in_pos_t = in_transform * in_pos;
  tf::Quaternion in_quat_t = in_transform * in_quat;
  
  out_pose.header = in_pose.header;
  out_pose.pose.position.x = in_pos_t.x();
  out_pose.pose.position.y = in_pos_t.y();
  out_pose.pose.position.z = in_pos_t.z();
  out_pose.pose.orientation.x = in_quat_t.x();
  out_pose.pose.orientation.y = in_quat_t.y();
  out_pose.pose.orientation.z = in_quat_t.z();

  return;
}

//Mapping Section
void TotalPlanner::callbackGetVMLanes(const vector_map_msgs::LaneArray& msg)
{
  std::cout << "Received Lanes" << endl;
  if(m_MapRaw.pLanes == nullptr)
    m_MapRaw.pLanes = new UtilityHNS::AisanLanesFileReader(msg);
}

void TotalPlanner::callbackGetVMPoints(const vector_map_msgs::PointArray& msg)
{
  std::cout << "Received Points" << endl;
  if(m_MapRaw.pPoints  == nullptr)
    m_MapRaw.pPoints = new UtilityHNS::AisanPointsFileReader(msg);
}

void TotalPlanner::callbackGetVMdtLanes(const vector_map_msgs::DTLaneArray& msg)
{
  std::cout << "Received dtLanes" << endl;
  if(m_MapRaw.pCenterLines == nullptr)
    m_MapRaw.pCenterLines = new UtilityHNS::AisanCenterLinesFileReader(msg);
}

void TotalPlanner::callbackGetVMIntersections(const vector_map_msgs::CrossRoadArray& msg)
{
  std::cout << "Received CrossRoads" << endl;
  if(m_MapRaw.pIntersections == nullptr)
    m_MapRaw.pIntersections = new UtilityHNS::AisanIntersectionFileReader(msg);
}

void TotalPlanner::callbackGetVMAreas(const vector_map_msgs::AreaArray& msg)
{
  std::cout << "Received Areas" << endl;
  if(m_MapRaw.pAreas == nullptr)
    m_MapRaw.pAreas = new UtilityHNS::AisanAreasFileReader(msg);
}

void TotalPlanner::callbackGetVMLines(const vector_map_msgs::LineArray& msg)
{
  std::cout << "Received Lines" << endl;
  if(m_MapRaw.pLines == nullptr)
    m_MapRaw.pLines = new UtilityHNS::AisanLinesFileReader(msg);
}

void TotalPlanner::callbackGetVMStopLines(const vector_map_msgs::StopLineArray& msg)
{
  std::cout << "Received StopLines" << endl;
  if(m_MapRaw.pStopLines == nullptr)
    m_MapRaw.pStopLines = new UtilityHNS::AisanStopLineFileReader(msg);
}

void TotalPlanner::callbackGetVMSignal(const vector_map_msgs::SignalArray& msg)
{
  std::cout << "Received Signals" << endl;
  if(m_MapRaw.pSignals  == nullptr)
    m_MapRaw.pSignals = new UtilityHNS::AisanSignalFileReader(msg);
}

void TotalPlanner::callbackGetVMVectors(const vector_map_msgs::VectorArray& msg)
{
  std::cout << "Received Vectors" << endl;
  if(m_MapRaw.pVectors  == nullptr)
    m_MapRaw.pVectors = new UtilityHNS::AisanVectorFileReader(msg);
}

void TotalPlanner::callbackGetVMCurbs(const vector_map_msgs::CurbArray& msg)
{
  std::cout << "Received Curbs" << endl;
  if(m_MapRaw.pCurbs == nullptr)
    m_MapRaw.pCurbs = new UtilityHNS::AisanCurbFileReader(msg);
}

void TotalPlanner::callbackGetVMRoadEdges(const vector_map_msgs::RoadEdgeArray& msg)
{
  std::cout << "Received Edges" << endl;
  if(m_MapRaw.pRoadedges  == nullptr)
    m_MapRaw.pRoadedges = new UtilityHNS::AisanRoadEdgeFileReader(msg);
}

void TotalPlanner::callbackGetVMWayAreas(const vector_map_msgs::WayAreaArray& msg)
{
  std::cout << "Received Wayareas" << endl;
  if(m_MapRaw.pWayAreas  == nullptr)
    m_MapRaw.pWayAreas = new UtilityHNS::AisanWayareaFileReader(msg);
}

void TotalPlanner::callbackGetVMCrossWalks(const vector_map_msgs::CrossWalkArray& msg)
{
  std::cout << "Received CrossWalks" << endl;
  if(m_MapRaw.pCrossWalks == nullptr)
    m_MapRaw.pCrossWalks = new UtilityHNS::AisanCrossWalkFileReader(msg);
}

void TotalPlanner::callbackGetVMNodes(const vector_map_msgs::NodeArray& msg)
{
  std::cout << "Received Nodes" << endl;
  if(m_MapRaw.pNodes == nullptr)
    m_MapRaw.pNodes = new UtilityHNS::AisanNodesFileReader(msg);
}


void TotalPlanner::UpdateMyParams()
{
  ros::NodeHandle _nh;
  _nh.getParam("/op_total_planner/weightPriority", m_PlanningParams.weightPriority);
  _nh.getParam("/op_total_planner/weightTransition", m_PlanningParams.weightTransition);
  _nh.getParam("/op_total_planner/weightLong", m_PlanningParams.weightLong);
  _nh.getParam("/op_total_planner/weightLat", m_PlanningParams.weightLat);
  _nh.getParam("/op_total_planner/LateralSkipDistance", m_PlanningParams.LateralSkipDistance);

  _nh.getParam("/op_total_planner/lateralBlockingThreshold", m_PlanningParams.lateralBlockingThreshold);
  _nh.getParam("/op_total_planner/frontLongitudinalBlockingThreshold", m_PlanningParams.frontLongitudinalBlockingThreshold);
  _nh.getParam("/op_total_planner/rearLongitudinalBlockingThreshold", m_PlanningParams.rearLongitudinalBlockingThreshold);
  _nh.getParam("/op_total_planner/enableDebug", m_PlanningParams.enableDebug);
}

bool TotalPlanner::UpdateTf()
{
  try{
    m_vtob_listener.waitForTransform("/velodyne", "/base_link", ros::Time(0), ros::Duration(0.001));
    m_vtob_listener.lookupTransform("/velodyne", "/base_link", ros::Time(0), m_velodyne_to_base_link);

    m_vtom_listener.waitForTransform("/velodyne", "/map", ros::Time(0), ros::Duration(0.001));
    m_vtom_listener.lookupTransform("/velodyne", "/map", ros::Time(0), m_velodyne_to_map);
    return true;
  }
  catch(tf::TransformException& ex){
    // if(TF_DEBUG)
    //   ROS_ERROR("%s", ex.what());
    return false;
  }
}
}