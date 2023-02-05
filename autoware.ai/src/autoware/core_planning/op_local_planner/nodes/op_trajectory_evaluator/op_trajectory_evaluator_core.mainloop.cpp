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

#include "op_trajectory_evaluator_core.h"
#include "op_ros_helpers/op_ROSHelpers.h"
#include "op_planner/MappingHelpers.h"
#include <rubis_lib/sched.hpp>

namespace TrajectoryEvaluatorNS
{

TrajectoryEval::TrajectoryEval()
{
  bNewCurrentPos = false;
  bVehicleStatus = false;
  bWayGlobalPath = false;
  bWayGlobalPathToUse = false;
  m_bUseMoveingObjectsPrediction = false;
  m_noVehicleCnt = 0;

  ros::NodeHandle _nh;
  UpdatePlanningParams(_nh);

  tf::StampedTransform transform;
  PlannerHNS::ROSHelpers::GetTransformFromTF("map", "world", transform);
  m_OriginPos.position.x  = transform.getOrigin().x();
  m_OriginPos.position.y  = transform.getOrigin().y();
  m_OriginPos.position.z  = transform.getOrigin().z();

  pub_CollisionPointsRviz = nh.advertise<visualization_msgs::MarkerArray>("dynamic_collision_points_rviz", 1);
  pub_LocalWeightedTrajectoriesRviz = nh.advertise<visualization_msgs::MarkerArray>("local_trajectories_eval_rviz", 1);
  pub_LocalWeightedTrajectories = nh.advertise<rubis_msgs::LaneArrayWithPoseTwist>("local_weighted_trajectories", 1);
  pub_LocalWeightedTrajectoriesWithPoseTwist = nh.advertise<rubis_msgs::LaneArrayWithPoseTwist>("local_weighted_trajectories_with_pose_twist", 1);
  pub_TrajectoryCost = nh.advertise<autoware_msgs::Lane>("local_trajectory_cost", 1);
  pub_SafetyBorderRviz = nh.advertise<visualization_msgs::Marker>("safety_border", 1);
  pub_DistanceToPedestrian = nh.advertise<std_msgs::Float64>("distance_to_pedestrian", 1);
  pub_IntersectionCondition = nh.advertise<autoware_msgs::IntersectionCondition>("intersection_condition", 1);
  pub_SprintSwitch = nh.advertise<std_msgs::Bool>("sprint_switch", 1);

  // sub_current_pose = nh.subscribe("/current_pose", 10, &TrajectoryEval::callbackGetCurrentPose, this);
  // sub_current_state = nh.subscribe("/current_state", 10, &TrajectoryEval::callbackGetCurrentState, this);

  int bVelSource = 1;
  _nh.getParam("/op_trajectory_evaluator/velocitySource", bVelSource);
  if(bVelSource == 0)
    sub_robot_odom = nh.subscribe("/odom", 10, &TrajectoryEval::callbackGetRobotOdom, this);
  // else if(bVelSource == 1)
  //   sub_current_velocity = nh.subscribe("/current_velocity", 10, &TrajectoryEval::callbackGetVehicleStatus, this);
  else if(bVelSource == 2)
    sub_can_info = nh.subscribe("/can_info", 10, &TrajectoryEval::callbackGetCANInfo, this);

  sub_GlobalPlannerPaths = nh.subscribe("/lane_waypoints_array", 1, &TrajectoryEval::callbackGetGlobalPlannerPath, this);
  sub_LocalPlannerPaths = nh.subscribe("/local_trajectories_with_pose_twist", 1, &TrajectoryEval::callbackGetLocalPlannerPath, this);
  sub_predicted_objects = nh.subscribe("/predicted_objects", 1, &TrajectoryEval::callbackGetPredictedObjects, this);
  sub_current_behavior = nh.subscribe("/current_behavior", 1, &TrajectoryEval::callbackGetBehaviorState, this);

  PlannerHNS::ROSHelpers::InitCollisionPointsMarkers(50, m_CollisionsDummy);

  while(1){
    if(UpdateTf() == true)
      break;
  }
}

TrajectoryEval::~TrajectoryEval()
{
}

void TrajectoryEval::UpdatePlanningParams(ros::NodeHandle& _nh)
{
  _nh.getParam("/op_trajectory_evaluator/enablePrediction", m_bUseMoveingObjectsPrediction);

  _nh.getParam("/op_common_params/horizontalSafetyDistance", m_PlanningParams.horizontalSafetyDistancel);
  _nh.getParam("/op_common_params/verticalSafetyDistance", m_PlanningParams.verticalSafetyDistance);
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

  std::cout << "Rolls Number: " << m_PlanningParams.rollOutNumber << std::endl;

  _nh.getParam("/op_common_params/horizonDistance", m_PlanningParams.horizonDistance);
  _nh.getParam("/op_common_params/minFollowingDistance", m_PlanningParams.minFollowingDistance);
  _nh.getParam("/op_common_params/minDistanceToAvoid", m_PlanningParams.minDistanceToAvoid);
  _nh.getParam("/op_common_params/maxDistanceToAvoid", m_PlanningParams.maxDistanceToAvoid);
  _nh.getParam("/op_common_params/speedProfileFactor", m_PlanningParams.speedProfileFactor);

  _nh.getParam("/op_common_params/enableLaneChange", m_PlanningParams.enableLaneChange);

  _nh.getParam("/op_common_params/width", m_CarInfo.width);
  _nh.getParam("/op_common_params/length", m_CarInfo.length);
  _nh.getParam("/op_common_params/wheelBaseLength", m_CarInfo.wheel_base);
  _nh.getParam("/op_common_params/turningRadius", m_CarInfo.turning_radius);
  _nh.getParam("/op_common_params/maxSteerAngle", m_CarInfo.max_steer_angle);
  _nh.getParam("/op_common_params/maxAcceleration", m_CarInfo.max_acceleration);
  _nh.getParam("/op_common_params/maxDeceleration", m_CarInfo.max_deceleration);
  m_CarInfo.max_speed_forward = m_PlanningParams.maxSpeed;
  m_CarInfo.min_speed_forward = m_PlanningParams.minSpeed;

  _nh.param("/op_trajectory_evaluator/PedestrianRightThreshold", m_PedestrianRightThreshold, 7.0);
  _nh.param("/op_trajectory_evaluator/PedestrianLeftThreshold", m_PedestrianLeftThreshold, 2.0);
  _nh.param("/op_trajectory_evaluator/PedestrianImageDetectionRange", m_PedestrianImageDetectionRange, 0.7);
  _nh.param("/op_trajectory_evaluator/PedestrianStopImgHeightThreshold", m_pedestrian_stop_img_height_threshold, 120);
  _nh.param("/op_trajectory_evaluator/ImageWidth", m_ImageWidth, 1920);
  _nh.param("/op_trajectory_evaluator/ImageHeight", m_ImageHeight, 1080);
  _nh.param("/op_trajectory_evaluator/VehicleImageDetectionRange", m_VehicleImageDetectionRange, 0.3);
  _nh.param("/op_trajectory_evaluator/VehicleImageWidthThreshold", m_VehicleImageWidthThreshold, 0.05);
  _nh.param("/op_trajectory_evaluator/SprintDecisionTime", m_SprintDecisionTime, 5.0);
  
  
  m_VehicleImageWidthThreshold = m_VehicleImageWidthThreshold * m_ImageWidth;
  m_PedestrianRightThreshold *= -1;

}

// void TrajectoryEval::callbackGetCurrentPose(const geometry_msgs::PoseStampedConstPtr& msg)
// {
//   m_CurrentPos = PlannerHNS::WayPoint(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z, tf::getYaw(msg->pose.orientation));
//   bNewCurrentPos = true;
// }

// void TrajectoryEval::callbackGetVehicleStatus(const geometry_msgs::TwistStampedConstPtr& msg)
// {
//   m_VehicleStatus.speed = msg->twist.linear.x;
//   m_CurrentPos.v = m_VehicleStatus.speed;
//   if(fabs(msg->twist.linear.x) > 0.25)
//     m_VehicleStatus.steer = atan(m_CarInfo.wheel_base * msg->twist.angular.z/msg->twist.linear.x);
//   UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
//   bVehicleStatus = true;
// }

void TrajectoryEval::callbackGetCANInfo(const autoware_can_msgs::CANInfoConstPtr &msg)
{
  m_VehicleStatus.speed = msg->speed/3.6;
  m_CurrentPos.v = m_VehicleStatus.speed;
  m_VehicleStatus.steer = msg->angle * m_CarInfo.max_steer_angle / m_CarInfo.max_steer_value;
  UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
  bVehicleStatus = true;
}

void TrajectoryEval::callbackGetRobotOdom(const nav_msgs::OdometryConstPtr& msg)
{
  m_VehicleStatus.speed = msg->twist.twist.linear.x;
  m_CurrentPos.v = m_VehicleStatus.speed;
  if(fabs(msg->twist.twist.linear.x) > 0.25)
    m_VehicleStatus.steer = atan(m_CarInfo.wheel_base * msg->twist.twist.angular.z/msg->twist.twist.linear.x);
  UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
  bVehicleStatus = true;
}

void TrajectoryEval::callbackGetGlobalPlannerPath(const autoware_msgs::LaneArrayConstPtr& msg)
{
  if(msg->lanes.size() > 0)
  {

    bool bOldGlobalPath = m_GlobalPaths.size() == msg->lanes.size();

    m_GlobalPaths.clear();

    for(unsigned int i = 0 ; i < msg->lanes.size(); i++)
    {
      PlannerHNS::ROSHelpers::ConvertFromAutowareLaneToLocalLane(msg->lanes.at(i), m_temp_path);

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
      std::cout << "Received New Global Path Evaluator! " << std::endl;
    }
    else
    {
      m_GlobalPaths.clear();
    }
  }
}

void TrajectoryEval::callbackGetLocalPlannerPath(const rubis_msgs::LaneArrayWithPoseTwistConstPtr& msg)
{
  rubis::instance_ = msg->instance;
  // Callback
  m_CurrentPos = PlannerHNS::WayPoint(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z, tf::getYaw(msg->pose.pose.orientation));
  bNewCurrentPos = true;

  m_VehicleStatus.speed = msg->twist.twist.linear.x;
  m_CurrentPos.v = m_VehicleStatus.speed;
  if(fabs(msg->twist.twist.linear.x) > 0.25)
    m_VehicleStatus.steer = atan(m_CarInfo.wheel_base * msg->twist.twist.angular.z/msg->twist.twist.linear.x);
  UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
  bVehicleStatus = true;  

  if(msg->lane_array.lanes.size() > 0)
  {
    m_GeneratedRollOuts.clear();
    int globalPathId_roll_outs = -1;

    for(unsigned int i = 0 ; i < msg->lane_array.lanes.size(); i++)
    {
      std::vector<PlannerHNS::WayPoint> path;
      PlannerHNS::ROSHelpers::ConvertFromAutowareLaneToLocalLane(msg->lane_array.lanes.at(i), path);
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
        bWayGlobalPath = false;
        m_GlobalPathsToUse = m_GlobalPaths;
        std::cout << "Synchronization At Trajectory Evaluator: GlobalID: " <<  globalPathId << ", LocalID: " << globalPathId_roll_outs << std::endl;
      }
    }

    bRollOuts = true;
  }
}

void TrajectoryEval::callbackGetPredictedObjects(const autoware_msgs::DetectedObjectArrayConstPtr& msg)
{  
  m_PredictedObjects.clear();
  bPredictedObjects = true;
  double distance_to_pedestrian = 1000;
  int image_person_detection_range_left = m_ImageWidth/2 - m_ImageWidth*m_PedestrianImageDetectionRange/2;
  int image_person_detection_range_right = m_ImageWidth/2 + m_ImageWidth*m_PedestrianImageDetectionRange/2;
  
  int image_vehicle_detection_range_left = m_ImageWidth/2 - m_ImageWidth*m_VehicleImageDetectionRange/2;
  int image_vehicle_detection_range_right = m_ImageWidth/2 + m_ImageWidth*m_VehicleImageDetectionRange/2;

  int vehicle_cnt = 0;

  PlannerHNS::DetectedObject obj;
  for(unsigned int i = 0 ; i <msg->objects.size(); i++)
  {    
    if(msg->objects.at(i).pose.position.y < -20 || msg->objects.at(i).pose.position.y > 20)
      continue;    
      
    if(msg->objects.at(i).pose.position.z > 1 || msg->objects.at(i).pose.position.z < -1.5)
      continue;

    autoware_msgs::DetectedObject msg_obj = msg->objects.at(i);     

    // #### Decison making for objects
    
    if(msg_obj.id > 0) // If fusion object is detected
    {
      // calculate distance to person first
      // if(msg_obj.label == "person"){        
      //   std::cout<<"Pedestrian box size(width x height):"<<msg_obj.width<<" "<<msg_obj.height<<std::endl;
      //   geometry_msgs::PoseStamped pose;
      //   pose.header = msg_obj.header;
      //   pose.pose = msg_obj.pose;
      //   try{
      //     m_vtob_listener.transformPose("/base_link", pose, pose);
      //     double temp_x_distance = pose.pose.position.x;
      //     double temp_y_distance = pose.pose.position.y;
      //     // y-axis: Left + / Right -          
      //     if(temp_y_distance > m_PedestrianLeftThreshold || temp_y_distance < m_PedestrianRightThreshold ) continue;
      //     if(abs(temp_x_distance) < abs(distance_to_pedestrian)) distance_to_pedestrian = temp_x_distance;          
      //   }
      //   catch(tf::TransformException& ex){
      //     // ROS_ERROR("Cannot transform person pose: %s", ex.what());

      //   }
      // }

      if(msg_obj.label == "car" || msg_obj.label == "truck" || msg_obj.label == "bus"){
        vehicle_cnt += 1;
      }

      PlannerHNS::ROSHelpers::ConvertFromAutowareDetectedObjectToOpenPlannerDetectedObject(msg->objects.at(i), obj);


      // transform center pose into map frame
      geometry_msgs::PoseStamped pose_in_map;
      pose_in_map.header = msg_obj.header;
      pose_in_map.pose = msg_obj.pose;
      try{
        m_vtom_listener.transformPose("/map", pose_in_map, pose_in_map);
      }
      catch(tf::TransformException& ex)
      {
        // ROS_ERROR("Cannot transform object pose: %s", ex.what());
        continue;
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

        try{
          m_vtom_listener.transformPose("/map", contour_point_in_map, contour_point_in_map);
        }
        catch(tf::TransformException& ex){
          // ROS_ERROR("Cannot transform contour pose: %s", ex.what());
          continue;
        }

        obj.contour.at(j).x = contour_point_in_map.pose.position.x;
        obj.contour.at(j).y = contour_point_in_map.pose.position.y;
        obj.contour.at(j).z = contour_point_in_map.pose.position.z;
      }

      msg_obj.header.frame_id = "map";

      m_PredictedObjects.push_back(obj);
    }
    /*
    else{ // If object is only detected at vision
      int image_obj_center_x = msg_obj.x+msg_obj.width/2;
      int image_obj_center_y = msg_obj.y+msg_obj.height/2;
      // if (msg_obj.label == "person"){// If person is detected only in image
      //   // TO ERASE
      //   std::cout<<"object height:" << msg_obj.height << " / threshold:" << m_pedestrian_stop_img_height_threshold << std::endl;
      //   if(image_obj_center_x >= image_person_detection_range_left && image_obj_center_x <= image_person_detection_range_right){ 
      //     double temp_x_distance = 1000;
      //     if(msg_obj.height >= m_pedestrian_stop_img_height_threshold) temp_x_distance = 10;
      //     if(abs(temp_x_distance) < abs(distance_to_pedestrian)) distance_to_pedestrian = temp_x_distance;
      //   }
      // }                    
      // else 
      if(msg_obj.label == "car" || msg_obj.label == "truck" || msg_obj.label == "bus"){            
        if((msg_obj.width > m_VehicleImageWidthThreshold) 
              && (image_obj_center_x > image_vehicle_detection_range_left) 
              && (image_obj_center_x < image_vehicle_detection_range_right)
        )
        {          
          vehicle_cnt+=1;        
        }
      }
    }
    */

    int image_obj_center_x = msg_obj.x+msg_obj.width/2;
    int image_obj_center_y = msg_obj.y+msg_obj.height/2;
    if (msg_obj.label == "person"){// If person is detected only in image
      
      // TO ERASE
      // ROS_WARN("object height:%d // thr: %d\n", msg_obj.height, m_pedestrian_stop_img_height_threshold);
      printf("center_x %d \n left: %d \n right %d\n\n\n", image_obj_center_x, image_person_detection_range_left, image_person_detection_range_right);
      if(image_obj_center_x >= image_person_detection_range_left && image_obj_center_x <= image_person_detection_range_right){ 
        double temp_x_distance = 1000;
        if(msg_obj.height >= m_pedestrian_stop_img_height_threshold) temp_x_distance = 10;
        if(abs(temp_x_distance) < abs(distance_to_pedestrian)) distance_to_pedestrian = temp_x_distance;
      }
    }
    
  }

  // Publish Sprint Switch
  std_msgs::Bool sprint_switch_msg;

  if(vehicle_cnt != 0){
    m_noVehicleCnt = 0;
    sprint_switch_msg.data = false;
  }
  else{ // No vehicle is exist in front of the car
    if(m_noVehicleCnt < m_SprintDecisionTime*10) {
      m_noVehicleCnt +=1;
      sprint_switch_msg.data = false;
    }
    else if (m_noVehicleCnt >= 5) sprint_switch_msg.data = true;
  }  
  pub_SprintSwitch.publish(sprint_switch_msg);

  // ROS_INFO("object # : %d", m_PredictedObjects.size());
  
  std_msgs::Float64 distanceToPedestrianMsg; 
  distanceToPedestrianMsg.data = distance_to_pedestrian;
  pub_DistanceToPedestrian.publish(distanceToPedestrianMsg);
}

void TrajectoryEval::callbackGetBehaviorState(const geometry_msgs::TwistStampedConstPtr& msg)
{
  m_CurrentBehavior.iTrajectory = msg->twist.angular.z;
}

void TrajectoryEval::callbackGetCurrentState(const std_msgs::Int32 & msg)
{
  m_CurrentBehavior.state = static_cast<PlannerHNS::STATE_TYPE>(msg.data);
}

void TrajectoryEval::UpdateMyParams()
{
  ros::NodeHandle _nh;
  _nh.getParam("/op_trajectory_evaluator/weightPriority", m_PlanningParams.weightPriority);
  _nh.getParam("/op_trajectory_evaluator/weightTransition", m_PlanningParams.weightTransition);
  _nh.getParam("/op_trajectory_evaluator/weightLong", m_PlanningParams.weightLong);
  _nh.getParam("/op_trajectory_evaluator/weightLat", m_PlanningParams.weightLat);
  _nh.getParam("/op_trajectory_evaluator/LateralSkipDistance", m_PlanningParams.LateralSkipDistance);
}

bool TrajectoryEval::UpdateTf()
{
  try{
    m_vtob_listener.waitForTransform("/velodyne", "/base_link", ros::Time(0), ros::Duration(0.001));
    m_vtob_listener.lookupTransform("/velodyne", "/base_link", ros::Time(0), m_velodyne_to_base_link);

    m_vtom_listener.waitForTransform("/velodyne", "/map", ros::Time(0), ros::Duration(0.001));
    m_vtom_listener.lookupTransform("/velodyne", "/map", ros::Time(0), m_velodyne_to_map);
    return true;
  }
  catch(tf::TransformException& ex){
    if(TF_DEBUG)
      ROS_ERROR("%s", ex.what());
    return false;
  }
}

void TrajectoryEval::MainLoop()
{
  ros::NodeHandle private_nh("~");

  // Scheduling Setup
  std::string task_response_time_filename;
  int rate;
  double task_minimum_inter_release_time;
  double task_execution_time;
  double task_relative_deadline; 

  private_nh.param<std::string>("/op_trajectory_evaluator/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/op_trajectory_evaluator.csv");
  private_nh.param<int>("/op_trajectory_evaluator/rate", rate, 10);
  private_nh.param("/op_trajectory_evaluator/task_minimum_inter_release_time", task_minimum_inter_release_time, (double)10);
  private_nh.param("/op_trajectory_evaluator/task_execution_time", task_execution_time, (double)10);
  private_nh.param("/op_trajectory_evaluator/task_relative_deadline", task_relative_deadline, (double)10);  

  rubis::init_task_profiling(task_response_time_filename);

  PlannerHNS::WayPoint prevState, state_change;

  // Add Crossing Info from yaml file
  XmlRpc::XmlRpcValue intersection_xml;  
  nh.getParam("/op_trajectory_evaluator/intersection_list", intersection_xml);
  PlannerHNS::MappingHelpers::ConstructIntersection_RUBIS(intersection_list_, intersection_xml);

  ros::Rate r(100);

  while(ros::ok()){
    // Before spin
    UpdateMyParams();
    UpdateTf();

    ros::spinOnce();

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
        pub_TrajectoryCost.publish(l);

        // hjw added : Check if ego is on intersection and obstacles are in risky area 
        int intersectionID = -1;
        double closestIntersectionDistance = -1;
        bool isInsideIntersection = false;
        bool riskyLeftTurn = false;
        bool riskyRightTurn = false;

        PlannerHNS::PlanningHelpers::GetIntersectionCondition(m_CurrentPos, intersection_list_, m_PredictedObjects, intersectionID, closestIntersectionDistance, isInsideIntersection, riskyLeftTurn, riskyRightTurn);

        autoware_msgs::IntersectionCondition ic_msg;
        ic_msg.intersectionID = intersectionID;
        ic_msg.intersectionDistance = closestIntersectionDistance;
        ic_msg.isIntersection = isInsideIntersection;
        ic_msg.riskyLeftTurn = riskyLeftTurn;
        ic_msg.riskyRightTurn = riskyRightTurn;

        pub_IntersectionCondition.publish(ic_msg);

      }

      if(m_TrajectoryCostsCalculator.m_TrajectoryCosts.size() == m_GeneratedRollOuts.size())
      {
        rubis_msgs::LaneArrayWithPoseTwist local_lanes;
        for(unsigned int i=0; i < m_GeneratedRollOuts.size(); i++)
        {
          autoware_msgs::Lane lane;
          PlannerHNS::ROSHelpers::ConvertFromLocalLaneToAutowareLane(m_GeneratedRollOuts.at(i), lane);
          lane.closest_object_distance = m_TrajectoryCostsCalculator.m_TrajectoryCosts.at(i).closest_obj_distance;
          lane.closest_object_velocity = m_TrajectoryCostsCalculator.m_TrajectoryCosts.at(i).closest_obj_velocity;
          lane.cost = m_TrajectoryCostsCalculator.m_TrajectoryCosts.at(i).cost;
          lane.is_blocked = m_TrajectoryCostsCalculator.m_TrajectoryCosts.at(i).bBlocked;
          lane.lane_index = i;
          local_lanes.lane_array.lanes.push_back(lane);
        }
        
        local_lanes.instance = rubis::instance_;
        local_lanes.pose = msg->pose;
        local_lanes.twist = msg->twist;

        pub_LocalWeightedTrajectoriesWithPoseTwist.publish(local_lanes);
        pub_LocalWeightedTrajectories.publish(local_lanes.lane_array);
        
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
    else
      sub_GlobalPlannerPaths = nh.subscribe("/lane_waypoints_array",   1,    &TrajectoryEval::callbackGetGlobalPlannerPath,   this);

    rubis::stop_task_profiling(0, 0);  

    r.sleep();
  }
}

}
