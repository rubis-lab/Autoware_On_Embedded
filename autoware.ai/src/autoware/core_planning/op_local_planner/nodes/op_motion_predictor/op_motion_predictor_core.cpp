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

#include "op_motion_predictor_core.h"
#include "op_planner/MappingHelpers.h"
#include "op_ros_helpers/op_ROSHelpers.h"
#include <rubis_lib/sched.hpp>

namespace MotionPredictorNS
{

MotionPrediction::MotionPrediction()
{
  bMap = false;
  bNewCurrentPos = false;
  bVehicleStatus = false;
  bTrackedObjects = false;
  m_bEnableCurbObstacles = false;
  m_DistanceBetweenCurbs = 1.0;
  m_VisualizationTime = 0.25;
  m_bGoNextStep = false;

  ros::NodeHandle _nh;
  UpdatePlanningParams(_nh);

  tf::StampedTransform transform;
  PlannerHNS::ROSHelpers::GetTransformFromTF("map", "world", transform);
  m_OriginPos.position.x  = transform.getOrigin().x();
  m_OriginPos.position.y  = transform.getOrigin().y();
  m_OriginPos.position.z  = transform.getOrigin().z();

  // pub_predicted_objects_trajectories = nh.advertise<autoware_msgs::DetectedObjectArray>("/predicted_objects", 1);
  pub_rubis_predicted_objects_trajectories = nh.advertise<rubis_msgs::DetectedObjectArray>("/rubis_predicted_objects", 1);
  pub_PredictedTrajectoriesRviz = nh.advertise<visualization_msgs::MarkerArray>("/predicted_trajectories_rviz", 1);
  pub_CurbsRviz          = nh.advertise<visualization_msgs::MarkerArray>("/map_curbs_rviz", 1);
  pub_ParticlesRviz = nh.advertise<visualization_msgs::MarkerArray>("prediction_particles", 1);

  sub_StepSignal = nh.subscribe("/pred_step_signal",     1, &MotionPrediction::callbackGetStepForwardSignals,     this);
  // sub_tracked_objects  = nh.subscribe("/tracked_objects",       1,    &MotionPrediction::callbackGetTrackedObjects,     this);
  
  // Setup tf
  std::string tf_str_list_str;
  _nh.param<std::string>("/op_motion_predictor/object_tf_list", tf_str_list_str, "[velodyne]");
  tf_str_list_ = ParseInputStr(tf_str_list_str);
  InitTF();

  // Init Objects Sub
  std::string input_object_list_str;
  _nh.param<std::string>("/op_motion_predictor/input_object_list", input_object_list_str, "[/tracked_objects]");
  std::vector<std::string> input_object_list = ParseInputStr(input_object_list_str);

  for(auto it = input_object_list.begin(); it != input_object_list.end(); ++it){    
    std::string topic = *it;
    std::cout<<topic<<std::endl;
    
    if(topic.find(std::string("rubis")) != std::string::npos){ // For rubis topic
      objects_subs_.push_back(nh.subscribe(topic.c_str(), 1, &MotionPrediction::callbackGetRubisTrackedObjects, this));  
    }
    else{// For normal topic      
      objects_subs_.push_back(nh.subscribe(topic.c_str(), 1, &MotionPrediction::callbackGetTrackedObjects, this));      
    }
    autoware_msgs::DetectedObjectArray msg;    
    object_msg_list_.push_back(msg);
  }
  
  sub_current_pose   = nh.subscribe("/current_pose", 1,  &MotionPrediction::callbackGetCurrentPose,     this); // Def: 1

  int bVelSource = 1;
  _nh.getParam("/op_motion_predictor/velocitySource", bVelSource);
  if(bVelSource == 0)
    sub_robot_odom = nh.subscribe("/odom", 1, &MotionPrediction::callbackGetRobotOdom, this); // Def: 1
  else if(bVelSource == 1)
    sub_current_velocity = nh.subscribe("/current_velocity", 1, &MotionPrediction::callbackGetVehicleStatus, this); // Def: 1
  else if(bVelSource == 2)
    sub_can_info = nh.subscribe("/can_info", 1, &MotionPrediction::callbackGetCANInfo, this); // Def: 1
  
  UtilityHNS::UtilityH::GetTickCount(m_VisualizationTimer);
  PlannerHNS::ROSHelpers::InitPredMarkers(100, m_PredictedTrajectoriesDummy);
  PlannerHNS::ROSHelpers::InitCurbsMarkers(100, m_CurbsDummy);
  PlannerHNS::ROSHelpers::InitPredParticlesMarkers(500, m_PredictedParticlesDummy);

  //Mapping Section
  sub_lanes = nh.subscribe("/vector_map_info/lane", 1, &MotionPrediction::callbackGetVMLanes,  this);
  sub_points = nh.subscribe("/vector_map_info/point", 1, &MotionPrediction::callbackGetVMPoints,  this);
  sub_dt_lanes = nh.subscribe("/vector_map_info/dtlane", 1, &MotionPrediction::callbackGetVMdtLanes,  this);
  sub_intersect = nh.subscribe("/vector_map_info/cross_road", 1, &MotionPrediction::callbackGetVMIntersections,  this);
  sup_area = nh.subscribe("/vector_map_info/area", 1, &MotionPrediction::callbackGetVMAreas,  this);
  sub_lines = nh.subscribe("/vector_map_info/line", 1, &MotionPrediction::callbackGetVMLines,  this);
  sub_stop_line = nh.subscribe("/vector_map_info/stop_line", 1, &MotionPrediction::callbackGetVMStopLines,  this);
  sub_signals = nh.subscribe("/vector_map_info/signal", 1, &MotionPrediction::callbackGetVMSignal,  this);
  sub_vectors = nh.subscribe("/vector_map_info/vector", 1, &MotionPrediction::callbackGetVMVectors,  this);
  sub_curbs = nh.subscribe("/vector_map_info/curb", 1, &MotionPrediction::callbackGetVMCurbs,  this);
  sub_edges = nh.subscribe("/vector_map_info/road_edge", 1, &MotionPrediction::callbackGetVMRoadEdges,  this);
  sub_way_areas = nh.subscribe("/vector_map_info/way_area", 1, &MotionPrediction::callbackGetVMWayAreas,  this);
  sub_cross_walk = nh.subscribe("/vector_map_info/cross_walk", 1, &MotionPrediction::callbackGetVMCrossWalks,  this);
  sub_nodes = nh.subscribe("/vector_map_info/node", 1, &MotionPrediction::callbackGetVMNodes,  this);

  std::cout << "OpenPlanner Motion Predictor initialized successfully " << std::endl;
}

MotionPrediction::~MotionPrediction()
{
}

void MotionPrediction::InitTF(){
  int tf_num = tf_str_list_.size();
  listener_list_ = std::vector<tf::TransformListener>(tf_num);
  transform_list_ = std::vector<tf::StampedTransform>(tf_num);

  for(int i = 0; i<tf_num; i++){
    tf::TransformListener listener;
    std::string target_frame = tf_str_list_[i];
    while(1){
      try{
        listener_list_[i].waitForTransform("velodyne", target_frame, ros::Time(0), ros::Duration(0.001));
        listener_list_[i].lookupTransform("velodyne", target_frame, ros::Time(0), transform_list_[i]);
        break;
      }
      catch(tf::TransformException& ex)
      {
        // ROS_ERROR("[2] Cannot transform object pose: %s", ex.what());
      }
    }
    std::cout<<"Init tf["<<i<<"] Transform "<<target_frame<<" -> "<<"velodyne is obtained."<<std::endl;
  }
}

std::vector<std::string> MotionPrediction::ParseInputStr(std::string str){
  std::vector<std::string> result;
  str.erase(str.begin());
  str.erase(str.end()-1);

  std::istringstream ss(str);
  std::string token;
  while(std::getline(ss, token, ',')){
    result.push_back(token);
  }

  return result;
}


void MotionPrediction::UpdatePlanningParams(ros::NodeHandle& _nh)
{
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
  m_CarInfo.max_speed_forward = m_PlanningParams.maxSpeed;
  m_CarInfo.min_speed_forward = m_PlanningParams.minSpeed;

  int iSource = 0;
  _nh.getParam("/op_common_params/mapSource" , iSource);
  if(iSource == 0)
    m_MapType = PlannerHNS::MAP_AUTOWARE;
  else if (iSource == 1)
    m_MapType = PlannerHNS::MAP_FOLDER;
  else if(iSource == 2)
    m_MapType = PlannerHNS::MAP_KML_FILE;

  _nh.getParam("/op_common_params/mapFileName" , m_MapPath);

  _nh.getParam("/op_motion_predictor/enableGenrateBranches" , m_PredictBeh.m_bGenerateBranches);
  _nh.getParam("/op_motion_predictor/max_distance_to_lane" , m_PredictBeh.m_MaxLaneDetectionDistance);
  _nh.getParam("/op_motion_predictor/prediction_distance" , m_PredictBeh.m_PredictionDistance);
  _nh.getParam("/op_motion_predictor/enableCurbObstacles"  , m_bEnableCurbObstacles);
  _nh.getParam("/op_motion_predictor/distanceBetweenCurbs", m_DistanceBetweenCurbs);
  _nh.getParam("/op_motion_predictor/visualizationTime", m_VisualizationTime);
  _nh.getParam("/op_motion_predictor/enableStepByStepSignal",   m_PredictBeh.m_bStepByStep );
  _nh.getParam("/op_motion_predictor/enableParticleFilterPrediction",   m_PredictBeh.m_bParticleFilter);


  UtilityHNS::UtilityH::GetTickCount(m_SensingTimer);
}

void MotionPrediction::callbackGetStepForwardSignals(const geometry_msgs::TwistStampedConstPtr& msg)
{
  if(msg->twist.linear.x == 1)
    m_bGoNextStep = true;
  else
    m_bGoNextStep = false;
}

void MotionPrediction::callbackGetCurrentPose(const geometry_msgs::PoseStampedConstPtr& msg)
{
  m_CurrentPos = PlannerHNS::WayPoint(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z, tf::getYaw(msg->pose.orientation));
  bNewCurrentPos = true;
}

void MotionPrediction::callbackGetVehicleStatus(const geometry_msgs::TwistStampedConstPtr& msg)
{
  m_VehicleStatus.speed = msg->twist.linear.x;
  m_CurrentPos.v = m_VehicleStatus.speed;
  if(fabs(msg->twist.linear.x) > 0.25)
    m_VehicleStatus.steer = atan(m_CarInfo.wheel_base * msg->twist.angular.z/msg->twist.linear.x);
  UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
  bVehicleStatus = true;
}

void MotionPrediction::callbackGetCANInfo(const autoware_can_msgs::CANInfoConstPtr &msg)
{
  m_VehicleStatus.speed = msg->speed/3.6;
  m_VehicleStatus.steer = msg->angle * m_CarInfo.max_steer_angle / m_CarInfo.max_steer_value;
  UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
  bVehicleStatus = true;
}

void MotionPrediction::callbackGetRobotOdom(const nav_msgs::OdometryConstPtr& msg)
{
  m_VehicleStatus.speed = msg->twist.twist.linear.x;
  m_VehicleStatus.steer += atan(m_CarInfo.wheel_base * msg->twist.twist.angular.z/msg->twist.twist.linear.x);
  UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
  bVehicleStatus = true;
}

autoware_msgs::DetectedObject MotionPrediction::TransformObjToVeldoyne(const autoware_msgs::DetectedObject& in_obj, tf::StampedTransform &transform){
  autoware_msgs::DetectedObject out_obj;
  out_obj = in_obj;
  // Transform pose
  geometry_msgs::PoseStamped pose;
  pose.pose.position.x = out_obj.pose.position.x;
  pose.pose.position.y = out_obj.pose.position.y;
  pose.pose.position.z = out_obj.pose.position.z;
  pose.pose.orientation.x = out_obj.pose.orientation.x;
  pose.pose.orientation.y = out_obj.pose.orientation.y;
  pose.pose.orientation.z = out_obj.pose.orientation.z;
  pose.pose.orientation.w = out_obj.pose.orientation.w;

  TransformPose(pose, pose, transform);                        
  out_obj.pose = pose.pose;

  // Transform convex hull
  for(unsigned int j = 0; j < out_obj.convex_hull.polygon.points.size(); j++){
    geometry_msgs::PoseStamped contour_point;
    contour_point.pose.position.x = out_obj.convex_hull.polygon.points.at(j).x;
    contour_point.pose.position.y = out_obj.convex_hull.polygon.points.at(j).y;
    contour_point.pose.position.z = out_obj.convex_hull.polygon.points.at(j).z;    
    TransformPose(contour_point, contour_point, transform);        
          
    out_obj.convex_hull.polygon.points.at(j).x = (float)contour_point.pose.position.x;
    out_obj.convex_hull.polygon.points.at(j).y = (float)contour_point.pose.position.y;
    out_obj.convex_hull.polygon.points.at(j).z = (float)contour_point.pose.position.z;
  }
  out_obj.header.frame_id = "velodyne";
  out_obj.convex_hull.header.frame_id = "velodyne";    
  out_obj.valid = true;   

  return out_obj;
}

autoware_msgs::DetectedObjectArray MotionPrediction::TrasformObjAryToVeldoyne(const autoware_msgs::DetectedObjectArray& in_obj, tf::StampedTransform &transform){ 
  autoware_msgs::DetectedObjectArray out_obj = in_obj;
  out_obj.objects.clear();
  for(unsigned int i = 0 ; i <in_obj.objects.size(); i++){
    autoware_msgs::DetectedObject msg_obj = in_obj.objects.at(i);
    msg_obj = TransformObjToVeldoyne(msg_obj, transform);
    out_obj.objects.push_back(msg_obj);
  }      
  
  return out_obj;
}

void MotionPrediction::callbackGetTrackedObjects(const autoware_msgs::DetectedObjectArrayConstPtr& in_msg)
{
  std::cout<<"ERROR non rubis"<<std::endl;
  UtilityHNS::UtilityH::GetTickCount(m_SensingTimer);
  m_TrackedObjects.clear();
  bTrackedObjects = true;

  // Check frame id of the object is valid
  std::string target_frame = in_msg->header.frame_id;
  int obj_idx = getIndex(tf_str_list_, target_frame);
  if(obj_idx == -1){      
    std::cout<<target_frame<<std::endl;
    ROS_ERROR("Cannot find index by frame id");      
    exit(0);
  }  

  autoware_msgs::DetectedObjectArray msg; 
  if(target_frame != "velodyne"){
    msg = TrasformObjAryToVeldoyne(*in_msg, transform_list_[obj_idx]);  
  }
  else{
    msg = *in_msg;
  }
  
  
  PlannerHNS::DetectedObject obj;

  for(unsigned int i = 0 ; i <msg.objects.size(); i++)
  {
    // if(msg->objects.at(i).id > 0)
    // {
      PlannerHNS::ROSHelpers::ConvertFromAutowareDetectedObjectToOpenPlannerDetectedObject(msg.objects.at(i), obj);
      m_TrackedObjects.push_back(obj);
    // }
  }
  if(bMap)
  {
    if(m_PredictBeh.m_bStepByStep && m_bGoNextStep)
    {
      m_bGoNextStep = false;
      m_PredictBeh.DoOneStep(m_TrackedObjects, m_CurrentPos, m_PlanningParams.minSpeed, m_CarInfo.max_deceleration,  m_Map);
    }
    else if(!m_PredictBeh.m_bStepByStep)
    {
      m_PredictBeh.DoOneStep(m_TrackedObjects, m_CurrentPos, m_PlanningParams.minSpeed, m_CarInfo.max_deceleration,  m_Map);
    }


    object_msg_list_[obj_idx].objects.clear();
    autoware_msgs::DetectedObject pred_obj;
    for(unsigned int i = 0 ; i <m_PredictBeh.m_ParticleInfo_II.size(); i++)
    {
      PlannerHNS::ROSHelpers::ConvertFromOpenPlannerDetectedObjectToAutowareDetectedObject(m_PredictBeh.m_ParticleInfo_II.at(i)->obj, false, pred_obj);
      if(m_PredictBeh.m_ParticleInfo_II.at(i)->best_beh_track)
        pred_obj.behavior_state = m_PredictBeh.m_ParticleInfo_II.at(i)->best_beh_track->best_beh;
      // pred_obj = TransformObjToVeldoyne(pred_obj, transform_list_[obj_idx]);
      object_msg_list_[obj_idx].objects.push_back(pred_obj);
    }
    
    if(m_bEnableCurbObstacles)
    {
      curr_curbs_obstacles.clear();
      GenerateCurbsObstacles(curr_curbs_obstacles);
      //std::cout << "Curbs No: " << curr_curbs_obstacles.size() << endl;
      for(unsigned int i = 0 ; i <curr_curbs_obstacles.size(); i++)
      {
        PlannerHNS::ROSHelpers::ConvertFromOpenPlannerDetectedObjectToAutowareDetectedObject(curr_curbs_obstacles.at(i), false, pred_obj);
        // pred_obj = TransformObjToVeldoyne(pred_obj, transform_list_[obj_idx]);
        object_msg_list_[obj_idx].objects.push_back(pred_obj);
      }
    }

    m_PredictedResultsResults.objects.clear();

    for(auto it1 = object_msg_list_.begin(); it1 != object_msg_list_.end(); ++it1){
  
      auto object_msg_entry = *it1;
      for(auto it2 = object_msg_entry.objects.begin(); it2 != object_msg_entry.objects.end(); ++it2){
        auto object_entry = *it2;
        object_entry.header.frame_id = "velodyne";
        object_entry.convex_hull.header.frame_id = "velodyne";
        object_entry.valid=true;
        m_PredictedResultsResults.objects.push_back(object_entry);
      }
    }

    m_PredictedResultsResults.header.stamp = ros::Time().now();
    m_PredictedResultsResults.header.frame_id = "velodyne";
    
    pub_predicted_objects_trajectories.publish(m_PredictedResultsResults);
  }
}

void MotionPrediction::callbackGetRubisTrackedObjects(const rubis_msgs::DetectedObjectArrayConstPtr& in_msg){
  rubis::start_task_profiling();

  rubis_msgs::DetectedObjectArray msg = *in_msg;
  _callbackGetRubisTrackedObjects(msg);

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
          m_MapRaw.pLanes, m_MapRaw.pPoints, m_MapRaw.pNodes, m_MapRaw.pLines, PlannerHNS::GPSPoint(), m_Map, true, m_PlanningParams.enableLaneChange, true);

      if(m_Map.roadSegments.size() > 0)
      {
        bMap = true;
        std::cout << " ******* Map V2 Is Loaded successfully from the Motion Predictor !! " << std::endl;
      }
    }
    else if(m_MapRaw.GetVersion()==1)
    {
      PlannerHNS::MappingHelpers::ConstructRoadNetworkFromROSMessage(m_MapRaw.pLanes->m_data_list, m_MapRaw.pPoints->m_data_list,
          m_MapRaw.pCenterLines->m_data_list, m_MapRaw.pIntersections->m_data_list,m_MapRaw.pAreas->m_data_list,
          m_MapRaw.pLines->m_data_list, m_MapRaw.pStopLines->m_data_list,  m_MapRaw.pSignals->m_data_list,
          m_MapRaw.pVectors->m_data_list, m_MapRaw.pCurbs->m_data_list, m_MapRaw.pRoadedges->m_data_list, m_MapRaw.pWayAreas->m_data_list,
          m_MapRaw.pCrossWalks->m_data_list, m_MapRaw.pNodes->m_data_list, conn_data,  PlannerHNS::GPSPoint(), m_Map, true);

      if(m_Map.roadSegments.size() > 0)
      {
        bMap = true;
        std::cout << " ******* Map V1 Is Loaded successfully from the Motion Predictor !! " << std::endl;
      }
    }
  }

  if(UtilityHNS::UtilityH::GetTimeDiffNow(m_VisualizationTimer) > m_VisualizationTime)
  {
    VisualizePrediction();
    UtilityHNS::UtilityH::GetTickCount(m_VisualizationTimer);
  }

  rubis::stop_task_profiling(rubis::instance_, 0);
}

void MotionPrediction::_callbackGetRubisTrackedObjects(rubis_msgs::DetectedObjectArray& objects_msg)
{
  rubis::instance_ = objects_msg.instance;
  rubis::obj_instance_ = objects_msg.instance;

  UtilityHNS::UtilityH::GetTickCount(m_SensingTimer);
  m_TrackedObjects.clear();
  bTrackedObjects = true;

  // Check frame id of the object is valid
  std::string target_frame = objects_msg.object_array.header.frame_id;
  int obj_idx = getIndex(tf_str_list_, target_frame);
  if(obj_idx == -1){      
    std::cout<<target_frame<<std::endl;
    ROS_ERROR("Cannot find index by frame id");      
    exit(0);
  }  

  autoware_msgs::DetectedObjectArray msg; 
  if(target_frame != "velodyne"){
    msg = TrasformObjAryToVeldoyne(objects_msg.object_array, transform_list_[obj_idx]);  
  }
  else{
    msg = objects_msg.object_array;
  }
  
  
  PlannerHNS::DetectedObject obj;

  for(unsigned int i = 0 ; i <msg.objects.size(); i++)
  {
    // if(msg->objects.at(i).id > 0)
    // {
      PlannerHNS::ROSHelpers::ConvertFromAutowareDetectedObjectToOpenPlannerDetectedObject(msg.objects.at(i), obj);
      m_TrackedObjects.push_back(obj);
    // }
  }
  
  if(bMap)
  {
    if(m_PredictBeh.m_bStepByStep && m_bGoNextStep)
    {
      m_bGoNextStep = false;
      m_PredictBeh.DoOneStep(m_TrackedObjects, m_CurrentPos, m_PlanningParams.minSpeed, m_CarInfo.max_deceleration,  m_Map);
    }
    else if(!m_PredictBeh.m_bStepByStep)
    {
      m_PredictBeh.DoOneStep(m_TrackedObjects, m_CurrentPos, m_PlanningParams.minSpeed, m_CarInfo.max_deceleration,  m_Map);
    }
    object_msg_list_[obj_idx].objects.clear();
    autoware_msgs::DetectedObject pred_obj;
    for(unsigned int i = 0 ; i <m_PredictBeh.m_ParticleInfo_II.size(); i++)
    {
      PlannerHNS::ROSHelpers::ConvertFromOpenPlannerDetectedObjectToAutowareDetectedObject(m_PredictBeh.m_ParticleInfo_II.at(i)->obj, false, pred_obj);
      if(m_PredictBeh.m_ParticleInfo_II.at(i)->best_beh_track)
        pred_obj.behavior_state = m_PredictBeh.m_ParticleInfo_II.at(i)->best_beh_track->best_beh;
      // pred_obj = TransformObjToVeldoyne(pred_obj, transform_list_[obj_idx]);
      object_msg_list_[obj_idx].objects.push_back(pred_obj);
    }
    
    if(m_bEnableCurbObstacles)
    {
      curr_curbs_obstacles.clear();
      GenerateCurbsObstacles(curr_curbs_obstacles);
      //std::cout << "Curbs No: " << curr_curbs_obstacles.size() << endl;
      for(unsigned int i = 0 ; i <curr_curbs_obstacles.size(); i++)
      {
        PlannerHNS::ROSHelpers::ConvertFromOpenPlannerDetectedObjectToAutowareDetectedObject(curr_curbs_obstacles.at(i), false, pred_obj);
        // pred_obj = TransformObjToVeldoyne(pred_obj, transform_list_[obj_idx]);
        object_msg_list_[obj_idx].objects.push_back(pred_obj);
      }
    }

    m_PredictedResultsResults.objects.clear();

    for(auto it1 = object_msg_list_.begin(); it1 != object_msg_list_.end(); ++it1){
  
      auto object_msg_entry = *it1;
      for(auto it2 = object_msg_entry.objects.begin(); it2 != object_msg_entry.objects.end(); ++it2){
        auto object_entry = *it2;
        object_entry.header.frame_id = "velodyne";
        object_entry.convex_hull.header.frame_id = "velodyne";
        object_entry.valid=true;
        m_PredictedResultsResults.objects.push_back(object_entry);
      }
    }

    m_PredictedResultsResults.header.stamp = ros::Time().now();
    m_PredictedResultsResults.header.frame_id = "velodyne";

    rubis_msgs::DetectedObjectArray output_msg;
    output_msg.instance = rubis::instance_;
    output_msg.obj_instance = rubis::obj_instance_;
    output_msg.object_array = m_PredictedResultsResults;
    pub_rubis_predicted_objects_trajectories.publish(output_msg);
    
  }

}

void MotionPrediction::GenerateCurbsObstacles(std::vector<PlannerHNS::DetectedObject>& curb_obstacles)
{
  if(!bNewCurrentPos) return;

  for(unsigned int ic = 0; ic < m_Map.curbs.size(); ic++)
  {

    if(m_Map.curbs.at(ic).points.size() > 0)
    {
      PlannerHNS::DetectedObject obj;
      obj.center.pos = m_Map.curbs.at(ic).points.at(0);

      if(curb_obstacles.size()>0)
      {
        double distance_to_prev = hypot(curb_obstacles.at(curb_obstacles.size()-1).center.pos.y-obj.center.pos.y, curb_obstacles.at(curb_obstacles.size()-1).center.pos.x-obj.center.pos.x);
        if(distance_to_prev < m_DistanceBetweenCurbs)
          continue;
      }

      double longitudinalDist = hypot(m_CurrentPos.pos.y -obj.center.pos.y, m_CurrentPos.pos.x-obj.center.pos.x);

      if(longitudinalDist > m_PlanningParams.horizonDistance)
        continue;

      obj.bDirection = false;
      obj.bVelocity = false;
      obj.id = -1;
      obj.t  = PlannerHNS::SIDEWALK;
      obj.label = "curb";
      for(unsigned int icp=0; icp< m_Map.curbs.at(ic).points.size(); icp++)
      {
        obj.contour.push_back(m_Map.curbs.at(ic).points.at(icp));
      }

      curb_obstacles.push_back(obj);
    }
  }
}

void MotionPrediction::VisualizePrediction()
{
//  m_all_pred_paths.clear();
//  for(unsigned int i=0; i< m_PredictBeh.m_PredictedObjects.size(); i++)
//    m_all_pred_paths.insert(m_all_pred_paths.begin(), m_PredictBeh.m_PredictedObjects.at(i).predTrajectories.begin(), m_PredictBeh.m_PredictedObjects.at(i).predTrajectories.end());
//
//  PlannerHNS::ROSHelpers::ConvertPredictedTrqajectoryMarkers(m_all_pred_paths, m_PredictedTrajectoriesActual, m_PredictedTrajectoriesDummy);
//  pub_PredictedTrajectoriesRviz.publish(m_PredictedTrajectoriesActual);
//
  PlannerHNS::ROSHelpers::ConvertCurbsMarkers(curr_curbs_obstacles, m_CurbsActual, m_CurbsDummy);
  pub_CurbsRviz.publish(m_CurbsActual);

  m_all_pred_paths.clear();
  m_particles_points.clear();
  visualization_msgs::MarkerArray behavior_rviz_arr;
  int number_of_particles = 0;
  for(unsigned int i=0; i< m_PredictBeh.m_ParticleInfo_II.size(); i++)
  {
    m_all_pred_paths.insert(m_all_pred_paths.begin(), m_PredictBeh.m_ParticleInfo_II.at(i)->obj.predTrajectories.begin(), m_PredictBeh.m_ParticleInfo_II.at(i)->obj.predTrajectories.end());

    for(unsigned int t=0; t < m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.size(); t++)
    {
      PlannerHNS::WayPoint p_wp;
      for(unsigned int j=0; j < m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->m_StopPart.size(); j++)
      {
        p_wp = m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->m_StopPart.at(j).pose;
        p_wp.bDir = PlannerHNS::STANDSTILL_DIR;
        m_particles_points.push_back(p_wp);
        number_of_particles++;
      }

      for(unsigned int j=0; j < m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->m_YieldPart.size(); j++)
      {
        p_wp = m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->m_YieldPart.at(j).pose;
        p_wp.bDir = PlannerHNS::BACKWARD_DIR;
        m_particles_points.push_back(p_wp);
        number_of_particles++;
      }

      if(m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->beh == PlannerHNS::BEH_FORWARD_STATE)
      {
        for(unsigned int j=0; j < m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->m_ForwardPart.size(); j++)
        {
          p_wp = m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->m_ForwardPart.at(j).pose;
          p_wp.bDir = PlannerHNS::FORWARD_DIR;
          m_particles_points.push_back(p_wp);
          number_of_particles++;
        }
      }

      if(m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->beh == PlannerHNS::BEH_BRANCH_LEFT_STATE)
      {
        //std::cout << "Left Particles : " << m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->m_LeftPart.size() << std::endl;
        for(unsigned int j=0; j < m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->m_LeftPart.size(); j++)
        {
          p_wp = m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->m_LeftPart.at(j).pose;
          p_wp.bDir = PlannerHNS::FORWARD_LEFT_DIR;
          m_particles_points.push_back(p_wp);
          number_of_particles++;
        }
      }

      if(m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->beh == PlannerHNS::BEH_BRANCH_RIGHT_STATE)
      {
        for(unsigned int j=0; j < m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->m_RightPart.size(); j++)
        {
          p_wp = m_PredictBeh.m_ParticleInfo_II.at(i)->m_TrajectoryTracker.at(t)->m_RightPart.at(j).pose;
          p_wp.bDir = PlannerHNS::FORWARD_RIGHT_DIR;
          m_particles_points.push_back(p_wp);
          number_of_particles++;
        }
      }
    }


//    visualization_msgs::Marker behavior_rviz;
//    std::ostringstream ns_beh;
//    ns_beh << "pred_beh_state_" << i;
//    ROSHelpers::VisualizeBehaviorState(m_PredictBeh.m_ParticleInfo_II.at(i).obj.center, m_PredictBeh.m_ParticleInfo_II.at(i).m_beh, false , 0, behavior_rviz, ns_beh.str(), 3);
//    behavior_rviz_arr.markers.push_back(behavior_rviz);

  }

//  pub_PredBehaviorStateRviz.publish(behavior_rviz_arr);

  PlannerHNS::ROSHelpers::ConvertParticles(m_particles_points,m_PredictedParticlesActual, m_PredictedParticlesDummy);
  //std::cout << "Original Particles: " << number_of_particles <<  ", Total Particles Num: " << m_PredictedParticlesActual.markers.size() << std::endl;
  pub_ParticlesRviz.publish(m_PredictedParticlesActual);

  //std::cout << "Start Tracking of Trajectories : " <<  m_all_pred_paths.size() << endl;
  PlannerHNS::ROSHelpers::ConvertPredictedTrqajectoryMarkers(m_all_pred_paths, m_PredictedTrajectoriesActual, m_PredictedTrajectoriesDummy);
  pub_PredictedTrajectoriesRviz.publish(m_PredictedTrajectoriesActual);

  UtilityHNS::UtilityH::GetTickCount(m_VisualizationTimer);
}

void MotionPrediction::MainLoop()
{

  ros::NodeHandle private_nh("~");

  // Scheduling & Profiling Setup
  std::string node_name = ros::this_node::getName();
  std::string task_response_time_filename;
  private_nh.param<std::string>(node_name+"/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/op_motion_predictor.csv");

  int rate;
  private_nh.param<int>(node_name+"/rate", rate, 10);

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

  ros::spin();
}

void MotionPrediction::TransformPose(const geometry_msgs::PoseStamped &in_pose, geometry_msgs::PoseStamped& out_pose, const tf::StampedTransform &in_transform)
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
  out_pose.pose.orientation.w = in_quat_t.w();


  return;
}

int MotionPrediction::getIndex(std::vector<std::string> v, std::string K)
{
    int index = -1;
    auto it = find(v.begin(), v.end(), K);
    // If element was found
    if (it != v.end()) 
    {
        // calculating the index
        // of K
        index = it - v.begin();
        return index;
    }
    else {
        // If the element is not
        // present in the vector
        index = -1;
    }

    K.erase(K.begin());
    it = find(v.begin(), v.end(), K);
    // If element was found
    if (it != v.end()) 
    {
        // calculating the index
        // of K
        index = it - v.begin();
        return index;
    }
    else {
        // If the element is not
        // present in the vector
        index = -1;
    }
    return index;
}

//Mapping Section

void MotionPrediction::callbackGetVMLanes(const vector_map_msgs::LaneArray& msg)
{
  std::cout << "Received Lanes" << endl;
  if(m_MapRaw.pLanes == nullptr)
    m_MapRaw.pLanes = new UtilityHNS::AisanLanesFileReader(msg);
}

void MotionPrediction::callbackGetVMPoints(const vector_map_msgs::PointArray& msg)
{
  std::cout << "Received Points" << endl;
  if(m_MapRaw.pPoints  == nullptr)
    m_MapRaw.pPoints = new UtilityHNS::AisanPointsFileReader(msg);
}

void MotionPrediction::callbackGetVMdtLanes(const vector_map_msgs::DTLaneArray& msg)
{
  std::cout << "Received dtLanes" << endl;
  if(m_MapRaw.pCenterLines == nullptr)
    m_MapRaw.pCenterLines = new UtilityHNS::AisanCenterLinesFileReader(msg);
}

void MotionPrediction::callbackGetVMIntersections(const vector_map_msgs::CrossRoadArray& msg)
{
  std::cout << "Received CrossRoads" << endl;
  if(m_MapRaw.pIntersections == nullptr)
    m_MapRaw.pIntersections = new UtilityHNS::AisanIntersectionFileReader(msg);
}

void MotionPrediction::callbackGetVMAreas(const vector_map_msgs::AreaArray& msg)
{
  std::cout << "Received Areas" << endl;
  if(m_MapRaw.pAreas == nullptr)
    m_MapRaw.pAreas = new UtilityHNS::AisanAreasFileReader(msg);
}

void MotionPrediction::callbackGetVMLines(const vector_map_msgs::LineArray& msg)
{
  std::cout << "Received Lines" << endl;
  if(m_MapRaw.pLines == nullptr)
    m_MapRaw.pLines = new UtilityHNS::AisanLinesFileReader(msg);
}

void MotionPrediction::callbackGetVMStopLines(const vector_map_msgs::StopLineArray& msg)
{
  std::cout << "Received StopLines" << endl;
  if(m_MapRaw.pStopLines == nullptr)
    m_MapRaw.pStopLines = new UtilityHNS::AisanStopLineFileReader(msg);
}

void MotionPrediction::callbackGetVMSignal(const vector_map_msgs::SignalArray& msg)
{
  std::cout << "Received Signals" << endl;
  if(m_MapRaw.pSignals  == nullptr)
    m_MapRaw.pSignals = new UtilityHNS::AisanSignalFileReader(msg);
}

void MotionPrediction::callbackGetVMVectors(const vector_map_msgs::VectorArray& msg)
{
  std::cout << "Received Vectors" << endl;
  if(m_MapRaw.pVectors  == nullptr)
    m_MapRaw.pVectors = new UtilityHNS::AisanVectorFileReader(msg);
}

void MotionPrediction::callbackGetVMCurbs(const vector_map_msgs::CurbArray& msg)
{
  std::cout << "Received Curbs" << endl;
  if(m_MapRaw.pCurbs == nullptr)
    m_MapRaw.pCurbs = new UtilityHNS::AisanCurbFileReader(msg);
}

void MotionPrediction::callbackGetVMRoadEdges(const vector_map_msgs::RoadEdgeArray& msg)
{
  std::cout << "Received Edges" << endl;
  if(m_MapRaw.pRoadedges  == nullptr)
    m_MapRaw.pRoadedges = new UtilityHNS::AisanRoadEdgeFileReader(msg);
}

void MotionPrediction::callbackGetVMWayAreas(const vector_map_msgs::WayAreaArray& msg)
{
  std::cout << "Received Wayareas" << endl;
  if(m_MapRaw.pWayAreas  == nullptr)
    m_MapRaw.pWayAreas = new UtilityHNS::AisanWayareaFileReader(msg);
}

void MotionPrediction::callbackGetVMCrossWalks(const vector_map_msgs::CrossWalkArray& msg)
{
  std::cout << "Received CrossWalks" << endl;
  if(m_MapRaw.pCrossWalks == nullptr)
    m_MapRaw.pCrossWalks = new UtilityHNS::AisanCrossWalkFileReader(msg);
}

void MotionPrediction::callbackGetVMNodes(const vector_map_msgs::NodeArray& msg)
{
  std::cout << "Received Nodes" << endl;
  if(m_MapRaw.pNodes == nullptr)
    m_MapRaw.pNodes = new UtilityHNS::AisanNodesFileReader(msg);
}

}
