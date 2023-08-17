/*
 * Copyright 2017-2019 Autoware Foundation. All rights reserved.
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

#ifndef OP_TOTAL_PLANNER_CORE
#define OP_TOTAL_PLANNER_CORE
#define TF_DEBUG false

#include <unistd.h>
#include <ros/ros.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>
#include <autoware_msgs/LaneArray.h>
#include <autoware_can_msgs/CANInfo.h>

#include "op_planner/PlannerH.h"
#include "op_planner/PlannerCommonDef.h"

#include <autoware_msgs/DetectedObjectArray.h>
#include <autoware_msgs/DetectedObject.h>
#include <autoware_msgs/IntersectionCondition.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>
#include "rubis_msgs/LaneArrayWithPoseTwist.h"

#include "op_planner/TrajectoryDynamicCosts.h"
#include "rubis_msgs/DetectedObjectArray.h"

#include "rubis_msgs/PoseTwistStamped.h"

#include "vector_map_msgs/PointArray.h"
#include "vector_map_msgs/LaneArray.h"
#include "vector_map_msgs/NodeArray.h"
#include "vector_map_msgs/StopLineArray.h"
#include "vector_map_msgs/DTLaneArray.h"
#include "vector_map_msgs/LineArray.h"
#include "vector_map_msgs/AreaArray.h"
#include "vector_map_msgs/SignalArray.h"
#include "vector_map_msgs/StopLine.h"
#include "vector_map_msgs/VectorArray.h"

#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <autoware_msgs/RUBISTrafficSignalArray.h>
#include <autoware_msgs/ControlCommand.h>
#include <XmlRpcException.h>

#include "op_planner/DecisionMaker.h"
#include "op_utility/DataRW.h"

#include <visualization_msgs/Marker.h>
#include <tf/transform_listener.h>

#include "rubis_msgs/LaneWithPoseTwist.h"

namespace TotalPlannerNS
{

class TotalPlanner
{
/* Trajectory generation*/
protected:
  geometry_msgs::PoseStamped current_pose_;
  geometry_msgs::TwistStamped current_twist_;

  PlannerHNS::PlannerH m_Planner;
  geometry_msgs::Pose m_OriginPos;
  PlannerHNS::WayPoint m_InitPos;
  bool bInitPos;

  PlannerHNS::WayPoint m_CurrentPos;
  bool bNewCurrentPos;

  PlannerHNS::VehicleState m_VehicleStatus;
  bool bVehicleStatus;

  std::vector<PlannerHNS::WayPoint> m_temp_path;
  std::vector<std::vector<PlannerHNS::WayPoint> > m_GlobalPaths;
  std::vector<std::vector<PlannerHNS::WayPoint> > m_GlobalPathSections;
  std::vector<PlannerHNS::WayPoint> t_centerTrajectorySmoothed;
  std::vector<std::vector<std::vector<PlannerHNS::WayPoint> > > m_RollOuts;
  bool bWayGlobalPath;
  struct timespec m_PlanningTimer;
    std::vector<std::string>    m_LogData;
    PlannerHNS::PlanningParams m_PlanningParams;
    PlannerHNS::CAR_BASIC_INFO m_CarInfo;


  //ROS messages (topics)
  ros::NodeHandle nh;

  //define publishers
  ros::Publisher pub_LocalTrajectories;
  ros::Publisher pub_LocalTrajectoriesWithPoseTwist;
  ros::Publisher pub_LocalTrajectoriesRviz;

  // define subscribers.
  ros::Subscriber sub_initialpose;
  ros::Subscriber sub_GlobalPlannerPaths;
  ros::Subscriber sub_pose_twist;

  // Others
  
  // Callback function for subscriber.
  void callbackGetInitPose(const geometry_msgs::PoseWithCovarianceStampedConstPtr &input);
  void callbackGetCurrentPoseTwist(const rubis_msgs::PoseTwistStampedPtr& msg);
  void callbackGetGlobalPlannerPath(const autoware_msgs::LaneArrayConstPtr& msg);
  rubis_msgs::LaneArrayWithPoseTwist trajectoryGeneration();

/* Trajectory evaluation */
protected:
  bool is_objects_updated_;

  PlannerHNS::TrajectoryDynamicCosts m_TrajectoryCostsCalculator;
  bool m_bUseMoveingObjectsPrediction;
  
  std::vector<std::vector<PlannerHNS::WayPoint> > m_GlobalPathsToUse;
  bool bWayGlobalPathToUse;
  std::vector<std::vector<PlannerHNS::WayPoint> > m_GeneratedRollOuts;
  bool bRollOuts;

  std::vector<PlannerHNS::DetectedObject> m_PredictedObjects;
  bool bPredictedObjects;
  // Added by PHY
  double m_SprintDecisionTime;
  int m_pedestrian_stop_img_height_threshold;

  PlannerHNS::BehaviorState   m_CurrentBehavior;
  visualization_msgs::MarkerArray m_CollisionsDummy;
  visualization_msgs::MarkerArray m_CollisionsActual;
  autoware_msgs::Lane trajectory_cost_;
  int m_ImageWidth;
  int m_ImageHeight;
  double m_PedestrianRightThreshold;
  double m_PedestrianLeftThreshold;
  double m_PedestrianImageDetectionRange;  
  double m_VehicleImageDetectionRange;
  double m_VehicleImageWidthThreshold;
  int m_noVehicleCnt;
  double distance_to_pedestrian_;

  //define publishers
  ros::Publisher pub_CollisionPointsRviz;
  ros::Publisher pub_SafetyBorderRviz;
  ros::Publisher pub_LocalWeightedTrajectoriesRviz;

  // define subscribers.
//   ros::Subscriber sub_LocalPlannerPaths;
  ros::Subscriber sub_predicted_objects;

  // HYP added
  bool sprint_switch_ = false;

  // TF
  tf::TransformListener m_vtob_listener;
  tf::TransformListener m_vtom_listener;
  tf::StampedTransform  m_velodyne_to_base_link;
  tf::StampedTransform  m_velodyne_to_map;

  // Others
  std::vector<PlannerHNS::Crossing> intersection_list_;
  autoware_msgs::DetectedObjectArray object_msg_;

  // Callback function for subscriber.  
  void callbackGetPredictedObjects(const rubis_msgs::DetectedObjectArrayConstPtr& msg);
  void _callbackGetPredictedObjects(const autoware_msgs::DetectedObjectArray& objects);
  rubis_msgs::LaneArrayWithPoseTwist trajectoryEvaluation(rubis_msgs::LaneArrayWithPoseTwist& local_lanes);

/* Behavior selection */
protected:
  double PI = 3.14159265;
  bool bWayGlobalPathLogs;
  std::vector<std::vector<PlannerHNS::WayPoint> > m_BehaviorRollOuts;

  PlannerHNS::MAP_SOURCE_TYPE m_MapType;
  std::string m_MapPath;

  PlannerHNS::RoadNetwork m_Map;
  bool bMap;

  PlannerHNS::TrajectoryCost m_TrajectoryBestCost;
  bool bBestCost;

  PlannerHNS::DecisionMaker m_BehaviorGenerator;

  autoware_msgs::Lane m_CurrentTrajectoryToSend;
  bool bNewLightStatus;
  bool bNewLightSignal;
  PlannerHNS::TrafficLightState  m_CurrLightStatus;
  std::vector<PlannerHNS::TrafficLight> m_CurrTrafficLight;
  std::vector<PlannerHNS::TrafficLight> m_PrevTrafficLight;
  tf::TransformListener m_map_base_listener;
  tf::StampedTransform m_map_base_transform;

  //Added by PHY
  double m_distanceToPedestrianThreshold;
  double m_turnAngle;
  double m_turnThreshold;
  double m_sprintSpeed;
  bool m_sprintSwitch;
  double m_obstacleWaitingTimeinIntersection;
  double distance_to_pdestrian_;

  //define publishers  
  ros::Publisher pub_LocalPathWithPosePub;
  ros::Publisher pub_LocalPath;
  ros::Publisher pub_LocalBasePath;
  ros::Publisher pub_BehaviorState;
  ros::Publisher pub_ClosestIndex;
  ros::Publisher pub_SimuBoxPose;
  ros::Publisher pub_SelectedPathRviz;
  ros::Publisher pub_BehaviorStateRviz;

  // Added by PHY & HJW
  ros::Publisher pub_EmergencyStop;
  ros::Publisher pub_turnMarker;
  ros::Publisher pub_turnAngle;

  // define subscribers.
  ros::Subscriber sub_TrafficLightStatus;
  ros::Subscriber sub_TrafficLightSignals;
  ros::Subscriber sub_Trajectory_Cost;

  ros::Subscriber sub_ctrl_cmd;

  //Added by PHY & HJW
  ros::Subscriber sub_SprintSwitch;
  ros::Subscriber sub_IntersectionCondition;

  // Others
  timespec planningTimer;
  std_msgs::Bool emergency_stop_msg;

  // Callback function for subscriber.
  void callbackGetLocalPlannerPath(const rubis_msgs::LaneArrayWithPoseTwistConstPtr& msg);      
//   void callbackGetV2XTrafficLightSignals(const autoware_msgs::RUBISTrafficSignalArray& msg);      

  void updatePedestrianAppearence();
  void updateTrajectoryCost();
  void behaviorSelection(rubis_msgs::LaneArrayWithPoseTwist& weighted_local_lanes);

  //Helper Functions
  void SendLocalPlanningTopics(const rubis_msgs::LaneArrayWithPoseTwist& msg);
  void VisualizeLocalPlanner();
  void LogLocalPlanningInfo(double dt);
  bool GetBaseMapTF();  
  void TransformPose(const geometry_msgs::PoseStamped &in_pose, geometry_msgs::PoseStamped& out_pose, const tf::StampedTransform &in_transform);
  void CalculateTurnAngle(PlannerHNS::WayPoint turn_point);

public:
  //Mapping Section

  UtilityHNS::MapRaw m_MapRaw;

  ros::Subscriber sub_lanes;
  ros::Subscriber sub_points;
  ros::Subscriber sub_dt_lanes;
  ros::Subscriber sub_intersect;
  ros::Subscriber sup_area;
  ros::Subscriber sub_lines;
  ros::Subscriber sub_stop_line;
  ros::Subscriber sub_signals;
  ros::Subscriber sub_vectors;
  ros::Subscriber sub_curbs;
  ros::Subscriber sub_edges;
  ros::Subscriber sub_way_areas;
  ros::Subscriber sub_cross_walk;
  ros::Subscriber sub_nodes;

  void callbackGetVMLanes(const vector_map_msgs::LaneArray& msg);
  void callbackGetVMPoints(const vector_map_msgs::PointArray& msg);
  void callbackGetVMdtLanes(const vector_map_msgs::DTLaneArray& msg);
  void callbackGetVMIntersections(const vector_map_msgs::CrossRoadArray& msg);
  void callbackGetVMAreas(const vector_map_msgs::AreaArray& msg);
  void callbackGetVMLines(const vector_map_msgs::LineArray& msg);
  void callbackGetVMStopLines(const vector_map_msgs::StopLineArray& msg);
  void callbackGetVMSignal(const vector_map_msgs::SignalArray& msg);
  void callbackGetVMVectors(const vector_map_msgs::VectorArray& msg);
  void callbackGetVMCurbs(const vector_map_msgs::CurbArray& msg);
  void callbackGetVMRoadEdges(const vector_map_msgs::RoadEdgeArray& msg);
  void callbackGetVMWayAreas(const vector_map_msgs::WayAreaArray& msg);
  void callbackGetVMCrossWalks(const vector_map_msgs::CrossWalkArray& msg);
  void callbackGetVMNodes(const vector_map_msgs::NodeArray& msg);
/* Helper functions */
protected:  
  void UpdatePlanningParams(ros::NodeHandle& _nh);
  void UpdateMyParams();
  bool UpdateTf();

public:
  TotalPlanner();
  ~TotalPlanner();
  void MainLoop();
};

}

#endif  // OP_TRAJECTORY_GENERATOR_CORE
