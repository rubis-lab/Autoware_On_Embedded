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

#ifndef OP_TRAJECTORY_EVALUATOR_CORE
#define OP_TRAJECTORY_EVALUATOR_CORE
#define TF_DEBUG false

#include <ros/ros.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>
#include <autoware_msgs/LaneArray.h>
#include <autoware_can_msgs/CANInfo.h>
#include <autoware_msgs/DetectedObjectArray.h>
#include <autoware_msgs/DetectedObject.h>
#include <autoware_msgs/IntersectionCondition.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>

#include "op_planner/PlannerCommonDef.h"
#include "op_planner/TrajectoryDynamicCosts.h"
#include "rubis_msgs/DetectedObjectArray.h"



namespace TrajectoryEvaluatorNS
{

class TrajectoryEval
{
protected:

  PlannerHNS::TrajectoryDynamicCosts m_TrajectoryCostsCalculator;
  bool m_bUseMoveingObjectsPrediction;

  geometry_msgs::Pose m_OriginPos;

  PlannerHNS::WayPoint m_CurrentPos;
  bool bNewCurrentPos;

  PlannerHNS::VehicleState m_VehicleStatus;
  bool bVehicleStatus;

  std::vector<PlannerHNS::WayPoint> m_temp_path;
  std::vector<std::vector<PlannerHNS::WayPoint> > m_GlobalPaths;
  std::vector<std::vector<PlannerHNS::WayPoint> > m_GlobalPathsToUse;
  std::vector<std::vector<PlannerHNS::WayPoint> > m_GlobalPathSections;
  std::vector<PlannerHNS::WayPoint> t_centerTrajectorySmoothed;
  bool bWayGlobalPath;
  bool bWayGlobalPathToUse;
  std::vector<std::vector<PlannerHNS::WayPoint> > m_GeneratedRollOuts;
  bool bRollOuts;

  std::vector<PlannerHNS::DetectedObject> m_PredictedObjects;
  bool bPredictedObjects;
  // Added by PHY
  double m_SprintDecisionTime;
  int m_pedestrian_stop_img_height_threshold;

  struct timespec m_PlanningTimer;
  std::vector<std::string>    m_LogData;
  PlannerHNS::PlanningParams  m_PlanningParams;
  PlannerHNS::CAR_BASIC_INFO  m_CarInfo;
  PlannerHNS::BehaviorState   m_CurrentBehavior;
  visualization_msgs::MarkerArray m_CollisionsDummy;
  visualization_msgs::MarkerArray m_CollisionsActual;
  int m_ImageWidth;
  int m_ImageHeight;
  double m_PedestrianRightThreshold;
  double m_PedestrianLeftThreshold;
  double m_PedestrianImageDetectionRange;  
  double m_VehicleImageDetectionRange;
  double m_VehicleImageWidthThreshold;
  int m_noVehicleCnt;

  //ROS messages (topics)
  ros::NodeHandle nh;

  //define publishers
  ros::Publisher pub_CollisionPointsRviz;
  ros::Publisher pub_LocalWeightedTrajectoriesRviz;
  ros::Publisher pub_LocalWeightedTrajectories;
  ros::Publisher pub_TrajectoryCost;
  ros::Publisher pub_SafetyBorderRviz;
  ros::Publisher pub_DistanceToPedestrian;
  ros::Publisher pub_IntersectionCondition;
  ros::Publisher pub_SprintSwitch;
  ros::Publisher pub_currentTraj;

  // define subscribers.
  ros::Subscriber sub_current_pose;
  ros::Subscriber sub_current_velocity;
  ros::Subscriber sub_robot_odom;
  ros::Subscriber sub_can_info;
  ros::Subscriber sub_GlobalPlannerPaths;
  ros::Subscriber sub_LocalPlannerPaths;
  ros::Subscriber sub_predicted_objects;
  ros::Subscriber sub_rubis_predicted_objects;
  ros::Subscriber sub_current_behavior;

  // HJW added
  ros::Subscriber sub_current_state;

  // TF
  tf::TransformListener m_vtob_listener;
  tf::TransformListener m_vtom_listener;
  tf::StampedTransform  m_velodyne_to_base_link;
  tf::StampedTransform  m_velodyne_to_map;


  // Callback function for subscriber.
  void callbackGetCurrentPose(const geometry_msgs::PoseStampedConstPtr& msg);
  void callbackGetVehicleStatus(const geometry_msgs::TwistStampedConstPtr& msg);
  void callbackGetCANInfo(const autoware_can_msgs::CANInfoConstPtr &msg);
  void callbackGetRobotOdom(const nav_msgs::OdometryConstPtr& msg);
  void callbackGetGlobalPlannerPath(const autoware_msgs::LaneArrayConstPtr& msg);
  void callbackGetLocalPlannerPath(const autoware_msgs::LaneArrayConstPtr& msg);
  void callbackGetPredictedObjects(const autoware_msgs::DetectedObjectArrayConstPtr& msg);
  void callbackGetRubisPredictedObjects(const rubis_msgs::DetectedObjectArrayConstPtr& msg);
  void callbackGetBehaviorState(const geometry_msgs::TwistStampedConstPtr & msg);
  void callbackGetCurrentState(const std_msgs::Int32 & msg);

  //Helper Functions
  void UpdatePlanningParams(ros::NodeHandle& _nh);

  void UpdateMyParams();
  bool UpdateTf();

public:
  TrajectoryEval();
  ~TrajectoryEval();
  void MainLoop();
};

}

#endif  // OP_TRAJECTORY_EVALUATOR_CORE
