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

#include "rubis_msgs/Bool.h"
#include "rubis_msgs/LaneArrayWithPoseTwist.h"
#include <autoware_can_msgs/CANInfo.h>
#include <autoware_msgs/DetectedObject.h>
#include <autoware_msgs/DetectedObjectArray.h>
#include <autoware_msgs/IntersectionCondition.h>
#include <autoware_msgs/LaneArray.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>

#include "op_planner/PlannerCommonDef.h"
#include "op_planner/TrajectoryDynamicCosts.h"
#include "rubis_msgs/DetectedObjectArray.h"
#include "rubis_msgs/Image.h"
#include "rubis_msgs/PlanningInfo.h"

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>

typedef message_filters::sync_policies::ExactTime<rubis_msgs::LaneArrayWithPoseTwist, rubis_msgs::DetectedObjectArray> SyncPolicy;
typedef message_filters::sync_policies::ExactTime<rubis_msgs::LaneArrayWithPoseTwist, rubis_msgs::DetectedObjectArray, rubis_msgs::Bool>
    LaneSyncPolicy;
typedef message_filters::Synchronizer<SyncPolicy> Sync;
typedef message_filters::Synchronizer<LaneSyncPolicy> LaneSync;

namespace TrajectoryEvaluatorNS {

class TrajectoryEval {
  protected:
    bool is_objects_updated_;

    PlannerHNS::TrajectoryDynamicCosts m_TrajectoryCostsCalculator;
    bool m_bUseMoveingObjectsPrediction;

    geometry_msgs::Pose m_OriginPos;

    PlannerHNS::WayPoint m_CurrentPos;
    bool bNewCurrentPos;

    PlannerHNS::VehicleState m_VehicleStatus;
    bool bVehicleStatus;

    std::vector<PlannerHNS::WayPoint> m_temp_path;
    std::vector<std::vector<PlannerHNS::WayPoint>> m_GlobalPaths;
    std::vector<std::vector<PlannerHNS::WayPoint>> m_GlobalPathsToUse;
    std::vector<std::vector<PlannerHNS::WayPoint>> m_GlobalPathSections;
    std::vector<PlannerHNS::WayPoint> t_centerTrajectorySmoothed;
    bool bWayGlobalPath;
    bool bWayGlobalPathToUse;
    std::vector<std::vector<PlannerHNS::WayPoint>> m_GeneratedRollOuts;
    bool bRollOuts;

    std::vector<PlannerHNS::DetectedObject> m_PredictedObjects;
    bool bPredictedObjects;
    // Added by PHY
    double m_SprintDecisionTime;
    int m_pedestrian_stop_img_height_threshold;

    struct timespec m_PlanningTimer;
    std::vector<std::string> m_LogData;
    PlannerHNS::PlanningParams m_PlanningParams;
    PlannerHNS::CAR_BASIC_INFO m_CarInfo;
    PlannerHNS::BehaviorState m_CurrentBehavior;
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
    std::string m_LaneTopic;
    std_msgs::Bool m_sprint_switch;
    std_msgs::Float64 m_distance_to_pedestrian;

    // ROS messages (topics)
    ros::NodeHandle nh;

    // define publishers
    ros::Publisher pub_CollisionPointsRviz;
    ros::Publisher pub_LocalWeightedTrajectoriesRviz;
    ros::Publisher pub_LocalWeightedTrajectories;
    // ros::Publisher pub_LocalWeightedTrajectoriesWithPoseTwist;
    // ros::Publisher pub_TrajectoryCost;
    ros::Publisher pub_SafetyBorderRviz;
    // ros::Publisher pub_DistanceToPedestrian;
    // ros::Publisher pub_IntersectionCondition;
    // ros::Publisher pub_SprintSwitch;
    ros::Publisher pub_currentTraj;
    ros::Publisher pub_PlanningInfo;

    // define subscribers.
    ros::Subscriber sub_current_pose;
    ros::Subscriber sub_current_velocity;
    ros::Subscriber sub_robot_odom;
    ros::Subscriber sub_can_info;
    ros::Subscriber sub_GlobalPlannerPaths;
    ros::Subscriber sub_LocalPlannerPaths;
    ros::Subscriber sub_predicted_objects;
    ros::Subscriber sub_rubis_predicted_objects;

    message_filters::Subscriber<rubis_msgs::LaneArrayWithPoseTwist> trajectories_sub_;
    message_filters::Subscriber<rubis_msgs::DetectedObjectArray> objects_sub_;
    message_filters::Subscriber<rubis_msgs::Bool> lane_sub_;
    boost::shared_ptr<Sync> sync_;
    boost::shared_ptr<LaneSync> lane_sync_;

    // HJW added
    ros::Subscriber sub_current_state;

    // TF
    tf::TransformListener m_vtob_listener;
    tf::TransformListener m_vtom_listener;
    tf::StampedTransform m_velodyne_to_base_link;
    tf::StampedTransform m_velodyne_to_map;

    // Others
    std::vector<PlannerHNS::Crossing> intersection_list_;
    autoware_msgs::DetectedObjectArray object_msg_;

    // Callback function for subscriber.
    void callbackGetGlobalPlannerPath(const autoware_msgs::LaneArrayConstPtr &msg);
    void _callbackGetLocalPlannerPath(const rubis_msgs::LaneArrayWithPoseTwistConstPtr &msg);
    void callbackGetPredictedObjects(const rubis_msgs::DetectedObjectArrayConstPtr &msg);
    void _callbackGetPredictedObjects(const autoware_msgs::DetectedObjectArray &objects);
    void callbackGetRubisPredictedObjects(const rubis_msgs::DetectedObjectArrayConstPtr &msg);
    void callbackGetBehaviorState(const geometry_msgs::TwistStampedConstPtr &msg);
    void callbackGetCurrentState(const std_msgs::Int32 &msg);
    void callback(const rubis_msgs::LaneArrayWithPoseTwist::ConstPtr &trajectories_msg, const rubis_msgs::DetectedObjectArray::ConstPtr &objects_msg);
    void callbackWithLane(const rubis_msgs::LaneArrayWithPoseTwist::ConstPtr &trajectories_msg,
                          const rubis_msgs::DetectedObjectArray::ConstPtr &objects_msg, const rubis_msgs::Bool::ConstPtr &lane_msg);

    // Helper Functions
    void UpdatePlanningParams(ros::NodeHandle &_nh);

    void UpdateMyParams();
    bool UpdateTf();

  public:
    TrajectoryEval();
    ~TrajectoryEval();
    void MainLoop();
};

} // namespace TrajectoryEvaluatorNS

#endif // OP_TRAJECTORY_EVALUATOR_CORE
