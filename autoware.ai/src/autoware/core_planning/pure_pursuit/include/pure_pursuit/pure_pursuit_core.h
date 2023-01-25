/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
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

#define SVL
// #define IONIC
// #define DEBUG

#ifndef PURE_PURSUIT_PURE_PURSUIT_CORE_H
#define PURE_PURSUIT_PURE_PURSUIT_CORE_H

// ROS includes
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Point.h>
#include <rubis_msgs/PoseStamped.h>
#include <rubis_msgs/TwistStamped.h>

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/Marker.h>

// User defined includes
#include <autoware_config_msgs/ConfigWaypointFollower.h>
#include <autoware_msgs/ControlCommandStamped.h>
#include <autoware_msgs/Lane.h>
#include <pure_pursuit/pure_pursuit.h>
#include <pure_pursuit/pure_pursuit_viz.h>

#include <vector>
#include <memory>

#include <XmlRpcException.h>

#include "rubis_msgs/LaneWithPoseTwist.h"

#ifdef IONIC
#include <can_data_msgs/Car_ctrl_output.h>
#endif

namespace waypoint_follower
{
enum class Mode : int32_t
{
  waypoint,
  dialog,

  unknown = -1,
};

template <class T>
typename std::underlying_type<T>::type enumToInteger(T t)
{
  return static_cast<typename std::underlying_type<T>::type>(t);
}

class DynamicParams
{
public:
  DynamicParams();

  double min_vel;
  double max_vel;
  double lookahead_ratio;
  double lookahead_distance;
};

class PurePursuitNode
{
public:
  PurePursuitNode();
  ~PurePursuitNode();

  void run();
  friend class PurePursuitNodeTestSuite;

private:
  // handle
  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;

  // class
  PurePursuit pp_;

  // publisher
  ros::Publisher twist_pub_, rubis_twist_pub_, pub2_,
    pub11_, pub12_, pub13_, pub14_, pub15_, pub16_, pub17_, pub18_;

  // subscriber
  ros::Subscriber sub1_, sub3_, final_waypoints_with_pose_twist_sub, car_ctrl_output_sub;

  // constant
  const int LOOP_RATE_;  // processing frequency

  // variables
  bool is_linear_interpolation_, publishes_for_steering_robot_,
    add_virtual_end_waypoints_;
  bool is_waypoint_set_, is_pose_set_, is_velocity_set_;
  double current_linear_velocity_, command_linear_velocity_;
  double wheel_base_;
  int expand_size_;
  LaneDirection direction_;
  int32_t velocity_source_;          // 0 = waypoint, 1 = Dialog
  double const_lookahead_distance_;  // meter
  double const_velocity_;            // km/h
  double lookahead_distance_ratio_;
  // the next waypoint must be outside of this threshold.
  double minimum_lookahead_distance_;

  // HJW added
  double angle_diff_;
  double lookahead_distance_ratio_from_param;
  double minimum_lookahead_distance_from_param;

  // Added by PHY
  bool dynamic_param_flag_;
  int task_profiling_flag_;
  std::vector<DynamicParams> dynamic_params;

  // callbacks
  void callbackFromConfig(
    const autoware_config_msgs::ConfigWaypointFollowerConstPtr& config);
  void callbackFromWayPoints(const autoware_msgs::LaneConstPtr& msg);
  void CallbackFinalWaypointsWithPoseTwist(const rubis_msgs::LaneWithPoseTwistConstPtr& msg);

  #ifdef IONIC
  void callbackCtrlOutput(const can_data_msgs::Car_ctrl_output::ConstPtr &msg);
  #endif

  bool use_algorithm_;
  std::vector<double> way_points_velocity_;
  std::vector<double> way_points_x_, way_points_y_;

  double findWayPointVelocity(autoware_msgs::Waypoint msg);

  // initializer
  void initForROS();

  // functions
  void publishTwistStamped(
    const bool& can_get_curvature, const double& kappa) const;
  void publishControlCommandStamped(
    const bool& can_get_curvature, const double& kappa) const;
  void publishDeviationCurrentPosition(
    const geometry_msgs::Point& point,
    const std::vector<autoware_msgs::Waypoint>& waypoints) const;
  void connectVirtualLastWaypoints(
    autoware_msgs::Lane* expand_lane, LaneDirection direction);
  inline void updateCurrentPose(geometry_msgs::PoseStampedConstPtr& msg);
  
  // Added by PHY
  void setLookaheadParamsByVel();


  int getSgn() const;
  double computeLookaheadDistance() const;
  double computeCommandVelocity() const;
  double computeCommandAccel() const;
  double computeAngularGravity(double velocity, double kappa) const;
};

double convertCurvatureToSteeringAngle(
  const double& wheel_base, const double& kappa);

inline double kmph2mps(double velocity_kmph)
{
  return (velocity_kmph * 1000) / (60 * 60);
}

}  // namespace waypoint_follower

#ifdef USE_WAYPOINT_ORIENTATION
static geometry_msgs::PoseStamped waypoint_pose_;
#endif

#endif  // PURE_PURSUIT_PURE_PURSUIT_CORE_H
