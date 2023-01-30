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

#include <vector>
#include <pure_pursuit/pure_pursuit_core.h>
#include <rubis_lib/sched.hpp>

namespace waypoint_follower
{
DynamicParams::DynamicParams(){
  min_vel = -1;
  max_vel = -1;
  lookahead_ratio = -1;
  lookahead_distance = -1;
}

// Constructor
PurePursuitNode::PurePursuitNode()
  : private_nh_("~")
  , pp_()
  , LOOP_RATE_(30)
  , is_waypoint_set_(false)
  , is_pose_set_(false)
  , is_velocity_set_(false)
  , current_linear_velocity_(0)
  , command_linear_velocity_(0)
  , direction_(LaneDirection::Forward)
  , velocity_source_(-1)
  , const_lookahead_distance_(4.0)
  , const_velocity_(5.0)
  , lookahead_distance_ratio_(2.0)
  , minimum_lookahead_distance_(6.0)
{
  initForROS();
  // initialize for PurePursuit
  pp_.setLinearInterpolationParameter(is_linear_interpolation_);
}

// Destructor
PurePursuitNode::~PurePursuitNode()
{
}

void PurePursuitNode::initForROS()
{
  // way point velocity setting 
  private_nh_.param("/vel_setting/use_algorithm", use_algorithm_, false);

  if(use_algorithm_){
    std::vector<int> curve_line_start, straight_line_start, curve_line_end, straight_line_end;
    double straight_velocity, curve_velocity;
    double prev_velocity, next_velocity;
    int prev_point, next_point;
    bool is_vel_set;

    private_nh_.param("/vel_setting/straight_velocity", straight_velocity, 5.0);
    private_nh_.param("/vel_setting/curve_velocity", curve_velocity, 3.0);

    private_nh_.getParam("/vel_setting/straight_line_start", straight_line_start);
    private_nh_.getParam("/vel_setting/straight_line_end", straight_line_end);

    private_nh_.getParam("/vel_setting/curve_line_start", curve_line_start);
    private_nh_.getParam("/vel_setting/curve_line_end", curve_line_end);
  
    private_nh_.getParam("/vel_setting/way_points_x", way_points_x_);
    private_nh_.getParam("/vel_setting/way_points_y", way_points_y_);

    for(int i = 1; i <= way_points_x_.size(); i++){
      is_vel_set = false;
      prev_point = 0;
      next_point = way_points_x_.size();

      for(int j = 0; j < straight_line_start.size(); j++){
        if ((i >= straight_line_start[j]) && (i <= straight_line_end[j])){
          is_vel_set = true;
          way_points_velocity_.push_back(straight_velocity);
        }
        else if(i < straight_line_start[j]){
          if(next_point > straight_line_start[j]){
            next_point = straight_line_start[j];
            next_velocity = straight_velocity;
          }
        }
        else if(i > straight_line_end[j]){
          if(prev_point < straight_line_end[j]){
            prev_point = straight_line_end[j];
            prev_velocity = straight_velocity;
          }
        }
      }

      for(int j = 0; j < curve_line_start.size(); j++){
        if ((i >= curve_line_start[j]) && (i <= curve_line_end[j])){
          is_vel_set = true;
          way_points_velocity_.push_back(curve_velocity);
        }
        else if(i < curve_line_start[j]){
          if(next_point > curve_line_start[j]){
            next_point = curve_line_start[j];
            next_velocity = curve_velocity;
          }
        }
        else if(i > curve_line_end[j]){
          if(prev_point < curve_line_end[j]){
            prev_point = curve_line_end[j];
            prev_velocity = curve_velocity;
          }
        }
      }

      if(!is_vel_set){
        way_points_velocity_.push_back((prev_velocity * (next_point - i) + next_velocity * (i - prev_point)) / (next_point - prev_point));
      }
    }
  }
  
  // ros parameter settings
  private_nh_.param("velocity_source", velocity_source_, 0);
  private_nh_.param("is_linear_interpolation", is_linear_interpolation_, true);
  private_nh_.param(
    "publishes_for_steering_robot", publishes_for_steering_robot_, false);
  private_nh_.param(
    "add_virtual_end_waypoints", add_virtual_end_waypoints_, false);
  private_nh_.param("const_lookahead_distance", const_lookahead_distance_, 4.0);
  private_nh_.param("const_velocity", const_velocity_, 5.0);
  private_nh_.param("lookahead_ratio", lookahead_distance_ratio_, 2.0);
  private_nh_.param(
    "minimum_lookahead_distance", minimum_lookahead_distance_, 6.0);
  nh_.param("vehicle_info/wheel_base", wheel_base_, 2.7);

  private_nh_.param("/pure_pursuit/dynamic_params_flag", dynamic_param_flag_, false);
  private_nh_.param("/pure_pursuit/instance_mode", rubis::instance_mode_, 0);
  
  if(dynamic_param_flag_){
    XmlRpc::XmlRpcValue xml_list;
    if(!nh_.getParam("/pure_pursuit/dynamic_params", xml_list)){
      ROS_ERROR("[pure_pursuit] Cannot load dynamic params");
      exit(1);
    }
    std::cout<<"Parameter is loaded / "<<xml_list.size()<<std::endl;
    for(int i=0; i<xml_list.size(); i++){
      XmlRpc::XmlRpcValue xml_param = xml_list[i];
      
      DynamicParams param;
      
      param.min_vel = (double)(xml_param[0]);
      param.max_vel = (double)(xml_param[1]);
      param.lookahead_ratio = (double)(xml_param[2]);
      param.lookahead_distance = (double)(xml_param[3]);
      dynamic_params.push_back(param);
    }
  
  }


  // setup subscriber
  pose_twist_sub_ = nh_.subscribe("/rubis_current_pose_twist", 1, &PurePursuitNode::CallbackTwistPose, this);
  final_waypoints_with_pose_twist_sub = nh_.subscribe("/final_waypoints_with_pose_twist", 10, &PurePursuitNode::CallbackFinalWaypointsWithPoseTwist, this);

  sub3_ = nh_.subscribe("config/waypoint_follower", 10,
    &PurePursuitNode::callbackFromConfig, this);
  
  // setup publisher
  twist_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("twist_raw", 10);
  if(rubis::instance_mode_) rubis_twist_pub_ = nh_.advertise<rubis_msgs::TwistStamped>("rubis_twist_raw", 10);

  pub2_ = nh_.advertise<autoware_msgs::ControlCommandStamped>("ctrl_raw", 10);
  pub11_ = nh_.advertise<visualization_msgs::Marker>("next_waypoint_mark", 0);
  pub12_ = nh_.advertise<visualization_msgs::Marker>("next_target_mark", 0);
  pub13_ = nh_.advertise<visualization_msgs::Marker>("search_circle_mark", 0);
  // debug tool
  pub14_ = nh_.advertise<visualization_msgs::Marker>("line_point_mark", 0);
  pub15_ =
    nh_.advertise<visualization_msgs::Marker>("trajectory_circle_mark", 0);
  pub16_ = nh_.advertise<std_msgs::Float32>("angular_gravity", 0);
  pub17_ = nh_.advertise<std_msgs::Float32>("deviation_of_current_position", 0);
  pub18_ =
    nh_.advertise<visualization_msgs::Marker>("expanded_waypoints_mark", 0);
  // pub7_ = nh.advertise<std_msgs::Bool>("wf_stat", 0);

}

void PurePursuitNode::run()
{
  ros::NodeHandle private_nh("~");

  // Scheduling Setup
  int task_scheduling_flag;
  std::string task_response_time_filename;
  int rate;
  double task_minimum_inter_release_time;
  double task_execution_time;
  double task_relative_deadline;

  private_nh.param<int>("/pure_pursuit/task_scheduling_flag", task_scheduling_flag, 0);
  private_nh.param<int>("/pure_pursuit/task_profiling_flag", task_profiling_flag_, 1);
  private_nh.param<std::string>("/pure_pursuit/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/pure_pursuit.csv");
  private_nh.param<int>("/pure_pursuit/rate", rate, 10);
  private_nh.param("/pure_pursuit/task_minimum_inter_release_time", task_minimum_inter_release_time, (double)10);
  private_nh.param("/pure_pursuit/task_execution_time", task_execution_time, (double)10);
  private_nh.param("/pure_pursuit/task_relative_deadline", task_relative_deadline, (double)10);
  if(task_profiling_flag_) rubis::sched::init_task_profiling(task_response_time_filename);  

  ros::spin();
}


void PurePursuitNode::publishTwistStamped(
  const bool& can_get_curvature, const double& kappa) const
{
  geometry_msgs::TwistStamped ts;
  ts.header.stamp = ros::Time::now();
  ts.twist.linear.x = can_get_curvature ? computeCommandVelocity() : 0;
  ts.twist.angular.z = can_get_curvature ? kappa * ts.twist.linear.x : 0;
  twist_pub_.publish(ts);

  if(rubis::instance_mode_ && rubis::instance_mode_ != RUBIS_NO_INSTANCE){
    rubis_msgs::TwistStamped rubis_ts;
    rubis_ts.instance = rubis::instance_;
    rubis_ts.obj_instance = rubis::obj_instance_;
    rubis_ts.msg = ts;
    rubis_twist_pub_.publish(rubis_ts);
  }

  if(rubis::sched::is_task_ready_ == TASK_NOT_READY) rubis::sched::init_task();
  rubis::sched::task_state_ = TASK_STATE_DONE;
}

void PurePursuitNode::publishControlCommandStamped(
  const bool& can_get_curvature, const double& kappa) const
{
  if (!publishes_for_steering_robot_)
  {
    return;
  }

  autoware_msgs::ControlCommandStamped ccs;
  ccs.header.stamp = ros::Time::now();
  ccs.cmd.linear_velocity = can_get_curvature ? computeCommandVelocity() : 0;
  ccs.cmd.linear_acceleration = can_get_curvature ? computeCommandAccel() : 0;
  ccs.cmd.steering_angle =
    can_get_curvature ? convertCurvatureToSteeringAngle(wheel_base_, kappa) : 0;

  pub2_.publish(ccs);
}

double PurePursuitNode::computeLookaheadDistance() const
{
  if (velocity_source_ == enumToInteger(Mode::dialog))
  {
    return const_lookahead_distance_;
  }

  // std::cout << "Curve : " << angle_diff_ << std::endl;
  // if(angle_diff_ < 5) // if curvature is too low, set lookahead distance to 25 statically.
  //   return 25;
  
  double ld = current_linear_velocity_ * lookahead_distance_ratio_;
  if(ld < minimum_lookahead_distance_)
    return minimum_lookahead_distance_;
  
  return ld;


  // Original Autoware Code
  // double maximum_lookahead_distance = current_linear_velocity_ * 10;
  // double ld = current_linear_velocity_ * lookahead_distance_ratio_;

  // if(ld < minimum_lookahead_distance_)
  //   return minimum_lookahead_distance_;
  // else if(ld > maximum_lookahead_distance)
  //   return maximum_lookahead_distance;
  // else
  //   return ld;
}

int PurePursuitNode::getSgn() const
{
  int sgn = 0;
  if (direction_ == LaneDirection::Forward)
  {
    sgn = 1;
  }
  else if (direction_ == LaneDirection::Backward)
  {
    sgn = -1;
  }
  return sgn;
}

double PurePursuitNode::computeCommandVelocity() const
{
  if (velocity_source_ == enumToInteger(Mode::dialog))
  {
    return getSgn() * kmph2mps(const_velocity_);
  }

  return command_linear_velocity_;
}

double PurePursuitNode::computeCommandAccel() const
{
  const geometry_msgs::Pose current_pose = pp_.getCurrentPose();
  const geometry_msgs::Pose target_pose =
    pp_.getCurrentWaypoints().at(1).pose.pose;

  // v^2 - v0^2 = 2ax
  const double x =
      std::hypot(current_pose.position.x - target_pose.position.x,
        current_pose.position.y - target_pose.position.y);
  const double v0 = current_linear_velocity_;
  const double v = computeCommandVelocity();
  const double a = getSgn() * (v * v - v0 * v0) / (2 * x);
  return a;
}

double PurePursuitNode::computeAngularGravity(
  double velocity, double kappa) const
{
  const double gravity = 9.80665;
  return (velocity * velocity) / (1.0 / kappa * gravity);
}

void PurePursuitNode::callbackFromConfig(
  const autoware_config_msgs::ConfigWaypointFollowerConstPtr& config)
{
  velocity_source_ = config->param_flag;
  const_lookahead_distance_ = config->lookahead_distance;
  const_velocity_ = config->velocity;
  lookahead_distance_ratio_ = config->lookahead_ratio;
  minimum_lookahead_distance_ = config->minimum_lookahead_distance;
}

void PurePursuitNode::publishDeviationCurrentPosition(
  const geometry_msgs::Point& point,
  const std::vector<autoware_msgs::Waypoint>& waypoints) const
{
  // Calculate the deviation of current position
  // from the waypoint approximate line

  if (waypoints.size() < 3)
  {
    return;
  }

  double a, b, c;
  getLinearEquation(
    waypoints.at(2).pose.pose.position, waypoints.at(1).pose.pose.position,
    &a, &b, &c);

  std_msgs::Float32 msg;
  msg.data = getDistanceBetweenLineAndPoint(point, a, b, c);

  pub17_.publish(msg);
}

inline void PurePursuitNode::updateCurrentPose(geometry_msgs::PoseStampedConstPtr& msg){
#ifndef USE_WAYPOINT_ORIENTATION
  pp_.setCurrentPose(msg);
#else
  geometry_msgs::PoseStamped updated_msg;
  updated_msg = *msg;
  updated_msg.pose.orientation = waypoint_pose_.pose.orientation;

  pp_.setCurrentPose(updated_msg);
#endif
  is_pose_set_ = true;
}

#ifdef IONIC
void PurePursuitNode::callbackCtrlOutput(const can_data_msgs::Car_ctrl_output::ConstPtr &msg)
{
  current_linear_velocity_ = kmph2mps(msg->real_speed);
  pp_.setCurrentVelocity(current_linear_velocity_);
  is_velocity_set_ = true;
}
#endif

void PurePursuitNode::setLookaheadParamsByVel(){
  for(auto it=dynamic_params.begin(); it != dynamic_params.end(); ++it){
    DynamicParams param = *it;
    if(current_linear_velocity_>param.min_vel && current_linear_velocity_ <= param.max_vel){
      lookahead_distance_ratio_ = param.lookahead_ratio;
      minimum_lookahead_distance_ = param.lookahead_distance;
      break;
    }
  }

  // std::cout<<"Waypoint Vel:"<<command_linear_velocity_<<"/ ratio"<<lookahead_distance_ratio_<<"/ disdtance"<<minimum_lookahead_distance_<<std::endl;
}

double PurePursuitNode::findWayPointVelocity(autoware_msgs::Waypoint msg){
  int len, idx = 0;
  
  double x, y;
  double minDist = 9999.0, tmp_dist;

  x = msg.pose.pose.position.x;
  y = msg.pose.pose.position.y;

  len = way_points_x_.size();
  for(int i = 0; i < len; i++){
    tmp_dist = pow(way_points_x_[i] - x, 2.0) + pow(way_points_y_[i] - y, 2.0);
    if(tmp_dist < minDist){
      minDist = tmp_dist;
      idx = i;
    }
  }

  return way_points_velocity_[idx];
}

void PurePursuitNode::CallbackTwistPose(const rubis_msgs::PoseTwistStampedConstPtr& msg)
{
  if(task_profiling_flag_) rubis::sched::start_task_profiling();
  rubis::instance_ = msg->instance;

  // Update pose
  geometry_msgs::PoseStampedConstPtr pose_ptr(new geometry_msgs::PoseStamped(msg->pose));
  updateCurrentPose(pose_ptr);

  // Update twist
  current_linear_velocity_ = msg->twist.twist.linear.x;
  pp_.setCurrentVelocity(current_linear_velocity_);
  is_velocity_set_ = true;

  if(use_algorithm_){
    command_linear_velocity_ = findWayPointVelocity(lane_.waypoints.at(0));
  }
  else{
    command_linear_velocity_ = (!lane_.waypoints.empty()) ? lane_.waypoints.at(0).twist.twist.linear.x : 0;
  }

  geometry_msgs::Point curr_point = lane_.waypoints.at(0).pose.pose.position;
  geometry_msgs::Point near_point = lane_.waypoints.at(std::min(3, (int)lane_.waypoints.size() - 1)).pose.pose.position;
  geometry_msgs::Point far_point = lane_.waypoints.at(std::min(30, (int)lane_.waypoints.size() - 1)).pose.pose.position;

  double deg_1 = atan2((near_point.y - curr_point.y), (near_point.x - curr_point.x)) / 3.14 * 180;
  double deg_2 = atan2((far_point.y - curr_point.y), (far_point.x - curr_point.x)) / 3.14 * 180;
  double angle_diff = std::abs(deg_1 - deg_2);
  if (angle_diff > 180){
    angle_diff = 360 - angle_diff;
  }

  angle_diff_ = angle_diff;

  // Update waypoints
  _CallbackFinalWaypointsWithPoseTwist();

  // After spinOnce
  pp_.setLookaheadDistance(computeLookaheadDistance());
  pp_.setMinimumLookaheadDistance(minimum_lookahead_distance_);

  double kappa = 0;
  bool can_get_curvature = pp_.canGetCurvature(&kappa);

  publishTwistStamped(can_get_curvature, kappa);
  publishControlCommandStamped(can_get_curvature, kappa);
  // for visualization with Rviz
  pub11_.publish(displayNextWaypoint(pp_.getPoseOfNextWaypoint()));
  pub13_.publish(displaySearchRadius(
    pp_.getCurrentPose().position, pp_.getLookaheadDistance()));
  pub12_.publish(displayNextTarget(pp_.getPoseOfNextTarget()));
  pub15_.publish(displayTrajectoryCircle(
      waypoint_follower::generateTrajectoryCircle(
        pp_.getPoseOfNextTarget(), pp_.getCurrentPose())));
  if (add_virtual_end_waypoints_)
  {
    pub18_.publish(
      displayExpandWaypoints(pp_.getCurrentWaypoints(), expand_size_));
  }
  std_msgs::Float32 angular_gravity_msg;
  angular_gravity_msg.data =
    computeAngularGravity(computeCommandVelocity(), kappa);
  pub16_.publish(angular_gravity_msg);

  publishDeviationCurrentPosition(
    pp_.getCurrentPose().position, pp_.getCurrentWaypoints());

  is_pose_set_ = false;
  is_velocity_set_ = false;
  is_waypoint_set_ = false;

  if(task_profiling_flag_) rubis::sched::stop_task_profiling(rubis::instance_, rubis::sched::task_state_);

}

void PurePursuitNode::_CallbackFinalWaypointsWithPoseTwist()
{
  if(dynamic_param_flag_){
    setLookaheadParamsByVel();
  }
  
  if (add_virtual_end_waypoints_)
  {
    const LaneDirection solved_dir = getLaneDirection(lane_);
    direction_ = (solved_dir != LaneDirection::Error) ? solved_dir : direction_;
    autoware_msgs::Lane expanded_lane(lane_);
    expand_size_ = -expanded_lane.waypoints.size();
    connectVirtualLastWaypoints(&expanded_lane, direction_);
    expand_size_ += expanded_lane.waypoints.size();

    pp_.setCurrentWaypoints(expanded_lane.waypoints);
  }
  else
  {
    pp_.setCurrentWaypoints(lane_.waypoints);
  }
  is_waypoint_set_ = true;

#ifdef USE_WAYPOINT_ORIENTATION
  waypoint_pose_ = lane_.waypoints[0].pose;
#endif
}

void PurePursuitNode::CallbackFinalWaypointsWithPoseTwist(const rubis_msgs::LaneWithPoseTwistConstPtr& msg)
{   
  rubis::obj_instance_ = msg->obj_instance;
  lane_ = msg->lane;
}

void PurePursuitNode::connectVirtualLastWaypoints(
  autoware_msgs::Lane* lane, LaneDirection direction)
{
  if (lane->waypoints.empty())
  {
    return;
  }
  static double interval = 1.0;
  const geometry_msgs::Pose& pn = lane->waypoints.back().pose.pose;
  autoware_msgs::Waypoint virtual_last_waypoint;
  virtual_last_waypoint.pose.pose.orientation = pn.orientation;
  virtual_last_waypoint.twist.twist.linear.x = 0.0;
  geometry_msgs::Point virtual_last_point_rlt;
  const int sgn = getSgn();
  for (double dist = minimum_lookahead_distance_; dist > 0.0; dist -= interval)
  {
    virtual_last_point_rlt.x += interval * sgn;
    virtual_last_waypoint.pose.pose.position =
      calcAbsoluteCoordinate(virtual_last_point_rlt, pn);
    lane->waypoints.emplace_back(virtual_last_waypoint);
  }
}

double convertCurvatureToSteeringAngle(
  const double& wheel_base, const double& kappa)
{
  return atan(wheel_base * kappa);
}

}  // namespace waypoint_follower
