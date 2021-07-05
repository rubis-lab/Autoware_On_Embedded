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
#include <rubis_sched/sched.hpp>

int scheduling_flag_;
int profiling_flag_;
std::string response_time_filename_;
int rate_;
double minimum_inter_release_time_;
double execution_time_;
double relative_deadline_;

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
  health_checker_ptr_ =
    std::make_shared<autoware_health_checker::HealthChecker>(nh_, private_nh_);
  health_checker_ptr_->ENABLE();
  // initialize for PurePursuit
  pp_.setLinearInterpolationParameter(is_linear_interpolation_);
}

// Destructor
PurePursuitNode::~PurePursuitNode()
{
}

void PurePursuitNode::initForROS()
{
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
  sub1_ = nh_.subscribe("final_waypoints", 10,
    &PurePursuitNode::callbackFromWayPoints, this);
  sub2_ = nh_.subscribe("current_pose", 10,
    &PurePursuitNode::callbackFromCurrentPose, this);
  sub3_ = nh_.subscribe("config/waypoint_follower", 10,
    &PurePursuitNode::callbackFromConfig, this);
  sub4_ = nh_.subscribe("current_velocity", 10,
    &PurePursuitNode::callbackFromCurrentVelocity, this);

  // setup publisher
  pub1_ = nh_.advertise<geometry_msgs::TwistStamped>("twist_raw", 10);
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

  private_nh.param<int>("/pure_pursuit/scheduling_flag", scheduling_flag_, 0);
  private_nh.param<int>("/pure_pursuit/profiling_flag", profiling_flag_, 0);
  private_nh.param<std::string>("/pure_pursuit/response_time_filename", response_time_filename_, "/home/hypark/Documents/profiling/response_time/pure_pursuit.csv");
  private_nh.param<int>("/pure_pursuit/rate", rate_, 10);
  private_nh.param("/pure_pursuit/minimum_inter_release_time", minimum_inter_release_time_, (double)10);
  private_nh.param("/pure_pursuit/execution_time", execution_time_, (double)10);
  private_nh.param("/pure_pursuit/relative_deadline", relative_deadline_, (double)10);

  ROS_INFO_STREAM("pure pursuit start");

  // FILE *fp;
  // if(profiling_flag_){      
  //   fp = fopen(response_time_filename_.c_str(), "a");
  // }

  ros::Rate loop_rate(LOOP_RATE_);
  // if(scheduling_flag_) loop_rate = ros::Rate(rate_);
  struct timespec start_time, end_time;

  while (ros::ok())
  {
    // if(profiling_flag_){        
    //   clock_gettime(CLOCK_MONOTONIC, &start_time);
    // }
    // if(scheduling_flag_){
    //   rubis::sched::set_sched_deadline(gettid(), 
    //     static_cast<uint64_t>(execution_time_), 
    //     static_cast<uint64_t>(relative_deadline_), 
    //     static_cast<uint64_t>(minimum_inter_release_time_)
    //   );
    // }      

    ros::spinOnce();
    if (!is_pose_set_ || !is_waypoint_set_ || !is_velocity_set_)
    {
      // ROS_WARN("Necessary topics are not subscribed yet ... ");
      
      // if(profiling_flag_){
      //   clock_gettime(CLOCK_MONOTONIC, &end_time);
      //   fprintf(fp, "%lld.%.9ld,%lld.%.9ld,%d\n",start_time.tv_sec,start_time.tv_nsec,end_time.tv_sec,end_time.tv_nsec,getpid());    
      //   fflush(fp);
      // }

      loop_rate.sleep();
      continue;
    }

    pp_.setLookaheadDistance(computeLookaheadDistance());
    pp_.setMinimumLookaheadDistance(minimum_lookahead_distance_);

    double kappa = 0;
    bool can_get_curvature = pp_.canGetCurvature(&kappa);

    publishTwistStamped(can_get_curvature, kappa);
    publishControlCommandStamped(can_get_curvature, kappa);
    health_checker_ptr_->NODE_ACTIVATE();
    health_checker_ptr_->CHECK_RATE("topic_rate_vehicle_cmd_slow", 8, 5, 1,
      "topic vehicle_cmd publish rate slow.");
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

    // if(profiling_flag_){
    //   clock_gettime(CLOCK_MONOTONIC, &end_time);
    //   fprintf(fp, "%lld.%.9ld,%lld.%.9ld,%d\n",start_time.tv_sec,start_time.tv_nsec,end_time.tv_sec,end_time.tv_nsec,getpid());    
    //   fflush(fp);
    // }

    loop_rate.sleep();
  }
  // fclose(fp);
}

void PurePursuitNode::publishTwistStamped(
  const bool& can_get_curvature, const double& kappa) const
{
  geometry_msgs::TwistStamped ts;
  ts.header.stamp = ros::Time::now();
  ts.twist.linear.x = can_get_curvature ? computeCommandVelocity() : 0;
  ts.twist.angular.z = can_get_curvature ? kappa * ts.twist.linear.x : 0;
  pub1_.publish(ts);
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

  double maximum_lookahead_distance = current_linear_velocity_ * 10;
  double ld = current_linear_velocity_ * lookahead_distance_ratio_;

  return ld < minimum_lookahead_distance_ ? minimum_lookahead_distance_ :
    ld > maximum_lookahead_distance ? maximum_lookahead_distance : ld;
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

void PurePursuitNode::callbackFromCurrentPose(
  const geometry_msgs::PoseStampedConstPtr& msg)
{
  pp_.setCurrentPose(msg);
  is_pose_set_ = true;
}

void PurePursuitNode::callbackFromCurrentVelocity(
  const geometry_msgs::TwistStampedConstPtr& msg)
{
  current_linear_velocity_ = msg->twist.linear.x;
  pp_.setCurrentVelocity(current_linear_velocity_);
  is_velocity_set_ = true;
}

void PurePursuitNode::setLookaheadParamsByVel(){
  for(auto it=dynamic_params.begin(); it != dynamic_params.end(); ++it){
    DynamicParams param = *it;
    if(command_linear_velocity_>param.min_vel && command_linear_velocity_ <= param.max_vel){
      lookahead_distance_ratio_ = param.lookahead_ratio;
      minimum_lookahead_distance_ = param.lookahead_distance;
      break;
    }
  }

  // std::cout<<"Waypoint Vel:"<<command_linear_velocity_<<"/ ratio"<<lookahead_distance_ratio_<<"/ disdtance"<<minimum_lookahead_distance_<<std::endl;
}

void PurePursuitNode::callbackFromWayPoints(
  const autoware_msgs::LaneConstPtr& msg)
{
  command_linear_velocity_ =
    (!msg->waypoints.empty()) ? msg->waypoints.at(0).twist.twist.linear.x : 0;
  setLookaheadParamsByVel();
  
  if (add_virtual_end_waypoints_)
  {
    const LaneDirection solved_dir = getLaneDirection(*msg);
    direction_ = (solved_dir != LaneDirection::Error) ? solved_dir : direction_;
    autoware_msgs::Lane expanded_lane(*msg);
    expand_size_ = -expanded_lane.waypoints.size();
    connectVirtualLastWaypoints(&expanded_lane, direction_);
    expand_size_ += expanded_lane.waypoints.size();

    pp_.setCurrentWaypoints(expanded_lane.waypoints);
  }
  else
  {
    pp_.setCurrentWaypoints(msg->waypoints);
  }
  is_waypoint_set_ = true;
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
