/*
 * Copyright 2015-2018 Autoware Foundation. All rights reserved.
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

#include "can_status_translator_core.h"
int is_topic_ready = 0;

namespace autoware_connector
{

// Constructor
CanStatusTranslatorNode::CanStatusTranslatorNode() : private_nh_("~"), v_info_()
{
  initForROS();
}

// Destructor
CanStatusTranslatorNode::~CanStatusTranslatorNode()
{
}

void CanStatusTranslatorNode::initForROS()
{
  // ros parameter settings
  if (!nh_.hasParam("/vehicle_info/wheel_base") || !nh_.hasParam("/vehicle_info/minimum_turning_radius") ||
      !nh_.hasParam("/vehicle_info/maximum_steering_wheel_angle_deg"))
  {
    v_info_.is_stored = false;
    ROS_INFO("vehicle_info is not set");
  }
  else
  {
    private_nh_.getParam("/vehicle_info/wheel_base", v_info_.wheel_base);
    // ROS_INFO_STREAM("wheel_base : " << wheel_base);

    private_nh_.getParam("/vehicle_info/minimum_turning_radius", v_info_.minimum_turning_radius);
    // ROS_INFO_STREAM("minimum_turning_radius : " << minimum_turning_radius);

    private_nh_.getParam("/vehicle_info/maximum_steering_wheel_angle_deg",
                         v_info_.maximum_steering_wheel_angle_deg);  //[degree]:
    // ROS_INFO_STREAM("maximum_steering_wheel_angle_deg : " << maximum_steering_wheel_angle_deg);

    v_info_.is_stored = true;
  }
  // setup subscriber
  sub1_ = nh_.subscribe("can_info", 100, &CanStatusTranslatorNode::callbackFromCANInfo, this);
  sub2_ = nh_.subscribe("vehicle_status", 10, &CanStatusTranslatorNode::callbackFromVehicleStatus, this);

  // setup publisher
  pub1_ = nh_.advertise<geometry_msgs::TwistStamped>("can_velocity", 10);
  pub2_ = nh_.advertise<std_msgs::Float32>("linear_velocity_viz", 10);
  pub3_ = nh_.advertise<autoware_msgs::VehicleStatus>("vehicle_status", 10);
}

void CanStatusTranslatorNode::run()
{
  // scheduling
  int task_scheduling_flag;
  int task_profiling_flag;
  std::string task_response_time_filename;
  int rate;
  double task_minimum_inter_release_time;
  double task_execution_time;
  double task_relative_deadline;

  private_nh_.param<int>("/can_status_translator/task_scheduling_flag", task_scheduling_flag, 0);
  private_nh_.param<int>("/can_status_translator/task_profiling_flag", task_profiling_flag, 0);
  private_nh_.param<std::string>("/can_status_translator/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/lidar_euclidean_cluster_detect.csv");
  private_nh_.param<int>("/can_status_translator/rate", rate, 10);
  private_nh_.param("/can_status_translator/task_minimum_inter_release_time", task_minimum_inter_release_time, (double)100000000);
  private_nh_.param("/can_status_translator/task_execution_time", task_execution_time, (double)100000000);
  private_nh_.param("/can_status_translator/task_relative_deadline", task_relative_deadline, (double)100000000);
 
  // SPIN
  if(!task_scheduling_flag && !task_profiling_flag){
    std::cout<<"can status translatr / task_scheduling_flag:"<<task_scheduling_flag<<std::endl;
    std::cout<<"can status translatr / is_topic_ready:"<<is_topic_ready<<std::endl;
    ros::spin();
  }
  else{    
    std::cout<<"can status translatr / task_scheduling_flag:"<<task_scheduling_flag<<std::endl;
    std::cout<<"can status translatr / is_topic_ready:"<<is_topic_ready<<std::endl;
    ros::Rate r(rate);
    while(ros::ok()){
      
      if(task_profiling_flag && is_topic_ready) rubis::sched::start_task_profiling();
      if(task_scheduling_flag && is_topic_ready){        
        rubis::sched::request_task_scheduling(task_minimum_inter_release_time, task_execution_time, task_relative_deadline);
      }
      ros::spinOnce();
      if(task_scheduling_flag && is_topic_ready) rubis::sched::yield_task_scheduling();
      if(task_profiling_flag && is_topic_ready) rubis::sched::stop_task_profiling();

      r.sleep();
    }  
  }
  
}

void CanStatusTranslatorNode::publishVelocity(const autoware_msgs::VehicleStatusConstPtr& msg)
{
  geometry_msgs::TwistStamped tw;
  tw.header = msg->header;

  // linear velocity
  tw.twist.linear.x = kmph2mps(msg->speed);  // km/h -> m/s

  // angular velocity
  tw.twist.angular.z = v_info_.convertSteeringAngleToAngularVelocity(
      kmph2mps(msg->speed), v_info_.getCurrentSteeringAngle(deg2rad(msg->angle)));

  pub1_.publish(tw);
  if(!is_topic_ready) is_topic_ready = 1;
}

void CanStatusTranslatorNode::publishVelocityViz(const autoware_msgs::VehicleStatusConstPtr& msg)
{
  std_msgs::Float32 fl;
  fl.data = msg->speed;
  pub2_.publish(fl);
}

void CanStatusTranslatorNode::publishVehicleStatus(const autoware_can_msgs::CANInfoConstPtr& msg)
{
  // currently, this function is only support to autoware_socket format.
  autoware_msgs::VehicleStatus vs;
  vs.header = msg->header;
  vs.tm = msg->tm;
  vs.drivemode = msg->devmode;  // I think devmode is typo in CANInfo...
  vs.steeringmode = msg->strmode;
  if (msg->driveshift == static_cast<int32_t>(GearShift::Drive)){
    vs.current_gear.gear = autoware_msgs::Gear::DRIVE;
  }
  else if (msg->driveshift == static_cast<int32_t>(GearShift::Reverse)){
    vs.current_gear.gear = autoware_msgs::Gear::REVERSE;
  }
  else if (msg->driveshift == static_cast<int32_t>(GearShift::Parking)){
    vs.current_gear.gear = autoware_msgs::Gear::PARK;
  }
  else if (msg->driveshift == static_cast<int32_t>(GearShift::Neutral)){
    vs.current_gear.gear = autoware_msgs::Gear::NEUTRAL;
  }
  else{
    vs.current_gear.gear = autoware_msgs::Gear::NEUTRAL;
  }

  vs.speed = msg->speed;
  if (vs.current_gear.gear == autoware_msgs::Gear::REVERSE)
  {
    vs.speed *= -1.0;
  }

  vs.drivepedal = msg->drivepedal;
  vs.brakepedal = msg->brakepedal;
  vs.angle = v_info_.getCurrentSteeringAngle(deg2rad(msg->angle));
  vs.lamp = 0;
  vs.light = msg->light;

  pub3_.publish(vs);
}

void CanStatusTranslatorNode::callbackFromVehicleStatus(const autoware_msgs::VehicleStatusConstPtr& msg)
{
  publishVelocity(msg);
  publishVelocityViz(msg);
}

void CanStatusTranslatorNode::callbackFromCANInfo(const autoware_can_msgs::CANInfoConstPtr& msg)
{
  publishVehicleStatus(msg);
}

}  // autoware_connector
