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

// ROS Includes
#include <ros/ros.h>

// User defined includes
#include "twist_gate/twist_gate.h"
#include <rubis_lib/sched.hpp>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "twist_gate");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  // Scheduling Setup
  int task_scheduling_flag;
  int task_profiling_flag;
  std::string task_response_time_filename;
  int rate;
  double task_minimum_inter_release_time;
  double task_execution_time;
  double task_relative_deadline;

  private_nh.param<int>("/twist_gate/task_scheduling_flag", task_scheduling_flag, 0);
  private_nh.param<int>("/twist_gate/task_profiling_flag", task_profiling_flag, 0);
  private_nh.param<std::string>("/twist_gate/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/twist_gate.csv");
  private_nh.param<int>("/twist_gate/rate", rate, 10);
  private_nh.param("/twist_gate/task_minimum_inter_release_time", task_minimum_inter_release_time, (double)10);
  private_nh.param("/twist_gate/task_execution_time", task_execution_time, (double)10);
  private_nh.param("/twist_gate/task_relative_deadline", task_relative_deadline, (double)10);
  private_nh.param<int>("/twist_gate/zero_flag", zero_flag_, 0);

  TwistGate twist_gate(nh, private_nh);

  if(task_profiling_flag) rubis::sched::init_task_profiling(task_response_time_filename);

  if(!task_scheduling_flag && !task_profiling_flag){
    ros::spin();
  }
  else{
    ros::Rate r(rate);
    // Initialize task ( Wait until first necessary topic is published )
    while(ros::ok()){
      if(rubis::sched::is_task_ready_ == TASK_READY) break;
      ros::spinOnce();
      r.sleep();      
    }

    // Executing task
    while(ros::ok()){
      if(task_profiling_flag) rubis::sched::start_task_profiling();

      if(rubis::sched::task_state_ == TASK_STATE_READY){
        if(task_scheduling_flag) rubis::sched::request_task_scheduling(task_minimum_inter_release_time, task_execution_time, task_relative_deadline); 
        rubis::sched::task_state_ = TASK_STATE_RUNNING;     
      }

      ros::spinOnce();

      if(task_profiling_flag) rubis::sched::stop_task_profiling(rubis::instance_, rubis::sched::task_state_);

      if(rubis::sched::task_state_ == TASK_STATE_DONE){
        if(task_scheduling_flag) rubis::sched::yield_task_scheduling();
        rubis::sched::task_state_ = TASK_STATE_READY;
      }
      
      r.sleep();
    }
  }

  return 0;
}
