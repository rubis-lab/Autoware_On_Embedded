/*********************************************************************
* Software License Agreement (BSD License)
* 
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
* 
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
* 
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
* 
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#include "image_transport/image_transport.h"
#include "image_transport/publisher_plugin.h"
#include <pluginlib/class_loader.h>
#include "rubis_lib/sched.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_republisher", ros::init_options::AnonymousName);
  if (argc < 2) {
    printf("Usage: %s in_transport in:=<in_base_topic> [out_transport] out:=<out_base_topic>\n", argv[0]);
    return 0;
  }
  ros::NodeHandle nh;
  std::string in_topic  = nh.resolveName("in");
  std::string in_transport = argv[1];
  std::string out_topic = nh.resolveName("out");

  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub;

  // scheduling
  ros::NodeHandle pnh("~");
  int task_scheduling_flag = 0;
  int task_profiling_flag = 0;
  std::string task_response_time_filename;
  int rate = 0;
  double task_minimum_inter_release_time = 0;
  double task_execution_time = 0;
  double task_relative_deadline = 0;

  pnh.param<int>("/republish/task_scheduling_flag", task_scheduling_flag, 0);
  pnh.param<int>("/republish/task_profiling_flag", task_profiling_flag, 0);
  pnh.param<std::string>("/republish/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/republish.csv");
  pnh.param<int>("/republish/rate", rate, 10);
  pnh.param("/republish/task_minimum_inter_release_time", task_minimum_inter_release_time, (double)100000000);
  pnh.param("/republish/task_execution_time", task_execution_time, (double)100000000);
  pnh.param("/republish/task_relative_deadline", task_relative_deadline, (double)100000000);
  
  if(task_profiling_flag) rubis::sched::init_task_profiling(task_response_time_filename);

  if (argc < 3) {
    // Use all available transports for output
    image_transport::Publisher pub = it.advertise(out_topic, 1);
    
    // Use Publisher::publish as the subscriber callback
    typedef void (image_transport::Publisher::*PublishMemFn)(const sensor_msgs::ImageConstPtr&) const;
    PublishMemFn pub_mem_fn = &image_transport::Publisher::publish;
    sub = it.subscribe(in_topic, 1, boost::bind(pub_mem_fn, &pub, _1), ros::VoidPtr(), in_transport);

    rubis::sched::task_state_ = TASK_STATE_READY;

    if(!task_scheduling_flag && !task_profiling_flag){
      ros::spin();
    }
    else{
      ros::Rate r(rate);
      // Executing task
      while(ros::ok()){
        if(task_profiling_flag) rubis::sched::start_task_profiling();

        if(rubis::sched::task_state_ == TASK_STATE_READY){          
          if(task_scheduling_flag) rubis::sched::request_task_scheduling(task_minimum_inter_release_time, task_execution_time, task_relative_deadline); 
          rubis::sched::task_state_ = TASK_STATE_RUNNING;     
        }

        ros::spinOnce();
        rubis::sched::task_state_ = TASK_STATE_DONE;

        if(task_profiling_flag) rubis::sched::stop_task_profiling(0, rubis::sched::task_state_);

        if(rubis::sched::task_state_ == TASK_STATE_DONE){          
          if(task_scheduling_flag) rubis::sched::yield_task_scheduling();
          rubis::sched::task_state_ = TASK_STATE_READY;
        }
        
        r.sleep();
      }
    }
  }
  else {
    // Use one specific transport for output
    std::string out_transport = argv[2];

    // Load transport plugin
    typedef image_transport::PublisherPlugin Plugin;
    pluginlib::ClassLoader<Plugin> loader("image_transport", "image_transport::PublisherPlugin");
    std::string lookup_name = Plugin::getLookupName(out_transport);
    boost::shared_ptr<Plugin> pub( loader.createInstance(lookup_name) );
    pub->advertise(nh, out_topic, 1, image_transport::SubscriberStatusCallback(),
                   image_transport::SubscriberStatusCallback(), ros::VoidPtr(), false);

    // Use PublisherPlugin::publish as the subscriber callback
    typedef void (Plugin::*PublishMemFn)(const sensor_msgs::ImageConstPtr&) const;
    PublishMemFn pub_mem_fn = &Plugin::publish;
    sub = it.subscribe(in_topic, 1, boost::bind(pub_mem_fn, pub.get(), _1), pub, in_transport);

    if(!task_scheduling_flag && !task_profiling_flag){
      ros::spin();
    }
    else{      
      ros::Rate r(rate);
      // Executing task      
      while(ros::ok()){
        if(task_profiling_flag) rubis::sched::start_task_profiling();

        if(rubis::sched::task_state_ == TASK_STATE_READY){          
          if(task_scheduling_flag) rubis::sched::request_task_scheduling(task_minimum_inter_release_time, task_execution_time, task_relative_deadline); 
          rubis::sched::task_state_ = TASK_STATE_RUNNING;     
        }

        ros::spinOnce();
        rubis::sched::task_state_ = TASK_STATE_DONE;

        if(task_profiling_flag) rubis::sched::stop_task_profiling(0, rubis::sched::task_state_);

        if(rubis::sched::task_state_ == TASK_STATE_DONE){          
          if(task_scheduling_flag) rubis::sched::yield_task_scheduling();
          rubis::sched::task_state_ = TASK_STATE_READY;
        }
        
        r.sleep();
      }
    }
  }

  return 0;
}
