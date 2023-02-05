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

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

#include "autoware_config_msgs/ConfigVoxelGridFilter.h"

#include <points_downsampler/PointsDownsamplerInfo.h>

#include <chrono>

#include "points_downsampler.h"

#include <rubis_lib/sched.hpp>
#include <rubis_msgs/PointCloud2.h>

#define SPIN_PROFILING

#define MAX_MEASUREMENT_RANGE 200.0

ros::Publisher filtered_points_pub;
ros::Publisher rubis_filtered_points_pub;

// Leaf size of VoxelGrid filter.
static double voxel_leaf_size = 2.0;

static ros::Publisher points_downsampler_info_pub;
static points_downsampler::PointsDownsamplerInfo points_downsampler_info_msg;

static std::chrono::time_point<std::chrono::system_clock> filter_start, filter_end;

static bool _output_log = false;
static std::ofstream ofs;
static std::string filename;

static std::string input_topic_name_;
static std::string output_topic_name_;
static double measurement_range = MAX_MEASUREMENT_RANGE;

static void config_callback(const autoware_config_msgs::ConfigVoxelGridFilter::ConstPtr& input)
{
  voxel_leaf_size = input->voxel_leaf_size;
  measurement_range = input->measurement_range;
}

inline static void publish_filtered_cloud(const sensor_msgs::PointCloud2::ConstPtr& input)
{  
  pcl::PointCloud<pcl::PointXYZI> scan;
  pcl::fromROSMsg(*input, scan);

  if(measurement_range != MAX_MEASUREMENT_RANGE){
    scan = removePointsByRange(scan, 0, measurement_range);
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan));
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());

  sensor_msgs::PointCloud2 filtered_msg;

  filter_start = std::chrono::system_clock::now();

  // if voxel_leaf_size < 0.1 voxel_grid_filter cannot down sample (It is specification in PCL)
  if (voxel_leaf_size >= 0.1)
  {
    // Downsampling the velodyne scan using VoxelGrid filter
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_grid_filter.setInputCloud(scan_ptr);
    voxel_grid_filter.filter(*filtered_scan_ptr);
    pcl::toROSMsg(*filtered_scan_ptr, filtered_msg);
  }
  else
  {
    pcl::toROSMsg(*scan_ptr, filtered_msg);
  }

  filter_end = std::chrono::system_clock::now();

  filtered_msg.header = input->header;
  filtered_points_pub.publish(filtered_msg);

  rubis_msgs::PointCloud2 rubis_filtered_msg;
  rubis_filtered_msg.msg = filtered_msg;
  rubis_filtered_msg.instance = rubis::instance_;
  rubis_filtered_points_pub.publish(rubis_filtered_msg);

  points_downsampler_info_msg.header = input->header;
  points_downsampler_info_msg.filter_name = "voxel_grid_filter";
  points_downsampler_info_msg.measurement_range = measurement_range;
  points_downsampler_info_msg.original_points_size = scan.size();
  if (voxel_leaf_size >= 0.1)
  {
    points_downsampler_info_msg.filtered_points_size = filtered_scan_ptr->size();
  }
  else
  {
    points_downsampler_info_msg.filtered_points_size = scan_ptr->size();
  }
  points_downsampler_info_msg.original_ring_size = 0;
  points_downsampler_info_msg.filtered_ring_size = 0;
  points_downsampler_info_msg.exe_time = std::chrono::duration_cast<std::chrono::microseconds>(filter_end - filter_start).count() / 1000.0;
  points_downsampler_info_pub.publish(points_downsampler_info_msg);

  if(_output_log == true){
    if(!ofs){
      std::cerr << "Could not open " << filename << "." << std::endl;
      exit(1);
    }
    ofs << points_downsampler_info_msg.header.seq << ","
      << points_downsampler_info_msg.header.stamp << ","
      << points_downsampler_info_msg.header.frame_id << ","
      << points_downsampler_info_msg.filter_name << ","
      << points_downsampler_info_msg.original_points_size << ","
      << points_downsampler_info_msg.filtered_points_size << ","
      << points_downsampler_info_msg.original_ring_size << ","
      << points_downsampler_info_msg.filtered_ring_size << ","
      << points_downsampler_info_msg.exe_time << ","
      << std::endl;
  }
  
}

static void scan_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
{
  rubis::instance_ = 0;
  publish_filtered_cloud(input);
}

static void rubis_scan_callback(const rubis_msgs::PointCloud2::ConstPtr& _input)
{
  rubis::start_task_profiling();

  sensor_msgs::PointCloud2::ConstPtr input = boost::make_shared<const sensor_msgs::PointCloud2>(_input->msg);
  rubis::instance_ = _input->instance;
  publish_filtered_cloud(input);

  rubis::stop_task_profiling(rubis::instance_, 0);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rubis_voxel_grid_filter");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  private_nh.param<std::string>("input_topic_name", input_topic_name_, std::string("points_raw"));
  private_nh.param<std::string>("output_topic_name", output_topic_name_, std::string("filtered_points"));
  private_nh.getParam("output_log", _output_log);
  if(_output_log == true){
    char buffer[80];
    std::time_t now = std::time(NULL);
    std::tm *pnow = std::localtime(&now);
    std::strftime(buffer,80,"%Y%m%d_%H%M%S",pnow);
    filename = "voxel_grid_filter_" + std::string(buffer) + ".csv";
    ofs.open(filename.c_str(), std::ios::app);
  }
  private_nh.param<double>("measurement_range", measurement_range, MAX_MEASUREMENT_RANGE);
  private_nh.param<double>("leaf_size", voxel_leaf_size, 0.1);

  // Scheduling & Profiling Setup
  std::string node_name = ros::this_node::getName();
  std::string task_response_time_filename;
  private_nh.param<std::string>(node_name+"/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/voexl_grid_filter.csv");

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


  // Publishers
  filtered_points_pub = nh.advertise<sensor_msgs::PointCloud2>(output_topic_name_, 10);
  rubis_filtered_points_pub = nh.advertise<rubis_msgs::PointCloud2>("/rubis_" + output_topic_name_, 10);
  points_downsampler_info_pub = nh.advertise<points_downsampler::PointsDownsamplerInfo>("/points_downsampler_info", 1000);

  // Subscribers
  ros::Subscriber config_sub = nh.subscribe("config/voxel_grid_filter", 10, config_callback);
  // ros::Subscriber scan_sub = nh.subscribe(input_topic_name_, 10, scan_callback);
  ros::Subscriber scan_sub;
  
  scan_sub = nh.subscribe("/rubis_"+input_topic_name_, 1, rubis_scan_callback); // Def: 10

  ros::spin();

  return 0;
}
