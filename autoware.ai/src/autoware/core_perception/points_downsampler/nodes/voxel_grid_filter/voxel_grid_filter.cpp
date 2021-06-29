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

#include <rubis_sched/sched.hpp>

#define SPIN_PROFILING

#define MAX_MEASUREMENT_RANGE 200.0

int scheduling_flag_;
int profiling_flag_;
std::string response_time_filename_;
int rate_;
double minimum_inter_release_time_;
double execution_time_;
double relative_deadline_;

ros::Publisher filtered_points_pub;

// Leaf size of VoxelGrid filter.
static double voxel_leaf_size = 2.0;

static ros::Publisher points_downsampler_info_pub;
static points_downsampler::PointsDownsamplerInfo points_downsampler_info_msg;

static std::chrono::time_point<std::chrono::system_clock> filter_start, filter_end;

static bool _output_log = false;
static std::ofstream ofs;
static std::string filename;

static std::string POINTS_TOPIC;
static double measurement_range = MAX_MEASUREMENT_RANGE;

static void config_callback(const autoware_config_msgs::ConfigVoxelGridFilter::ConstPtr& input)
{
  voxel_leaf_size = input->voxel_leaf_size;
  measurement_range = input->measurement_range;
}

static void scan_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
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

int main(int argc, char** argv)
{
  ros::init(argc, argv, "voxel_grid_filter");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  private_nh.getParam("points_topic", POINTS_TOPIC);
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

  private_nh.param<int>("/voxel_grid_filter/scheduling_flag", scheduling_flag_, 0);
  private_nh.param<int>("/voxel_grid_filter/profiling_flag", profiling_flag_, 0);
  private_nh.param<std::string>("/voxel_grid_filter/response_time_filename", response_time_filename_, "/home/hypark/Documents/profiling/response_time/voxel_grid_filter.csv");
  private_nh.param<int>("/voxel_grid_filter/rate", rate_, 10);
  private_nh.param("/voxel_grid_filter/minimum_inter_release_time", minimum_inter_release_time_, (double)10);
  private_nh.param("/voxel_grid_filter/execution_time", execution_time_, (double)10);
  private_nh.param("/voxel_grid_filter/relative_deadline", relative_deadline_, (double)10);

  // Publishers
  filtered_points_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_points", 10);
  points_downsampler_info_pub = nh.advertise<points_downsampler::PointsDownsamplerInfo>("/points_downsampler_info", 1000);

  // Subscribers
  ros::Subscriber config_sub = nh.subscribe("config/voxel_grid_filter", 10, config_callback);
  ros::Subscriber scan_sub = nh.subscribe(POINTS_TOPIC, 10, scan_callback);

  // if(!scheduling_flag_ && !profiling_flag_){
    ros::spin();
  // }
  // else{
  //   FILE *fp;
  //   if(profiling_flag_){      
  //     fp = fopen(response_time_filename_.c_str(), "a");
  //   }

  //   ros::Rate r(rate_);
  //   struct timespec start_time, end_time;
  //   while(ros::ok()){
  //     if(profiling_flag_){        
  //       clock_gettime(CLOCK_MONOTONIC, &start_time);
  //     }
  //     if(scheduling_flag_){
  //       rubis::sched::set_sched_deadline(gettid(), 
  //         static_cast<uint64_t>(execution_time_), 
  //         static_cast<uint64_t>(relative_deadline_), 
  //         static_cast<uint64_t>(minimum_inter_release_time_)
  //       );
  //     }      

  //     ros::spinOnce();

  //     if(profiling_flag_){
  //       clock_gettime(CLOCK_MONOTONIC, &end_time);
  //       fprintf(fp, "%lld.%.9ld,%lld.%.9ld,%d\n",start_time.tv_sec,start_time.tv_nsec,end_time.tv_sec,end_time.tv_nsec,getpid());    
  //       fflush(fp);
  //     }

  //     r.sleep();
  //   }  
  // fclose(fp);
  // }

  return 0;
}
