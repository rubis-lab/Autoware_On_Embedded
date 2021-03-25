#include <string>
#include "ros/ros.h"
#include "std_msgs/String.h"

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#define LIDAR_NUM 2

ros::Publisher combined_pub;
ros::Subscriber points_sub[LIDAR_NUM];

struct TFInfo {
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};

struct MultiVLPCombinerConfig
{
  double rpm;
  std::string points_topic_1;
  std::string points_topic_2;
  std::string frame_id_1;
  std::string frame_id_2;
  std::string output_topic;
  std::string output_frame_id;
  struct TFInfo tf_1;
  struct TFInfo tf_2;
};

struct MultiVLPCombinerConfig config_;

Eigen::Matrix4f rotation_matrix_1, rotation_matrix_2;

void lidar1_points_callback(const sensor_msgs::PointCloud2ConstPtr& msg);
void lidar2_points_callback(const sensor_msgs::PointCloud2ConstPtr& msg);

pcl::PointCloud<pcl::PointXYZI>::Ptr latest_points_1(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr latest_points_2(new pcl::PointCloud<pcl::PointXYZI>);