#include "multi_vlp_combiner/combiner.h"

void init_rotational_matrix(){
  Eigen::Translation3f init_translation_1(config_.tf_1.x, config_.tf_1.y, config_.tf_1.z);
  Eigen::AngleAxisf init_rotation_x_1(config_.tf_1.roll, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf init_rotation_y_1(config_.tf_1.pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf init_rotation_z_1(config_.tf_1.yaw, Eigen::Vector3f::UnitZ());

  Eigen::Translation3f init_translation_2(config_.tf_2.x, config_.tf_2.y, config_.tf_2.z);
  Eigen::AngleAxisf init_rotation_x_2(config_.tf_2.roll, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf init_rotation_y_2(config_.tf_2.pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf init_rotation_z_2(config_.tf_2.yaw, Eigen::Vector3f::UnitZ());

  rotation_matrix_1 = (init_translation_1 * init_rotation_z_1 * init_rotation_y_1 * init_rotation_x_1).matrix();
  rotation_matrix_2 = (init_translation_2 * init_rotation_z_2 * init_rotation_y_2 * init_rotation_x_2).matrix();
}

void lidar1_points_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr raw_points_1(new pcl::PointCloud<pcl::PointXYZI>);

  pcl::fromROSMsg(*msg, *raw_points_1);

  // rotation with x, y, z, roll, pitch, yaw
  pcl::transformPointCloud (*raw_points_1, *latest_points_1, rotation_matrix_1);
}

void lidar2_points_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr raw_points_2(new pcl::PointCloud<pcl::PointXYZI>);

  pcl::fromROSMsg(*msg, *raw_points_2);

  // rotation with x, y, z, roll, pitch, yaw
  pcl::transformPointCloud (*raw_points_2, *latest_points_2, rotation_matrix_2);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "multi_vlp_combiner");
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  pnh_.param<double>("/multi_vlp_combiner/rpm", config_.rpm, 600.0);
  pnh_.param<std::string>("/multi_vlp_combiner/points_topic_1", config_.points_topic_1, "/points_raw_1");
  pnh_.param<std::string>("/multi_vlp_combiner/points_topic_2", config_.points_topic_2, "/points_raw_2");
  pnh_.param<std::string>("/multi_vlp_combiner/output_topic", config_.output_topic, "/points_raw");
  pnh_.param<std::string>("/multi_vlp_combiner/output_frame_id", config_.output_frame_id, "velodyne");

  ros::Rate loop_rate(config_.rpm / 60);

  // Make Rotation Matrix

  // Publish / Subscribe Info
  combined_pub = nh_.advertise<sensor_msgs::PointCloud2>(config_.output_topic, 1000);
  points_sub[0] = nh_.subscribe(config_.points_topic_1, 10, lidar1_points_callback);
  points_sub[1] = nh_.subscribe(config_.points_topic_2, 10, lidar2_points_callback);

  while(ros::ok()){
    pnh_.param<double>("/multi_vlp_combiner/tf1_x", config_.tf_1.x, 0.0);
    pnh_.param<double>("/multi_vlp_combiner/tf1_y", config_.tf_1.y, 0.0);
    pnh_.param<double>("/multi_vlp_combiner/tf1_z", config_.tf_1.z, 0.0);
    pnh_.param<double>("/multi_vlp_combiner/tf1_roll", config_.tf_1.roll, 0.0);
    pnh_.param<double>("/multi_vlp_combiner/tf1_pitch", config_.tf_1.pitch, 0.0);
    pnh_.param<double>("/multi_vlp_combiner/tf1_yaw", config_.tf_1.yaw, 0.0);
    pnh_.param<double>("/multi_vlp_combiner/tf2_x", config_.tf_2.x, 0.0);
    pnh_.param<double>("/multi_vlp_combiner/tf2_y", config_.tf_2.y, 0.0);
    pnh_.param<double>("/multi_vlp_combiner/tf2_z", config_.tf_2.z, 0.0);
    pnh_.param<double>("/multi_vlp_combiner/tf2_roll", config_.tf_2.roll, 0.0);
    pnh_.param<double>("/multi_vlp_combiner/tf2_pitch", config_.tf_2.pitch, 0.0);
    pnh_.param<double>("/multi_vlp_combiner/tf2_yaw", config_.tf_2.yaw, 0.0);
    init_rotational_matrix();

    pcl::PointCloud<pcl::PointXYZI>::Ptr combined_pcd(new pcl::PointCloud<pcl::PointXYZI>);
    sensor_msgs::PointCloud2 cloud_msg;

    (*combined_pcd).reserve((*latest_points_1).size() + (*latest_points_2).size());
    (*combined_pcd).insert((*combined_pcd).end(), (*latest_points_1).begin(), (*latest_points_1).end());
    (*combined_pcd).insert((*combined_pcd).end(), (*latest_points_2).begin(), (*latest_points_2).end());

    pcl::toROSMsg(*combined_pcd, cloud_msg);
    cloud_msg.header.frame_id = config_.output_frame_id;
    cloud_msg.header.stamp = ros::Time::now();
    combined_pub.publish(cloud_msg);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}