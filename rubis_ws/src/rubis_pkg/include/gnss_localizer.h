#ifndef NMEA2TFPOSE_CORE_H
#define NMEA2TFPOSE_CORE_H

// C++ includes
#include <string>
#include <memory>

// ROS includes
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nmea_msgs/Sentence.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/Imu.h>
#include <gnss/geo_pos_conv.hpp>
#include <geometry_msgs/TwistStamped.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

namespace gnss_localizer
{
struct Pose
{
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};

class Nmea2TFPoseNode
{
public:
  Nmea2TFPoseNode();
  ~Nmea2TFPoseNode();

  void run();

private:
  // handle
  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;

  // publisher
  ros::Publisher pub1_;
  ros::Publisher vel_pub_;

  // subscriber
  ros::Subscriber sub1_;
  ros::Subscriber sub2_;

  // constants
  const std::string MAP_FRAME_;
  const std::string GPS_FRAME_;

  // variables
  int32_t plane_number_;
  geo_pos_conv geo_;
  geo_pos_conv last_geo_;
  double roll_, pitch_, yaw_;
  double orientation_time_, position_time_, current_time_, prev_time_;
  ros::Time orientation_stamp_;
  tf::TransformBroadcaster br_;
  bool orientation_ready_;  // true if position history is long enough to compute orientation
  geometry_msgs::TwistStamped gnss_vel_;
  geometry_msgs::PoseStamped cur_pose_;
  Pose cur_pose_data_, prev_pose_data_;
  tf::TransformListener listener_;
  tf::StampedTransform transform_;

  // callbacks
  void callbackFromNmeaSentence(const nmea_msgs::Sentence::ConstPtr &msg);
  void callbackFromIMU(const sensor_msgs::Imu& msg);

  // initializer
  void initForROS();
  void InitTF();

  // functions
  void publishPoseStamped();
  void publishTF();
  void publishVelocity();
  void createOrientation();
  void convert(std::vector<std::string> nmea, ros::Time current_stamp);
  void TransformPose(const geometry_msgs::PoseStamped &in_pose, geometry_msgs::PoseStamped& out_pose, const tf::StampedTransform &in_transform);
};

std::vector<std::string> split(const std::string &string);

}  // namespace gnss_localizer
#endif  // NMEA2TFPOSE_CORE_H
