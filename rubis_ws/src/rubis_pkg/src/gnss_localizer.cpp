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

#include "gnss_localizer.h"

namespace gnss_localizer
{
// Constructor
Nmea2TFPoseNode::Nmea2TFPoseNode()
  : private_nh_("~")
  , MAP_FRAME_("map")
  , GPS_FRAME_("gnss")
  , roll_(0)
  , pitch_(0)
  , yaw_(0)
  , orientation_time_(-std::numeric_limits<double>::infinity())
  , position_time_(-std::numeric_limits<double>::infinity())
  , current_time_(0)
  , prev_time_(0)
  , orientation_stamp_(0)
  , orientation_ready_(false)
{
  initForROS();
  // InitTF();
  geo_.set_plane(plane_number_);
}

// Destructor
Nmea2TFPoseNode::~Nmea2TFPoseNode()
{
}

void Nmea2TFPoseNode::initForROS()
{
  // ros parameter settings
  private_nh_.getParam("plane", plane_number_);
  private_nh_.param<bool>("enable_noise", enable_noise_, false);

  private_nh_.param<bool>("enable_offset", enable_offset_, false);
  if(enable_offset_){
    private_nh_.param<double>("offset_bx", offset_bx_, 0);
    private_nh_.param<double>("offset_by", offset_by_, 0);
    private_nh_.param<double>("offset_theta", offset_theta_, 0);    

    std::vector<double> transformation_vec;
    if( !nh_.getParam("gnss_transformation",transformation_vec)){
      ROS_ERROR("Cannot load gnss_transformation");
    }

    createTransformationOffsetMatrix();
    createTransformationMatrix(transformation_vec);
  }

  if(enable_noise_){
    std::srand((unsigned int)time(NULL));
    private_nh_.param<double>("max_noise", max_noise_, 2.0);
  }

  // setup subscriber
  sub1_ = nh_.subscribe("nmea_sentence", 100, &Nmea2TFPoseNode::callbackFromNmeaSentence, this);
  sub2_ = nh_.subscribe("imu_raw", 100, &Nmea2TFPoseNode::callbackFromIMU, this);

  // setup publisher
  if(enable_offset_){
    pub1_ = nh_.advertise<geometry_msgs::PoseStamped>("gnss_offset_pose", 10);
    pub2_ = nh_.advertise<geometry_msgs::PoseStamped>("gnss_transformed_pose", 10);
  }
  else
    pub1_ = nh_.advertise<geometry_msgs::PoseStamped>("gnss_pose", 10);
  vel_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("gnss_vel", 10);
}

void Nmea2TFPoseNode::run()
{
  ros::spin();
}

void Nmea2TFPoseNode::createTransformationOffsetMatrix(){
  double offset_theta_radian = offset_theta_ * M_PI / 180.0;
  T_offset_ <<  cos(offset_theta_radian), -1 * sin(offset_theta_radian),  offset_bx_,
                sin(offset_theta_radian), cos(offset_theta_radian),       offset_by_,
                0,                        0,                              1; 

  T_offset_inv_ = T_offset_.inverse();
}

void Nmea2TFPoseNode::createTransformationMatrix(std::vector<double>& transformation_vec){
  T_ << transformation_vec[0], transformation_vec[1], transformation_vec[2],
        transformation_vec[3], transformation_vec[4], transformation_vec[5],
        transformation_vec[6], transformation_vec[7], transformation_vec[8];
}

void Nmea2TFPoseNode::publishPoseStamped()
{ 
  cur_pose_.header.frame_id = GPS_FRAME_;
  cur_pose_.header.stamp = ros::Time::now();
  cur_pose_.pose.position.x = geo_.y();
  cur_pose_.pose.position.y = geo_.x();
  cur_pose_.pose.position.z = geo_.z();

  if(enable_noise_){
    int noise_100 = (int)(max_noise_ * 100);
    double x_noise = (double)(rand() % (int)(noise_100 * 2) - noise_100) / 100;
    double y_noise = (double)(rand() % (int)(noise_100 * 2) - noise_100) / 100;
    cur_pose_.pose.position.x += x_noise;
    cur_pose_.pose.position.y += y_noise;
  }

  if(enable_offset_){    
    Eigen::Matrix<double, 3, 1> pos_offset, transposed_pose_offset;
    pos_offset(0) = cur_pose_.pose.position.x; pos_offset(1) = cur_pose_.pose.position.y; pos_offset(2) = 1;
    transposed_pose_offset = T_offset_ * pos_offset;
    cur_pose_.pose.position.x = transposed_pose_offset(0);
    cur_pose_.pose.position.y = transposed_pose_offset(1);
  }

  // TransformPose(cur_pose_, cur_pose_, transform_);
  cur_pose_.header.frame_id = MAP_FRAME_;
  cur_pose_.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll_, pitch_, yaw_);

  pub1_.publish(cur_pose_);
  if(enable_offset_){
    Eigen::Matrix<double, 3, 1> pos, transposed_pose;
    pos(0) = cur_pose_.pose.position.x; pos(1) = cur_pose_.pose.position.y; pos(2) = 1;
    transposed_pose = T_ * pos;
    cur_pose_.pose.position.x = transposed_pose(0);
    cur_pose_.pose.position.y = transposed_pose(1);

    pub2_.publish(cur_pose_);
  }
}

void Nmea2TFPoseNode::publishTF()
{
  tf::Transform transform;
  transform.setOrigin(tf::Vector3(cur_pose_data_.x, cur_pose_data_.y, cur_pose_data_.z));
  tf::Quaternion quaternion;
  quaternion.setRPY(roll_, pitch_, yaw_);
  transform.setRotation(quaternion);
  br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), MAP_FRAME_, GPS_FRAME_));
}

void Nmea2TFPoseNode::createOrientation()
{
  yaw_ = atan2(geo_.x() - last_geo_.x(), geo_.y() - last_geo_.y());
  roll_ = 0;
  pitch_ = 0;
}

void Nmea2TFPoseNode::convert(std::vector<std::string> nmea, ros::Time current_stamp)
{
  try
  {
    if (nmea.at(0).compare(0, 2, "QQ") == 0)
    {
      orientation_time_ = stod(nmea.at(3));
    //   roll_ = stod(nmea.at(4)) * M_PI / 180.;
    //   pitch_ = -1 * stod(nmea.at(5)) * M_PI / 180.;
    //   yaw_ = -1 * stod(nmea.at(6)) * M_PI / 180. + M_PI / 2;
      orientation_stamp_ = current_stamp;
      orientation_ready_ = true;
    //   ROS_INFO("QQ is subscribed.");
    }
    else if (nmea.at(0) == "$PASHR")
    {
      orientation_time_ = stod(nmea.at(1));
    //   roll_ = stod(nmea.at(4)) * M_PI / 180.;
    //   pitch_ = -1 * stod(nmea.at(5)) * M_PI / 180.;
    //   yaw_ = -1 * stod(nmea.at(2)) * M_PI / 180. + M_PI / 2;
      orientation_ready_ = true;
    //   ROS_INFO("PASHR is subscribed.");
    }
    else if (nmea.at(0).compare(3, 3, "GGA") == 0)
    {
      std::cout.precision(20);
      current_time_ = stod(nmea.at(1));
      position_time_ = stod(nmea.at(1));
      double lat = stod(nmea.at(2));
      double lon = stod(nmea.at(4));
      double h = stod(nmea.at(9));

      if (nmea.at(3) == "S")
        lat = -lat;

      if (nmea.at(5) == "W")
        lon = -lon;

      geo_.set_llh_nmea_degrees(lat, lon, h);

    //   ROS_INFO("GGA is subscribed.");
    }
    else if (nmea.at(0) == "$GPRMC")
    {
      current_time_ = stod(nmea.at(1));
      position_time_ = stoi(nmea.at(1));
      double lat = stod(nmea.at(3));
      double lon = stod(nmea.at(5));
      double h = 0.0;

      if (nmea.at(4) == "S")
        lat = -lat;

      if (nmea.at(6) == "W")
        lon = -lon;

      geo_.set_llh_nmea_degrees(lat, lon, h);

      ROS_INFO("GPRMC is subscribed.");
    }
  }
  catch (const std::exception &e)
  {
    ROS_WARN_STREAM("Message is invalid : " << e.what());
  }
}

void Nmea2TFPoseNode::callbackFromIMU(const sensor_msgs::Imu& msg){
    tf::Quaternion q(
                    msg.orientation.x, 
                    msg.orientation.y, 
                    msg.orientation.z, 
                    msg.orientation.w);
    tf::Matrix3x3 m(q);
    m.getRPY(roll_, pitch_, yaw_);
}

void Nmea2TFPoseNode::publishVelocity(){
    static int zero_cnt = 0;
    double diff_x, diff_y, diff_z, diff_yaw, diff, current_velocity, angular_velocity;


    diff_x = abs(cur_pose_data_.x - prev_pose_data_.x);
    diff_y = abs(cur_pose_data_.y - prev_pose_data_.y);
    diff_z = abs(cur_pose_data_.z - prev_pose_data_.z);
    diff_yaw = cur_pose_data_.yaw - prev_pose_data_.yaw;
    diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

    if(diff_x == 0 && diff_y == 0 && diff_z == 0){
        if(zero_cnt<=5){
            zero_cnt++;
            return;
        }
    }
    else{
        zero_cnt=0;
    }

    const double diff_time = current_time_ - prev_time_;
    current_velocity = (diff_time > 0) ? (diff / diff_time) : 0;
    // std::cout<<"current_velocity:"<<current_velocity<<std::endl;
    // current_velocity =  (cur_pose_.pose.position.x - prev_pose_.pose.position.x >= 0) ? current_velocity : -current_velocity;
    // current_velocity_x = (diff_time > 0) ? (diff_x / diff_time) : 0;
    // current_velocity_y = (diff_time > 0) ? (diff_y / diff_time) : 0;
    // current_velocity_z = (diff_time > 0) ? (diff_z / diff_time) : 0;
    angular_velocity = (diff_time > 0) ? (diff_yaw / diff_time) : 0;

    geometry_msgs::TwistStamped twist_msg;
    twist_msg.header.stamp = ros::Time::now();
    twist_msg.header.frame_id = "/base_link";
    twist_msg.twist.linear.x = current_velocity;
    twist_msg.twist.linear.y = 0.0;
    twist_msg.twist.linear.z = 0.0;
    twist_msg.twist.angular.x = 0.0;
    twist_msg.twist.angular.y = 0.0;
    twist_msg.twist.angular.z = angular_velocity;
    vel_pub_.publish(twist_msg);

    prev_time_ = current_time_;
}


void Nmea2TFPoseNode::callbackFromNmeaSentence(const nmea_msgs::Sentence::ConstPtr &msg)
{
  // std::cout<<msg->sentence<<std::endl;
  convert(split(msg->sentence), msg->header.stamp);

  double timeout = 10.0;

  publishPoseStamped();
  cur_pose_data_.x = cur_pose_.pose.position.x;
  cur_pose_data_.y = cur_pose_.pose.position.y;
  cur_pose_data_.z = cur_pose_.pose.position.z;
  cur_pose_data_.roll = roll_;
  cur_pose_data_.pitch = pitch_;
  cur_pose_data_.yaw = yaw_;

  publishVelocity();
  publishTF();

  prev_pose_data_.x = cur_pose_data_.x;
  prev_pose_data_.y = cur_pose_data_.y;
  prev_pose_data_.z = cur_pose_data_.z;
  prev_pose_data_.roll = cur_pose_data_.roll;
  prev_pose_data_.pitch = cur_pose_data_.pitch;
  prev_pose_data_.yaw = cur_pose_data_.yaw;

  return;


  if (orientation_stamp_.isZero()
      || fabs(orientation_stamp_.toSec() - msg->header.stamp.toSec()) > timeout)
  {
    double dt = sqrt(pow(geo_.x() - last_geo_.x(), 2) + pow(geo_.y() - last_geo_.y(), 2));
    double threshold = 0.2;
    if (dt > threshold)
    {
      /* If orientation data is not available it is generated based on translation
         from the previous position. For the first message the previous position is
         simply the origin, which gives a wildly incorrect orientation. Some nodes
         (e.g. ndt_matching) rely on that first message to initialise their pose guess,
         and cannot recover from such incorrect orientation.
         Therefore the first message is not published, ensuring that orientation is
         only calculated from sensible positions.
      */
      if (orientation_ready_)
      {
        ROS_INFO("QQ is not subscribed. Orientation is created by atan2");
        createOrientation();
        publishPoseStamped();
        publishTF();
      }
      else
      {
        orientation_ready_ = true;
      }
      last_geo_ = geo_;
    }
    return;
  }

  double e = 1e-2;
  if ((fabs(orientation_time_ - position_time_) < e) && orientation_ready_)
  {
    publishPoseStamped();
    publishTF();
    return;
  }
}

void Nmea2TFPoseNode::TransformPose(const geometry_msgs::PoseStamped &in_pose, geometry_msgs::PoseStamped& out_pose, const tf::StampedTransform &in_transform)
{

  tf::Vector3 in_pos(in_pose.pose.position.x,
                     in_pose.pose.position.y,
                     in_pose.pose.position.z);
  tf::Quaternion in_quat(in_pose.pose.orientation.x,
                         in_pose.pose.orientation.y,
                         in_pose.pose.orientation.w,
                         in_pose.pose.orientation.z);

  tf::Vector3 in_pos_t = in_transform * in_pos;
  tf::Quaternion in_quat_t = in_transform * in_quat;
  
  out_pose.header = in_pose.header;
  out_pose.pose.position.x = in_pos_t.x();
  out_pose.pose.position.y = in_pos_t.y();
  out_pose.pose.position.z = in_pos_t.z();
  out_pose.pose.orientation.x = in_quat_t.x();
  out_pose.pose.orientation.y = in_quat_t.y();
  out_pose.pose.orientation.z = in_quat_t.z();

  return;
}

void Nmea2TFPoseNode::InitTF(){
  while(1){
      try{
      listener_.waitForTransform("base_link", "gnss", ros::Time(0), ros::Duration(0.001));
      listener_.lookupTransform("base_link", "gnss", ros::Time(0), transform_);
      break;
      }
      catch(tf::TransformException& ex)
      {
      // ROS_ERROR("[2] Cannot transform object pose: %s", ex.what());
      }
  }
  std::cout<<"Init tf Transform gps -> base_link is obtained."<<std::endl;  
}

std::vector<std::string> split(const std::string &string)
{
  std::vector<std::string> str_vec_ptr;
  std::string token;
  std::stringstream ss(string);

  while (getline(ss, token, ','))
    str_vec_ptr.push_back(token);

  return str_vec_ptr;
}

}  // namespace gnss_localizer
