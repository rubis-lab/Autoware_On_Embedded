#ifndef SVL_SENSING_H
#define SVL_SENSING_H

#include <algorithm>
#include <cstdlib>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <rubis_lib/sched.hpp>

#include <sensor_msgs/PointCloud2.h>
#include <rubis_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <rubis_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <rubis_msgs/PoseTwistStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

class SvlSensing
{
public:
    SvlSensing();
    ~SvlSensing();
    void run();
private:
    int* data_;
    int rate_;
    int data_size_, n_, last_publish_id_, id_;
    ros::NodeHandle nh_;


    nav_msgs::Odometry::ConstPtr latest_odom_msg_;
    sensor_msgs::Image::ConstPtr latest_image_msg1_;
    sensor_msgs::Image::ConstPtr latest_image_msg2_;
    sensor_msgs::Image::ConstPtr latest_image_msg3_;
    sensor_msgs::Image::ConstPtr latest_image_msg4_;
    sensor_msgs::PointCloud2::ConstPtr latest_lidar_msg_;
 
    void callbackOdom(const nav_msgs::Odometry::ConstPtr &odom_msg);
    void callbackImage1(const sensor_msgs::Image::ConstPtr &image_msg1);
    void callbackImage2(const sensor_msgs::Image::ConstPtr &image_msg2);
    void callbackImage3(const sensor_msgs::Image::ConstPtr &image_msg3);
    void callbackImage4(const sensor_msgs::Image::ConstPtr &image_msg4);
    void callbackLidar(const sensor_msgs::PointCloud2::ConstPtr &lidar_msg);
    ros::Publisher lidar_pub_, pose_twist_pub_, image_pub1_, image_pub2_, image_pub3_, image_pub4_; 
    ros::Subscriber lidar_sub_, odom_sub_, image_sub_1_, image_sub_2_, image_sub_3_, image_sub_4_; 

    
    void read_data(int n);
    void read_sequential_data(int m);
};

#endif