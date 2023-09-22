#ifndef SVL_SENSING_H
#define SVL_SENSING_H

#include <algorithm>

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

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry, sensor_msgs::Image> SyncPolicyImage1;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry, sensor_msgs::Image, sensor_msgs::Image> SyncPolicyImage2;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> SyncPolicyImage3;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> SyncPolicyImage4;
typedef message_filters::Synchronizer<SyncPolicyImage1> Sync1;
typedef message_filters::Synchronizer<SyncPolicyImage2> Sync2;
typedef message_filters::Synchronizer<SyncPolicyImage3> Sync3;
typedef message_filters::Synchronizer<SyncPolicyImage4> Sync4;

class SvlSensing
{
public:
    SvlSensing();
    ~SvlSensing();
    void run();
private:
    ros::NodeHandle nh_;
        
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;    
    message_filters::Subscriber<sensor_msgs::Image> image_sub_1_;
    message_filters::Subscriber<sensor_msgs::Image> image_sub_2_;
    message_filters::Subscriber<sensor_msgs::Image> image_sub_3_;
    message_filters::Subscriber<sensor_msgs::Image> image_sub_4_;
    boost::shared_ptr<Sync1> sync1_;
    boost::shared_ptr<Sync2> sync2_;
    boost::shared_ptr<Sync3> sync3_;
    boost::shared_ptr<Sync4> sync4_;

    ros::Publisher lidar_pub_, pose_twist_pub_, image_pub1_, image_pub2_, image_pub3_, image_pub4_; 


    void callback_image1(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg, const nav_msgs::Odometry::ConstPtr& odom_msg, const sensor_msgs::Image::ConstPtr& image_msg1);
    void callback_image2(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg, const nav_msgs::Odometry::ConstPtr& odom_msg, const sensor_msgs::Image::ConstPtr& image_msg1, const sensor_msgs::Image::ConstPtr& image_msg2);
    void callback_image3(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg, const nav_msgs::Odometry::ConstPtr& odom_msg, const sensor_msgs::Image::ConstPtr& image_msg1, const sensor_msgs::Image::ConstPtr& image_msg2, const sensor_msgs::Image::ConstPtr& image_msg3);
    void callback_image4(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg, const nav_msgs::Odometry::ConstPtr& odom_msg, const sensor_msgs::Image::ConstPtr& image_msg1, const sensor_msgs::Image::ConstPtr& image_msg2, const sensor_msgs::Image::ConstPtr& image_msg3, const sensor_msgs::Image::ConstPtr& image_msg4);
    

};

#endif