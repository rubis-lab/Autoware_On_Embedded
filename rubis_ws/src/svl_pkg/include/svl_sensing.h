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
#include <nav_msgs/Odometry.h>
#include <rubis_msgs/PoseTwistStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> SyncPolicy;
typedef message_filters::Synchronizer<SyncPolicy> Sync;

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
    boost::shared_ptr<Sync> sync_;

    ros::Publisher lidar_pub_, pose_twist_pub_; 


    void callback(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg, const nav_msgs::Odometry::ConstPtr& odom_msg);
    

};

#endif