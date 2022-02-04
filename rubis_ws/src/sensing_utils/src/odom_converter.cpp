#include <ros/ros.h>
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/TwistStamped.h"

float linear_velocity_ = 0;
ros::Publisher pub_;
ros::Subscriber sub_;

void odom_callback(const nav_msgs::OdometryConstPtr& msg){
    geometry_msgs::TwistStamped twist_msg;
    twist_msg.header = msg->header;
    twist_msg.twist = msg->twist.twist;

    pub_.publish(twist_msg);
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "odom_converter");
    ros::NodeHandle nh;

    sub_ = nh.subscribe("/odom", 1, odom_callback);
    pub_ = nh.advertise<geometry_msgs::TwistStamped>("odom_twist", 1);

    ros::Rate rate(12.5);
    while(ros::ok()){
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}