#include <ros/ros.h>
#include <ros/time.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

static ros::Subscriber sub;
static ros::Publisher pub;

void callbackGetInitialPose(const geometry_msgs::PoseWithCovarianceStamped& msg){
    geometry_msgs::PoseStamped m_InitialPose;
    m_InitialPose.pose.position.x = msg.pose.pose.position.x;
    m_InitialPose.pose.position.y = msg.pose.pose.position.y;
    m_InitialPose.pose.position.z = msg.pose.pose.position.z;
    m_InitialPose.pose.orientation.x = msg.pose.pose.orientation.x;
    m_InitialPose.pose.orientation.y = msg.pose.pose.orientation.y;
    m_InitialPose.pose.orientation.z = msg.pose.pose.orientation.z;
    m_InitialPose.pose.orientation.w = msg.pose.pose.orientation.w;
    m_InitialPose.header.frame_id = msg.header.frame_id;
    m_InitialPose.header.stamp = ros::Time::now();
    pub.publish(m_InitialPose);   
}


int main(int argc, char* argv[]){   
    ros::init(argc, argv, "fake_current_pose");
    ros::NodeHandle nh;
    sub = nh.subscribe("/initialpose", 1, callbackGetInitialPose);
    pub = nh.advertise<geometry_msgs::PoseStamped>("/current_pose",1);

    while(ros::ok())
        ros::spin();
    return 0;
}