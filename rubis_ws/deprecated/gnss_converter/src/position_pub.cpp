#include <ros/ros.h>
#include <gnss_converter/LLH2UTM.h>

#include <geometry_msgs/PoseStamped.h>
#include <inertiallabs_msgs/gps_data.h>
#include <inertiallabs_msgs/ins_data.h>

#define M_PI 3.14159265358979323846

static ros::Publisher gnss_pose_pub_;
static geometry_msgs::PoseStamped gnss_pose_;
static double x_offset_, y_offset_, z_offset_, yaw_offset_;
static double roll_, pitch_, yaw_;

void inertial_gps_cb(const inertiallabs_msgs::gps_data::ConstPtr &msg_gps){
    gnss_pose_.header = msg_gps->header;
    gnss_pose_.header.frame_id = "/map";

    /* coordinate transform (LLH2 to UTM) */ 
    LLH2UTM(msg_gps->LLH.x, msg_gps->LLH.y, msg_gps->LLH.z, gnss_pose_);

    /* position offset calculation */ 
    gnss_pose_.pose.position.x = gnss_pose_.pose.position.x - x_offset_;
    gnss_pose_.pose.position.y = gnss_pose_.pose.position.y - y_offset_;
    gnss_pose_.pose.position.z = gnss_pose_.pose.position.z - z_offset_;

    gnss_pose_pub_.publish(gnss_pose_);
}

int main(int argc, char *argv[]){
    ros::init(argc, argv, "position_pub");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    private_nh.param("x_offset", x_offset_, 0.0);
    private_nh.param("y_offset", y_offset_, 0.0);
    private_nh.param("z_offset", z_offset_, 0.0);

    ros::Subscriber inertial_gps_sub;
    inertial_gps_sub = nh.subscribe("/Inertial_Labs/gps_data", 10, inertial_gps_cb);

    gnss_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/gnss_pose", 10);

    ros::spin();

    // ros::Rate rate(10);

    // while(nh.ok()){
    //     ros::spinOnce();
    //     rate.sleep();
    // }
}
