#include <gnss_converter/LLH2UTM.h>
#include <gnss_converter/quaternion_euler.h>

#include <geometry_msgs/PoseStamped.h>
#include <inertiallabs_msgs/gps_data.h>
#include <inertiallabs_msgs/ins_data.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#define M_PI 3.14159265358979323846

using namespace std;
using namespace message_filters;

typedef sync_policies::ExactTime<inertiallabs_msgs::gps_data, inertiallabs_msgs::ins_data> SyncPolicy_;

static ros::Publisher gnss_pose_pub_;
static double x_offset_, y_offset_, yaw_offset_;

void gnss_pose_pub_cb(const inertiallabs_msgs::gps_data::ConstPtr &msg_gps, const inertiallabs_msgs::ins_data::ConstPtr &msg_ins){
    static tf::TransformBroadcaster br;

    geometry_msgs::PoseStamped gnss_pose;

    gnss_pose.header = msg_gps->header;
    gnss_pose.header.frame_id = "/map";

    LLH2UTM(msg_gps->LLH.x, msg_gps->LLH.y, msg_gps->LLH.z, gnss_pose);

    gnss_pose.pose.position.x = gnss_pose.pose.position.x - x_offset_;
    gnss_pose.pose.position.y = gnss_pose.pose.position.y - y_offset_;

    double roll, pitch, yaw;

    roll = msg_ins->YPR.z;
    pitch = msg_ins->YPR.y;
    
    yaw = msg_ins->YPR.x;
    yaw *= -1;
    yaw -= yaw_offset_;
    if(yaw > 180.0) yaw -= 360.0;
    if(yaw < -180.0) yaw += 360.0;

    roll = roll * M_PI/180.0;
    pitch = pitch * M_PI/180.0;
    yaw = yaw * M_PI/180.0;

    ToQuaternion(yaw, pitch, roll, gnss_pose.pose.orientation);

    tf::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    
    tf::StampedTransform transform;
    transform.setOrigin(tf::Vector3(gnss_pose.pose.position.x, gnss_pose.pose.position.y, 0.0));
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, msg_gps->header.stamp, "/map", "/base_link"));

    gnss_pose_pub_.publish(gnss_pose);
}

int main(int argc, char *argv[]){
    ros::init(argc, argv, "gnss_pose_pub");

    ros::NodeHandle nh;

    nh.param("/gnss_pose_pub/x_offset", x_offset_, 0.0);
    nh.param("/gnss_pose_pub/y_offset", y_offset_, 0.0);
    nh.param("/ins_twist_generator/yaw_offset", yaw_offset_, 0.0);

    gnss_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/ndt_pose", 2);

    message_filters::Subscriber<inertiallabs_msgs::gps_data> gps_sub(nh, "/Inertial_Labs/gps_data", 2);
    message_filters::Subscriber<inertiallabs_msgs::ins_data> ins_sub(nh, "/Inertial_Labs/ins_data", 2);
    
    Synchronizer<SyncPolicy_> sync(SyncPolicy_(2), gps_sub, ins_sub);

    sync.registerCallback(boost::bind(&gnss_pose_pub_cb, _1, _2));

    ros::spin();
}