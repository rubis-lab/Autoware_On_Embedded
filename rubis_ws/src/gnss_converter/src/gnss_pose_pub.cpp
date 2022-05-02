#include <gnss_converter/LLH2UTM.h>

#include <geometry_msgs/PoseStamped.h>
#include <inertiallabs_msgs/gps_data.h>
#include <inertiallabs_msgs/ins_data.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

using namespace std;
using namespace message_filters;

typedef sync_policies::ExactTime<inertiallabs_msgs::gps_data, inertiallabs_msgs::ins_data> SyncPolicy_;

static ros::Publisher gnss_pose_pub_;

void gnss_pose_pub_cb(const inertiallabs_msgs::gps_data::ConstPtr &msg_gps, const inertiallabs_msgs::ins_data::ConstPtr &msg_ins){
    geometry_msgs::PoseStamped gnss_pose;

    gnss_pose.header = msg_gps->header;
    gnss_pose.header.frame_id = "/map";

    LLH2UTM(msg_gps->LLH.x, msg_gps->LLH.y, msg_gps->LLH.z, gnss_pose);

    gnss_pose.pose.position.x = gnss_pose.pose.position.x;
    gnss_pose.pose.position.y = gnss_pose.pose.position.y;

    gnss_pose_pub_.publish(gnss_pose);
}

int main(int argc, char *argv[]){
    ros::init(argc, argv, "gnss_pose_pub");

    ros::NodeHandle nh;

    gnss_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/gnss_pose", 2);

    message_filters::Subscriber<inertiallabs_msgs::gps_data> gps_sub(nh, "/Inertial_Labs/gps_data", 2);
    message_filters::Subscriber<inertiallabs_msgs::ins_data> ins_sub(nh, "/Inertial_Labs/ins_data", 2);
    
    Synchronizer<SyncPolicy_> sync(SyncPolicy_(2), gps_sub, ins_sub);

    sync.registerCallback(boost::bind(&gnss_pose_pub_cb, _1, _2));

    ros::spin();
}