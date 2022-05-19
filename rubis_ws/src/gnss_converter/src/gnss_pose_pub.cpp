#include <gnss_converter/LLH2UTM.h>

#include <geometry_msgs/PoseStamped.h>
#include <inertiallabs_msgs/gps_data.h>
#include <inertiallabs_msgs/ins_data.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#define M_PI 3.14159265358979323846

using namespace std;
using namespace message_filters;

typedef sync_policies::ExactTime<inertiallabs_msgs::gps_data, inertiallabs_msgs::ins_data> SyncPolicy_;

static ros::Publisher gnss_pose_pub_;
static geometry_msgs::PoseStamped gnss_pose_;
static double x_offset_, y_offset_, z_offset_, yaw_offset_;
static double roll_, pitch_, yaw_;

void gnss_pose_pub_cb(const inertiallabs_msgs::gps_data::ConstPtr &msg_gps, const inertiallabs_msgs::ins_data::ConstPtr &msg_ins){
    gnss_pose_.header = msg_gps->header;
    gnss_pose_.header.frame_id = "/map";

    /* coordinate transform (LLH2 to UTM) */ 
    LLH2UTM(msg_gps->LLH.x, msg_gps->LLH.y, msg_gps->LLH.z, gnss_pose_);

    /* position offset calculation */ 
    gnss_pose_.pose.position.x = gnss_pose_.pose.position.x - x_offset_;
    gnss_pose_.pose.position.y = gnss_pose_.pose.position.y - y_offset_;
    gnss_pose_.pose.position.z = gnss_pose_.pose.position.z - z_offset_;

    /* orientation */ 
    roll_ = msg_ins->YPR.z;
    pitch_ = msg_ins->YPR.y;

    /* yaw offset calculation */
    yaw_ = -1 * (msg_ins->YPR.x) - yaw_offset_;
    yaw_ = (yaw_ > 180.0) ? (yaw_ - 360) : ((yaw_ < -180) ? (yaw_ + 360) : yaw_);
    
    /* unit conversion */ 
    roll_ = roll_ * M_PI/180.0; pitch_ = pitch_ * M_PI/180.0; yaw_ = yaw_ * M_PI/180.0;
}

int main(int argc, char *argv[]){
    ros::init(argc, argv, "gnss_pose_pub");

    ros::NodeHandle nh;

    nh.param("/gnss_pose_pub/x_offset", x_offset_, 0.0);
    nh.param("/gnss_pose_pub/y_offset", y_offset_, 0.0);
    nh.param("/gnss_pose_pub/z_offset", z_offset_, 0.0);
    nh.param("/ins_twist_generator/yaw_offset", yaw_offset_, 0.0);

    gnss_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/ndt_pose", 2);

    message_filters::Subscriber<inertiallabs_msgs::gps_data> gps_sub(nh, "/Inertial_Labs/gps_data", 2);
    message_filters::Subscriber<inertiallabs_msgs::ins_data> ins_sub(nh, "/Inertial_Labs/ins_data", 2);
    
    Synchronizer<SyncPolicy_> sync(SyncPolicy_(2), gps_sub, ins_sub);

    sync.registerCallback(boost::bind(&gnss_pose_pub_cb, _1, _2));

    ros::Rate rate(10);

    tf::TransformListener listener;
    tf::TransformBroadcaster broadcaster;
    tf::StampedTransform transform;
    tf::Transform tf_gnss_to_base, tf_map_to_gnss, tf_map_to_base;
    tf::Quaternion q;

    while(nh.ok()){
        try{
            /* lookup /gnss to /base_link static transform */ 
            listener.lookupTransform("/gnss", "/base_link", ros::Time(0), transform);
            tf_gnss_to_base = transform;
            break;
        }
        catch (tf::TransformException ex){
            ROS_ERROR("%s", ex.what());
            rate.sleep();
        }
    }

    while(nh.ok()){
        ros::spinOnce();
        
        /* /map to /gnss tf */ 
        tf_map_to_gnss.setOrigin(tf::Vector3(gnss_pose_.pose.position.x, gnss_pose_.pose.position.y, gnss_pose_.pose.position.z));
        q.setRPY(roll_, pitch_, yaw_);
        tf_map_to_gnss.setRotation(q);

        /* /map to /base tf calculation */ 
        tf_map_to_base = tf_map_to_gnss * tf_gnss_to_base;

        /* msg */  
        gnss_pose_.pose.position.x = tf_map_to_base.getOrigin().x();
        gnss_pose_.pose.position.y = tf_map_to_base.getOrigin().y();
        gnss_pose_.pose.position.z = tf_map_to_base.getOrigin().z();

        gnss_pose_.pose.orientation.w = tf_map_to_base.getRotation().w();
        gnss_pose_.pose.orientation.x = tf_map_to_base.getRotation().x();
        gnss_pose_.pose.orientation.y = tf_map_to_base.getRotation().y();
        gnss_pose_.pose.orientation.z = tf_map_to_base.getRotation().z();

        /* broadcast & publish */ 
        broadcaster.sendTransform(tf::StampedTransform(tf_map_to_base, ros::Time::now(), "/map", "/base_link"));
        gnss_pose_pub_.publish(gnss_pose_); 

        rate.sleep();
    }
}
