#include <ros/ros.h>

#include <geometry_msgs/PoseStamped.h>
#include <autoware_msgs/NDTStat.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

static std::string _baselink_frame;
static std::vector<double> score_list;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "multi_ndt_combiner");

    std::vector<std::string> pose_topic_list;
    std::vector<std::string> stat_topic_list;
    std::vector<ros::Subscriber> pose_sub_list;
    std::vector<ros::Subscriber> stat_sub_list;

    ros::NodeHandle private_nh("~");

    private_nh.param<std::string>("baselink_frame", _baselink_frame, std::string("base_link"));
    private_nh.getParam("pose_topics", pose_topic_list);
    private_nh.getParam("stat_topics", stat_topic_list);

    // static tf::TransformBroadcaster br;
    // tf::Transform transform;
    // transform.setOrigin(tf::Vector3(current_pose.x, current_pose.y, current_pose.z));
    // transform.setRotation(current_q);
    // br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/map", _baselink_frame));

    ros::Rate ros_rate(10);
    while(ros::ok()){
        ros::spinOnce();
        ros_rate.sleep();
    }
}