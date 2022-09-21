#include <ros/ros.h>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <autoware_msgs/NDTStat.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

static std::string _baselink_frame;

static std::vector<std::string> pose_topic_list;
static std::vector<std::string> stat_topic_list;

static std::vector<double> score_list;
static std::vector<geometry_msgs::Pose> pose_list;

static std::vector<ros::Subscriber> pose_sub_list;
static std::vector<ros::Subscriber> stat_sub_list;

void pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg, int idx){
    pose_list.at(idx).position.x = msg->pose.position.x;
    pose_list.at(idx).position.y = msg->pose.position.y;
    pose_list.at(idx).position.z = msg->pose.position.z;
    pose_list.at(idx).orientation.x = msg->pose.orientation.x;
    pose_list.at(idx).orientation.y = msg->pose.orientation.y;
    pose_list.at(idx).orientation.z = msg->pose.orientation.z;
    pose_list.at(idx).orientation.w = msg->pose.orientation.w;
}

void stat_callback(const autoware_msgs::NDTStat::ConstPtr& msg, int idx){
    // msg->iteration
    score_list.at(idx) = msg->score;
}

void subscriber_init(ros::NodeHandle nh)
{
    for(int i=0; i<pose_topic_list.size(); i++){
        geometry_msgs::Pose p;
        pose_list.push_back(p);
        ros::Subscriber pose_sub = nh.subscribe<geometry_msgs::PoseStamped>(pose_topic_list.at(i), 10, boost::bind(pose_callback, _1, i));
        pose_sub_list.push_back(pose_sub);

        double sc;
        score_list.push_back(sc);
        ros::Subscriber stat_sub = nh.subscribe<autoware_msgs::NDTStat>(stat_topic_list.at(i), 10, boost::bind(stat_callback, _1, i));
        stat_sub_list.push_back(stat_sub);
    }
}

void publishTransform(void){
    // TODO: select pose number
    // static tf::TransformBroadcaster br;
    // tf::Transform transform;
    // transform.setOrigin(tf::Vector3(current_pose.x, current_pose.y, current_pose.z));
    // transform.setRotation(current_q);
    // br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/map", _baselink_frame));
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "multi_ndt_combiner");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    private_nh.getParam("pose_topics", pose_topic_list);
    private_nh.getParam("stat_topics", stat_topic_list);

    subscriber_init(nh);

    private_nh.param<std::string>("baselink_frame", _baselink_frame, std::string("base_link"));

    ros::Rate ros_rate(10);
    while(ros::ok()){
        ros::spinOnce();
        ros_rate.sleep();
    }
}