#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>

static ros::Subscriber sub;
static ros::Publisher pub;

void points_cb(const sensor_msgs::PointCloud2& msg){
    sensor_msgs::PointCloud2 msg_with_intensity = msg;
    msg_with_intensity.fields.at(3).datatype = 7;

    msg_with_intensity.header.stamp = ros::Time::now();
    pub.publish(msg_with_intensity);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "lidar_republisher");
    ros::NodeHandle nh;
    std::string input_topic;
    std::string output_topic;

    std::string node_name = ros::this_node::getName();
    std::string input_topic_name = node_name + "/input_topic";
    std::string output_topic_name = node_name + "/output_topic";

    nh.param<std::string>(input_topic_name, input_topic, "/points_raw_origin");
    nh.param<std::string>(output_topic_name, output_topic, "/points_raw");

    sub = nh.subscribe(input_topic, 1, points_cb);    
    pub = nh.advertise<sensor_msgs::PointCloud2>(output_topic, 1);

    while(ros::ok())
        ros::spin();
    
    return 0;
}