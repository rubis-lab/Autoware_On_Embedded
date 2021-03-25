#include <ros/ros.h>
#include <ros/time.h>
#include <nmea_msgs/Sentence.h>

static ros::Subscriber sub;
static ros::Publisher pub;

void gnss_cb(const nmea_msgs::Sentence& msg){
    nmea_msgs::Sentence out;
    out = msg;
    out.header.stamp = ros::Time::now();
    pub.publish(out);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "gnss_republisher");
    ros::NodeHandle nh;    
    std::string input_topic;
    std::string output_topic;

    std::string node_name = ros::this_node::getName();
    std::string input_topic_name = node_name + "/input_topic";
    std::string output_topic_name = node_name + "/output_topic";

    nh.param<std::string>(input_topic_name, input_topic, "/nmea_sentence_origin");
    nh.param<std::string>(output_topic_name, output_topic, "/nmea_sentence");

    sub = nh.subscribe(input_topic, 1, gnss_cb);
    pub = nh.advertise<nmea_msgs::Sentence>(output_topic, 1);    

    while(ros::ok())
        ros::spin();
    
    return 0;
}