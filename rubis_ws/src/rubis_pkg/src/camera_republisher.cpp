#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/Image.h>

static ros::Subscriber sub;
static ros::Publisher pub;

void camera_cb(const sensor_msgs::Image& msg){
    sensor_msgs::Image out;
    out = msg;
    out.header.stamp = ros::Time::now();
    pub.publish(out);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "lidar_republisher");
    ros::NodeHandle nh;    
    std::string input_topic;
    nh.param<std::string>("/camera_republisher/input_topic", input_topic, "/image_raw_origin");

    sub = nh.subscribe(input_topic, 1, camera_cb);
    pub = nh.advertise<sensor_msgs::Image>("/image_raw", 1);    

    while(ros::ok())
        ros::spin();
    
    return 0;
}