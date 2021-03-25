#include <ros/ros.h>
#include <autoware_msgs/VehicleCmd.h>

static ros::Subscriber sub;
static ros::Publisher pub;

void ndt_cb(const autoware_msgs::VehicleCmd& msg){
    sensor_msgs::Image out;
    out = msg;
    out.header.stamp = ros::Time::now();
    pub.publish(out);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "lkas_combiner");
    ros::NodeHandle nh;

    sub = nh.subscribe("/vehicle_cmd_ndt", 1, camera_cb);
    pub = nh.advertise<autoware_msgs::VehicleCmd>("/vehicle_cmd", 1);
    

    while(ros::ok())
        ros::spin();
    
    return 0;
}