#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>

static ros::Subscriber sub;
static ros::Publisher pub;
int is_topic_ready = 0;

void points_cb(const sensor_msgs::PointCloud2& msg){
    sensor_msgs::PointCloud2 msg_with_intensity = msg;
    msg_with_intensity.fields.at(3).datatype = 7;

    msg_with_intensity.header.stamp = ros::Time::now();
    pub.publish(msg_with_intensity);
    if(!is_topic_ready) is_topic_ready = 1;
}

int main(int argc, char** argv){
    ros::init(argc, argv, "lidar_republisher");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    std::string input_topic;
    std::string output_topic;

    std::string node_name = ros::this_node::getName();
    std::string input_topic_name = node_name + "/input_topic";
    std::string output_topic_name = node_name + "/output_topic";

    nh.param<std::string>(input_topic_name, input_topic, "/points_raw_origin");
    nh.param<std::string>(output_topic_name, output_topic, "/points_raw");

    sub = nh.subscribe(input_topic, 1, points_cb);    
    pub = nh.advertise<sensor_msgs::PointCloud2>(output_topic, 1);

    // Scheduling Setup
    int task_scheduling_flag;
    int task_profiling_flag;
    std::string task_response_time_filename;
    int rate;
    double task_minimum_inter_release_time;
    double task_execution_time;
    double task_relative_deadline; 

    private_nh.param<int>("/lidar_republisher/task_scheduling_flag", task_scheduling_flag, 0);
    private_nh.param<int>("/lidar_republisher/task_profiling_flag", task_profiling_flag, 0);
    private_nh.param<std::string>("/lidar_republisher/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/lidar_republisher.csv");
    private_nh.param<int>("/lidar_republisher/rate", rate, 10);
    private_nh.param("/lidar_republisher/task_minimum_inter_release_time", task_minimum_inter_release_time, (double)10);
    private_nh.param("/lidar_republisher/task_execution_time", task_execution_time, (double)10);
    private_nh.param("/lidar_republisher/task_relative_deadline", task_relative_deadline, (double)10);

    /* For Task scheduling */
    if(!task_scheduling_flag && !task_profiling_flag){
        ros::spin();
    }
    else{
        
        ros::Rate r(rate);    
        while(ros::ok()){
            ros::spinOnce();
            r.sleep();
        }
    }
    
    return 0;
}