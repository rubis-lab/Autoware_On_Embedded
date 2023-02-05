#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>
#include <rubis_msgs/PointCloud2.h>
#include <rubis_lib/sched.hpp>

static ros::Subscriber sub;
static ros::Publisher pub, pub_rubis;
int is_topic_ready = 1;

void points_cb(const sensor_msgs::PointCloud2ConstPtr& msg){
    rubis::start_task_profiling();

    sensor_msgs::PointCloud2 msg_with_intensity = *msg;
    
    msg_with_intensity.fields.at(3).datatype = 7;
    msg_with_intensity.header.stamp = ros::Time::now();

    pub.publish(msg_with_intensity);

    rubis_msgs::PointCloud2 rubis_msg_with_intensity;
    rubis_msg_with_intensity.instance = rubis::instance_;
    rubis_msg_with_intensity.msg = msg_with_intensity;
    pub_rubis.publish(rubis_msg_with_intensity);

    rubis::stop_task_profiling(rubis::instance_, 0);
    rubis::instance_ = rubis::instance_+1;
    rubis::obj_instance_ = rubis::obj_instance_+1;
}

std::string exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
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
    std::string rubis_output_topic;

    nh.param<std::string>(input_topic_name, input_topic, "/points_raw_origin");
    nh.param<std::string>(output_topic_name, output_topic, "/points_raw");

    sub = nh.subscribe(input_topic, 1, points_cb);      
    pub = nh.advertise<sensor_msgs::PointCloud2>(output_topic, 1);
    rubis_output_topic = "/rubis_"+output_topic.substr(1);
    pub_rubis = nh.advertise<rubis_msgs::PointCloud2>(rubis_output_topic, 1);
    
    // Scheduling & Profiling Setup
    std::string task_response_time_filename;
    private_nh.param<std::string>(node_name+"/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/lidar_republisher.csv");

    int rate;
    private_nh.param<int>(node_name+"/rate", rate, 10);

    struct rubis::sched_attr attr;
    std::string policy;
    int priority, exec_time ,deadline, period;

    private_nh.param(node_name+"/task_scheduling_configs/policy", policy, std::string("NONE"));    
    private_nh.param(node_name+"/task_scheduling_configs/priority", priority, 99);
    private_nh.param(node_name+"/task_scheduling_configs/exec_time", exec_time, 0);
    private_nh.param(node_name+"/task_scheduling_configs/deadline", deadline, 0);
    private_nh.param(node_name+"/task_scheduling_configs/period", period, 0);
    attr = rubis::create_sched_attr(priority, exec_time, deadline, period);    
    rubis::init_task_scheduling(policy, attr);

    rubis::init_task_profiling(task_response_time_filename);
    
    ros::spin();
    
    return 0;
}