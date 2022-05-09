#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>
#include <rubis_msgs/PointCloud2.h>
#include <rubis_lib/sched.hpp>

static ros::Subscriber sub;
static ros::Publisher pub, pub_rubis;
int is_topic_ready = 1;

void points_cb(const sensor_msgs::PointCloud2ConstPtr& msg){
    sensor_msgs::PointCloud2 msg_with_intensity = *msg;
    
    msg_with_intensity.fields.at(3).datatype = 7;
    msg_with_intensity.header.stamp = ros::Time::now();

    pub.publish(msg_with_intensity);

    if(rubis::instance_mode_){
        rubis_msgs::PointCloud2 rubis_msg_with_intensity;
        rubis_msg_with_intensity.instance = rubis::instance_;
        rubis_msg_with_intensity.msg = msg_with_intensity;
        rubis::instance_ = rubis::instance_+1;
        pub_rubis.publish(rubis_msg_with_intensity);
    }

    if(rubis::sched::is_task_ready_ == TASK_NOT_READY) rubis::sched::init_task();
    rubis::sched::task_state_ = TASK_STATE_DONE;
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

    nh.param<int>(node_name+"/instance_mode", rubis::instance_mode_, 0);
    nh.param<std::string>(input_topic_name, input_topic, "/points_raw_origin");
    nh.param<std::string>(output_topic_name, output_topic, "/points_raw");

    sub = nh.subscribe(input_topic, 1, points_cb);      
    pub = nh.advertise<sensor_msgs::PointCloud2>(output_topic, 1);
    if(rubis::instance_mode_){
        rubis_output_topic = "/rubis_"+output_topic.substr(1);
        pub_rubis = nh.advertise<rubis_msgs::PointCloud2>(rubis_output_topic, 1);
    }

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
    if(task_profiling_flag) rubis::sched::init_task_profiling(task_response_time_filename);

    if(!task_scheduling_flag && !task_profiling_flag){
        ros::spin();
    }
    else{
        ros::Rate r(rate);
        // Initialize task ( Wait until first necessary topic is published )
        while(ros::ok()){
            if(rubis::sched::is_task_ready_ == TASK_READY) break;
            ros::spinOnce();
            r.sleep();      
        }

        // Executing task
        while(ros::ok()){
            if(task_profiling_flag) rubis::sched::start_task_profiling();

            if(rubis::sched::task_state_ == TASK_STATE_READY){                
                if(task_scheduling_flag) rubis::sched::request_task_scheduling(task_minimum_inter_release_time, task_execution_time, task_relative_deadline); 
                rubis::sched::task_state_ = TASK_STATE_RUNNING;     
            }

            ros::spinOnce();

            if(task_profiling_flag) rubis::sched::stop_task_profiling(rubis::instance_, rubis::sched::task_state_);

            if(rubis::sched::task_state_ == TASK_STATE_DONE){
                if(task_scheduling_flag) rubis::sched::yield_task_scheduling();
                rubis::sched::task_state_ = TASK_STATE_READY;
            }
        
            r.sleep();
        }
    }
    
    return 0;
}