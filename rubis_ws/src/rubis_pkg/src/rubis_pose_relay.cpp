#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <rubis_msgs/PoseStamped.h>
#include <rubis_lib/sched.hpp>

std::string input_topic_, rubis_input_topic_;
ros::Subscriber rubis_sub_, sub_;
ros::Publisher rubis_pub_, pub_;

inline void relay(const geometry_msgs::PoseStampedConstPtr& msg){
    rubis_msgs::PoseStamped rubis_msg;
    rubis_msg.instance = rubis::instance_;
    rubis_msg.msg = *msg;
    rubis_pub_.publish(rubis_msg);
    pub_.publish(msg);
}

void cb(const geometry_msgs::PoseStampedConstPtr& msg){
    rubis::instance_ = 0;
    relay(msg);
}

void rubis_cb(const rubis_msgs::PoseStampedConstPtr& _msg){
    geometry_msgs::PoseStampedConstPtr msg = boost::make_shared<const geometry_msgs::PoseStamped>(_msg->msg);
    rubis::instance_ = _msg->instance;
    relay(msg);
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "pose_relay");  

    // Scheduling Setup
    std::string task_response_time_filename;
    int rate;
    double task_minimum_inter_release_time;
    double task_execution_time;
    double task_relative_deadline; 

    ros::NodeHandle nh;

    nh.param<std::string>("/pose_relay/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/pose_relay.csv");
    nh.param<int>("/pose_relay/rate", rate, 10);
    nh.param("/pose_relay/task_minimum_inter_release_time", task_minimum_inter_release_time, (double)10);
    nh.param("/pose_relay/task_execution_time", task_execution_time, (double)10);
    nh.param("/pose_relay/task_relative_deadline", task_relative_deadline, (double)10);

    input_topic_ = std::string(argv[1]);

    std::cout<<"!!! input topic  "<<input_topic_<<std::endl;

    rubis_input_topic_ = "/rubis_"+input_topic_.substr(1);
    rubis_sub_ = nh.subscribe(rubis_input_topic_, 10, rubis_cb);
    rubis_pub_ = nh.advertise<rubis_msgs::PoseStamped>("/rubis_current_pose", 10);

    pub_ = nh.advertise<geometry_msgs::PoseStamped>("/current_pose", 10);

    /* For Task scheduling */
    rubis::init_task_profiling(task_response_time_filename);

    ros::Rate r(rate);
    while(ros::ok()){
        rubis::start_task_profiling();

        ros::spinOnce();

        rubis::stop_task_profiling(rubis::instance_, 0);

        r.sleep();
    }

    return 0;
}