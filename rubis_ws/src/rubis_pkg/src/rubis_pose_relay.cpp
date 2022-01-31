#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <rubis_msgs/PoseStamped.h>
#include <rubis_lib/sched.hpp>

std::string input_topic_, rubis_input_topic_;
ros::Subscriber rubis_sub_, sub_;
ros::Publisher rubis_pub_, pub_;

inline void relay(const geometry_msgs::PoseStampedConstPtr& msg){
    if(rubis::instance_mode_ && rubis::instance_ != RUBIS_NO_INSTANCE){
        rubis_msgs::PoseStamped rubis_msg;
        rubis_msg.instance = rubis::instance_;
        rubis_msg.msg = *msg;
        rubis_pub_.publish(rubis_msg);
    }
    pub_.publish(msg);

    if(rubis::sched::is_task_ready_ == TASK_NOT_READY) rubis::sched::init_task();
    rubis::sched::task_state_ = TASK_STATE_DONE;
}

void cb(const geometry_msgs::PoseStampedConstPtr& msg){
    rubis::instance_ = RUBIS_NO_INSTANCE;
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
    int task_scheduling_flag;
    int task_profiling_flag;
    std::string task_response_time_filename;
    int rate;
    double task_minimum_inter_release_time;
    double task_execution_time;
    double task_relative_deadline; 

    ros::NodeHandle nh;

    nh.param<int>("/pose_relay/task_scheduling_flag", task_scheduling_flag, 0);
    nh.param<int>("/pose_relay/task_profiling_flag", task_profiling_flag, 0);
    nh.param<std::string>("/pose_relay/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/pose_relay.csv");
    nh.param<int>("/pose_relay/rate", rate, 10);
    nh.param("/pose_relay/task_minimum_inter_release_time", task_minimum_inter_release_time, (double)10);
    nh.param("/pose_relay/task_execution_time", task_execution_time, (double)10);
    nh.param("/pose_relay/task_relative_deadline", task_relative_deadline, (double)10);
    nh.param<int>("/pose_relay/instance_mode", rubis::instance_mode_, 0);

    input_topic_ = std::string(argv[1]);

    std::cout<<"!!! input topic  "<<input_topic_<<std::endl;

    if(rubis::instance_mode_){
        rubis_input_topic_ = "/rubis_"+input_topic_.substr(1);
        rubis_sub_ = nh.subscribe(rubis_input_topic_, 10, rubis_cb);
        rubis_pub_ = nh.advertise<rubis_msgs::PoseStamped>("/rubis_current_pose", 10);
    }
    else{
        rubis_sub_ = nh.subscribe(input_topic_, 10, cb);
    }

    pub_ = nh.advertise<geometry_msgs::PoseStamped>("/current_pose", 10);

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