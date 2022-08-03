#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <rubis_msgs/PoseStamped.h>
#include <rubis_msgs/TwistStamped.h>
#include <rubis_lib/sched.hpp>

std::string input_topic_pose_, rubis_input_topic_pose_, input_topic_velocity_, rubis_input_topic_velocity_;
ros::Subscriber rubis_pose_sub_, pose_sub_, rubis_velocity_sub_, velocity_sub_;
ros::Publisher rubis_pose_pub_, pose_pub_, rubis_velocity_pub_, velocity_pub_;
unsigned long pose_instance_, twist_instance_;

inline void relay_pose(const geometry_msgs::PoseStampedConstPtr& msg){
    if(rubis::instance_mode_ && rubis::instance_ != RUBIS_NO_INSTANCE){
        rubis_msgs::PoseStamped rubis_msg;
        rubis_msg.instance = rubis::instance_;
        rubis_msg.msg = *msg;
        rubis_pose_pub_.publish(rubis_msg);
    }
    pose_pub_.publish(msg);

    if(rubis::sched::is_task_ready_ == TASK_NOT_READY) rubis::sched::init_task();
    rubis::sched::task_state_ = TASK_STATE_DONE;
}

inline void relay_twist(const geometry_msgs::TwistStampedConstPtr& msg){
    if(rubis::instance_mode_ && rubis::instance_ != RUBIS_NO_INSTANCE){
        rubis_msgs::TwistStamped rubis_msg;
        rubis_msg.instance = rubis::instance_;
        rubis_msg.msg = *msg;
        rubis_velocity_pub_.publish(rubis_msg);
    }
    velocity_pub_.publish(msg);
}

void pose_cb(const geometry_msgs::PoseStampedConstPtr& msg){
    rubis::instance_ = RUBIS_NO_INSTANCE;
    relay_pose(msg);
}

void rubis_pose_cb(const rubis_msgs::PoseStampedConstPtr& _msg){
    geometry_msgs::PoseStampedConstPtr msg = boost::make_shared<const geometry_msgs::PoseStamped>(_msg->msg);
    rubis::instance_ = _msg->instance;
    pose_instance_ = _msg->instance;
    relay_pose(msg);
}

void twist_cb(const geometry_msgs::TwistStampedConstPtr& msg){
    twist_instance_ = RUBIS_NO_INSTANCE;
    relay_twist(msg);
}

void rubis_twist_cb(const rubis_msgs::TwistStampedConstPtr& _msg){
    geometry_msgs::TwistStampedConstPtr msg = boost::make_shared<const geometry_msgs::TwistStamped>(_msg->msg);
    twist_instance_ = _msg->instance;
    relay_twist(msg);
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "relay");  

    // Scheduling Setup
    int task_scheduling_flag;
    int task_profiling_flag;
    std::string task_response_time_filename;
    int rate;
    double task_minimum_inter_release_time;
    double task_execution_time;
    double task_relative_deadline; 

    ros::NodeHandle nh;

    nh.param<int>("/relay/task_scheduling_flag", task_scheduling_flag, 0);
    nh.param<int>("/relay/task_profiling_flag", task_profiling_flag, 0);
    nh.param<std::string>("/relay/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/relay.csv");
    nh.param<int>("/relay/rate", rate, 10);
    nh.param("/relay/task_minimum_inter_release_time", task_minimum_inter_release_time, (double)10);
    nh.param("/relay/task_execution_time", task_execution_time, (double)10);
    nh.param("/relay/task_relative_deadline", task_relative_deadline, (double)10);
    nh.param<int>("/relay/instance_mode", rubis::instance_mode_, 0);
    nh.param<int>("/infinite_spin_rate_mode", rubis::infinite_spin_rate_mode_, 0);

    input_topic_pose_ = "/ndt_pose";
    input_topic_velocity_ = "/estimate_twist";

    if(rubis::instance_mode_){
        rubis_input_topic_pose_ = "/rubis_"+input_topic_pose_.substr(1);
        rubis_input_topic_velocity_ = "/rubis_"+input_topic_velocity_.substr(1);
        rubis_pose_sub_ = nh.subscribe(rubis_input_topic_pose_, 10, rubis_pose_cb);
        rubis_velocity_sub_ = nh.subscribe(rubis_input_topic_velocity_, 10, rubis_twist_cb);
        rubis_pose_pub_ = nh.advertise<rubis_msgs::PoseStamped>("/rubis_current_pose", 10);
        rubis_velocity_pub_ = nh.advertise<rubis_msgs::TwistStamped>("/rubis_current_velocity", 10);
    }
    else{
        pose_sub_ = nh.subscribe(rubis_input_topic_pose_, 10, pose_cb);
        velocity_sub_ = nh.subscribe(rubis_input_topic_velocity_, 10, twist_cb);
    }

    pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/current_pose", 10);
    velocity_pub_ = nh.advertise<geometry_msgs::TwistStamped>("/current_velocity", 10);

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
        bool topic_ready_=false;

        // Executing task
        while(ros::ok()){
            if(rubis::infinite_spin_rate_mode_) {
                //wait until 'ndt_matching' publish
                while(!topic_ready_){
                    nh.getParam("/ndt_pub_", topic_ready_);
                }
                nh.setParam("/ndt_pub_", false);
                topic_ready_=false;
            }
            if(task_profiling_flag) rubis::sched::start_task_profiling();

            if(rubis::sched::task_state_ == TASK_STATE_READY){                
                if(task_scheduling_flag) rubis::sched::request_task_scheduling(task_minimum_inter_release_time, task_execution_time, task_relative_deadline); 
                rubis::sched::task_state_ = TASK_STATE_RUNNING;     
            }

            ros::spinOnce();
            if(pose_instance_ != twist_instance_){
                ros::Rate ms(10000);
                ms.sleep();
                continue;
            }

            if(task_profiling_flag) rubis::sched::stop_task_profiling(rubis::instance_, rubis::sched::task_state_);

            if(rubis::sched::task_state_ == TASK_STATE_DONE){
                if(task_scheduling_flag) rubis::sched::yield_task_scheduling();
                rubis::sched::task_state_ = TASK_STATE_READY;
            }
            if(rubis::infinite_spin_rate_mode_){
                nh.setParam("/relay_pub_", true);
            }
            else{
                r.sleep();
            }
        }
    }

    return 0;
}