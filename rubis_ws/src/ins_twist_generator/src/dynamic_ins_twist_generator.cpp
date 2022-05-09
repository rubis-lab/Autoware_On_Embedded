#include <ros/ros.h>
#include <vector>
#include "inertiallabs_msgs/ins_data.h"
#include "inertiallabs_msgs/gps_data.h"
#include "inertiallabs_msgs/sensor_data.h"
#include <tf/transform_listener.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <rubis_msgs/InsStat.h>
#include <tf/transform_listener.h>

#define M_PI           3.14159265358979323846
#define DEBUG

ros::Time cur_time_;
double yaw_, yaw_offset_, linear_velocity_, angular_velocity_, linear_acceleration_;
double yaw_diff_threshold_ = 5.0;

bool calculate_yaw_offset(double yaw_diff, double& yaw_offset){
    static bool is_running = false;
    static int n = 30; // 3 seconds
    static std::vector<double> yaw_diff_vec;
    static double prev_yaw_diff;

    if(!is_running){
        prev_yaw_diff = yaw_diff;
        is_running = true;
        return false;
    }

    if(abs(yaw_diff - prev_yaw_diff) > yaw_diff_threshold_){
        yaw_diff_vec.clear();
    }
    else{
        yaw_diff_vec.push_back(yaw_diff);
    }

    if(yaw_diff_vec.size() == n){
        double avg = 0.0;
        for(int i = 0; i < yaw_diff_vec.size(); i++)
            avg += yaw_diff_vec[i];
        avg = avg / (double)(yaw_diff_vec.size());
        yaw_offset = avg;
        yaw_diff_vec.clear();
        is_running = false;
        return true;
    }

    prev_yaw_diff = yaw_diff;
    return false;
}

void ins_callback(const inertiallabs_msgs::ins_dataConstPtr msg){
    
    static bool yaw_offset_calculation_flag = false;
    double ins_roll, ins_pitch, ins_yaw;

    ins_roll = msg->YPR.z;
    ins_pitch = msg->YPR.y;    
    ins_yaw = msg->YPR.x;

    ins_yaw *= -1;
    yaw_ = ins_yaw;
    yaw_ -= yaw_offset_;
    if(ins_yaw > 180.0) ins_yaw -= 360.0;
    if(ins_yaw < -180.0) ins_yaw += 360.0;
    if(yaw_ > 180.0) yaw_ -= 360.0;
    if(yaw_ < -180.0) yaw_ += 360.0;

    double tf_yaw, current_yaw_diff;
    
    static tf::StampedTransform tf;
    static tf::TransformListener listener;
    try{
        listener.lookupTransform("/map", "/base_link", ros::Time(0), tf);
        auto q = tf.getRotation();
        tf::Matrix3x3 m(q);
        double tf_roll, tf_pitch, tf_yaw;
        m.getRPY(tf_roll, tf_pitch, tf_yaw);

        tf_roll *= 180/M_PI;
        tf_pitch *= 180/M_PI;
        tf_yaw *= 180/M_PI;

        current_yaw_diff = yaw_ - tf_yaw;
        #ifdef DEBUG
            std::cout<<"Current yaw diff: "<<current_yaw_diff<<std::endl;
        #endif     
        if(!yaw_offset_calculation_flag && current_yaw_diff > yaw_diff_threshold_){
            yaw_offset_calculation_flag = true;
            #ifdef DEBUG
                std::cout<<" -----> Calculate new yaw offset"<<std::endl;
            #endif        
        }
    }
    catch(tf::TransformException ex){}

    if(yaw_offset_calculation_flag){
        double new_yaw_offset = 0.0;
        bool is_finished = calculate_yaw_offset(ins_yaw - tf_yaw, new_yaw_offset);
        if(is_finished){
            yaw_offset_calculation_flag = false;
            yaw_offset_ = new_yaw_offset;
            #ifdef DEBUG
                std::cout<<"New yaw offset: "<<yaw_offset_<<std::endl;
            #endif        
        }
    }

}

void gps_callback(const inertiallabs_msgs::gps_dataConstPtr msg){
    static ros::Time prev_time = cur_time_;
    static double prev_velocity = linear_velocity_;
    
    cur_time_ = msg->header.stamp;
    linear_velocity_ = msg->HorSpeed; // m/s    

    double diff_time = (cur_time_ - prev_time).toSec();
    double diff_velocity = linear_velocity_ - prev_velocity;
    
    if(diff_time == 0.0) return;

    linear_acceleration_ = diff_velocity/diff_time;

    prev_time = cur_time_;
    prev_velocity = linear_velocity_;
}

void sensor_callback(const inertiallabs_msgs::sensor_dataConstPtr msg){
    angular_velocity_ = msg->Gyro.z; 
    angular_velocity_ = angular_velocity_ * M_PI/180; // rad/s
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "dynamic_ins_twist_generator");
    ros::NodeHandle nh;

    ros::Subscriber ins_sub = nh.subscribe("/Inertial_Labs/ins_data", 1, ins_callback);
    ros::Subscriber sensor_sub = nh.subscribe("/Inertial_Labs/sensor_data", 1, sensor_callback);
    ros::Subscriber gps_sub = nh.subscribe("/Inertial_Labs/gps_data", 1, gps_callback);
    ros::Publisher ins_twist_pub = nh.advertise<geometry_msgs::TwistStamped>("/ins_twist", 1);
    ros::Publisher ins_stat_pub = nh.advertise<rubis_msgs::InsStat>("/ins_stat", 1);

    nh.param("/dynamic_ins_twist_generator/yaw_offset", yaw_offset_, 0.0);
    nh.param("/dynamic_ins_twist_generator/yaw_diff_threshold", yaw_diff_threshold_, 5.0);

    cur_time_ = ros::Time::now();
    ros::Rate rate(10);

    while(nh.ok()){
        ros::spinOnce();

        geometry_msgs::TwistStamped ins_twist_msg;

        ins_twist_msg.header.stamp = ros::Time::now();
        ins_twist_msg.header.frame_id = "/base_link";

        ins_twist_msg.twist.linear.x = linear_velocity_;
        ins_twist_msg.twist.linear.y = 0.0;
        ins_twist_msg.twist.linear.z = 0.0;

        ins_twist_msg.twist.angular.x = 0.0;
        ins_twist_msg.twist.angular.y = 0.0;
        ins_twist_msg.twist.angular.z = angular_velocity_;

        ins_twist_pub.publish(ins_twist_msg);

        rubis_msgs::InsStat ins_stat_msg;

        ins_stat_msg.header = ins_twist_msg.header;

        ins_stat_msg.vel_x = linear_velocity_ * cos(yaw_ * M_PI/180.0);
        ins_stat_msg.vel_y = linear_velocity_ * sin(yaw_ * M_PI/180.0);
        ins_stat_msg.acc_x = linear_acceleration_ * cos(yaw_ * M_PI/180.0);
        ins_stat_msg.acc_y = linear_acceleration_ * sin(yaw_ * M_PI/180.0);
        ins_stat_msg.linear_velocity = linear_velocity_;
        ins_stat_msg.linear_acceleration = linear_acceleration_;
        ins_stat_msg.angular_velocity = angular_velocity_;
        ins_stat_msg.yaw = yaw_;

        ins_stat_pub.publish(ins_stat_msg);

        rate.sleep();
    }

    return 0;
}