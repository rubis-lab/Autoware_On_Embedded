#include <ros/ros.h>
#include <vector>
#include "inertiallabs_msgs/ins_data.h"
#include "inertiallabs_msgs/sensor_data.h"
#include "inertiallabs_msgs/gps_data.h"
#include <tf/transform_listener.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>

#define M_PI           3.14159265358979323846

double ins_yaw_default = 0, ins_yaw_modified = 0, ins_yaw_offset = 0, ndt_yaw = 0;
bool is_done = 0;

void ins_callback(const inertiallabs_msgs::ins_dataConstPtr msg){
    double roll, pitch, yaw;

    roll = msg->YPR.z;
    pitch = msg->YPR.y;
    yaw = msg->YPR.x;
    ins_yaw_default = yaw;
    std::cout<<"# INS RPY(default): "<<roll<<" "<<pitch<<" "<<yaw<<std::endl;

    yaw *= -1;
    if(yaw > 180.0) yaw -= 360.0;
    if(yaw < -180.0) yaw += 360.0;
    ins_yaw_modified = yaw;
    std::cout<<"# INS RPY(modified): "<<roll<<" "<<pitch<<" "<<yaw<<std::endl;

    double velocity;
    velocity = sqrt(pow(msg->Vel_ENU.x,2) + pow(msg->Vel_ENU.y,2));
    // std::cout<<"# INS Vel: "<<velocity<<" "<<msg->Vel_ENU.x<<" "<<msg->Vel_ENU.y<<std::endl;
    
    yaw += 88.9;
    if(yaw > 180.0) yaw -= 360.0;
    if(yaw < -180.0) yaw += 360.0;
    ins_yaw_offset = yaw;
    std::cout<<"# INS RPY(offset): "<<roll<<" "<<pitch<<" "<<yaw + 88.9<<std::endl<<std::endl;
    
    
}

void sensor_callback(const inertiallabs_msgs::sensor_dataConstPtr msg){
    double acc_x = msg->Accel.x * 9.80665 * 0.000001;
    double acc_y = msg->Accel.y * 9.80665 * 0.000001;
    double acc_z = msg->Accel.z * 9.80665 * 0.000001;
    

    // std::cout<<"# Accel: "<<sqrt(pow(acc_x,2)+pow(acc_y,2))<<" "<<acc_x<<" "<<acc_y<<" "<<acc_z<<std::endl;
}


void ndt_callback(const geometry_msgs::PoseStampedConstPtr msg){
    static double prev_pose_x;
    static double prev_pose_y;
    static double prev_t;
    static bool is_init = false;
    if(!is_init){
        prev_pose_x = msg->pose.position.x;
        prev_pose_y = msg->pose.position.y;
        prev_t = msg->header.stamp.sec + msg->header.stamp.nsec * 0.000000001;
        is_init = true;
        return;
    }

    tf::Quaternion q(   msg->pose.orientation.x, 
                        msg->pose.orientation.y, 
                        msg->pose.orientation.z, 
                        msg->pose.orientation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    roll *= 180/M_PI;
    pitch *= 180/M_PI;
    yaw *= 180/M_PI;
    std::cout<<"# NDT RPY: "<<roll<<" "<<pitch<<" "<<yaw<<std::endl;
    
    ndt_yaw = yaw;

    double cur_pose_x = msg->pose.position.x;
    double cur_pose_y = msg->pose.position.y;
    double cur_t = msg->header.stamp.sec + msg->header.stamp.nsec * 0.000000001;
    
    double theta_t = cur_t - prev_t;
    double theta_x = cur_pose_x - prev_pose_x;
    double theta_y = cur_pose_y - prev_pose_y;
    double vel_x = theta_x * theta_t;
    double vel_y = theta_y * theta_t;

    double velocity;
    velocity = sqrt(pow(vel_x,2) + pow(vel_y,2)) * 3.6;
    // std::cout<<"# NDT Vel: "<<velocity<<" "<<vel_x<<" "<<vel_y<<" "<<std::endl;

    prev_pose_x = cur_pose_x;
    prev_pose_y = cur_pose_y;
    prev_t = cur_t;
}

void twist_callback(const geometry_msgs::TwistStampedConstPtr msg){
    double velocity;
    velocity = sqrt(pow(msg->twist.linear.x,2) + pow(msg->twist.linear.y,2) + pow(msg->twist.linear.z,2)) * 3.6;

    // std::cout<<"## NDT Vel: "<<velocity<<" "<<msg->twist.linear.x<<" "<<msg->twist.linear.y<<" "<<msg->twist.linear.z<<std::endl;

    return;
}

void gps_callback(const inertiallabs_msgs::gps_dataConstPtr msg){
    
    double velocity = msg->HorSpeed * 3.6;

    // std::cout<<"## HorSpeed: "<<velocity<<std::endl;

    return;
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "ins_sync");
    ros::NodeHandle nh;

    ros::Subscriber sub1  = nh.subscribe("/Inertial_Labs/ins_data", 1, ins_callback);    
    // ros::Subscriber sub2  = nh.subscribe("/ndt_pose", 1, ndt_callback);
    // ros::Subscriber sub3  = nh.subscribe("/estimate_twist", 1, twist_callback);
    ros::Subscriber sub4  = nh.subscribe("/Inertial_Labs/sensor_data", 1, sensor_callback);
    ros::Subscriber sub5  = nh.subscribe("/Inertial_Labs/gps_data", 1, gps_callback);
    
    tf::TransformListener listener;

    int cnt = 0, n = 100;
    std::vector<int> yaw_diff_vec;

    ros::Publisher ndt_yaw_pub = nh.advertise<geometry_msgs::PoseStamped>("/ndt_yaw", 1);

    ros::Rate rate(10);
    while(nh.ok()){
        tf::StampedTransform tf;
        try{
            listener.lookupTransform("/map", "/base_link", ros::Time(0), tf);
            auto q = tf.getRotation();
            tf::Matrix3x3 m(q);
            double tf_roll, tf_pitch, tf_yaw;
            m.getRPY(tf_roll, tf_pitch, tf_yaw);

            tf_roll *= 180/M_PI;
            tf_pitch *= 180/M_PI;
            tf_yaw *= 180/M_PI;
            
            std::cout<<"## TF RPY: "<<tf_roll<<" "<<tf_pitch<<" "<<tf_yaw<<std::endl;            
            std::cout<<"## ins yaw default - tf yaw: "<<ins_yaw_default-tf_yaw<<std::endl;
            std::cout<<"## ins yaw modified - tf yaw: "<<ins_yaw_modified-tf_yaw<<std::endl;
            std::cout<<"## ins yaw offset - tf yaw: "<<ins_yaw_offset-tf_yaw<<std::endl<<std::endl;            

            geometry_msgs::PoseStamped msg;
            msg.pose.position.x = tf_yaw;
            ndt_yaw_pub.publish(msg);

            double yaw_diff = ins_yaw_default - tf_yaw;    
        }
        catch(tf::TransformException ex){

        }
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}