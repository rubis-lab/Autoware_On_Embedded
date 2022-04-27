#include <ros/ros.h>
#include <vector>
#include "inertiallabs_msgs/ins_data.h"
#include <tf/transform_listener.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>

#define M_PI           3.14159265358979323846
// #define DEBUG

double ins_yaw = 0, ndt_yaw = 0;
bool is_done = 0;

void ins_callback(const inertiallabs_msgs::ins_dataConstPtr msg){
    double roll, pitch, yaw;

    roll = msg->YPR.z;
    pitch = msg->YPR.y;
    yaw = msg->YPR.x;
    
    yaw *= -1;
    if(yaw > 180.0) yaw -= 360.0;
    if(yaw < -180.0) yaw += 360.0;

    #ifdef DEBUG
        std::cout<<"# INS RPY: "<<roll<<" "<<pitch<<" "<<yaw<<std::endl;
    #endif

    double velocity;
    velocity = sqrt(pow(msg->Vel_ENU.x,2) + pow(msg->Vel_ENU.y,2) + pow(msg->Vel_ENU.z,2)) * 3.6;

    #ifdef DEBUG
        std::cout<<"# INS Vel: "<<velocity<<" "<<msg->Vel_ENU.x<<" "<<msg->Vel_ENU.y<<" "<<msg->Vel_ENU.z<<std::endl;
    #endif
    ins_yaw = yaw;
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "yaw_offset_calculator");
    ros::NodeHandle nh;

    ros::Subscriber sub1  = nh.subscribe("/Inertial_Labs/ins_data", 1, ins_callback);
    
    tf::TransformListener listener;

    int cnt = 0, n = 50;
    double prev_yaw_diff = 0;
    std::vector<double> yaw_diff_vec;

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

            #ifdef DEBUG
                std::cout<<"## TF RPY: "<<tf_roll<<" "<<tf_pitch<<" "<<tf_yaw<<std::endl;            
                std::cout<<"## ins yaw - tf yaw: "<<ins_yaw-tf_yaw<<std::endl<<std::endl;
            #endif

            double yaw_diff = ins_yaw - tf_yaw;
            
            if(cnt++ > 10){
                yaw_diff_vec.push_back(yaw_diff);
                if(prev_yaw_diff - yaw_diff > 5.0){
                    std::cout<<"[ERROR] NDT matching fail!"<<std::endl;
                    exit(1);
                }
            }            

            if(yaw_diff_vec.size() == n){
                double avg = 0.0;
                for(int i = 0; i < yaw_diff_vec.size(); i++)
                    avg += yaw_diff_vec[i];
                avg = avg / (double)(yaw_diff_vec.size());
                std::cout<<"[RESULT] Yaw offset: "<<avg<<std::endl;
                break;
            }

            prev_yaw_diff = yaw_diff;
        }
        catch(tf::TransformException ex){

        }
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}