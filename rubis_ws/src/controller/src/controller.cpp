#include <ros/ros.h>
#include "autoware_msgs/ControlCommandStamped.h"
#include "geometry_msgs/TwistStamped.h"

#define DEBUG

#ifdef DEBUG
#include "autoware_msgs/VehicleCmd.h"
#endif

static float dt_;
static float kp_, ki_, kd_;
static float max_acc_, min_acc_;
static float current_velocity_; // m/s
static float set_point_, process_variable_, output_; // m/s^2
ros::Subscriber sub_ctrl_cmd_;

#ifdef DEBUG
ros::Publisher debug_pub_vehicle_cmd_;
static float debug_steering_angle_;

ros::Subscriber debug_sub_twist_cmd_;
geometry_msgs::TwistStamped debug_twist_;
#endif

inline double kmph2mps(double kmph){
    return (kmph * 1000.0 / 60.0 / 60.0); 
}

inline double mps2kmph(double mps)
{
  return (mps * 3.6);
}

inline void calculate_linear_acceleration(){
    float err = set_point_ - process_variable_;
    output_ = kp_ * err + ki_* err * dt_ + kd_ * err / dt_;
    if(output_ > max_acc_) output_ = max_acc_;
    else if (output_ < min_acc_) output_ = min_acc_;

    return;
}

inline void send_control_signal(){
    calculate_linear_acceleration();

    // TODO: Send control signal to interface
    #ifdef DEBUG
    autoware_msgs::VehicleCmd msg;
    msg.ctrl_cmd.linear_velocity = current_velocity_;
    msg.ctrl_cmd.linear_acceleration = output_;
    msg.ctrl_cmd.steering_angle = debug_steering_angle_;
    msg.twist_cmd = debug_twist_;
    msg.twist_cmd.twist.linear.x += output_ * dt_;
    debug_pub_vehicle_cmd_.publish(msg);
    #endif
}

void ctrl_callback(const autoware_msgs::ControlCommandStampedConstPtr& msg){
    set_point_ = msg->cmd.linear_acceleration;

    // TODO: Get process_variable from the Vehicle
    #ifdef DEBUG
    static float _prev_velocity = 0.0;
    current_velocity_ = msg->cmd.linear_velocity; // TODO: Change to IonicautoRealSPEED
    process_variable_ = (current_velocity_ - _prev_velocity)/100;    // TODO: Change to IonicautoControlAx
                                                                    // When subscription rate is 10hz
    _prev_velocity = current_velocity_;
    debug_steering_angle_ = msg->cmd.steering_angle;
    #endif

    return;
}


#ifdef DEBUG
void debug_twist_cmd_callback(const geometry_msgs::TwistStampedConstPtr& msg){
    debug_twist_.twist = msg->twist;
}
#endif

int main(int argc, char* argv[]){
    ros::init(argc, argv, "controller");
    ros::NodeHandle nh;

    nh.param("/controller/max_acc", max_acc_, (float)1.0);
    nh.param("/controller/min_acc", min_acc_, (float)-3.0);

    // PID parameters
    nh.param("/controller/kp", kp_, (float)0.5);
    nh.param("/controller/ki", ki_, (float)0.5);
    nh.param("/controller/kd", kd_, (float)0.001);

    nh.param("/controller/dt", dt_, (float)10.0); //ms

    sub_ctrl_cmd_ = nh.subscribe("/ctrl_cmd", 1, ctrl_callback);

    #ifdef DEBUG
    debug_pub_vehicle_cmd_ = nh.advertise<autoware_msgs::VehicleCmd>("/vehicle_cmd_test", 1);
    debug_sub_twist_cmd_ = nh.subscribe("/twist_cmd", 1, debug_twist_cmd_callback);
    #endif

    ros::Rate rate(1000/dt_);

    // Initialization
    set_point_ = 0;
    process_variable_ = 0;
    output_ = 0;

    while(ros::ok()){
        send_control_signal();
        ros::spinOnce();
        rate.sleep();
    }

    // TODO: Steering control



    return 0;
}