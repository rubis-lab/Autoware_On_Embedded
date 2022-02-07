#include "rubis_logger/rubis_logger.h"
// #define DEBUG

// static float dt_;
static std_msgs::Header header_;
static std::vector<std::string> target_topics_;
static int paramtest_;

inline double kmph2mps(double kmph){
    return (kmph * 1000.0 / 60.0 / 60.0); 
}

inline double mps2kmph(double mps)
{
    return (mps * 3.6);
}

//수정.
void usefullcallback() {
    // printf("sfsfsf\n");
    for(int i=0; i<target_topics_.size(); i++) {
        printf("topic name: %s\n", target_topics_[i].c_str());
    }

    return;
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "rubis_logger");
    ros::NodeHandle nh;

    nh.param("/rubis_logger/target_topics", target_topics_, std::vector<std::string>());
    nh.param("paramtest", paramtest_, (int)1010);

    printf("%d\n %d\n", target_topics_.size(), paramtest_);
    // PID parameters
    // nh.param("/controller/kp", kp_, (float)0.5);
    // nh.param("/controller/ki", ki_, (float)0.5);
    // nh.param("/controller/kd", kd_, (float)0.001);

    // nh.param("/controller/dt", dt_, (float)10.0); //ms

    // svl_pub_vehicle_cmd_ = nh.advertise<autoware_msgs::VehicleCmd>("/vehicle_cmd_test", 1);
    // svl_sub_twist_cmd_ = nh.subscribe("/odom", 1, svl_twist_cmd_callback);

    ros::Rate rate(1);

    while(ros::ok()){

        usefullcallback();
        ros::spinOnce();
        rate.sleep();
    }

    // TODO: Steering control

    return 0;
}