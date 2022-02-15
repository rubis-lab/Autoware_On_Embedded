#include "rubis_logger/rubis_logger.h"
// #define DEBUG

using std::getenv;
using std::string;
using std::vector;
using std::ofstream;
using std::to_string;
using std::time;
using std::endl;
using std::map;

static std_msgs::Header header_;
static vector<string> target_topics_;
static string logdirpath_;
static string logfilepath_;
static ofstream logfile_;
static int paramtest_;
static int log_start_;

static vector<ros::Subscriber> sub_topics_;
static map<string, string> log_topics_;

string YYYYMMDDHHmmSS() {
    time_t time_now;
    struct tm *tm;
    time(&time_now);
    tm=localtime(&time_now);

    string ret = "";
    ret += to_string(tm->tm_year+1900);
    
    int mon = tm->tm_mon+1;
    if(mon < 10)
        ret += to_string(0);
    ret += to_string(mon);
    
    int day = tm->tm_mday;
    if(day < 10) 
        ret += to_string(0);
    ret += to_string(day);

    int hour = tm->tm_hour;
    if(hour < 10) 
        ret += to_string(0);
    ret += to_string(hour);

    int min = tm->tm_min;
    if(min < 10) 
        ret += to_string(0);
    ret += to_string(min);

    int sec = tm->tm_sec;
    if(sec < 10) 
        ret += to_string(0);
    ret += to_string(sec);

    printf("%s\n", ret.c_str());
    return ret;
}

int u_timestamp() {
    time_t time_now;
    time(&time_now);
    printf("%ld\n", time_now);      //int, unix timestamp
    return time_now;
}

void debugcallback() {
    printf("log topics: ");
    for(int i=0; i<target_topics_.size(); i++) {
        printf("%s ", target_topics_[i].c_str());
    }
    printf("\n");
    return;
}

void writelogcallback() {
    string log_instance = "";
    log_instance += ("-------------------------------------------------------\n");
    log_instance += "time: ";
    log_instance += to_string(u_timestamp());
    log_instance += "\n";
    
    for(int i=0; i<target_topics_.size(); i++) {
        log_instance += log_topics_[target_topics_[i]];
    }
    log_instance += ("-------------------------------------------------------\n");
    if(logfile_.is_open()) {
        logfile_ << log_instance;
    }

    return;
}


void sub_ctrl_cmd(const autoware_msgs::ControlCommandStampedConstPtr& msg) {
    
    string log_topic = "";

    log_topic += "target_topic: /ctrl_cmd\n";
    log_topic += "ctrl_cmd.cmd.linear_velocity: " + to_string(msg->cmd.linear_velocity) + "\n";
    log_topic += "ctrl_cmd.cmd.steering_angle: " + to_string(msg->cmd.steering_angle) + "\n";

    log_topics_["/ctrl_cmd"] = log_topic;
}

//SVL
void sub_odom(const nav_msgs::Odometry::ConstPtr& msg) {

    string log_topic = "";

    log_topic += "target_topic: /odom\n";
    log_topic += "odom.twist.twist.linear.x: " + to_string(msg->twist.twist.linear.x) + "\n";       //m/s
    log_topic += "odom.twist.twist.angular.z: " + to_string(msg->twist.twist.angular.z) + "\n";     //radian

    log_topics_["/odom"] = log_topic;
}

void sub_vehicle_cmd_test(const autoware_msgs::VehicleCmd::ConstPtr& msg) {

    string log_topic = "";

    log_topic += "target_topic: /vehicle_cmd_test\n";
    log_topic += "vehicle_cmd_test.ctrl_cmd.linear_acceleration: " + to_string(msg->ctrl_cmd.linear_acceleration) + "\n";       //m/s
    log_topic += "vehicle_cmd_test.ctrl_cmd.steering_angle: " + to_string(msg->ctrl_cmd.steering_angle) + "\n";       //m/s

    log_topics_["/vehicle_cmd_test"] = log_topic;
}

//IONIC
// car -> ros
void sub_car_ctrl_output(const can_data_msgs::Car_ctrl_output::ConstPtr& msg) {
    log_topics_["/car_ctrl_output"] = "";
}

// ros -> car
void sub_car_ctrl_input(const can_data_msgs::Car_ctrl_input::ConstPtr& msg) {
    log_topics_["/car_ctrl_input"] = "";

}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "rubis_logger");
    ros::NodeHandle nh;

    nh.param("/rubis_logger/target_topics", target_topics_, vector<string>());
    nh.param("paramtest", paramtest_, (int)1010);
    nh.param("logdirpath", logdirpath_, (string)"");

    int u_ts = u_timestamp();
    string ts = YYYYMMDDHHmmSS();

    if(!logdirpath_.empty()) {    
        // printf("%ld\n\n\n\n", time_now.tv_sec);
        logfilepath_ = logdirpath_ + ts + ".log";
        logfile_.open(logfilepath_, std::ios_base::app);
    }
    
    printf("logdirpath: %s\n", logdirpath_.c_str());
    printf("logfilepath: %s\n", logfilepath_.c_str());
    // printf("%d\n %d\n", target_topics_.size(), paramtest_);

//   - "/odom"
//   - "/vehicle_cmd_test"
//   - "/ctrl_cmd"

    // svl_pub_vehicle_cmd_ = nh.advertise<autoware_msgs::VehicleCmd>("/vehicle_cmd_test", 1);
    for(int i=0; i<target_topics_.size(); i++) {
        if(!target_topics_[i].compare("/ctrl_cmd")) {
            sub_topics_.push_back(nh.subscribe(target_topics_[i], 1, sub_ctrl_cmd));
            log_topics_["/ctrl_cmd"] = "";
        } else if(!target_topics_[i].compare("/odom")) {
            sub_topics_.push_back(nh.subscribe(target_topics_[i], 1, sub_odom));
            log_topics_["/odom"] = "";
        } else if(!target_topics_[i].compare("/vehicle_cmd_test")) {
            sub_topics_.push_back(nh.subscribe(target_topics_[i], 1, sub_vehicle_cmd_test));
            log_topics_["/vehicle_cmd_test"] = "";
        } else if(!target_topics_[i].compare("/car_ctrl_output")) {
            sub_topics_.push_back(nh.subscribe(target_topics_[i], 1, sub_car_ctrl_input));
            log_topics_["/car_ctrl_output"] = "";
        } else if(!target_topics_[i].compare("/car_ctrl_input")) {
            sub_topics_.push_back(nh.subscribe(target_topics_[i], 1, sub_car_ctrl_output));
            log_topics_["/car_ctrl_input"] = "";
        } else {
            printf("not available topic\n");
        }
    }

    ros::Rate rate(10);

    while(ros::ok()){

        debugcallback();
        writelogcallback();
        ros::spinOnce();
        rate.sleep();
    }

    // TODO: Steering control

    return 0;
}