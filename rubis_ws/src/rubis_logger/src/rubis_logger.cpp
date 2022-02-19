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
static int u_stamp_recent_;
static int time_small_unit_;
static int logger_rate_;

static vector<ros::Subscriber> sub_topics_;
static map<string, string> log_topics_;
static map<string, int> log_topics_ok_;

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

void writelogheader() {
    u_stamp_recent_ = u_timestamp();
    
    string log_instance = "";
    log_instance += ("-------------------------------------------------------\n");
    log_instance += "start_time: ";
    log_instance += to_string(u_stamp_recent_);
    log_instance += "\n";
    log_instance += "number_target_topics: ";
    log_instance += to_string(target_topics_.size());
    log_instance += "\n";
    for(int i=0; i<target_topics_.size(); i++) {
        log_instance += "topic: ";
        log_instance += target_topics_[i];
        log_instance += "\n";
    }

    log_instance += ("-------------------------------------------------------\n");
    if(logfile_.is_open()) {
        logfile_ << log_instance;
    }
    return;
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
    
    if(u_timestamp() == u_stamp_recent_) {
        time_small_unit_ += 1;
    } else {
        time_small_unit_ = 0;
    }
    u_stamp_recent_ = u_timestamp();
    
    string log_instance = "";
    log_instance += ("-------------------------------------------------------\n");
    log_instance += "time: ";
    log_instance += to_string(u_stamp_recent_);
    log_instance += "-";
    log_instance += to_string(time_small_unit_);
    log_instance += "\n";
    
    for(int i=0; i<target_topics_.size(); i++) {
        if(!log_topics_[target_topics_[i]].compare(""))
            return;
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

    string log_topic = "";

    log_topic += "target_topic: /car_ctrl_output\n";
    log_topic += "car_ctrl_output.real_speed: " + to_string(msg->real_speed) + "\n";       //m/s
    log_topic += "car_ctrl_output.steering_angle: " + to_string(msg->steering_angle) + "\n";       // radian

    log_topics_["/car_ctrl_output"] = log_topic;
    
}

// ros -> car
void sub_car_ctrl_input(const can_data_msgs::Car_ctrl_input::ConstPtr& msg) {
    
    string log_topic = "";

    log_topic += "target_topic: /car_ctrl_input\n";
    log_topic += "car_ctrl_input.acceleration: " + to_string(msg->acceleration) + "\n";       //m/s
    log_topic += "car_ctrl_input.steering_angle: " + to_string(msg->steering_angle) + "\n";       // radian

    log_topics_["/car_ctrl_input"] = log_topic;
}

void sub_rubis_log_handler(const rubis_logger_msgs::rubis_log_handler::ConstPtr& msg) {
    
    string log_topic = "";

    log_topic += "target_topic: /rubis_log_handler\n";
    log_topic += "rubis_log_handler.writeon: " + to_string(msg->writeon) + "\n";       //1 0
    // log_topic += "rubis_log_handler.teststart: " + to_string(msg->teststart) + "\n";       //1 0
    // log_topic += "rubis_log_handler.testend: " + to_string(msg->testend) + "\n";       //1 0

    log_topics_["/rubis_log_handler"] = log_topic;
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "rubis_logger");
    ros::NodeHandle nh;

    nh.param("target_topics", target_topics_, vector<string>());
    nh.param("paramtest", paramtest_, (int)1010);
    nh.param("logdirpath", logdirpath_, (string)"");
    nh.param("logger_rate", logger_rate_, (int)10);

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
            sub_topics_.push_back(nh.subscribe(target_topics_[i], 1, sub_car_ctrl_output));
            log_topics_["/car_ctrl_output"] = "";
        } else if(!target_topics_[i].compare("/car_ctrl_input")) {
            sub_topics_.push_back(nh.subscribe(target_topics_[i], 1, sub_car_ctrl_input));
            log_topics_["/car_ctrl_input"] = "";
        } else if(!target_topics_[i].compare("/rubis_log_handler")) {
            sub_topics_.push_back(nh.subscribe(target_topics_[i], 1, sub_rubis_log_handler));
            log_topics_["/rubis_log_handler"] = "";
        } else {
            printf("not available topic\n");
        }
    }

    writelogheader();

    ros::Rate rate(logger_rate_);

    while(ros::ok()){

        debugcallback();
        writelogcallback();
        ros::spinOnce();
        rate.sleep();
    }

    // TODO: Steering control

    return 0;
}