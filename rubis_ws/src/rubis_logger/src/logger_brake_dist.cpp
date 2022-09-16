#include "rubis_logger/logger_brake_dist.h"
// #define DEBUG

using std::getenv;
using std::string;
using std::vector;
using std::ofstream;
using std::to_string;
using std::time;
using std::endl;
using std::cout;
using std::map;

static std_msgs::Header header_;
static vector<string> target_topics_;
static string dist_filename_;
static ofstream logfile_;
static int paramtest_;
static int log_start_;
static int u_stamp_recent_;
static int time_small_unit_;
static int rate_;


static double walker_pose_x_;
static double walker_pose_y_;
static double walker_pose_z_;

static double vehicle_pose_x_;
static double vehicle_pose_y_;
static double vehicle_pose_z_;

static double distance_;
static double vehicle_offset_;
struct timespec time_now_;
static long long time_sec_;
static long long time_nsec_;
static double time_msec_;

// static vector<ros::Subscriber> sub_topics_;
// static map<string, string> log_topics_;
// static map<string, int> log_topics_ok_;

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
    // printf("%ld\n", time_now);      //int, unix timestamp
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
    return;
}

void walker_pose_callback(const nav_msgs::Odometry::ConstPtr& msg) {

    walker_pose_x_ = msg->pose.pose.position.x;
    walker_pose_y_ = msg->pose.pose.position.y;
    walker_pose_z_ = msg->pose.pose.position.z;
    // cout << "walker_pose_callback" << endl;

    return;
}

void vehicle_pose_callback(const nav_msgs::Odometry::ConstPtr& msg) {

    vehicle_pose_x_ = msg->pose.pose.position.x;
    vehicle_pose_y_ = msg->pose.pose.position.y;
    vehicle_pose_z_ = msg->pose.pose.position.z;
    // cout << "vehicle_pose_callback" << endl;

    return;
}

void calculate_dist() {

    // cout << "walker pose x,y,z: " << walker_pose_x_ << ", " << walker_pose_y_ << ", " << walker_pose_z_ << endl;
    // cout << "vehicle pose x,y,z: " << vehicle_pose_x_ << ", " << vehicle_pose_y_ << ", " << vehicle_pose_z_ << endl;
    double delta_x = (walker_pose_x_ - vehicle_pose_x_);
    double delta_y = (walker_pose_y_ - vehicle_pose_y_);
    distance_ = sqrt(pow(delta_x, 2) + pow(delta_y, 2)) - 2.5;
    return;
}

void get_ctime() {
    clock_gettime(CLOCK_MONOTONIC, &time_now_);
    time_sec_ = (long long)time_now_.tv_sec;
    time_nsec_ = (long long)time_now_.tv_nsec;
}

void write_log_callback() {
    get_ctime();
    calculate_dist();
    // cout << "time: " << time_sec_ << "." << time_nsec_ << endl << "dist: " << distance_ << endl;

    return;
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "logger_brake_dist");
    ros::NodeHandle nh;

    int u_ts = u_timestamp();
    string ts = YYYYMMDDHHmmSS();

    nh.param("dist_filename", dist_filename_, (string)"");
    nh.param("rate", rate_, (int)10);
    nh.param("vehicle_offset", vehicle_offset_, (double)2.5);

    ros::Subscriber sub_walker_pose = nh.subscribe("/carla/walker/odometry", 1, walker_pose_callback);
    ros::Subscriber sub_vehicle_pose = nh.subscribe("/carla/ego_vehicle/odometry", 1, vehicle_pose_callback);

    if(!dist_filename_.empty()) {    
        // printf("%ld\n\n\n\n", time_now.tv_sec);
        logfile_.open(dist_filename_, std::ios_base::app);
    }
    
    printf("dist_filename_: %s\n", dist_filename_.c_str());

    // writelogheader();
    ros::Rate rate(rate_);

    while(ros::ok()){
        debugcallback();
        write_log_callback();
        ros::spinOnce();
        rate.sleep();
    }

    // TODO: Steering control

    return 0;
}