#include <ros/ros.h>
#include <ros_autorunner_lib/ros_autorunner.h>

// Include subscribe message type
#include <sensor_msgs/PointCloud2.h>
#include <autoware_msgs/NDTStat.h>
#include <autoware_msgs/DetectedObjectArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#define SLEEP_PERIOD 1

class CubetownAutorunner : public AutorunnerBase{
private:    
    ros::NodeHandle     nh_;
    ROSAutorunner       ros_autorunner_;
private:
    virtual void register_subscribers();
private:
    void points_raw_cb(const sensor_msgs::PointCloud2& msg);
    void ndt_pose_cb(const geometry_msgs::PoseStamped& msg);
    void detection_cb(const autoware_msgs::DetectedObjectArray& msg);
    void behavior_state_cb(const visualization_msgs::MarkerArray& msg);

public:
    Sub_v               sub_v_;
    ros::Publisher      initial_pose_pub_;
public:
    CubetownAutorunner() {}
    CubetownAutorunner(ros::NodeHandle nh) : nh_(nh){}
    virtual void Run();
};
