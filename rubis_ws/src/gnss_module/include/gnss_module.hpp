#ifndef GNSS_MODULE_H
#define GNSS_MODULE_H

#include <ros/ros.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <rubis_msgs/InsStat.h>

#include <inertiallabs_msgs/ins_data.h>
#include <inertiallabs_msgs/gps_data.h>
#include <inertiallabs_msgs/sensor_data.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <rubis_msgs/InsStat.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include "LLH2UTM.h"
#include "LKF.h"
#include "quaternion_euler.h"
#include <eigen3/Eigen/Eigen>
#endif

#define M_PI 3.14159265358979323846

using namespace std;
using namespace message_filters;

class GnssModule{
public:
    GnssModule();
    void run();

private:
    void observation_cb(const inertiallabs_msgs::gps_data::ConstPtr &gps_msg, const inertiallabs_msgs::ins_data::ConstPtr &ins_msg, const inertiallabs_msgs::sensor_data::ConstPtr &sensor_msg);
    void run_kalman_filter(geometry_msgs::PoseStamped& pose, geometry_msgs::TwistStamped& twist, rubis_msgs::InsStat& ins_stat);
    void gps_data_cb(const inertiallabs_msgs::gps_data::ConstPtr &gps_msg);
    void ins_data_cb(const inertiallabs_msgs::ins_data::ConstPtr &ins_msg);
    void sensor_data_cb(const inertiallabs_msgs::sensor_data::ConstPtr &sensor_msg);
    void createTransformationMatrix(std::vector<double>& transformation_vec);

private:
    ros::NodeHandle nh_;
    ros::Publisher gnss_pose_pub_, ins_twist_pub_, ins_stat_pub_;
    ros::Subscriber gps_data_sub_, ins_data_sub_, sensor_data_sub_;
    ros::Time cur_time_;

    message_filters::Subscriber<inertiallabs_msgs::gps_data> gps_sync_sub_;
    message_filters::Subscriber<inertiallabs_msgs::ins_data> ins_sync_sub_;
    message_filters::Subscriber<inertiallabs_msgs::sensor_data> sensor_sync_sub_;
    typedef sync_policies::ExactTime<inertiallabs_msgs::gps_data, inertiallabs_msgs::ins_data, inertiallabs_msgs::sensor_data> SyncPolicy;
    typedef Synchronizer<SyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;

    tf::Transform tf_gnss_to_base_;
    geometry_msgs::PoseStamped gnss_pose_;
    geometry_msgs::TwistStamped ins_twist_;
    rubis_msgs::InsStat ins_stat_;

    double x_offset_;
    double y_offset_;
    double z_offset_;
    double yaw_offset_;
    bool debug_;
    bool use_kalman_filter_;
    bool use_gnss_tf_;
    bool use_sync_;
    LKF lkf_;

    Eigen::Matrix<double, 3, 3> T_;

    double time_diff_;
    bool is_updated_ = false;
};

