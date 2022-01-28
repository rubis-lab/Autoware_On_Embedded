#include <iostream>
#include <angles/angles.h>
#include <cmath>
#include <vector>

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <inertiallabs_msgs/gps_data.h>
#include <inertiallabs_msgs/ins_data.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>

#include <eigen3/Eigen/Eigen>

#include <tf/LinearMath/Quaternion.h>
#include <tf/LinearMath/Transform.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

#include <opencv2/opencv.hpp>

#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>

using namespace std;
using namespace message_filters;
using namespace Eigen;

#define ENTER_BUTTON 10
#define ESC_BUTTON 27

struct gps_stat
{
    std_msgs::Header header;
    geometry_msgs::Point gps_pose;
    geometry_msgs::Vector3 gps_ypr;
    geometry_msgs::Point ndt_pose;
    geometry_msgs::Vector3 ndt_ypr;
    double ndt_score;
};

typedef sync_policies::ApproximateTime<inertiallabs_msgs::gps_data, inertiallabs_msgs::ins_data, geometry_msgs::PoseStamped> SyncPolicy_1;
typedef sync_policies::ExactTime<inertiallabs_msgs::gps_data, inertiallabs_msgs::ins_data> SyncPolicy_2;

static Matrix<double, 4, 4> pos_tf_, ori_tf_;

static ros::Publisher gnss_pose_pub_;

static vector<gps_stat> gps_backup_;
static int ndt_pose_x_max_ = -9999999, ndt_pose_y_max_ = -9999999;
static int ndt_pose_x_min_ = 9999999, ndt_pose_y_min_ = 9999999;
static int scale_factor_ = 10;

static int points_idx_;
static gps_stat selected_points_[4];

void gps_ndt_data_cb(const inertiallabs_msgs::gps_data::ConstPtr &msg_gps, const inertiallabs_msgs::ins_data::ConstPtr &msg_ins,
                     const geometry_msgs::PoseStamped::ConstPtr &msg_ndt_pose);
void pub_gnss_pose_cb(const inertiallabs_msgs::gps_data::ConstPtr &msg_gps, const inertiallabs_msgs::ins_data::ConstPtr &msg_ins);
void track_bar_cb(int pos, void *userdata);
void mouse_cb(int event, int x, int y, int flags, void *userdata);
void points_select();
void calculate_tf_matrix();