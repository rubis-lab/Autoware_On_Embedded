#include <iostream>
#include <angles/angles.h>
#include <cmath>

#include <ros/ros.h>
#include <inertiallabs_msgs/gps_data.h>
#include <inertiallabs_msgs/ins_data.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>

#include <eigen3/Eigen/Eigen>

#include <tf/LinearMath/Quaternion.h>
#include <tf/LinearMath/Transform.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

using namespace message_filters;
using namespace Eigen;

typedef sync_policies::ApproximateTime<inertiallabs_msgs::gps_data, inertiallabs_msgs::ins_data, geometry_msgs::PoseStamped> SyncPolicy_1;
typedef sync_policies::ExactTime<inertiallabs_msgs::gps_data, inertiallabs_msgs::ins_data> SyncPolicy_2;

#define WGS84_A		6378137.0		// major axis
#define WGS84_B		6356752.31424518	// minor axis
#define WGS84_F		0.0033528107		// ellipsoid flattening
#define WGS84_E		0.0818191908		// first eccentricity
#define WGS84_EP	0.0820944379		// second eccentricity

// UTM Parameters
#define UTM_K0		0.9996			// scale factor
#define UTM_FE		500000.0		// false easting
#define UTM_FN_N	0.0           // false northing, northern hemisphere
#define UTM_FN_S	10000000.0    // false northing, southern hemisphere
#define UTM_E2		(WGS84_E*WGS84_E)	// e^2
#define UTM_E4		(UTM_E2*UTM_E2)		// e^4
#define UTM_E6		(UTM_E4*UTM_E2)		// e^6
#define UTM_EP2		(UTM_E2/(1-UTM_E2))	// e'^2

double pos_x, pos_y, pos_z;
double quat_x, quat_y, quat_z, quat_w;

double yaw_diff;
int count;

Matrix<double, 4, 4> pos_tf;
Matrix<double, 4, 4> gps_pos;
Matrix<double, 4, 4> ndt_pos;

Matrix<double, 4, 4> ori_tf;
Matrix<double, 4, 4> gps_qt;
Matrix<double, 4, 4> ndt_qt;

ros::Publisher gnss_pose_pub;

void calculate_tf_with_gps_ndt_cb(const inertiallabs_msgs::gps_data::ConstPtr& msg_gps, const inertiallabs_msgs::ins_data::ConstPtr& msg_ins,
                                     const geometry_msgs::PoseStamped::ConstPtr& msg_ndt_pose);
void pub_gnss_pose_cb(const inertiallabs_msgs::gps_data::ConstPtr& msg_gps, const inertiallabs_msgs::ins_data::ConstPtr& msg_ins);                               
void LLH2UTM(double Lat, double Long, double H, geometry_msgs::PoseStamped& pose);
void ToEulerAngles(geometry_msgs::Quaternion q, double &yaw, double &pitch, double &roll);
void ToQuaternion(double yaw, double pitch, double roll, geometry_msgs::Quaternion &q);