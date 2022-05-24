#include <iostream>
#include <cmath>

#include <geometry_msgs/Quaternion.h>

static void ToEulerAngles(geometry_msgs::Quaternion q, double &yaw, double &pitch, double &roll);
static void ToQuaternion(double yaw, double pitch, double roll, geometry_msgs::Quaternion &q);
static double NormalizeRadian(double value, double min, double max);