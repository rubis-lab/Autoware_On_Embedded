#include <iostream>
#include <cmath>

#include <geometry_msgs/Quaternion.h>

void ToEulerAngles(geometry_msgs::Quaternion q, double &yaw, double &pitch, double &roll);
void ToQuaternion(double yaw, double pitch, double roll, geometry_msgs::Quaternion &q);