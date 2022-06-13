#ifndef QUATERNION_EULER_H
#define QUATERNION_EULER_H

#include <iostream>
#include <cmath>
#include <math.h>
#include <geometry_msgs/Quaternion.h>

static void ToEulerAngles(geometry_msgs::Quaternion q, double &yaw, double &pitch, double &roll){
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    double sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        pitch = std::asin(sinp);

    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    yaw = std::atan2(siny_cosp, cosy_cosp);
}

static void ToQuaternion(double yaw, double pitch, double roll, geometry_msgs::Quaternion &q)
{
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;
}

static double NormalizeRadian(double value, double min, double max){
    value -= min;
    max -= min;
    
    if (max == 0)
        return min;
    value = fmod(value, max);
    value += min;

    while (value < min)
    {
        value += max;
    }

    return value;
}

#endif