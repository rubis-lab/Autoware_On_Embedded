#ifndef __RUBIS_CONVERSION_H__
#define __RUBIS_CONVERSION_H__

#include <sensor_msgs/PointCloud2.h>
#include <rubis_msgs/PointCloud2.h>

#include <geometry_msgs/TwistStamped.h>
#include <rubis_msgs/TwistStamped.h>

#include <geometry_msgs/PoseStamped.h>
#include <rubis_msgs/PoseStamped.h>

namespace rubis{
    void toRubisPointCloud2(const sensor_msgs::PointCloud2 &input, rubis_msgs::PointCloud2 &output, unsigned long &instance);
    void fromRubisPointCloud2(const rubis_msgs::PointCloud2 &input, sensor_msgs::PointCloud2 &output, unsigned long &instance);

    void toRubisTwistStamped(const geometry_msgs::TwistStamped &input, rubis_msgs::TwistStamped &output, unsigned long &instance);
    void fromRubisTwistStamped(const rubis_msgs::TwistStamped &input, geometry_msgs::TwistStamped &output, unsigned long &instance);  

    void toRubisPoseStamped(const geometry_msgs::PoseStamped &input, rubis_msgs::PoseStamped &output, unsigned long &instance);
    void fromRubisPoseStamped(const rubis_msgs::PoseStamped &input, geometry_msgs::PoseStamped &output, unsigned long &instance);  
}

#endif