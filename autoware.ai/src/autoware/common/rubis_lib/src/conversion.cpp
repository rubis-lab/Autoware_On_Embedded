#include <rubis_lib/conversion.h>
#include <sensor_msgs/PointCloud2.h>


namespace rubis{
void toRubisPointCloud2(const sensor_msgs::PointCloud2 &input, rubis_msgs::PointCloud2 &output, unsigned long& instance){
    output.instance = instance;
    output.msg = input;

    return;
}

void fromRubisPointCloud2(const rubis_msgs::PointCloud2 &input, sensor_msgs::PointCloud2 &output, unsigned long& instance){
    instance = input.instance;
    output = input.msg;

    return;
}

void toRubisTwistStamped(const geometry_msgs::TwistStamped &input, rubis_msgs::TwistStamped &output, unsigned long &instance){
    output.instance = instance;
    output.msg = input;

    return;
}

void fromRubisTwistStamped(const rubis_msgs::TwistStamped &input, geometry_msgs::TwistStamped &output, unsigned long &instance){
    instance = input.instance;
    output = input.msg;

    return;
}


void toRubisPoseStamped(const geometry_msgs::PoseStamped &input, rubis_msgs::PoseStamped &output, unsigned long &instance){
    output.instance = instance;
    output.msg = input;

    return;
}

void fromRubisPoseStamped(const rubis_msgs::PoseStamped &input, geometry_msgs::PoseStamped &output, unsigned long &instance){
    instance = input.instance;
    output = input.msg;

    return;
}


}