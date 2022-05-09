#include <rubis_testbed_autorunner/rubis_testbed_autorunner.h>

int main(int argc, char* argv[]){
    ros::init(argc, argv, "rubis_testbed_autorunner");
    ros::NodeHandle nh;

    RubisTestbedAutorunner rubis_testbed_autorunner(nh);
    rubis_testbed_autorunner.Run();

    return 0;
}