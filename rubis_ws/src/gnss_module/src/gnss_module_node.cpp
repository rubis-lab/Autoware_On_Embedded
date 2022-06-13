#include "gnss_module.hpp"

int main(int argc, char* argv[]){
    ros::init(argc, argv, "gnss_module");

    GnssModule gnss_module;
    gnss_module.run();

    return 0;
}