#include <lgsvl_triple_lidar_autorunner/lgsvl_triple_lidar_autorunner.h>

int main(int argc, char* argv[]){
    ros::init(argc, argv, "lgsvl_triple_lidar_autorunner_node");
    ros::NodeHandle nh;

    LGSVLTripleLiDARAutorunner lgsvl_triple_lidar_autorunner(nh);
    lgsvl_triple_lidar_autorunner.Run();

    return 0;
}