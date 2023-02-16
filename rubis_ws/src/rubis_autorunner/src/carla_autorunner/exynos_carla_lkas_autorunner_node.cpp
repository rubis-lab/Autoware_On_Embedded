#include <carla_autorunner/carla_autorunner.h>

int main(int argc, char* argv[]){
    ros::init(argc, argv, "carla_autorunner");
    ros::NodeHandle nh;

    CarlaAutorunner carla_autorunner(nh);
    carla_autorunner.Run();

    return 0;
}