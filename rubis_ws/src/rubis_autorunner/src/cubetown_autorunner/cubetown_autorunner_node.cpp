#include <cubetown_autorunner/cubetown_autorunner.h>

int main(int argc, char* argv[]){
    ros::init(argc, argv, "cubetown_autorunner");
    ros::NodeHandle nh;

    CubetownAutorunner cubetown_autorunner(nh);
    cubetown_autorunner.Run();

    return 0;
}