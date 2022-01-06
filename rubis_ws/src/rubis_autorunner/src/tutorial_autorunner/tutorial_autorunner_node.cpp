#include <tutorial_autorunner/tutorial_autorunner.h>

int main(int argc, char* argv[]){
    ros::init(argc, argv, "tutorial_autorunner");
    ros::NodeHandle nh;

    TutorialAutorunner tutorial_autorunner(nh);
    tutorial_autorunner.Run();

    return 0;
}