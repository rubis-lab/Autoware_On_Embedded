#include <desktop_lane_keeping/desktop_lane_keeping.h>

int main(int argc, char* argv[]){
    ros::init(argc, argv, "desktop_lane_keeping");
    ros::NodeHandle nh;

    DesktopLaneKeeping desktop_lane_keeping(nh);
    desktop_lane_keeping.Run();

    return 0;
}