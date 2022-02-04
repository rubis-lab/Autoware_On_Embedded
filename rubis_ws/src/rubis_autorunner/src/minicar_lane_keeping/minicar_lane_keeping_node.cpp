#include <minicar_lane_keeping/minicar_lane_keeping.h>

int main(int argc, char* argv[]){
    ros::init(argc, argv, "minicar_lane_keeping");
    ros::NodeHandle nh;

    MinicarLaneKeeping minicar_lane_keeping(nh);
    minicar_lane_keeping.Run();

    return 0;
}