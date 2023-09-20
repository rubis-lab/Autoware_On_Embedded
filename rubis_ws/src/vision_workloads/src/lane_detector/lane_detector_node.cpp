#include <ros/ros.h>
#include "lane_detector.hpp"

int main(int argc, char* argv[]){
    ros::init(argc, argv, "lane_detector");
    LaneDetector lane_detector;
    lane_detector.run();

    return 0;
}