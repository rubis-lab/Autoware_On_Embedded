#include <ros/ros.h>

#include "svl_sensing.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "svl_sensing");  
  SvlSensing svl_sensing;
  svl_sensing.run();

  return 0;
}
