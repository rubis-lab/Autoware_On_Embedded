#include <ros/ros.h>
#include <geometry_msgs/TwistStamped.h>

geometry_msgs::TwistStamped msg;
float target_velocity;

// void cb(const geometry_msgs::TwistStamped in_msg){
//   msg = in_msg;
//   msg.twist.linear.x = target_velocity;
// }

int main(int argc, char **argv)
{
  ros::init(argc, argv, "twist_cmd_publisher");
  ros::NodeHandle nh;
  ros::Rate rate(10);
  nh.param<float>("/twist_cmd_publisher/target_velocity", target_velocity, 2.0);

  ros::Publisher pub;
  // ros::Subscriber sub;
  pub = nh.advertise<geometry_msgs::TwistStamped>("/twist_cmd", 10);
  // sub = nh.subscribe("/twist_raw", 1, cb);
  msg.twist.linear.x = target_velocity;


  while(ros::ok()){
    // ros::spinOnce();
    
    pub.publish(msg);

    rate.sleep();
    
  }
  

  return 0;
}
