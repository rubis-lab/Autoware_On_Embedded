#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>

ros::Publisher twist_pub;
ros::Subscriber twist_sub;
geometry_msgs::Twist twist;

void twist_cb
(const geometry_msgs::Twist& in_twist)
{
	twist = in_twist;
	return;
}

int main
(int argc, char* argv[])
{
	ros::init(argc, argv, "twist_converter");	
	ros::NodeHandle nh;
	ros::Rate rate(10);
	
	twist_sub = nh.subscribe("/cmd_vel", 10, twist_cb);
	twist_pub = nh.advertise<geometry_msgs::TwistStamped>("/twist_cmd", 10);
	
	int seq = 0;
	geometry_msgs::TwistStamped out_twist;
	out_twist.twist = geometry_msgs::Twist();

	while(ros::ok()){
		out_twist.header.stamp = ros::Time::now();
		out_twist.header.seq = seq++;
		out_twist.twist = twist;
		twist_pub.publish(out_twist);
		ros::spinOnce();
		rate.sleep();
	}

	return 0;
}
