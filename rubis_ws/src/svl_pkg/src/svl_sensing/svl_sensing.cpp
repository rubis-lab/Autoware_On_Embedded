#include "svl_sensing.h"

SvlSensing::SvlSensing()
{
	lidar_sub_.subscribe(nh_, "/points_raw_origin", 1);
	odom_sub_.subscribe(nh_, "/odom", 1);
	sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), lidar_sub_, odom_sub_));
	sync_->registerCallback(boost::bind(&SvlSensing::callback, this, _1, _2));
	lidar_pub_ = nh_.advertise<rubis_msgs::PointCloud2>("/rubis_points_raw", 1);
	pose_twist_pub_ = nh_.advertise<rubis_msgs::PoseTwistStamped>("/svl_pose_twist", 1);

	std::string node_name = ros::this_node::getName();
  std::string task_response_time_filename;
  nh_.param<std::string>(node_name+"/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/svl_sensing.csv");

	struct rubis::sched_attr attr;
  std::string policy;
  int priority, exec_time ,deadline, period;
    
  nh_.param(node_name+"/task_scheduling_configs/policy", policy, std::string("NONE"));    
  nh_.param(node_name+"/task_scheduling_configs/priority", priority, 99);
  nh_.param(node_name+"/task_scheduling_configs/exec_time", exec_time, 0);
  nh_.param(node_name+"/task_scheduling_configs/deadline", deadline, 0);
  nh_.param(node_name+"/task_scheduling_configs/period", period, 0);
  attr = rubis::create_sched_attr(priority, exec_time, deadline, period);    
  rubis::init_task_scheduling(policy, attr);
  rubis::init_task_profiling(task_response_time_filename);
}

SvlSensing::~SvlSensing()
{
}

void SvlSensing::callback(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg, const nav_msgs::Odometry::ConstPtr& odom_msg)
{
	rubis::start_task_profiling_at_initial_node(std::max(lidar_msg->header.stamp.sec, odom_msg->header.stamp.sec), std::max(lidar_msg->header.stamp.nsec, odom_msg->header.stamp.nsec));

	rubis_msgs::PointCloud2 out_lidar_msg;	
	out_lidar_msg.instance = rubis::instance_;
	out_lidar_msg.msg = *lidar_msg;
	out_lidar_msg.msg.fields.at(3).datatype = 7;

	rubis_msgs::PoseTwistStamped out_pose_twist_msg;
	out_pose_twist_msg.instance = rubis::instance_;
	out_pose_twist_msg.pose.header = odom_msg->header;	
	out_pose_twist_msg.pose.header.frame_id = "/map";
	out_pose_twist_msg.pose.pose = odom_msg->pose.pose;
	out_pose_twist_msg.twist.header = odom_msg->header;
	out_pose_twist_msg.twist.header.frame_id = "/map";
	out_pose_twist_msg.twist.twist = odom_msg->twist.twist;

	lidar_pub_.publish(out_lidar_msg);
	pose_twist_pub_.publish(out_pose_twist_msg);

	rubis::stop_task_profiling(rubis::instance_++, rubis::obj_instance_++);
	return;
}


void SvlSensing::run()
{
    ros::spin();
}