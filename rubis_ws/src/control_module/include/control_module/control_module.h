#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/MarkerArray.h>
#include <math.h>
#include <mutex>

#define PARAM_DEBUG 1
#define PI 3.14

class ControlModule{
private:
	ros::NodeHandle nh_;
	ros::Subscriber twist_sub_;
	ros::Subscriber goal_sub_;
	ros::Subscriber lane_following_speed_sub_;
	ros::Subscriber lane_following_steer_sub_;
	ros::Subscriber current_pose_sub_;
	ros::Subscriber behavior_state_sub_;
	ros::Publisher vel_pub_;
	ros::Publisher pos_pub_;
	ros::Publisher state_pub_;

	double speed_to_erpm_gain_;
	double speed_to_erpm_offset_;
	double steering_angle_to_servo_gain_;
	double steering_angle_to_servo_offset_;
	double wheelbase_;
	
	bool check_arrived_ = false;
	bool from_lane_following_;
	bool from_autoware_;	
	bool is_arrived_to_goal_ = false;
	bool is_end_ = false;
	double end_threshold_;
	double autoware_speed_;
	double autoware_steer_;
	double lane_following_speed_;
	double lane_following_steer_;
	std::string lane_following_speed_str_;
	std::string lane_following_steer_str_;
	std::string twist_str_;	
	std::string current_state_;
	geometry_msgs::Pose goal_pose_;
	geometry_msgs::Pose current_pose_;

private:
	void init();
	void autoware_cb(const geometry_msgs::TwistStamped& in_twist);
	void lane_following_speed_cb(const std_msgs::Float64& speed);
	void lane_following_steer_cb(const std_msgs::Float64& steer);
	void current_pose_cb(const geometry_msgs::PoseStamped& current_pose);
	void goal_cb(const geometry_msgs::PoseStamped& msg);
	void behavior_state_cb(const visualization_msgs::MarkerArray& msg);
	void cal_vesc_value(double linear_vel, double angular_vel);
	void pub_vesc_value();
	bool is_arrived_to_goal();
	double valMap(double val, double in_min, double in_max, double out_min, double out_max);
	double radian_to_degree(double radian);

public:
	ControlModule(){};
	void Run();
};


