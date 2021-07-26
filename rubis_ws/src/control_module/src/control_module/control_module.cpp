#include <control_module.h>

void ControlModule::autoware_cb(const geometry_msgs::TwistStamped& in_twist){
	double linear_vel = in_twist.twist.linear.x;
	double angular_vel = in_twist.twist.angular.z;

	cal_vesc_value(linear_vel, angular_vel);
	ControlModule::pub_vesc_value();
}

void ControlModule::lane_following_speed_cb(const std_msgs::Float64& speed){
	ROS_WARN("Send lane following speed!");
	lane_following_speed_ = speed.data;
	ControlModule::pub_vesc_value();
}

void ControlModule::lane_following_steer_cb(const std_msgs::Float64& steer){
	lane_following_steer_ = steer.data;
	ControlModule::pub_vesc_value();
}

void ControlModule::current_pose_cb(const geometry_msgs::PoseStamped& current_pose){
	current_pose_ = current_pose.pose;
}

void ControlModule::goal_cb(const geometry_msgs::PoseStamped& msg){
	check_arrived_ = true;
	goal_pose_ = msg.pose;
}

void ControlModule::behavior_state_cb(const visualization_msgs::MarkerArray& msg){
	if(msg.markers.empty()){
		current_state_ = std::string("Init");
		return;
	}

	std::string state_text = msg.markers.front().text;
	
	if(state_text.find(std::string("Init"))!=std::string::npos)
		current_state_ = std::string("Init");
	else if(state_text.find(std::string("Forward"))!=std::string::npos)
		current_state_ = std::string("Forward");
	else if(state_text.find(std::string("End"))!=std::string::npos)
		current_state_ = std::string("End");
	else
		current_state_ = std::string("Init");	

	return;
}

void ControlModule::cal_vesc_value(double linear_vel, double angular_vel)
	{	

	// std::cout<<"Linear velocity : " << twist_vel << std::endl;
	autoware_speed_ = linear_vel * speed_to_erpm_gain_ + speed_to_erpm_offset_; //twist -> vesc(rpm)

	double current_steering_angle;
	current_steering_angle = atan(wheelbase_* angular_vel / linear_vel);
	if(std::isnan(current_steering_angle) == 1 ) current_steering_angle = 0;
	//std::cout<<"\n\nAngle : "<<radian_to_degree(current_steering_angle)<<std::endl;
	autoware_steer_ = current_steering_angle * steering_angle_to_servo_gain_ + steering_angle_to_servo_offset_;
		
}

void ControlModule::pub_vesc_value(){
	std_msgs::Float64 speed_msg;
	std_msgs::Float64 steer_msg;
	std_msgs::String state_msg;

	double linear_vel = 0;
	double angular_vel = 0;

	/*if(is_arrived_to_goal_ == true){
		// Arrived to the goal
		ROS_WARN("Arrived to goal!");
		linear_vel = 0;
		angular_vel = 0;
	}*/
	//else if(from_autoware_ == true && from_lane_following_ == false && is_end_ == false){
		// ROS_WARN("Autoware forward");
		// Only Autoware
		//ROS_WARN("speed : %lf / steer : %lf", autoware_speed_, autoware_steer_);
		linear_vel = autoware_speed_;
		angular_vel = autoware_steer_;
	/*}
	else if(from_autoware_ == true && from_lane_following_ == false && is_end_ == true){
		ROS_WARN("Autoware End");
		// Only Autoware
		current_state_ = std::string("Fail");
		linear_vel = 0;
		angular_vel = 0;
	}
	else if(from_autoware_ == false && from_lane_following_ == true){ // Don't care is_end_
		// Only lane following
		ROS_WARN("D");
		ROS_WARN("Lane speed %lf, steer %lf", lane_following_speed_, lane_following_steer_);
		current_state_ = std::string("Lane Following");
		linear_vel = lane_following_speed_;
		angular_vel = lane_following_steer_;
	}
	else if(from_autoware_ == true && from_lane_following_ == true && is_end_ == false){
		// Both of them but state is not end
		linear_vel = autoware_speed_;
		angular_vel = autoware_steer_;
	}
	else if(from_autoware_ == true && from_lane_following_ == true && is_end_ == true){
		// Both of them and state is end
		current_state_ = std::string("Lane Following");
		linear_vel = lane_following_speed_;
		angular_vel = lane_following_steer_;
	}
	else{
		current_state_ = std::string("Empty");
		linear_vel = 0;
		angular_vel = 0;
	}*/

	speed_msg.data = linear_vel;
	steer_msg.data = angular_vel;
	state_msg.data = current_state_;
	vel_pub_.publish(speed_msg);
	pos_pub_.publish(steer_msg);
	state_pub_.publish(state_msg);

	return;
}

bool ControlModule::is_arrived_to_goal(){
	double current_to_end = sqrt( pow(goal_pose_.position.x - current_pose_.position.x, 2) 
								+ pow(goal_pose_.position.y - current_pose_.position.y, 2) );
	
	return current_to_end <= end_threshold_;
}

double ControlModule::valMap(double val, double in_min, double in_max, double out_min, double out_max){
	return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

double ControlModule::radian_to_degree(double radian){
	return radian * 180 / PI;
}


void ControlModule::init(){	
	nh_.param<double>("speed_to_erpm_gain", speed_to_erpm_gain_, 0.0);
	nh_.param<double>("speed_to_erpm_offset", speed_to_erpm_offset_, 0.0);
	nh_.param<double>("steering_angle_to_servo_gain", steering_angle_to_servo_gain_, 0.0);
	nh_.param<double>("steering_angle_to_servo_offset", steering_angle_to_servo_offset_, 0.0);
	nh_.param<double>("wheelbase", wheelbase_, 0.25);

	nh_.param("from_lane_following", from_lane_following_, false);
	nh_.param("from_autoware", from_autoware_, false);	
	nh_.param("lane_following_speed_str", lane_following_speed_str_, std::string("/lane_following/speed"));
	nh_.param("lane_following_steer_str", lane_following_steer_str_, std::string("/lane_following/steer"));
	nh_.param("twist_str", twist_str_, std::string("/twist_cmd"));
	nh_.param("end_threshold", end_threshold_, 0.5);

	if(PARAM_DEBUG){
		ROS_WARN("from lane : %d / from auto : %d", from_lane_following_?1:0, from_autoware_?1:0);
		ROS_WARN("lane str : %s / %s", lane_following_speed_str_.c_str(), lane_following_steer_str_.c_str());
		ROS_WARN("twsit str : %s / end threshold : %lf", twist_str_.c_str(), end_threshold_);
	}


	vel_pub_ = nh_.advertise<std_msgs::Float64>("/commands/motor/speed", 1);
	pos_pub_ = nh_.advertise<std_msgs::Float64>("/commands/servo/position", 1);
	state_pub_ = nh_.advertise<std_msgs::String>("/control_module_state", 1);

	if(from_lane_following_){
		lane_following_speed_sub_ = nh_.subscribe(lane_following_speed_str_, 1, &ControlModule::lane_following_speed_cb, this);
		lane_following_steer_sub_ = nh_.subscribe(lane_following_steer_str_, 1, &ControlModule::lane_following_steer_cb, this);
	}
	if(from_autoware_){
		twist_sub_ = nh_.subscribe(twist_str_, 1, &ControlModule::autoware_cb, this);		
	}
	current_pose_sub_ = nh_.subscribe("/ndt_pose", 1, &ControlModule::current_pose_cb, this);
	goal_sub_ = nh_.subscribe("move_base_simple_goal", 1, &ControlModule::goal_cb, this);	
	behavior_state_sub_ = nh_.subscribe("/behavior_state", 1, &ControlModule::behavior_state_cb, this);

	
	//autoware_speed_ = 0;
	//autoware_steer_ = 0.5;
	lane_following_speed_ = 0;
	lane_following_steer_ = 0;	
	current_state_ = std::string("Empty");
	goal_pose_ = geometry_msgs::Pose();	
	current_pose_ = geometry_msgs::Pose();
	
	// Exit if source is not selected
	if(from_lane_following_ == false && from_autoware_ == false){
		ROS_ERROR("You need to select at least one source for actuating! [ lane_following / Autoware ]");
		exit(1);
	}		
}

void ControlModule::Run()
{
	init();
	ros::Rate rate(20);
	cal_vesc_value(0,0);	
	std_msgs::Float64 speed_msg;
	std_msgs::Float64 steer_msg;
	speed_msg.data = autoware_speed_;
	steer_msg.data = autoware_steer_;
		
	ROS_WARN("Control module START ! ");
	
	while(ros::ok()){
		ros::spinOnce();
		if(check_arrived_) is_arrived_to_goal_ = is_arrived_to_goal();
		is_end_ = ( current_state_ == std::string("End") ) ? true : false;
		pub_vesc_value();
		rate.sleep();
	}
}

