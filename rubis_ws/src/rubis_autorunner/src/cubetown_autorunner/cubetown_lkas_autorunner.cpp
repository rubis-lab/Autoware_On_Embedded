#include "cubetown_autorunner/cubetown_autorunner.h"

void CubetownAutorunner::Run(){
    register_subscribers();             // Register subscribers that shoud check can go next or not
    ros_autorunner_.init(nh_, sub_v_);   // Initialize the ROS-Autorunner
    ros::Rate rate(1);                  // Rate can be changed
    while(ros::ok()){               
        if(!ros_autorunner_.Run()) break;           // Run Autorunner
        ros::spinOnce();
        rate.sleep();
    }    
}

void CubetownAutorunner::register_subscribers(){
    int total_step_num = nh_.param("/total_step_num", -1);
    if(total_step_num < 0){
        std::cout<<"Parameter total_step_num is invalid"<<std::endl;
        exit(1);
    }    
    sub_v_.resize(total_step_num);          // Resizing the subscriber vectors. Its size must be same with number of steps

    // Set the check function(subscriber)
    sub_v_[STEP(1)] = nh_.subscribe("/points_raw", 1, &CubetownAutorunner::points_raw_cb, this);   
    sub_v_[STEP(2)] = nh_.subscribe("/ndt_stat", 1, &CubetownAutorunner::ndt_stat_cb, this); 
    sub_v_[STEP(3)] = nh_.subscribe("/behavior_state", 1, &CubetownAutorunner::behavior_state_cb, this); 
}

 void CubetownAutorunner::points_raw_cb(const sensor_msgs::PointCloud2& msg){
    if(!msg.fields.empty() && !ros_autorunner_.step_info_list_[STEP(2)].is_prepared){
        ROS_WARN("[STEP 1] Map and Sensors are prepared");
    	sleep(SLEEP_PERIOD);
        ros_autorunner_.step_info_list_[STEP(2)].is_prepared = true;
    }
 }

 void CubetownAutorunner::ndt_stat_cb(const autoware_msgs::NDTStat& msg){
    if(msg.score < 0.7 && !ros_autorunner_.step_info_list_[STEP(3)].is_prepared){
        ROS_WARN("[STEP 2] Localization is success");
    	sleep(SLEEP_PERIOD);
        ros_autorunner_.step_info_list_[STEP(3)].is_prepared = true;
    }
 }

 void CubetownAutorunner::behavior_state_cb(const visualization_msgs::MarkerArray& msg){
    std::string state = msg.markers.front().text;    
    if(!msg.markers.empty() && state.find(std::string("Forward"))!=std::string::npos){
        ROS_WARN("[STEP 3] Global & local planning success");
        ros_autorunner_.step_info_list_[STEP(4)].is_prepared = true;
    }
}



