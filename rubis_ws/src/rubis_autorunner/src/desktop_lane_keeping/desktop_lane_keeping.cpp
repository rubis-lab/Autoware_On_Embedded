#include "desktop_lane_keeping/desktop_lane_keeping.h"

void DesktopLaneKeeping::Run(){
    register_subscribers();             // Register subscribers that shoud check can go next or not
    ros_autorunner_.init(nh_, sub_v_);   // Initialize the ROS-Autorunner
    ros::Rate rate(1);                  // Rate can be changed
    while(ros::ok()){               
        ros_autorunner_.Run();           // Run Autorunner
        ros::spinOnce();
        rate.sleep();
    }    
}

void DesktopLaneKeeping::register_subscribers(){
    sub_v_.resize(TOTAL_STEP_NUM);          // Resizing the subscriber vectors. Its size must be same with number of steps

    // Set the check function(subscriber)
    sub_v_[STEP(1)] = nh_.subscribe("/points_no_ground_center", 1, &DesktopLaneKeeping::points_no_ground_cb, this);   
    sub_v_[STEP(2)] = nh_.subscribe("/ndt_stat", 1, &DesktopLaneKeeping::ndt_stat_cb, this);   
    sub_v_[STEP(3)] = nh_.subscribe("/behavior_state", 1, &DesktopLaneKeeping::behavior_state_cb, this);   
}

 void DesktopLaneKeeping::points_no_ground_cb(const sensor_msgs::PointCloud2& msg){
    if(!msg.fields.empty() && !ros_autorunner_.step_info_list_[STEP(2)].is_prepared){
        ROS_WARN("[STEP 1] Map and Sensors are prepared");
    	sleep(SLEEP_PERIOD);
        ros_autorunner_.step_info_list_[STEP(2)].is_prepared = true;
    }
 }

 void DesktopLaneKeeping::ndt_stat_cb(const autoware_msgs::NDTStat& msg){
    if(msg.score < 0.3 && !ros_autorunner_.step_info_list_[STEP(3)].is_prepared){
        ROS_WARN("[STEP 2] Localization is success");
    	sleep(SLEEP_PERIOD);
        ros_autorunner_.step_info_list_[STEP(3)].is_prepared = true;
    }
 }

 void DesktopLaneKeeping::behavior_state_cb(const visualization_msgs::MarkerArray& msg){
    std::string state = msg.markers.front().text;    
    if(!msg.markers.empty() && state.find(std::string("Forward"))!=std::string::npos){
        ROS_WARN("[STEP 3] Global & local planning success");
        ros_autorunner_.step_info_list_[STEP(4)].is_prepared = true;
    }
}



