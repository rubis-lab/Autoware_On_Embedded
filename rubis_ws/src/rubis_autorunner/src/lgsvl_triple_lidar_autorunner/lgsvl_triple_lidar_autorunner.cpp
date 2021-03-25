#include <lgsvl_triple_lidar_autorunner/lgsvl_triple_lidar_autorunner.h>

void LGSVLTripleLiDARAutorunner::Run(){
    register_subscribers();             // Register subscribers that shoud check can go next or not
    ros_autorunner_.init(nh_, sub_v_);   // Initialize the ROS-Autorunner
    ros::Rate rate(1);                  // Rate can be changed
    while(ros::ok()){               
        ros_autorunner_.Run();           // Run Autorunner
        ros::spinOnce();
        rate.sleep();
    }    
}

void LGSVLTripleLiDARAutorunner::register_subscribers(){
    sub_v_.resize(TOTAL_STEP_NUM);          // Resizing the subscriber vectors. Its size must be same with number of steps

    // Set the check function(subscriber)
    sub_v_[STEP(1)] = nh_.subscribe("/nmea_sentence", 1, &LGSVLTripleLiDARAutorunner::nmea_sentence_cb, this);
    sub_v_[STEP(2)] = nh_.subscribe("/points_no_ground_left", 1, &LGSVLTripleLiDARAutorunner::points_no_ground_left_cb, this);
    sub_v_[STEP(3)] = nh_.subscribe("/current_pose", 1, &LGSVLTripleLiDARAutorunner::current_pose_cb, this);
    sub_v_[STEP(4)] = nh_.subscribe("/detection/image_detector/objects", 1, &LGSVLTripleLiDARAutorunner::detection_objects_cb, this);
    sub_v_[STEP(5)] = nh_.subscribe("/detection/object_tracker/objects_left", 1, &LGSVLTripleLiDARAutorunner::detection_objects_from_tracker_cb, this);
    sub_v_[STEP(6)] = nh_.subscribe("/lane_waypoints_array", 1, &LGSVLTripleLiDARAutorunner::lane_waypoints_array_cb, this);
    sub_v_[STEP(7)] = nh_.subscribe("/local_trajectory_cost", 1, &LGSVLTripleLiDARAutorunner::local_traj_cost_cb, this);
    sub_v_[STEP(8)] = nh_.subscribe("/behavior_state", 1, &LGSVLTripleLiDARAutorunner::behavior_state_cb, this);
    // sub_v_[STEP(9)]
}

void LGSVLTripleLiDARAutorunner::nmea_sentence_cb(const nmea_msgs::Sentence& msg){
    if(!msg.sentence.empty() && !ros_autorunner_.step_info_list_[STEP(2)].is_prepared){
        ROS_WARN("[STEP 1] Connected with LGSVL");
        ROS_WARN("[STEP 1] GNSS Info Detected");
        sleep(SLEEP_PERIOD);
        ros_autorunner_.step_info_list_[STEP(2)].is_prepared = true;
    }
}

void LGSVLTripleLiDARAutorunner::points_no_ground_left_cb(const sensor_msgs::PointCloud2& msg){
    if(!msg.fields.empty() && !ros_autorunner_.step_info_list_[STEP(3)].is_prepared){
        ROS_WARN("[STEP 2] ray ground filter activated");
    	sleep(SLEEP_PERIOD);
        ros_autorunner_.step_info_list_[STEP(3)].is_prepared = true;
    }
}

void LGSVLTripleLiDARAutorunner::current_pose_cb(const geometry_msgs::PoseStamped& msg){
    if(!ros_autorunner_.step_info_list_[STEP(4)].is_prepared){
        ROS_WARN("[STEP 3] Localization Success!");
        ROS_WARN("[STEP 3] gnss pose Detected");
        sleep(SLEEP_PERIOD);
        ros_autorunner_.step_info_list_[STEP(4)].is_prepared = true;
    }
}

void LGSVLTripleLiDARAutorunner::detection_objects_cb(const autoware_msgs::DetectedObjectArray& msg){
    if(!ros_autorunner_.step_info_list_[STEP(5)].is_prepared){
        ROS_WARN("[STEP 4] Object Deteced from Vision");
        sleep(SLEEP_PERIOD);
        ros_autorunner_.step_info_list_[STEP(5)].is_prepared = true;
    }
}

void LGSVLTripleLiDARAutorunner::detection_objects_from_tracker_cb(const autoware_msgs::DetectedObjectArray& msg){
    if(!ros_autorunner_.step_info_list_[STEP(6)].is_prepared){
        ROS_WARN("[STEP 5] Object tracking Complete");
        sleep(SLEEP_PERIOD);
        ros_autorunner_.step_info_list_[STEP(6)].is_prepared = true;
    }
}

void LGSVLTripleLiDARAutorunner::lane_waypoints_array_cb(const autoware_msgs::LaneArray& msg){
    if(!msg.lanes.empty() && !ros_autorunner_.step_info_list_[STEP(7)].is_prepared){
        ROS_WARN("[STEP 6] Global path is created");
        ros_autorunner_.step_info_list_[STEP(7)].is_prepared = true;
    }
}

void LGSVLTripleLiDARAutorunner::local_traj_cost_cb(const autoware_msgs::Lane& msg){
    if(!ros_autorunner_.step_info_list_[STEP(8)].is_prepared){
        ROS_WARN("[STEP 7] Trajectory Evaluate Complete");
        ros_autorunner_.step_info_list_[STEP(8)].is_prepared = true;
    }
}

void LGSVLTripleLiDARAutorunner::behavior_state_cb(const visualization_msgs::MarkerArray& msg){
    std::string state = msg.markers.front().text;
    ROS_WARN("[STEP 8] Behavior State %s", state.c_str());
    if(!msg.markers.empty() && state.find(std::string("Forward"))!=std::string::npos){
        ROS_WARN("[STEP 8] Behvior state is set to forward");
        ros_autorunner_.step_info_list_[STEP(9)].is_prepared = true;
    }
}