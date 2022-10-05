
#include "gnss_module.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>

#define DEBUG

GnssModule::GnssModule(){
    std::string pose_topic_name;
    nh_.param("/gnss_module/pose_topic_name", pose_topic_name, std::string("/gnss_pose"));
    nh_.param("/gnss_module/x_offset", x_offset_, 0.0);
    nh_.param("/gnss_module/y_offset", y_offset_, 0.0);
    nh_.param("/gnss_module/z_offset", z_offset_, 0.0);
    nh_.param("/gnss_module/yaw_offset", yaw_offset_, 0.0);
    nh_.param("/gnss_module/debug", debug_, false);
    nh_.param("/gnss_module/use_kalman_filter", use_kalman_filter_, true);
    nh_.param("/gnss_module/use_gnss_tf", use_gnss_tf_, false);
    nh_.param("/gnss_module/use_sync", use_sync_, false);
    
    std::vector<double> transformation_vec;
    if( !nh_.getParam("gnss_transformation",transformation_vec)){
      ROS_ERROR("Cannot load gnss_transformation");      
    }
    createTransformationMatrix(transformation_vec);

    gnss_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(pose_topic_name.c_str(), 1);
    ins_twist_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("/ins_twist", 1);
    ins_stat_pub_ = nh_.advertise<rubis_msgs::InsStat>("/ins_stat", 1);

    if(use_sync_){
        gps_sync_sub_.subscribe(nh_, "/Inertial_Labs/gps_data", 2);
        ins_sync_sub_.subscribe(nh_, "/Inertial_Labs/ins_data", 2);
        sensor_sync_sub_.subscribe(nh_, "/Inertial_Labs/sensor_data", 2);
        sync_.reset(new Sync(SyncPolicy(10), gps_sync_sub_, ins_sync_sub_, sensor_sync_sub_));
        sync_->registerCallback(boost::bind(&GnssModule::observation_cb, this, _1, _2, _3));
    }
    else{
        gps_data_sub_ = nh_.subscribe("/Inertial_Labs/gps_data", 1, &GnssModule::gps_data_cb, this);
        ins_data_sub_ = nh_.subscribe("/Inertial_Labs/ins_data", 1, &GnssModule::ins_data_cb, this);
        sensor_data_sub_ = nh_.subscribe("/Inertial_Labs/sensor_data", 1, &GnssModule::sensor_data_cb, this);
    }

    std::vector<float> H_k_vec, Q_k_vec, R_k_vec, P_k_vec;
    if(!nh_.getParam("/gnss_module/H_k", H_k_vec)){
      ROS_ERROR("Failed to get parameter H_k");
      exit(1);
    }
    if(!nh_.getParam("/gnss_module/Q_k", Q_k_vec)){
      ROS_ERROR("Failed to get parameter Q_k");
      exit(1);
    }
    if(!nh_.getParam("/gnss_module/R_k", R_k_vec)){
      ROS_ERROR("Failed to get parameter R_k");
      exit(1);
    }
    if(!nh_.getParam("/gnss_module/P_k", P_k_vec)){
      ROS_ERROR("Failed to get parameter P_k");
      exit(1);
    }

    Eigen::Matrix6f H_k = Eigen::Matrix6f(H_k_vec.data());
    Eigen::Matrix6f Q_k = Eigen::Matrix6f(Q_k_vec.data());
    Eigen::Matrix6f R_k = Eigen::Matrix6f(R_k_vec.data());
    Eigen::Matrix6f P_k = Eigen::Matrix6f(P_k_vec.data());

    std::cout<< H_k <<std::endl;
    std::cout<< Q_k <<std::endl;
    std::cout<< R_k <<std::endl;
    std::cout<< P_k <<std::endl;

    lkf_ = LKF(H_k, Q_k, R_k, P_k);

    /* lookup /gnss to /base_link static transform */ 
    ros::Rate rate(100);
    tf::TransformListener listener;
    tf::StampedTransform transform;
    while(nh_.ok()){
        try{
            listener.lookupTransform("/gnss", "/base_link", ros::Time(0), transform);
            tf_gnss_to_base_ = transform;
            break;
        }
        catch (tf::TransformException ex){
            ROS_ERROR("%s", ex.what());
            rate.sleep();
        }
    }

    return;
}

void GnssModule::createTransformationMatrix(std::vector<double>& transformation_vec){
    T_ << transformation_vec[0], transformation_vec[1], transformation_vec[2],
        transformation_vec[3], transformation_vec[4], transformation_vec[5],
        transformation_vec[6], transformation_vec[7], transformation_vec[8];
    return;
}

void GnssModule::gps_data_cb(const inertiallabs_msgs::gps_data::ConstPtr &gps_msg){
    static ros::Time prev_time = cur_time_;
    static double prev_linear_velocity = 0.0;
    double roll, pitch, yaw, linear_velocity, angular_velocity;

    cur_time_ = gps_msg->header.stamp;
    gnss_pose_.header = gps_msg->header;
    gnss_pose_.header.frame_id = "/map";

    /* coordinate transform (LLH2 to UTM) */ 
    LLH2UTM(gps_msg->LLH.x, gps_msg->LLH.y, gps_msg->LLH.z, gnss_pose_);

    /* position offset calculation */ 
    gnss_pose_.pose.position.x = gnss_pose_.pose.position.x - x_offset_;
    gnss_pose_.pose.position.y = gnss_pose_.pose.position.y - y_offset_;
    gnss_pose_.pose.position.z = gnss_pose_.pose.position.z - z_offset_;

    /* pose transform */
    Eigen::Matrix<double, 3, 1> pos, transposed_pose;
    pos(0) = gnss_pose_.pose.position.x; pos(1) = gnss_pose_.pose.position.y; pos(2) = 1;
    transposed_pose = T_ * pos;
    gnss_pose_.pose.position.x = transposed_pose(0);
    gnss_pose_.pose.position.y = transposed_pose(1);

    /* velocity */
    linear_velocity = gps_msg->HorSpeed; // m/s

    /* acceleration */
    time_diff_ = (cur_time_ - prev_time).toSec();
    if(time_diff_ == 0.0) time_diff_ = 0.000000001; // 1ns
    double linear_acceleration = (linear_velocity - prev_linear_velocity)/time_diff_;

    /* ins_twist */
    ins_twist_.header = gnss_pose_.header;
    ins_twist_.header.frame_id = "/base_link";
    
    ins_twist_.twist.linear.x = linear_velocity;
    ins_twist_.twist.linear.y = 0.0;
    ins_twist_.twist.linear.z = 0.0;
    ins_twist_.twist.angular.x = 0.0;
    ins_twist_.twist.angular.y = 0.0;

    /* ins_stat */
    ins_stat_.header = gnss_pose_.header;
    ins_stat_.header.frame_id = "/base_link";

    ins_stat_.linear_velocity = linear_velocity;
    ins_stat_.linear_acceleration = linear_acceleration;
    ins_stat_.angular_velocity = angular_velocity;
    ins_stat_.yaw = yaw;

    /* previous value */
    prev_time = cur_time_;
    prev_linear_velocity = linear_velocity;

    /* Check data is updated */
    is_updated_ = true;

    return;
}

void GnssModule::ins_data_cb(const inertiallabs_msgs::ins_data::ConstPtr &ins_msg){

}

void GnssModule::sensor_data_cb(const inertiallabs_msgs::sensor_data::ConstPtr &sensor_msg){

}

void GnssModule::observation_cb(const inertiallabs_msgs::gps_data::ConstPtr &gps_msg, const inertiallabs_msgs::ins_data::ConstPtr &ins_msg, const inertiallabs_msgs::sensor_data::ConstPtr &sensor_msg){
    static ros::Time prev_time = cur_time_;
    static double prev_linear_velocity = 0.0;
    static double prev_angular_velocity = 0.0;
    double roll, pitch, yaw, linear_velocity, angular_velocity;

    cur_time_ = gps_msg->header.stamp;
    gnss_pose_.header = gps_msg->header;
    gnss_pose_.header.frame_id = "/map";

    /* coordinate transform (LLH2 to UTM) */ 
    LLH2UTM(gps_msg->LLH.x, gps_msg->LLH.y, gps_msg->LLH.z, gnss_pose_);

    /* position offset calculation */ 
    gnss_pose_.pose.position.x = gnss_pose_.pose.position.x - x_offset_;
    gnss_pose_.pose.position.y = gnss_pose_.pose.position.y - y_offset_;
    gnss_pose_.pose.position.z = gnss_pose_.pose.position.z - z_offset_;

    /* orientation */ 
    roll = ins_msg->YPR.z;
    pitch = ins_msg->YPR.y;

    /* yaw offset calculation */
    yaw = -1 * (ins_msg->YPR.x) - yaw_offset_;
    yaw = (yaw > 180.0) ? (yaw - 360) : ((yaw < -180) ? (yaw + 360) : yaw);
    
    /* unit conversion(deg -> rad) */ 
    roll = roll * M_PI/180.0; pitch = pitch * M_PI/180.0; yaw = yaw * M_PI/180.0;

    /* set orientation */
    geometry_msgs::Quaternion q;
    ToQuaternion(roll, pitch, yaw, q);
    gnss_pose_.pose.orientation = q;

    /* velocity */
    linear_velocity = gps_msg->HorSpeed; // m/s
    angular_velocity = sensor_msg->Gyro.z * M_PI/180; // rad/s

    /* acceleration */
    time_diff_ = (cur_time_ - prev_time).toSec();
    if(time_diff_ == 0.0) time_diff_ = 0.000000001; // 1ns
    double linear_acceleration = (linear_velocity - prev_linear_velocity)/time_diff_;
    double angular_acceleration = (angular_velocity - prev_angular_velocity)/time_diff_;

    /* ins_twist */
    ins_twist_.header = gnss_pose_.header;
    ins_twist_.header.frame_id = "/base_link";
    
    ins_twist_.twist.linear.x = linear_velocity;
    ins_twist_.twist.linear.y = 0.0;
    ins_twist_.twist.linear.z = 0.0;
    ins_twist_.twist.angular.x = 0.0;
    ins_twist_.twist.angular.y = 0.0;
    ins_twist_.twist.angular.z = angular_velocity;

    /* ins_stat */
    ins_stat_.header = gnss_pose_.header;
    ins_stat_.header.frame_id = "/base_link";

    ins_stat_.linear_velocity = linear_velocity;
    ins_stat_.linear_acceleration = linear_acceleration;
    ins_stat_.angular_velocity = angular_velocity;
    ins_stat_.angular_acceleration = angular_acceleration;
    ins_stat_.vel_x = ins_stat_.linear_velocity * cos(yaw);
    ins_stat_.vel_y = ins_stat_.linear_velocity * sin(yaw);
    ins_stat_.acc_x = ins_stat_.angular_velocity * cos(yaw);
    ins_stat_.acc_y = ins_stat_.angular_velocity * sin(yaw);
    ins_stat_.yaw = yaw;

    /* previous value */
    prev_time = cur_time_;
    prev_linear_velocity = linear_velocity;
    prev_angular_velocity = angular_velocity;

    /* Check data is updated */
    is_updated_ = true;
    return;
}

void GnssModule::run_kalman_filter(geometry_msgs::PoseStamped& pose, geometry_msgs::TwistStamped& twist, rubis_msgs::InsStat& ins_stat){
    Eigen::Vector6f u_k, z_k; // u_k: control vector, z_k: observation_vector

    u_k << ins_stat.acc_x, ins_stat.acc_y, ins_stat.angular_acceleration, ins_stat.acc_x, ins_stat.acc_y, ins_stat.angular_acceleration;
    z_k << pose.pose.position.x, pose.pose.position.y, ins_stat.yaw, ins_stat.vel_x, ins_stat.vel_y, ins_stat.angular_velocity;

    // u_k << ins_stat.linear_velocity, ins_stat.linear_acceleration, ins_stat.angular_acceleration, 0, 0, 0;
    // z_k << pose.pose.position.x, pose.pose.position.y, ins_stat.yaw, ins_stat.vel_x, ins_stat.vel_y, ins_stat.angular_velocity;

    Eigen::Vector6f x_hat_prime_k = lkf_.run(time_diff_, u_k, z_k); // x_hat_prime_k: filtered result
    pose.pose.position.x = x_hat_prime_k(0);
    pose.pose.position.y = x_hat_prime_k(1);
    ins_stat.yaw = NormalizeRadian(x_hat_prime_k(2), -1 * M_PI, M_PI);
    ins_stat.vel_x = ins_stat.linear_velocity * cos(ins_stat.yaw);
    ins_stat.vel_y = ins_stat.linear_velocity * sin(ins_stat.yaw);
    ins_stat.angular_velocity = x_hat_prime_k(5);
    ins_stat.acc_x = ins_stat.linear_acceleration * cos(ins_stat.yaw);
    ins_stat.acc_y = ins_stat.linear_acceleration * sin(ins_stat.yaw);

    twist.twist.linear.x = sqrt(pow(ins_stat.vel_x, 2) + pow(ins_stat.vel_y, 2));
    twist.twist.angular.z = ins_stat.angular_velocity;        
}

void GnssModule::run(){
    ros::Rate rate(100);

    tf::TransformBroadcaster broadcaster;
    tf::StampedTransform transform;
    tf::Transform tf_map_to_gnss, tf_map_to_base, tf_map_to_kalman;
    tf::Quaternion q;

    bool is_kalman_filter_on = false;
    double roll, pitch, yaw;
    
    cur_time_ = ros::Time::now();
    while(nh_.ok()){
        ros::spinOnce();
        if(!is_updated_){
            rate.sleep();
            continue;
        }

        if(!use_kalman_filter_){ // No Kalman filtering
        /* Update TF */
            tf_map_to_gnss.setOrigin(tf::Vector3(gnss_pose_.pose.position.x, gnss_pose_.pose.position.y, gnss_pose_.pose.position.z));
            ToEulerAngles(gnss_pose_.pose.orientation, roll, pitch, yaw);
            q.setRPY(0, 0, yaw);
            tf_map_to_gnss.setRotation(q);

            /* /map to /base tf calculation */ 
            tf_map_to_base = tf_map_to_gnss * tf_gnss_to_base_;

            /* update gnss_pose */  
            gnss_pose_.pose.position.x = tf_map_to_base.getOrigin().x();
            gnss_pose_.pose.position.y = tf_map_to_base.getOrigin().y();
            gnss_pose_.pose.position.z = tf_map_to_base.getOrigin().z();

            gnss_pose_.pose.orientation.w = tf_map_to_base.getRotation().w();
            gnss_pose_.pose.orientation.x = tf_map_to_base.getRotation().x();
            gnss_pose_.pose.orientation.y = tf_map_to_base.getRotation().y();
            gnss_pose_.pose.orientation.z = tf_map_to_base.getRotation().z();

            /* broadcast & publish */ 
            if(use_gnss_tf_) broadcaster.sendTransform(tf::StampedTransform(tf_map_to_base, ros::Time::now(), "/map", "/base_link"));
            broadcaster.sendTransform(tf::StampedTransform(tf_map_to_base, ros::Time::now(), "/map", "/gnss_pose"));

            gnss_pose_pub_.publish(gnss_pose_);
            ins_twist_pub_.publish(ins_twist_);
            ins_stat_pub_.publish(ins_stat_);

        }
        else if(!debug_){ // Use Kalman filter
            /* kalman_filtering */ 
            if(use_kalman_filter_){
                if(!is_kalman_filter_on){
                    is_kalman_filter_on = true;
                    lkf_.set_init_value(gnss_pose_.pose.position.x, gnss_pose_.pose.position.y, ins_stat_.yaw, ins_stat_.vel_x, ins_stat_.vel_y, ins_stat_.angular_velocity); 
                }
                run_kalman_filter(gnss_pose_, ins_twist_, ins_stat_);
            }       
            
            /* Update TF */
            tf_map_to_gnss.setOrigin(tf::Vector3(gnss_pose_.pose.position.x, gnss_pose_.pose.position.y, gnss_pose_.pose.position.z));
            ToEulerAngles(gnss_pose_.pose.orientation, roll, pitch, yaw);
            q.setRPY(0, 0, yaw);
            tf_map_to_gnss.setRotation(q);
    
            /* /map to /base tf calculation */ 
            tf_map_to_base = tf_map_to_gnss * tf_gnss_to_base_;
    
            /* update gnss_pose */  
            gnss_pose_.pose.position.x = tf_map_to_base.getOrigin().x();
            gnss_pose_.pose.position.y = tf_map_to_base.getOrigin().y();
            gnss_pose_.pose.position.z = tf_map_to_base.getOrigin().z();
    
            gnss_pose_.pose.orientation.w = tf_map_to_base.getRotation().w();
            gnss_pose_.pose.orientation.x = tf_map_to_base.getRotation().x();
            gnss_pose_.pose.orientation.y = tf_map_to_base.getRotation().y();
            gnss_pose_.pose.orientation.z = tf_map_to_base.getRotation().z();
    
            /* broadcast & publish */ 
            if(use_gnss_tf_) broadcaster.sendTransform(tf::StampedTransform(tf_map_to_base, ros::Time::now(), "/map", "/base_link"));

            gnss_pose_pub_.publish(gnss_pose_);
            ins_twist_pub_.publish(ins_twist_);
            ins_stat_pub_.publish(ins_stat_);
        }
        else{ // Debug mode
            geometry_msgs::PoseStamped kalman_pose = gnss_pose_;

            /* Update TF */
            tf_map_to_gnss.setOrigin(tf::Vector3(gnss_pose_.pose.position.x, gnss_pose_.pose.position.y, gnss_pose_.pose.position.z));
            ToEulerAngles(gnss_pose_.pose.orientation, roll, pitch, yaw);
            q.setRPY(0, 0, yaw);
            tf_map_to_gnss.setRotation(q);
    
            /* /map to /base tf calculation */ 
            tf_map_to_base = tf_map_to_gnss * tf_gnss_to_base_;
    
            /* update gnss_pose */  
            gnss_pose_.pose.position.x = tf_map_to_base.getOrigin().x();
            gnss_pose_.pose.position.y = tf_map_to_base.getOrigin().y();
            gnss_pose_.pose.position.z = tf_map_to_base.getOrigin().z();
    
            gnss_pose_.pose.orientation.w = tf_map_to_base.getRotation().w();
            gnss_pose_.pose.orientation.x = tf_map_to_base.getRotation().x();
            gnss_pose_.pose.orientation.y = tf_map_to_base.getRotation().y();
            gnss_pose_.pose.orientation.z = tf_map_to_base.getRotation().z();

            /* kalman_filtering */ 
            if(use_kalman_filter_){
                if(!is_kalman_filter_on){
                    is_kalman_filter_on = true;                    
                    lkf_.set_init_value(kalman_pose.pose.position.x, kalman_pose.pose.position.y, ins_stat_.yaw, ins_stat_.vel_x, ins_stat_.vel_y, ins_stat_.angular_velocity);                     
                }
                run_kalman_filter(kalman_pose, ins_twist_, ins_stat_);
            }       

            /* Update TF */
            tf_map_to_kalman.setOrigin(tf::Vector3(kalman_pose.pose.position.x, kalman_pose.pose.position.y, kalman_pose.pose.position.z));
            ToEulerAngles(kalman_pose.pose.orientation, roll, pitch, yaw);
            q.setRPY(0, 0, yaw);
            tf_map_to_kalman.setRotation(q);
    
            /* /map to /base tf calculation */ 
            tf_map_to_kalman = tf_map_to_kalman * tf_gnss_to_base_;
        
            /* broadcast & publish */ 
            if(use_gnss_tf_) broadcaster.sendTransform(tf::StampedTransform(tf_map_to_base, ros::Time::now(), "/map", "/base_link"));
            broadcaster.sendTransform(tf::StampedTransform(tf_map_to_kalman, ros::Time::now(), "/map", "/kalman"));
    
            gnss_pose_pub_.publish(gnss_pose_);
            ins_twist_pub_.publish(ins_twist_);
            ins_stat_pub_.publish(ins_stat_);
        }

        rate.sleep();
        is_updated_ = false;
    }

    return;
}