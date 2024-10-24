#include "svl_sensing.h"
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <cstdlib>
#include <mutex>
#include <atomic>

// Constructor
SvlSensing::SvlSensing() : id_(0), last_publish_id_(0) {
    ros::NodeHandle private_nh("~");
    std::string node_name = ros::this_node::getName();
    std::string lidar_topic, odom_topic, image_topic_1, image_topic_2, image_topic_3, image_topic_4, pose_twist_topic;
    int image_topic_cnt = 0;
    int data_size_mb = 0;

    // Parameters
    private_nh.param<int>(node_name + "/rate", rate_, 5);
    private_nh.param<std::string>(node_name + "/pose_twist_topic", pose_twist_topic, "/rubis_current_pose_twist");
    private_nh.param<std::string>(node_name + "/lidar_topic", lidar_topic, "/points_raw_origin");
    private_nh.param<std::string>(node_name + "/odom_topic", odom_topic, "/odom");
    private_nh.param<std::string>(node_name + "/image_topic_1", image_topic_1, "none");
    private_nh.param<std::string>(node_name + "/image_topic_2", image_topic_2, "none");
    private_nh.param<std::string>(node_name + "/image_topic_3", image_topic_3, "none");
    private_nh.param<std::string>(node_name + "/image_topic_4", image_topic_4, "none");
    private_nh.param<int>(node_name + "/data_size_mb", data_size_mb, 64);

    // Subscribers
    lidar_sub_ = nh_.subscribe(lidar_topic, 2, &SvlSensing::callbackLidar, this);
    odom_sub_ = nh_.subscribe(odom_topic, 10, &SvlSensing::callbackOdom, this);

    if (image_topic_1 != "none") {
        image_sub_1_ = nh_.subscribe(image_topic_1, 10, &SvlSensing::callbackImage1, this);
        image_topic_cnt++;
    }

    if (image_topic_2 != "none") {
        if (image_topic_cnt != 1) {
            ROS_ERROR("# [ERROR] Set image_topic_1 first");
            exit(EXIT_FAILURE);
        }
        image_sub_2_ = nh_.subscribe(image_topic_2, 10, &SvlSensing::callbackImage2, this);
        image_topic_cnt++;
    }

    if (image_topic_3 != "none") {
        if (image_topic_cnt != 2) {
            ROS_ERROR("# [ERROR] Set image_topic_2 first");
            exit(EXIT_FAILURE);
        }
        image_sub_3_ = nh_.subscribe(image_topic_3, 10, &SvlSensing::callbackImage3, this);
        image_topic_cnt++;
    }

    if (image_topic_4 != "none") {
        if (image_topic_cnt != 3) {
            ROS_ERROR("# [ERROR] Set image_topic_3 first");
            exit(EXIT_FAILURE);
        }
        image_sub_4_ = nh_.subscribe(image_topic_4, 10, &SvlSensing::callbackImage4, this);
        image_topic_cnt++;
    }

    // Publishers
    lidar_pub_ = nh_.advertise<rubis_msgs::PointCloud2>("/rubis_points_raw", 1);
    pose_twist_pub_ = nh_.advertise<rubis_msgs::PoseTwistStamped>(pose_twist_topic, 1);

    if (image_topic_cnt >= 1) {
        image_pub1_ = nh_.advertise<rubis_msgs::Image>("/image_raw1", 1);
    }
    if (image_topic_cnt >= 2) {
        image_pub2_ = nh_.advertise<rubis_msgs::Image>("/image_raw2", 1);
    }
    if (image_topic_cnt >= 3) {
        image_pub3_ = nh_.advertise<rubis_msgs::Image>("/image_raw3", 1);
    }
    if (image_topic_cnt >= 4) {
        image_pub4_ = nh_.advertise<rubis_msgs::Image>("/image_raw4", 1);
    }

    // Task Profiling Configuration
    std::string task_response_time_filename;
    nh_.param<std::string>(node_name + "/task_response_time_filename", task_response_time_filename,
                           "~/Documents/profiling/response_time/svl_sensing.csv");

    struct rubis::sched_attr attr;
    std::string policy;
    int priority, exec_time, deadline, period;

    nh_.param(node_name + "/task_scheduling_configs/policy", policy, std::string("NONE"));
    nh_.param(node_name + "/task_scheduling_configs/priority", priority, 99);
    nh_.param(node_name + "/task_scheduling_configs/exec_time", exec_time, 0);
    nh_.param(node_name + "/task_scheduling_configs/deadline", deadline, 0);
    nh_.param(node_name + "/task_scheduling_configs/period", period, 0);
    attr = rubis::create_sched_attr(priority, exec_time, deadline, period);
    rubis::init_task_scheduling(policy, attr);
    rubis::init_task_profiling(task_response_time_filename);

    // Data Initialization
    nh_.param(node_name + "/n", n_, 10);
    data_size_ = data_size_mb*1024*1024;
    data_ = (int *)aligned_alloc(32, data_size_);
    memset(data_, 0, data_size_);
}

// Destructor
SvlSensing::~SvlSensing() {
    if (data_) {
        free(data_);
    }
}

// Non-LiDAR Callbacks: Update data and increment id_
void SvlSensing::callbackOdom(const nav_msgs::Odometry::ConstPtr &odom_msg) {
    latest_odom_msg_ = odom_msg;
}

void SvlSensing::callbackImage1(const sensor_msgs::Image::ConstPtr &image_msg1) {
    latest_image_msg1_ = image_msg1;
}

void SvlSensing::callbackImage2(const sensor_msgs::Image::ConstPtr &image_msg2) {
    latest_image_msg2_ = image_msg2;
}

void SvlSensing::callbackImage3(const sensor_msgs::Image::ConstPtr &image_msg3) {
    latest_image_msg3_ = image_msg3;
    id_++;
}

void SvlSensing::callbackImage4(const sensor_msgs::Image::ConstPtr &image_msg4) {
    latest_image_msg4_ = image_msg4;
}

// LiDAR Callback: Check for updates and publish if necessary
void SvlSensing::callbackLidar(const sensor_msgs::PointCloud2::ConstPtr &lidar_msg) {

    // Check if id_ has changed since last publish
    if (id_ == last_publish_id_) {
        // No new data, skip publishing
        return;
    }

    // Ensure that odom message is available
    if (!latest_odom_msg_) {
        ROS_WARN("Odom message not received yet. Skipping publish.");
        return;
    }

    // Update last_publish_id_
    last_publish_id_ = id_;

    // Start task profiling
    rubis::start_task_profiling_at_initial_node(
        std::max(lidar_msg->header.stamp.sec, latest_odom_msg_->header.stamp.sec),
        std::max(lidar_msg->header.stamp.nsec, latest_odom_msg_->header.stamp.nsec));

    // Current time for synchronization
    auto cur_time = ros::Time::now();

    // Prepare and publish PointCloud2 message
    rubis_msgs::PointCloud2 out_lidar_msg;
    out_lidar_msg.header = lidar_msg->header;
    out_lidar_msg.header.stamp = cur_time;
    out_lidar_msg.instance = rubis::instance_;
    out_lidar_msg.msg = *lidar_msg;
    out_lidar_msg.msg.header.stamp = cur_time;
    if (out_lidar_msg.msg.fields.size() > 3) {
        out_lidar_msg.msg.fields.at(3).datatype = 7;
    }

    // Prepare and publish PoseTwistStamped message
    rubis_msgs::PoseTwistStamped out_pose_twist_msg;
    out_pose_twist_msg.header = latest_odom_msg_->header;
    out_pose_twist_msg.header.stamp = cur_time;
    out_pose_twist_msg.instance = rubis::instance_;
    out_pose_twist_msg.pose.header = latest_odom_msg_->header;
    out_pose_twist_msg.pose.header.stamp = cur_time;
    out_pose_twist_msg.pose.header.frame_id = "/map";
    out_pose_twist_msg.pose.pose = latest_odom_msg_->pose.pose;
    out_pose_twist_msg.twist.header.frame_id = "/map";
    out_pose_twist_msg.twist.header.stamp = cur_time;
    out_pose_twist_msg.twist.twist = latest_odom_msg_->twist.twist;

    // Prepare and publish Image messages if available
    rubis_msgs::Image out_image_msg1, out_image_msg2, out_image_msg3, out_image_msg4;
    bool has_image1 = false, has_image2 = false, has_image3 = false, has_image4 = false;

    if (latest_image_msg1_) {
        out_image_msg1.header = latest_odom_msg_->header;
        out_image_msg1.header.stamp = cur_time;
        out_image_msg1.instance = rubis::instance_;
        out_image_msg1.msg = *latest_image_msg1_;
        has_image1 = true;
    }

    if (latest_image_msg2_) {
        out_image_msg2.header = latest_odom_msg_->header;
        out_image_msg2.header.stamp = cur_time;
        out_image_msg2.instance = rubis::instance_;
        out_image_msg2.msg = *latest_image_msg2_;
        has_image2 = true;
    }

    if (latest_image_msg3_) {
        out_image_msg3.header = latest_odom_msg_->header;
        out_image_msg3.header.stamp = cur_time;
        out_image_msg3.instance = rubis::instance_;
        out_image_msg3.msg = *latest_image_msg3_;
        has_image3 = true;
    }

    if (latest_image_msg4_) {
        out_image_msg4.header = latest_odom_msg_->header;
        out_image_msg4.header.stamp = cur_time;
        out_image_msg4.instance = rubis::instance_;
        out_image_msg4.msg = *latest_image_msg4_;
        has_image4 = true;
    }

    // Perform data processing
    read_data(n_);

    // Publish LiDAR and PoseTwist messages
    lidar_pub_.publish(out_lidar_msg);
    pose_twist_pub_.publish(out_pose_twist_msg);

    // Publish Image messages if available
    if (has_image1) {
        image_pub1_.publish(out_image_msg1);
    }
    if (has_image2) {
        image_pub2_.publish(out_image_msg2);
    }
    if (has_image3) {
        image_pub3_.publish(out_image_msg3);
    }
    if (has_image4) {
        image_pub4_.publish(out_image_msg4);
    }

    // Broadcast TF Transform
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion gnss_q(out_pose_twist_msg.pose.pose.orientation.x,
                          out_pose_twist_msg.pose.pose.orientation.y,
                          out_pose_twist_msg.pose.pose.orientation.z,
                          out_pose_twist_msg.pose.pose.orientation.w);
    transform.setOrigin(tf::Vector3(out_pose_twist_msg.pose.pose.position.x,
                                    out_pose_twist_msg.pose.pose.position.y,
                                    out_pose_twist_msg.pose.pose.position.z));
    transform.setRotation(gnss_q);

    br.sendTransform(tf::StampedTransform(transform, cur_time, "/map", "/base_link"));

    // Stop task profiling
    rubis::stop_task_profiling(rubis::instance_++, rubis::lidar_instance_++, rubis::vision_instance_++);
}

void SvlSensing::read_data(int n){    
    int stride_size = 64;
    int stride = 64 / sizeof(int);
    int data_num = data_size_ / sizeof(int);    
    int index_stride = (int)(data_num / 16);

    for(int j = 0; j < n; j++){
        int index = 0;
        int limit = index_stride;
        for(int i = 0; i <16; i++){
            for(; index < limit; index += stride){        
                data_[index] = data_[index] + 1;                          
            }
            limit += index_stride;
        }
    }

    return;
}

// Run method remains largely unchanged
void SvlSensing::run() {
    if(rate_ < 0)
        ros::spin();
    else{
        ros::Rate r(rate_);
        while (ros::ok()) {
            ros::spinOnce();
            id_++;
            r.sleep();
        }
    }
}
