#include "svl_sensing.h"
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <cstdlib>

SvlSensing::SvlSensing() {
    ros::NodeHandle private_nh("~");
    std::string node_name = ros::this_node::getName();
    std::string lidar_topic, odom_topic, image_topic_1, image_topic_2, image_topic_3, image_topic_4, pose_twist_topic;
    int image_topic_cnt = 0;

    private_nh.param<int>(node_name + "/rate", rate_, 5);
    
    private_nh.param<std::string>(node_name + "/pose_twist_topic", pose_twist_topic, "/rubis_current_pose_twist");

    private_nh.param<std::string>(node_name + "/lidar_topic", lidar_topic, "/points_raw_origin");
    lidar_sub_.subscribe(nh_, lidar_topic, 2);

    private_nh.param<std::string>(node_name + "/odom_topic", odom_topic, "/odom");
    odom_sub_.subscribe(nh_, odom_topic, 2);

    private_nh.param<std::string>(node_name + "/image_topic_1", image_topic_1, "none");

    if (image_topic_1.compare(std::string("none"))) {
        image_sub_1_.subscribe(nh_, image_topic_1, 2);
        image_topic_cnt++;
    }

    private_nh.param<std::string>(node_name + "/image_topic_2", image_topic_2, "none");
    if (image_topic_2.compare(std::string("none"))) {
        if (image_topic_cnt != 1) {
            std::cout << "# [ERROR] Set image_topic_1 first" << std::endl;
            exit(0);
        }
        image_sub_2_.subscribe(nh_, image_topic_2, 2);
        image_topic_cnt++;
    }

    private_nh.param<std::string>(node_name + "/image_topic_3", image_topic_3, "none");
    if (image_topic_3.compare(std::string("none"))) {
        if (image_topic_cnt != 2) {
            std::cout << "# [ERROR] Set image_topic_2 first" << std::endl;
            exit(0);
        }
        image_sub_3_.subscribe(nh_, image_topic_3, 2);
        image_topic_cnt++;
    }

    private_nh.param<std::string>(node_name + "/image_topic_4", image_topic_4, "none");
    if (image_topic_4.compare(std::string("none"))) {
        if (image_topic_cnt != 3) {
            std::cout << "# [ERROR] Set image_topic_3 first" << std::endl;
            exit(0);
        }
        image_sub_4_.subscribe(nh_, image_topic_4, 2);
        image_topic_cnt++;
    }

    if (image_topic_cnt == 0) {
        sync0_.reset(new message_filters::Synchronizer<SyncPolicyNoImage>(SyncPolicyNoImage(1), lidar_sub_, odom_sub_));
        sync0_->registerCallback(boost::bind(&SvlSensing::callback_no_image, this, _1, _2));
    } else if (image_topic_cnt == 1) {
        sync1_.reset(new message_filters::Synchronizer<SyncPolicyImage1>(SyncPolicyImage1(1), lidar_sub_, odom_sub_, image_sub_1_));
        sync1_->registerCallback(boost::bind(&SvlSensing::callback_image1, this, _1, _2, _3));
        image_pub1_ = nh_.advertise<rubis_msgs::Image>("/image_raw1", 1);
    } else if (image_topic_cnt == 2) {
        sync2_.reset(new message_filters::Synchronizer<SyncPolicyImage2>(SyncPolicyImage2(1), lidar_sub_, odom_sub_, image_sub_1_, image_sub_2_));
        sync2_->registerCallback(boost::bind(&SvlSensing::callback_image2, this, _1, _2, _3, _4));
        image_pub1_ = nh_.advertise<rubis_msgs::Image>("/image_raw1", 1);
        image_pub2_ = nh_.advertise<rubis_msgs::Image>("/image_raw2", 1);
    } else if (image_topic_cnt == 3) {
        sync3_.reset(new message_filters::Synchronizer<SyncPolicyImage3>(SyncPolicyImage3(1), lidar_sub_, odom_sub_, image_sub_1_, image_sub_2_,
                                                                         image_sub_3_));
        sync3_->registerCallback(boost::bind(&SvlSensing::callback_image3, this, _1, _2, _3, _4, _5));
        image_pub1_ = nh_.advertise<rubis_msgs::Image>("/image_raw1", 1);
        image_pub2_ = nh_.advertise<rubis_msgs::Image>("/image_raw2", 1);
        image_pub3_ = nh_.advertise<rubis_msgs::Image>("/image_raw3", 1);
    } else if (image_topic_cnt == 4) {
        sync4_.reset(new message_filters::Synchronizer<SyncPolicyImage4>(SyncPolicyImage4(1), lidar_sub_, odom_sub_, image_sub_1_, image_sub_2_,
                                                                         image_sub_3_, image_sub_4_));
        sync4_->registerCallback(boost::bind(&SvlSensing::callback_image4, this, _1, _2, _3, _4, _5, _6));
        image_pub1_ = nh_.advertise<rubis_msgs::Image>("/image_raw1", 1);
        image_pub2_ = nh_.advertise<rubis_msgs::Image>("/image_raw2", 1);
        image_pub3_ = nh_.advertise<rubis_msgs::Image>("/image_raw3", 1);
        image_pub4_ = nh_.advertise<rubis_msgs::Image>("/image_raw4", 1);
    }

    lidar_pub_ = nh_.advertise<rubis_msgs::PointCloud2>("/rubis_points_raw", 1);
    pose_twist_pub_ = nh_.advertise<rubis_msgs::PoseTwistStamped>(pose_twist_topic, 1);

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


    nh_.param(node_name + "/n", n_, 10);
    data_size_ = data_size_mb*1024*1024;
    data_ = (int *)aligned_alloc(32, data_size_);
    memset(data_, 0, data_size_);
}

SvlSensing::~SvlSensing() {}

void SvlSensing::read_data(int n){    
    int stride_size = 64;
    int stride = 64 / sizeof(int);
    int data_num = data_size_ / sizeof(int);    
    int index_stride = (int)(data_num / 16);

    int index = 0;
    int limit = index_stride;
    
    std::cout<<"n: "<<n<<std::endl;
    for(int j = 0; j < n; j++){
        for(int i = 0; i <16; i++){
            for(; index < limit; index += stride){
                data_[index] = data_[index] + 1;
            }
            limit += index_stride;
        }
    }

    return;
}

void SvlSensing::read_sequential_data(int m){    
    int stride_size = 64;
    int stride = 64 / sizeof(int);
    int data_num = data_size_ / sizeof(int);    
    int index_stride = (int)(data_num / 16);

    int index = 0;
    int limit = index_stride;
    
    for(int i = 0; i <16; i++){
        for(; index < limit; index += 1){
            data_[index] = data_[index] + 1;
        }
        limit += index_stride;
    }

    return;
}


void SvlSensing::callback_no_image(const sensor_msgs::PointCloud2::ConstPtr &lidar_msg, const nav_msgs::Odometry::ConstPtr &odom_msg) {
    rubis::start_task_profiling_at_initial_node(std::max(lidar_msg->header.stamp.sec, odom_msg->header.stamp.sec),
                                                std::max(lidar_msg->header.stamp.nsec, odom_msg->header.stamp.nsec));

    auto cur_time = ros::Time::now();
    rubis_msgs::PointCloud2 out_lidar_msg;
    out_lidar_msg.header = lidar_msg->header;
    out_lidar_msg.header.stamp = cur_time;
    out_lidar_msg.instance = rubis::instance_;
    out_lidar_msg.msg = *lidar_msg;
    out_lidar_msg.msg.header.stamp = cur_time;
    out_lidar_msg.msg.fields.at(3).datatype = 7;

    rubis_msgs::PoseTwistStamped out_pose_twist_msg;
    out_pose_twist_msg.header = odom_msg->header;
    out_pose_twist_msg.header.stamp = cur_time;
    out_pose_twist_msg.instance = rubis::instance_;
    out_pose_twist_msg.pose.header = odom_msg->header;
    out_pose_twist_msg.pose.header.stamp = cur_time;
    out_pose_twist_msg.pose.header.frame_id = "/map";
    out_pose_twist_msg.pose.pose = odom_msg->pose.pose;
    out_pose_twist_msg.twist.header.frame_id = "/map";
    out_pose_twist_msg.twist.header.stamp = cur_time;
    out_pose_twist_msg.twist.twist = odom_msg->twist.twist;

    read_data(n_);

    lidar_pub_.publish(out_lidar_msg);
    pose_twist_pub_.publish(out_pose_twist_msg);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion gnss_q(out_pose_twist_msg.pose.pose.orientation.x, out_pose_twist_msg.pose.pose.orientation.y, out_pose_twist_msg.pose.pose.orientation.z,
                          out_pose_twist_msg.pose.pose.orientation.w);
    transform.setOrigin(tf::Vector3(out_pose_twist_msg.pose.pose.position.x, out_pose_twist_msg.pose.pose.position.y, out_pose_twist_msg.pose.pose.position.z));
    transform.setRotation(gnss_q);
    
    br.sendTransform(tf::StampedTransform(transform, cur_time, "/map", "/base_link"));


    rubis::stop_task_profiling(rubis::instance_++, rubis::lidar_instance_++, rubis::vision_instance_++);
    return;
}

void SvlSensing::callback_image1(const sensor_msgs::PointCloud2::ConstPtr &lidar_msg, const nav_msgs::Odometry::ConstPtr &odom_msg,
                                 const sensor_msgs::Image::ConstPtr &image_msg1) {
    rubis::start_task_profiling_at_initial_node(std::max(lidar_msg->header.stamp.sec, odom_msg->header.stamp.sec),
                                                std::max(lidar_msg->header.stamp.nsec, odom_msg->header.stamp.nsec));

    auto cur_time = ros::Time::now();
    rubis_msgs::PointCloud2 out_lidar_msg;
    out_lidar_msg.header = lidar_msg->header;
    out_lidar_msg.header.stamp = cur_time;
    out_lidar_msg.instance = rubis::instance_;
    out_lidar_msg.msg = *lidar_msg;
    out_lidar_msg.msg.header.stamp = cur_time;
    out_lidar_msg.msg.fields.at(3).datatype = 7;

    rubis_msgs::PoseTwistStamped out_pose_twist_msg;
    out_pose_twist_msg.header = odom_msg->header;
    out_pose_twist_msg.header.stamp = cur_time;
    out_pose_twist_msg.instance = rubis::instance_;
    out_pose_twist_msg.pose.header = odom_msg->header;
    out_pose_twist_msg.pose.header.stamp = cur_time;
    out_pose_twist_msg.pose.header.frame_id = "/map";
    out_pose_twist_msg.pose.pose = odom_msg->pose.pose;
    out_pose_twist_msg.twist.header.frame_id = "/map";
    out_pose_twist_msg.twist.header.stamp = cur_time;
    out_pose_twist_msg.twist.twist = odom_msg->twist.twist;

    rubis_msgs::Image out_image_msg1;
    out_image_msg1.header = odom_msg->header;
    out_image_msg1.header.stamp = cur_time;
    out_image_msg1.instance = rubis::instance_;
    out_image_msg1.msg = *image_msg1;

    // read_data(n_);

    lidar_pub_.publish(out_lidar_msg);
    pose_twist_pub_.publish(out_pose_twist_msg);
    image_pub1_.publish(out_image_msg1);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion gnss_q(out_pose_twist_msg.pose.pose.orientation.x, out_pose_twist_msg.pose.pose.orientation.y, out_pose_twist_msg.pose.pose.orientation.z,
                          out_pose_twist_msg.pose.pose.orientation.w);
    transform.setOrigin(tf::Vector3(out_pose_twist_msg.pose.pose.position.x, out_pose_twist_msg.pose.pose.position.y, out_pose_twist_msg.pose.pose.position.z));
    transform.setRotation(gnss_q);
    
    br.sendTransform(tf::StampedTransform(transform, cur_time, "/map", "/base_link"));


    rubis::stop_task_profiling(rubis::instance_++, rubis::lidar_instance_++, rubis::vision_instance_++);
    return;
}

void SvlSensing::callback_image2(const sensor_msgs::PointCloud2::ConstPtr &lidar_msg, const nav_msgs::Odometry::ConstPtr &odom_msg,
                                 const sensor_msgs::Image::ConstPtr &image_msg1, const sensor_msgs::Image::ConstPtr &image_msg2) {
    rubis::start_task_profiling_at_initial_node(std::max(lidar_msg->header.stamp.sec, odom_msg->header.stamp.sec),
                                                std::max(lidar_msg->header.stamp.nsec, odom_msg->header.stamp.nsec));

    auto cur_time = ros::Time::now();
    rubis_msgs::PointCloud2 out_lidar_msg;
    out_lidar_msg.header = lidar_msg->header;
    out_lidar_msg.header.stamp = cur_time;
    out_lidar_msg.instance = rubis::instance_;
    out_lidar_msg.msg = *lidar_msg;
    out_lidar_msg.msg.header.stamp = cur_time;
    out_lidar_msg.msg.fields.at(3).datatype = 7;

    rubis_msgs::PoseTwistStamped out_pose_twist_msg;
    out_pose_twist_msg.header = odom_msg->header;
    out_pose_twist_msg.header.stamp = cur_time;
    out_pose_twist_msg.instance = rubis::instance_;
    out_pose_twist_msg.pose.header = odom_msg->header;
    out_pose_twist_msg.pose.header.stamp = cur_time;
    out_pose_twist_msg.pose.header.frame_id = "/map";
    out_pose_twist_msg.pose.pose = odom_msg->pose.pose;
    out_pose_twist_msg.twist.header.frame_id = "/map";
    out_pose_twist_msg.twist.header.stamp = cur_time;
    out_pose_twist_msg.twist.twist = odom_msg->twist.twist;

    rubis_msgs::Image out_image_msg1;
    out_image_msg1.header = odom_msg->header;
    out_image_msg1.header.stamp = cur_time;
    out_image_msg1.instance = rubis::instance_;
    out_image_msg1.msg = *image_msg1;

    rubis_msgs::Image out_image_msg2;
    out_image_msg2.header = odom_msg->header;
    out_image_msg2.header.stamp = cur_time;
    out_image_msg2.instance = rubis::instance_;
    out_image_msg2.msg = *image_msg2;

    read_data(n_);

    lidar_pub_.publish(out_lidar_msg);
    pose_twist_pub_.publish(out_pose_twist_msg);
    image_pub1_.publish(out_image_msg1);
    image_pub2_.publish(out_image_msg2);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion gnss_q(out_pose_twist_msg.pose.pose.orientation.x, out_pose_twist_msg.pose.pose.orientation.y, out_pose_twist_msg.pose.pose.orientation.z,
                          out_pose_twist_msg.pose.pose.orientation.w);
    transform.setOrigin(tf::Vector3(out_pose_twist_msg.pose.pose.position.x, out_pose_twist_msg.pose.pose.position.y, out_pose_twist_msg.pose.pose.position.z));
    transform.setRotation(gnss_q);
    
    br.sendTransform(tf::StampedTransform(transform, cur_time, "/map", "/base_link"));


    rubis::stop_task_profiling(rubis::instance_++, rubis::lidar_instance_++, rubis::vision_instance_++);
    return;
}

void SvlSensing::callback_image3(const sensor_msgs::PointCloud2::ConstPtr &lidar_msg, const nav_msgs::Odometry::ConstPtr &odom_msg,
                                 const sensor_msgs::Image::ConstPtr &image_msg1, const sensor_msgs::Image::ConstPtr &image_msg2,
                                 const sensor_msgs::Image::ConstPtr &image_msg3) {
    rubis::start_task_profiling_at_initial_node(std::max(lidar_msg->header.stamp.sec, odom_msg->header.stamp.sec),
                                                std::max(lidar_msg->header.stamp.nsec, odom_msg->header.stamp.nsec));

    auto cur_time = ros::Time::now();
    rubis_msgs::PointCloud2 out_lidar_msg;
    out_lidar_msg.header = lidar_msg->header;
    out_lidar_msg.header.stamp = cur_time;
    out_lidar_msg.instance = rubis::instance_;
    out_lidar_msg.msg = *lidar_msg;
    out_lidar_msg.msg.header.stamp = cur_time;
    out_lidar_msg.msg.fields.at(3).datatype = 7;

    rubis_msgs::PoseTwistStamped out_pose_twist_msg;
    out_pose_twist_msg.header = odom_msg->header;
    out_pose_twist_msg.header.stamp = cur_time;
    out_pose_twist_msg.instance = rubis::instance_;
    out_pose_twist_msg.pose.header = odom_msg->header;
    out_pose_twist_msg.pose.header.stamp = cur_time;
    out_pose_twist_msg.pose.header.frame_id = "/map";
    out_pose_twist_msg.pose.pose = odom_msg->pose.pose;
    out_pose_twist_msg.twist.header.frame_id = "/map";
    out_pose_twist_msg.twist.header.stamp = cur_time;
    out_pose_twist_msg.twist.twist = odom_msg->twist.twist;

    rubis_msgs::Image out_image_msg1;
    out_image_msg1.header = odom_msg->header;
    out_image_msg1.header.stamp = cur_time;
    out_image_msg1.instance = rubis::instance_;
    out_image_msg1.msg = *image_msg1;

    rubis_msgs::Image out_image_msg2;
    out_image_msg2.header = odom_msg->header;
    out_image_msg2.header.stamp = cur_time;
    out_image_msg2.instance = rubis::instance_;
    out_image_msg2.msg = *image_msg2;

    rubis_msgs::Image out_image_msg3;
    out_image_msg3.header = odom_msg->header;
    out_image_msg3.header.stamp = cur_time;
    out_image_msg3.instance = rubis::instance_;
    out_image_msg3.msg = *image_msg3;

    read_data(n_);

    lidar_pub_.publish(out_lidar_msg);
    pose_twist_pub_.publish(out_pose_twist_msg);
    image_pub1_.publish(out_image_msg1);
    image_pub2_.publish(out_image_msg2);
    image_pub3_.publish(out_image_msg3);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion gnss_q(out_pose_twist_msg.pose.pose.orientation.x, out_pose_twist_msg.pose.pose.orientation.y, out_pose_twist_msg.pose.pose.orientation.z,
                          out_pose_twist_msg.pose.pose.orientation.w);
    transform.setOrigin(tf::Vector3(out_pose_twist_msg.pose.pose.position.x, out_pose_twist_msg.pose.pose.position.y, out_pose_twist_msg.pose.pose.position.z));
    transform.setRotation(gnss_q);
    
    br.sendTransform(tf::StampedTransform(transform, cur_time, "/map", "/base_link"));


    rubis::stop_task_profiling(rubis::instance_++, rubis::lidar_instance_++, rubis::vision_instance_++);
    return;
}

void SvlSensing::callback_image4(const sensor_msgs::PointCloud2::ConstPtr &lidar_msg, const nav_msgs::Odometry::ConstPtr &odom_msg,
                                 const sensor_msgs::Image::ConstPtr &image_msg1, const sensor_msgs::Image::ConstPtr &image_msg2,
                                 const sensor_msgs::Image::ConstPtr &image_msg3, const sensor_msgs::Image::ConstPtr &image_msg4) {
    rubis::start_task_profiling_at_initial_node(std::max(lidar_msg->header.stamp.sec, odom_msg->header.stamp.sec),
                                                std::max(lidar_msg->header.stamp.nsec, odom_msg->header.stamp.nsec));

    auto cur_time = ros::Time::now();
    rubis_msgs::PointCloud2 out_lidar_msg;
    out_lidar_msg.header = lidar_msg->header;
    out_lidar_msg.header.stamp = cur_time;
    out_lidar_msg.instance = rubis::instance_;
    out_lidar_msg.msg = *lidar_msg;
    out_lidar_msg.msg.header.stamp = cur_time;
    out_lidar_msg.msg.fields.at(3).datatype = 7;

    rubis_msgs::PoseTwistStamped out_pose_twist_msg;
    out_pose_twist_msg.header = odom_msg->header;
    out_pose_twist_msg.header.stamp = cur_time;
    out_pose_twist_msg.instance = rubis::instance_;
    out_pose_twist_msg.pose.header = odom_msg->header;
    out_pose_twist_msg.pose.header.stamp = cur_time;
    out_pose_twist_msg.pose.header.frame_id = "/map";
    out_pose_twist_msg.pose.pose = odom_msg->pose.pose;
    out_pose_twist_msg.twist.header.frame_id = "/map";
    out_pose_twist_msg.twist.header.stamp = cur_time;
    out_pose_twist_msg.twist.twist = odom_msg->twist.twist;

    rubis_msgs::Image out_image_msg1;
    out_image_msg1.header = odom_msg->header;
    out_image_msg1.header.stamp = cur_time;
    out_image_msg1.instance = rubis::instance_;
    out_image_msg1.msg = *image_msg1;

    rubis_msgs::Image out_image_msg2;
    out_image_msg2.header = odom_msg->header;
    out_image_msg2.header.stamp = cur_time;
    out_image_msg2.instance = rubis::instance_;
    out_image_msg2.msg = *image_msg2;

    rubis_msgs::Image out_image_msg3;
    out_image_msg3.header = odom_msg->header;
    out_image_msg3.header.stamp = cur_time;
    out_image_msg3.instance = rubis::instance_;
    out_image_msg3.msg = *image_msg3;

    rubis_msgs::Image out_image_msg4;
    out_image_msg4.header = odom_msg->header;
    out_image_msg4.header.stamp = cur_time;
    out_image_msg4.instance = rubis::instance_;
    out_image_msg4.msg = *image_msg4;

    read_data(n_);

    lidar_pub_.publish(out_lidar_msg);
    pose_twist_pub_.publish(out_pose_twist_msg);
    image_pub1_.publish(out_image_msg1);
    image_pub2_.publish(out_image_msg2);
    image_pub3_.publish(out_image_msg3);
    image_pub4_.publish(out_image_msg4);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion gnss_q(out_pose_twist_msg.pose.pose.orientation.x, out_pose_twist_msg.pose.pose.orientation.y, out_pose_twist_msg.pose.pose.orientation.z,
                          out_pose_twist_msg.pose.pose.orientation.w);
    transform.setOrigin(tf::Vector3(out_pose_twist_msg.pose.pose.position.x, out_pose_twist_msg.pose.pose.position.y, out_pose_twist_msg.pose.pose.position.z));
    transform.setRotation(gnss_q);
    
    br.sendTransform(tf::StampedTransform(transform, cur_time, "/map", "/base_link"));

    rubis::stop_task_profiling(rubis::instance_++, rubis::lidar_instance_++, rubis::vision_instance_++);
    return;
}

void SvlSensing::run() {
    if(rate_ < 0)
        ros::spin();
    else{
    
    ros::Rate r(rate_);
    while (ros::ok()) {
        ros::spinOnce();
        r.sleep();
    }

    }
}