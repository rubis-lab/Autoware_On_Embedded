#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <sstream>
#include <string>
#include <ros/ros.h>

#include <diagnostic_msgs/DiagnosticArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float32.h>

#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

// #define DEBUG_ENABLE

struct pose
{
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};

class GicpLocalizer{
public:

    GicpLocalizer(ros::NodeHandle &nh, ros::NodeHandle &private_nh);
    ~GicpLocalizer();

private:
    ros::NodeHandle nh_, private_nh_;

    ros::Subscriber initial_pose_sub_;
    ros::Subscriber map_points_sub_;
    ros::Subscriber sensor_points_sub_;
    ros::Subscriber gnss_pose_sub_;

    ros::Publisher sensor_aligned_pose_pub_;
    ros::Publisher gicp_pose_pub_;
    ros::Publisher gicp_vel_pub_;
    ros::Publisher exe_time_pub_;
    ros::Publisher transform_probability_pub_;
    ros::Publisher iteration_num_pub_;
    ros::Publisher diagnostics_pub_;

    // pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> gicp_;
    fast_gicp::FastGICPSingleThread<pcl::PointXYZ, pcl::PointXYZ> fgicp_st_;
    fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> fgicp_mt_;
    fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> vgicp_;


    tf2_ros::Buffer tf2_buffer_;
    tf2_ros::TransformListener tf2_listener_;
    tf2_ros::TransformBroadcaster tf2_broadcaster_;

    Eigen::Matrix4f base_to_sensor_matrix_;
    Eigen::Matrix4f pre_trans, delta_trans;
    bool pose_initialized = false;

    std::string base_frame_;
    std::string map_frame_;
    std::string sensor_frame_;
    bool enable_gnss_backup_ = false;
    double gnss_backup_thr_;

    // init guess for gicp
    geometry_msgs::PoseWithCovarianceStamped initial_pose_cov_msg_;

    std::mutex gicp_map_mtx_;

    double resolution_{};
    float leafsize_{};
    int numThreads_{};
    std::string neighborSearchMethod_;

    // downsampling
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid_;

    double converged_param_transform_probability_{};
    std::thread diagnostic_thread_;
    std::map<std::string, std::string> key_value_stdmap_;

    // for twist
    bool pose_published = false;
    bool should_backup = false;
    ros::Time previous_ts;
    struct pose init_pose;
    struct pose previous_pose;
    struct pose gnss_pose;

    // function
    void init_params();
    void timer_diagnostic();

    bool get_transform(const std::string & target_frame, const std::string & source_frame,
                       const geometry_msgs::TransformStamped::Ptr & transform_stamped_ptr,
                       const ros::Time & time_stamp);
    bool get_transform(const std::string & target_frame, 
                       const std::string & source_frame,
                       const geometry_msgs::TransformStamped::Ptr & transform_stamped_ptr);
    void publish_tf(const std::string & frame_id, const std::string & child_frame_id,
                    const geometry_msgs::PoseStamped & pose_msg);

    void callback_pointsmap(const sensor_msgs::PointCloud2::ConstPtr & pointcloud2_msg_ptr);
    void callback_init_pose(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr & pose_conv_msg_ptr);
    void callback_pointcloud(const sensor_msgs::PointCloud2::ConstPtr & pointcloud2_msg_ptr);
    void callback_gnss_pose(const geometry_msgs::PoseStamped::ConstPtr & gnss_pose_msg_ptr);

    double calcDiffForRadian(const double, const double);
    bool convertPoseIntoRelativeCoordinate(const struct pose target_pose, const struct pose reference_pose);

};// GicpLocalizer Core