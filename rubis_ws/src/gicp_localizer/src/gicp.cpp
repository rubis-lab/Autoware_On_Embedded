#include "gicp.h"

GicpLocalizer::GicpLocalizer(ros::NodeHandle &nh, ros::NodeHandle &private_nh)
:nh_(nh)
,private_nh_(private_nh)
,tf2_listener_(tf2_buffer_)
{
    key_value_stdmap_["state"] = "Initializing";
    init_params();

    sensor_aligned_pose_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("points_aligned", 10);
    gicp_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("gicp_pose", 10);
    gicp_vel_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("gicp_vel", 10);
    exe_time_pub_ = nh_.advertise<std_msgs::Float32>("exe_time_ms", 10);
    diagnostics_pub_ = nh_.advertise<diagnostic_msgs::DiagnosticArray>("diagnostics", 10);

    initial_pose_sub_ = nh_.subscribe("initialpose", 100, &GicpLocalizer::callback_init_pose, this);
    map_points_sub_ = nh_.subscribe("points_map", 1, &GicpLocalizer::callback_pointsmap, this);
    sensor_points_sub_ = nh_.subscribe("filtered_points", 1, &GicpLocalizer::callback_pointcloud, this);

    gnss_pose_sub_ = nh.subscribe("/gnss_pose", 8, &GicpLocalizer::callback_gnss_pose, this);

    pose_initialized = false;

    diagnostic_thread_ = std::thread(&GicpLocalizer::timer_diagnostic, this);
    diagnostic_thread_.detach();
}

GicpLocalizer::~GicpLocalizer() = default;

void GicpLocalizer::timer_diagnostic(){
    ros::Rate rate(100);
    while(ros::ok()){
        diagnostic_msgs::DiagnosticStatus diag_status_msg;
        diag_status_msg.name = "gicp_scan_matcher";
        diag_status_msg.hardware_id = "";

        for (const auto & key_value : key_value_stdmap_) {
            diagnostic_msgs::KeyValue key_value_msg;
            key_value_msg.key = key_value.first;
            key_value_msg.value = key_value.second;
            diag_status_msg.values.push_back(key_value_msg);
        }

        diag_status_msg.level = diagnostic_msgs::DiagnosticStatus::OK;
        diag_status_msg.message = "";
        if (key_value_stdmap_.count("state") && key_value_stdmap_["state"] == "Initializing") {
            diag_status_msg.level = diagnostic_msgs::DiagnosticStatus::WARN;
            diag_status_msg.message += "Initializing State. ";
        }
        if (key_value_stdmap_.count("skipping_publish_num") &&
            std::stoi(key_value_stdmap_["skipping_publish_num"]) > 1) {
            diag_status_msg.level = diagnostic_msgs::DiagnosticStatus::WARN;
            diag_status_msg.message += "skipping_publish_num > 1. ";
        }
        if (key_value_stdmap_.count("skipping_publish_num") &&
            std::stoi(key_value_stdmap_["skipping_publish_num"]) >= 5) {
            diag_status_msg.level = diagnostic_msgs::DiagnosticStatus::ERROR;
            diag_status_msg.message += "skipping_publish_num exceed limit. ";
        }

        diagnostic_msgs::DiagnosticArray diag_msg;
        diag_msg.header.stamp = ros::Time::now();
        diag_msg.status.push_back(diag_status_msg);
        diagnostics_pub_.publish(diag_msg);
        rate.sleep();
    }
}

void GicpLocalizer::callback_gnss_pose(const geometry_msgs::PoseStamped::ConstPtr & gnss_pose_msg_ptr){
    gnss_pose.x = gnss_pose_msg_ptr->pose.position.x;
    gnss_pose.y = gnss_pose_msg_ptr->pose.position.y;
    gnss_pose.z = gnss_pose_msg_ptr->pose.position.z;

    // Use orientation info not from IMU, but previous success pose

    // tf::Quaternion quat(gnss_pose_msg_ptr->pose.orientation.x,
    //     gnss_pose_msg_ptr->pose.orientation.y,
    //     gnss_pose_msg_ptr->pose.orientation.z,
    //     gnss_pose_msg_ptr->pose.orientation.w);
    // tf::Matrix3x3(quat).getRPY(gnss_pose.roll, gnss_pose.pitch, gnss_pose.yaw);
}

void GicpLocalizer::callback_init_pose(
   const geometry_msgs::PoseWithCovarianceStamped::ConstPtr & initial_pose_msg_ptr){
    if (initial_pose_msg_ptr->header.frame_id == map_frame_) {
        initial_pose_cov_msg_ = *initial_pose_msg_ptr;
    }else{
        // get TF from pose_frame to map_frame
        geometry_msgs::TransformStamped::Ptr TF_pose_to_map_ptr(new geometry_msgs::TransformStamped);
        get_transform(map_frame_, initial_pose_msg_ptr->header.frame_id, TF_pose_to_map_ptr);

        // transform pose_frame to map_frame
        geometry_msgs::PoseWithCovarianceStamped::Ptr mapTF_initial_pose_msg_ptr(
                new geometry_msgs::PoseWithCovarianceStamped);
        tf2::doTransform(*initial_pose_msg_ptr, *mapTF_initial_pose_msg_ptr, *TF_pose_to_map_ptr);
        // mapTF_initial_pose_msg_ptr->header.stamp = initial_pose_msg_ptr->header.stamp;
        initial_pose_cov_msg_ = *mapTF_initial_pose_msg_ptr;
    }

    pose_initialized = false;
}

void GicpLocalizer::callback_pointsmap(
   const sensor_msgs::PointCloud2::ConstPtr & map_points_msg_ptr){
    fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> vgicp_new;

    vgicp_new.setResolution(resolution_);
    vgicp_new.setNumThreads(numThreads_);

    if(neighborSearchMethod_ == "DIRECT7"){
        vgicp_new.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
    }
    else if(neighborSearchMethod_ == "DIRECT27"){
        vgicp_new.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT27);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr map_points_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*map_points_msg_ptr, *map_points_ptr);
//    vgicp_new.setInputTarget(map_points_ptr);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
    voxelgrid_.setInputCloud(map_points_ptr);
    voxelgrid_.filter(*filtered);
    map_points_ptr = filtered;

    vgicp_.setInputTarget(map_points_ptr);
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//    vgicp_new.align(*output_cloud, Eigen::Matrix4f::Identity());
    vgicp_.align(*output_cloud, Eigen::Matrix4f::Identity());

}

void GicpLocalizer::callback_pointcloud(
    const sensor_msgs::PointCloud2::ConstPtr & sensor_points_sensorTF_msg_ptr)
{
    const auto exe_start_time = std::chrono::system_clock::now();
    // add map mutex
    std::lock_guard<std::mutex> lock(gicp_map_mtx_);

    // const std::string sensor_frame = sensor_points_sensorTF_msg_ptr->header.frame_id;
    const auto sensor_ros_time = sensor_points_sensorTF_msg_ptr->header.stamp;

    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> sensor_points_sensorTF_ptr(
            new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*sensor_points_sensorTF_msg_ptr, *sensor_points_sensorTF_ptr);
    // get TF base to sensor
    geometry_msgs::TransformStamped::Ptr TF_base_to_sensor_ptr(new geometry_msgs::TransformStamped);
    // get_transform(base_frame_, sensor_frame, TF_base_to_sensor_ptr);
    get_transform(base_frame_, sensor_frame_, TF_base_to_sensor_ptr);

    const Eigen::Affine3d base_to_sensor_affine = tf2::transformToEigen(*TF_base_to_sensor_ptr);
    const Eigen::Matrix4f base_to_sensor_matrix = base_to_sensor_affine.matrix().cast<float>();
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> sensor_points_baselinkTF_ptr(
            new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(
            *sensor_points_sensorTF_ptr, *sensor_points_baselinkTF_ptr, base_to_sensor_matrix);

    vgicp_.setInputSource(sensor_points_baselinkTF_ptr);

    if(vgicp_.getInputTarget() == nullptr){
        ROS_WARN_STREAM_THROTTLE(1, "No MAP!");
        return;
    }
    // align
    Eigen::Matrix4f initial_pose_matrix;
    if (!pose_initialized){
        Eigen::Affine3d initial_pose_affine;
        tf2::fromMsg(initial_pose_cov_msg_.pose.pose, initial_pose_affine);
        initial_pose_matrix = initial_pose_affine.matrix().cast<float>();
        // for the first time, we don't know the pre_trans, so just use the init_trans,
        // which means, the delta trans for the second time is 0

        if(init_pose.x != 0.0){
            Eigen::Translation3f init_translation(init_pose.x, init_pose.y, init_pose.z);
            Eigen::AngleAxisf init_rotation_x(init_pose.roll, Eigen::Vector3f::UnitX());
            Eigen::AngleAxisf init_rotation_y(init_pose.pitch, Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf init_rotation_z(init_pose.yaw, Eigen::Vector3f::UnitZ());
            initial_pose_matrix = (init_translation * init_rotation_z * init_rotation_y * init_rotation_x).matrix();
            init_pose.x = 0.0;
        }

        pre_trans = initial_pose_matrix;
        pose_initialized = true;
    }
    else if(should_backup){
        std::cout << "[GICP Localizer]Backup by gnss" << std::endl;
        Eigen::Translation3f init_translation(gnss_pose.x, gnss_pose.y, gnss_pose.z);
        Eigen::AngleAxisf init_rotation_x(gnss_pose.roll, Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf init_rotation_y(gnss_pose.pitch, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf init_rotation_z(gnss_pose.yaw, Eigen::Vector3f::UnitZ());
        initial_pose_matrix = (init_translation * init_rotation_z * init_rotation_y * init_rotation_x).matrix();

        pre_trans = initial_pose_matrix;
        should_backup = false;
    }
    else
    {
        // use predicted pose as init guess (currently we only impl linear model)
        initial_pose_matrix = pre_trans * delta_trans;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    const auto align_start_time = std::chrono::system_clock::now();
    key_value_stdmap_["state"] = "Aligning";
    vgicp_.align(*output_cloud, initial_pose_matrix);
    key_value_stdmap_["state"] = "Sleeping";
    const auto align_end_time = std::chrono::system_clock::now();
    const float align_time = std::chrono::duration_cast<std::chrono::microseconds>(align_end_time - align_start_time).count() /1000.0;

    const Eigen::Matrix4f result_pose_matrix = vgicp_.getFinalTransformation();
    Eigen::Affine3d result_pose_affine;
    result_pose_affine.matrix() = result_pose_matrix.cast<double>();
    const geometry_msgs::Pose result_pose_msg = tf2::toMsg(result_pose_affine);

    const auto exe_end_time = std::chrono::system_clock::now();
    const auto exe_time = std::chrono::duration_cast<std::chrono::microseconds>(exe_end_time - exe_start_time).count() / 1000.0;
    // calculate the delta tf from pre_trans to current_trans
    delta_trans = pre_trans.inverse() * result_pose_matrix;

    Eigen::Vector3f delta_translation = delta_trans.block<3, 1>(0, 3);

    #ifdef DEBUG_ENABLE
    std::cout<<"delta x: "<<delta_translation(0) << " y: "<<delta_translation(1)<<
             " z: "<<delta_translation(2)<<std::endl;
    
    std::cout<<"delta yaw: "<<delta_euler(0) << " pitch: "<<delta_euler(1)<<
             " roll: "<<delta_euler(2)<<std::endl;
    #endif

    Eigen::Matrix3f delta_rotation_matrix = delta_trans.block<3, 3>(0, 0);
    Eigen::Vector3f delta_euler = delta_rotation_matrix.eulerAngles(2,1,0);

    pre_trans = result_pose_matrix;

    // publish pose
    geometry_msgs::PoseStamped result_pose_stamped_msg;
    result_pose_stamped_msg.header.stamp = sensor_ros_time;
    result_pose_stamped_msg.header.frame_id = map_frame_;
    result_pose_stamped_msg.pose = result_pose_msg;

    gicp_pose_pub_.publish(result_pose_stamped_msg);

    // publish twist
    if(!pose_published){
        previous_ts = sensor_ros_time;
        previous_pose.x = result_pose_msg.position.x;
        previous_pose.y = result_pose_msg.position.y;
        previous_pose.z = result_pose_msg.position.z;

        tf::Quaternion quat(result_pose_msg.orientation.x, result_pose_msg.orientation.y, result_pose_msg.orientation.z, result_pose_msg.orientation.w);
        tf::Matrix3x3(quat).getRPY(previous_pose.roll, previous_pose.pitch, previous_pose.yaw);

        gnss_pose.roll = previous_pose.roll;
        gnss_pose.pitch = previous_pose.pitch;
        gnss_pose.yaw = previous_pose.yaw;

        pose_published = true;
    }
    else{
        struct pose current_pose;
        geometry_msgs::TwistStamped twist_stamped_msg;
        result_pose_stamped_msg.header.stamp = sensor_ros_time;
        result_pose_stamped_msg.header.frame_id = map_frame_;

        current_pose.x = result_pose_msg.position.x;
        current_pose.y = result_pose_msg.position.y;
        current_pose.z = result_pose_msg.position.z;

        // get Quaternion
        tf::Quaternion quat(result_pose_msg.orientation.x, result_pose_msg.orientation.y, result_pose_msg.orientation.z, result_pose_msg.orientation.w);
        // converted to RPY[-pi : pi]
        tf::Matrix3x3(quat).getRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

        double diff_time = (sensor_ros_time - previous_ts).toSec();
        previous_ts = sensor_ros_time;

        double diff_x = current_pose.x - previous_pose.x;
        double diff_y = current_pose.y - previous_pose.y;
        double diff_z = current_pose.z - previous_pose.z;

        double diff_yaw = calcDiffForRadian(current_pose.yaw, previous_pose.yaw);
        double diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

        if(!convertPoseIntoRelativeCoordinate(current_pose, previous_pose)){
            diff *= -1;
        }

        double current_velocity = (diff_time > 0) ? (diff / diff_time) : 0;
        double angular_velocity = (diff_time > 0) ? (diff_yaw / diff_time) : 0;

        twist_stamped_msg.twist.linear.x = current_velocity;
        twist_stamped_msg.twist.linear.y = 0.0;
        twist_stamped_msg.twist.linear.z = 0.0;
        twist_stamped_msg.twist.angular.x = 0.0;
        twist_stamped_msg.twist.angular.y = 0.0;
        twist_stamped_msg.twist.angular.z = angular_velocity;

        // Save previous angular info for backup
        double gnss_gicp_diff = (current_pose.x - gnss_pose.x) * (current_pose.x - gnss_pose.x) + (current_pose.y - gnss_pose.y) * (current_pose.y - gnss_pose.y);

        if(enable_gnss_backup_ && gnss_gicp_diff > gnss_backup_thr_ * gnss_backup_thr_){
            previous_pose.x = gnss_pose.x;
            previous_pose.y = gnss_pose.y;
            previous_pose.z = gnss_pose.z;
            should_backup = true;
        }
        else{
            previous_pose.x = current_pose.x;
            previous_pose.y = current_pose.y;
            previous_pose.z = current_pose.z;
            previous_pose.roll = current_pose.roll;
            previous_pose.pitch = current_pose.pitch;
            previous_pose.yaw = current_pose.yaw;

            gnss_pose.roll = current_pose.roll;
            gnss_pose.pitch = current_pose.pitch;
            gnss_pose.yaw = current_pose.yaw;
        }

        gicp_vel_pub_.publish(twist_stamped_msg);
    }

    // publish tf(map frame to base frame)
    publish_tf(map_frame_, base_frame_, result_pose_stamped_msg);

    // publish aligned point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr sensor_points_mapTF_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(
            *sensor_points_baselinkTF_ptr, *sensor_points_mapTF_ptr, result_pose_matrix);
    sensor_msgs::PointCloud2 sensor_points_mapTF_msg;
    pcl::toROSMsg(*sensor_points_mapTF_ptr, sensor_points_mapTF_msg);
    sensor_points_mapTF_msg.header.stamp = sensor_ros_time;
    sensor_points_mapTF_msg.header.frame_id = map_frame_;
    sensor_aligned_pose_pub_.publish(sensor_points_mapTF_msg);

    std_msgs::Float32 exe_time_msg;
    exe_time_msg.data = exe_time;
    exe_time_pub_.publish(exe_time_msg);

    key_value_stdmap_["seq"] = std::to_string(sensor_points_sensorTF_msg_ptr->header.seq);
    #ifdef DEBUG_ENABLE
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "align_time: " << align_time << "ms" << std::endl;
    std::cout << "exe_time: " << exe_time << "ms" << std::endl;
    #endif
}

double GicpLocalizer::calcDiffForRadian(const double lhs_rad, const double rhs_rad)
{
  double diff_rad = lhs_rad - rhs_rad;
  if (diff_rad >= M_PI)
    diff_rad = diff_rad - 2 * M_PI;
  else if (diff_rad < -M_PI)
    diff_rad = diff_rad + 2 * M_PI;
  return diff_rad;
}

bool GicpLocalizer::convertPoseIntoRelativeCoordinate(const struct pose target_pose, const struct pose reference_pose)
{
    tf::Quaternion target_q;
    target_q.setRPY(target_pose.roll, target_pose.pitch, target_pose.yaw);
    tf::Vector3 target_v(target_pose.x, target_pose.y, target_pose.z);
    tf::Transform target_tf(target_q, target_v);

    tf::Quaternion reference_q;
    reference_q.setRPY(reference_pose.roll, reference_pose.pitch, reference_pose.yaw);
    tf::Vector3 reference_v(reference_pose.x, reference_pose.y, reference_pose.z);
    tf::Transform reference_tf(reference_q, reference_v);

    tf::Transform trans_target_tf = reference_tf.inverse() * target_tf;

    pose trans_target_pose;
    trans_target_pose.x = trans_target_tf.getOrigin().getX();
    trans_target_pose.y = trans_target_tf.getOrigin().getY();
    trans_target_pose.z = trans_target_tf.getOrigin().getZ();
    tf::Matrix3x3 tmp_m(trans_target_tf.getRotation());
    tmp_m.getRPY(trans_target_pose.roll, trans_target_pose.pitch, trans_target_pose.yaw);

    return trans_target_pose.x >= 0;
}

void GicpLocalizer::init_params(){
    private_nh_.param("base_frame", base_frame_, std::string("base_link"));
    private_nh_.param("sensor_frame", sensor_frame_, std::string("velodyne"));
    ROS_INFO("base_frame_id: %s", base_frame_.c_str());

    leafsize_ = 0.1;
    resolution_ = 1.0;
    numThreads_ = 2;

    private_nh_.getParam("resolution", resolution_);
    private_nh_.getParam("numthreads", numThreads_);
    private_nh_.getParam("leafsize", leafsize_);

    private_nh_.param("neighborSearchMethod", neighborSearchMethod_, std::string("DIRECT1"));

    private_nh_.param("init_x", init_pose.x, 0.0);
    private_nh_.param("init_y", init_pose.y, 0.0);
    private_nh_.param("init_z", init_pose.z, 0.0);
    private_nh_.param("init_roll", init_pose.roll, 0.0);
    private_nh_.param("init_pitch", init_pose.pitch, 0.0);
    private_nh_.param("init_yaw", init_pose.yaw, 0.0);

    private_nh_.param("enable_gnss_backup", enable_gnss_backup_, false);
    private_nh_.param("gnss_backup_threshold", gnss_backup_thr_, 10.0);

    voxelgrid_.setLeafSize(leafsize_, leafsize_, leafsize_);
    vgicp_.setResolution(resolution_);
    vgicp_.setNumThreads(numThreads_);

    map_frame_ = "map";

    ROS_INFO( "resolution: %lf, numthreads: %d", resolution_, numThreads_);
}

bool GicpLocalizer::get_transform(
   const std::string & target_frame, const std::string & source_frame,
   const geometry_msgs::TransformStamped::Ptr & transform_stamped_ptr, const ros::Time & time_stamp)
{
    if (target_frame == source_frame) {
        transform_stamped_ptr->header.stamp = time_stamp;
        transform_stamped_ptr->header.frame_id = target_frame;
        transform_stamped_ptr->child_frame_id = source_frame;
        transform_stamped_ptr->transform.translation.x = 0.0;
        transform_stamped_ptr->transform.translation.y = 0.0;
        transform_stamped_ptr->transform.translation.z = 0.0;
        transform_stamped_ptr->transform.rotation.x = 0.0;
        transform_stamped_ptr->transform.rotation.y = 0.0;
        transform_stamped_ptr->transform.rotation.z = 0.0;
        transform_stamped_ptr->transform.rotation.w = 1.0;
        return true;
    }

    try{
        *transform_stamped_ptr =
            tf2_buffer_.lookupTransform(target_frame, source_frame, time_stamp);
    } catch (tf2::TransformException & ex) {
        ROS_WARN("%s", ex.what());
        ROS_ERROR("Please publish TF %s to %s", target_frame.c_str(), source_frame.c_str());

        transform_stamped_ptr->header.stamp = time_stamp;
        transform_stamped_ptr->header.frame_id = target_frame;
        transform_stamped_ptr->child_frame_id = source_frame;
        transform_stamped_ptr->transform.translation.x = 0.0;
        transform_stamped_ptr->transform.translation.y = 0.0;
        transform_stamped_ptr->transform.translation.z = 0.0;
        transform_stamped_ptr->transform.rotation.x = 0.0;
        transform_stamped_ptr->transform.rotation.y = 0.0;
        transform_stamped_ptr->transform.rotation.z = 0.0;
        transform_stamped_ptr->transform.rotation.w = 1.0;
        return false;
    }
    return true;
}

bool GicpLocalizer::get_transform(
    const std::string & target_frame, const std::string & source_frame,
    const geometry_msgs::TransformStamped::Ptr & transform_stamped_ptr)
{
    if (target_frame == source_frame) {
        transform_stamped_ptr->header.stamp = ros::Time::now();
        transform_stamped_ptr->header.frame_id = target_frame;
        transform_stamped_ptr->child_frame_id = source_frame;
        transform_stamped_ptr->transform.translation.x = 0.0;
        transform_stamped_ptr->transform.translation.y = 0.0;
        transform_stamped_ptr->transform.translation.z = 0.0;
        transform_stamped_ptr->transform.rotation.x = 0.0;
        transform_stamped_ptr->transform.rotation.y = 0.0;
        transform_stamped_ptr->transform.rotation.z = 0.0;
        transform_stamped_ptr->transform.rotation.w = 1.0;
        return true;
    }

    try {
        *transform_stamped_ptr =
                tf2_buffer_.lookupTransform(target_frame, source_frame, ros::Time(0), ros::Duration(1.0));
    } catch (tf2::TransformException & ex) {
        ROS_WARN("%s", ex.what());
        ROS_ERROR("Please publish TF %s to %s", target_frame.c_str(), source_frame.c_str());

        transform_stamped_ptr->header.stamp = ros::Time::now();
        transform_stamped_ptr->header.frame_id = target_frame;
        transform_stamped_ptr->child_frame_id = source_frame;
        transform_stamped_ptr->transform.translation.x = 0.0;
        transform_stamped_ptr->transform.translation.y = 0.0;
        transform_stamped_ptr->transform.translation.z = 0.0;
        transform_stamped_ptr->transform.rotation.x = 0.0;
        transform_stamped_ptr->transform.rotation.y = 0.0;
        transform_stamped_ptr->transform.rotation.z = 0.0;
        transform_stamped_ptr->transform.rotation.w = 1.0;
        return false;
    }
    return true;
}

void GicpLocalizer::publish_tf(
        const std::string & frame_id, const std::string & child_frame_id,
        const geometry_msgs::PoseStamped & pose_msg)
{
    geometry_msgs::TransformStamped transform_stamped;
    transform_stamped.header.frame_id = frame_id;
    transform_stamped.child_frame_id = child_frame_id;
    transform_stamped.header.stamp = pose_msg.header.stamp;

    transform_stamped.transform.translation.x = pose_msg.pose.position.x;
    transform_stamped.transform.translation.y = pose_msg.pose.position.y;
    transform_stamped.transform.translation.z = pose_msg.pose.position.z;

    tf2::Quaternion tf_quaternion;
    tf2::fromMsg(pose_msg.pose.orientation, tf_quaternion);
    tf_quaternion.normalize();
    transform_stamped.transform.rotation.x = tf_quaternion.x();
    transform_stamped.transform.rotation.y = tf_quaternion.y();
    transform_stamped.transform.rotation.z = tf_quaternion.z();
    transform_stamped.transform.rotation.w = tf_quaternion.w();

    tf2_broadcaster_.sendTransform(transform_stamped);
}

int main(int argc, char **argv){
    ros::init(argc, argv, "gicp_localizer");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    GicpLocalizer gicp_localizer(nh, private_nh);
    ROS_INFO("\033[1;32m---->\033[0m Gicp Localizer Started.");
    ros::spin();

    return 0;
}