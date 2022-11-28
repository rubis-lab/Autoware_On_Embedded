#include <pthread.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ndt_cpu/NormalDistributionsTransform.h>
#include <pcl/registration/ndt.h>
#ifdef CUDA_FOUND
#include <ndt_gpu/NormalDistributionsTransform.h>
#endif
#ifdef USE_PCL_OPENMP
#include <pcl_omp_registration/ndt.h>
#endif

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <autoware_msgs/NDTStat.h>

//headers in Autoware Health Checker
#include <autoware_health_checker/health_checker/health_checker.h>

#include <rubis_lib/sched.hpp>
#include <rubis_msgs/PointCloud2.h>
#include <rubis_msgs/PoseStamped.h>
#include <rubis_msgs/InsStat.h>

#define SPIN_PROFILING
#define PREDICT_POSE_THRESHOLD 0.5

#define Wa 0.4
#define Wb 0.3
#define Wc 0.3
#define M_PI 3.14159265358979323846

#define DEBUG

static std::shared_ptr<autoware_health_checker::HealthChecker> health_checker_ptr_;

struct pose
{
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

enum class MethodType
{
  PCL_GENERIC = 0,
  PCL_ANH = 1,
  PCL_ANH_GPU = 2,
  PCL_OPENMP = 3,
};
static MethodType _method_type = MethodType::PCL_GENERIC;

static pose initial_pose, predict_pose, previous_pose, previous_gnss_pose,
    ndt_pose, current_pose, localizer_pose;

static double offset_x, offset_y, offset_z, offset_yaw;  // current_pos - previous_pose

// For GPS backup method
static pose current_gnss_pose;
static double previous_score = 0.0;

static pcl::PointCloud<pcl::PointXYZ> map, add;

// If the map is loaded, map_loaded will be 1.
static int map_loaded = 0;
static int init_pos_set = 0;

static pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
static cpu::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> anh_ndt;
#ifdef CUDA_FOUND
static std::shared_ptr<gpu::GNormalDistributionsTransform> anh_gpu_ndt_ptr =
    std::make_shared<gpu::GNormalDistributionsTransform>();
#endif
#ifdef USE_PCL_OPENMP
static pcl_omp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> omp_ndt;
#endif

// Configurable Parameters
static int _max_iter = 30;
static float _ndt_res = 1.0;
static double _step_size = 0.1;
static double _trans_eps = 0.01;

static int _use_init_pose;
static bool _use_gnss;
static std::string _offset = "linear";  // linear, zero, quadratic
static int _queue_size = 1000;
static bool _get_height = false;
static bool _publish_tf = true;
static double _gnss_pose_diff_thr = 3.0;

static std::string _baselink_frame = "base_link";
static std::string _localizer_frame = "velodyne";

static std::string _input_topic = "filtered_points";
static std::string _output_pose_topic = "ndt_pose";
static std::string _ndt_stat_topic = "ndt_stat";
static std::string _twist_topic = "estimated_twist";
static std::string _ndt_time_topic = "time_ndt_matching";

// TF base_link <-> localizer
static double _tf_x = 1.2;
static double _tf_y = 0.0;
static double _tf_z = 2.0;
static double _tf_roll = 0.0;
static double _tf_pitch = 0.0;
static double _tf_yaw = 0.0;
static Eigen::Matrix4f tf_btol;

// Publisher
static ros::Publisher ndt_pose_pub;
static geometry_msgs::PoseStamped ndt_pose_msg;

static ros::Publisher rubis_ndt_pose_pub;
static rubis_msgs::PoseStamped rubis_ndt_pose_msg;

static ros::Publisher localizer_pose_pub;
static geometry_msgs::PoseStamped localizer_pose_msg;

static ros::Publisher estimate_twist_pub;
static geometry_msgs::TwistStamped estimate_twist_msg;

static ros::Publisher time_ndt_matching_pub;
static std_msgs::Float32 time_ndt_matching;

static ros::Publisher ndt_stat_pub;
static autoware_msgs::NDTStat ndt_stat_msg;

static ros::Duration scan_duration;

static double exe_time = 0.0;
static bool has_converged;
static int iteration = 0;
static double fitness_score = 0.0;
static double trans_probability = 0.0;
static int matching_fail_cnt = 0;

// reference for comparing fitness_score, default value set to 500.0
static double _gnss_reinit_fitness = 500.0;

static double diff = 0.0;
static double diff_x = 0.0, diff_y = 0.0, diff_z = 0.0, diff_yaw;

static double current_velocity = 0.0, previous_velocity = 0.0, previous_previous_velocity = 0.0;  // [m/s]
static double current_velocity_x = 0.0, previous_velocity_x = 0.0;
static double current_velocity_y = 0.0, previous_velocity_y = 0.0;
static double current_velocity_z = 0.0, previous_velocity_z = 0.0;
// static double current_velocity_yaw = 0.0, previous_velocity_yaw = 0.0;
static double current_velocity_smooth = 0.0;

static double current_accel = 0.0, previous_accel = 0.0;  // [m/s^2]
static double current_accel_x = 0.0;
static double current_accel_y = 0.0;
static double current_accel_z = 0.0;
// static double current_accel_yaw = 0.0;

static double angular_velocity = 0.0;

static int use_predict_pose = 0;

// INS Stat information
static bool _is_ins_stat_received = false;
static double _current_ins_stat_vel_x = 0.0, _current_ins_stat_vel_y = 0.0;
static double _current_ins_stat_acc_x = 0.0, _current_ins_stat_acc_y = 0.0;
static double _current_ins_stat_yaw = 0.0;
static double _current_ins_stat_linear_velocity = 0.0, _current_ins_stat_linear_acceleration = 0.0, _current_ins_stat_angular_velocity = 0.0;

static double _previous_ins_stat_vel_x = 0.0, _previous_ins_stat_vel_y = 0.0;
static double _previous_ins_stat_acc_x = 0.0, _previous_ins_stat_acc_y = 0.0;
static double _previous_ins_stat_yaw = 0.0;
static double _previous_ins_stat_linear_velocity = 0.0, _previous_ins_stat_linear_acceleration = 0.0, _previous_ins_stat_angular_velocity = 0.0;

static double _previous_success_score;
static bool _is_matching_failed = false;

static std::chrono::time_point<std::chrono::system_clock> matching_start, matching_end;

static double predict_pose_error = 0.0;

static unsigned int points_map_num = 0;

pthread_mutex_t mutex;

static bool _is_init_match_finished = false;
static float _init_match_threshold = 8.0;

static void init_params()
{
  ros::NodeHandle private_nh("~");

  int method_type_tmp = 0;
  private_nh.param<int>("method_type", method_type_tmp, 0);
  _method_type = static_cast<MethodType>(method_type_tmp);

  private_nh.param<int>("max_iter", _max_iter, 30);
  private_nh.param<float>("resolution", _ndt_res, 1.0);
  private_nh.param<double>("step_size", _step_size, 0.1);
  private_nh.param<double>("trans_epsilon", _trans_eps, 0.01);

  private_nh.param<int>("use_init_pose", _use_init_pose, 1);

  if(_use_init_pose){
    private_nh.param<double>("init_x", initial_pose.x, 0.0);
    private_nh.param<double>("init_y", initial_pose.y, 0.0);
    private_nh.param<double>("init_z", initial_pose.z, 0.0);
    private_nh.param<double>("init_roll", initial_pose.roll, 0.0);
    private_nh.param<double>("init_pitch", initial_pose.pitch, 0.0);
    private_nh.param<double>("init_yaw", initial_pose.yaw, 0.0);

    init_pos_set = 1;
  }

  private_nh.param<bool>("use_gnss", _use_gnss, false);
  private_nh.param<double>("gnss_pose_diff_threshold", _gnss_pose_diff_thr, 3.0);

  private_nh.param<int>("queue_size", _queue_size, 1);
  private_nh.param<std::string>("offset", _offset, std::string("linear"));
  private_nh.param<bool>("get_height", _get_height, false);
  private_nh.param<bool>("publish_tf", _publish_tf, true);

  private_nh.param<std::string>("baselink_frame", _baselink_frame, std::string("base_link"));
  private_nh.param<std::string>("localizer_frame", _localizer_frame, std::string("velodyne"));

  private_nh.param<std::string>("input_topic", _input_topic, std::string("filtered_points"));
  private_nh.param<std::string>("output_pose_topic", _output_pose_topic, std::string("ndt_pose"));
  private_nh.param<std::string>("ndt_stat_topic", _ndt_stat_topic, std::string("ndt_stat"));
  private_nh.param<std::string>("twist_topic", _twist_topic, std::string("estimated_twist"));
  private_nh.param<std::string>("ndt_time_topic", _ndt_time_topic, std::string("time_ndt_matching"));

  localizer_pose.x = initial_pose.x;
  localizer_pose.y = initial_pose.y;
  localizer_pose.z = initial_pose.z;
  localizer_pose.roll = initial_pose.roll;
  localizer_pose.pitch = initial_pose.pitch;
  localizer_pose.yaw = initial_pose.yaw;

  previous_pose.x = initial_pose.x;
  previous_pose.y = initial_pose.y;
  previous_pose.z = initial_pose.z;
  previous_pose.roll = initial_pose.roll;
  previous_pose.pitch = initial_pose.pitch;
  previous_pose.yaw = initial_pose.yaw;

  current_pose.x = initial_pose.x;
  current_pose.y = initial_pose.y;
  current_pose.z = initial_pose.z;
  current_pose.roll = initial_pose.roll;
  current_pose.pitch = initial_pose.pitch;
  current_pose.yaw = initial_pose.yaw;

  current_velocity = 0;
  current_velocity_x = 0;
  current_velocity_y = 0;
  current_velocity_z = 0;
  angular_velocity = 0;
}

static pose convertPoseIntoRelativeCoordinate(const pose &target_pose, const pose &reference_pose)
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

    return trans_target_pose;
}

static void map_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
{
  if (points_map_num != input->width)
  {
    std::cout << "Update points_map." << std::endl;

    points_map_num = input->width;

    // Convert the data type(from sensor_msgs to pcl).
    pcl::fromROSMsg(*input, map);

    pcl::PointCloud<pcl::PointXYZ>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZ>(map));

    // Setting point cloud to be aligned to.
    if (_method_type == MethodType::PCL_GENERIC)
    { 
      pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> new_ndt;
      pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      new_ndt.setResolution(_ndt_res);
      
      new_ndt.setInputTarget(map_ptr);
      new_ndt.setMaximumIterations(_max_iter);
      new_ndt.setStepSize(_step_size);
      new_ndt.setTransformationEpsilon(_trans_eps);

      new_ndt.align(*output_cloud, Eigen::Matrix4f::Identity());

      pthread_mutex_lock(&mutex);
      ndt = new_ndt;
      pthread_mutex_unlock(&mutex);
    }
    else if (_method_type == MethodType::PCL_ANH)
    {
      cpu::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> new_anh_ndt;
      new_anh_ndt.setResolution(_ndt_res);
      new_anh_ndt.setInputTarget(map_ptr);
      new_anh_ndt.setMaximumIterations(_max_iter);
      new_anh_ndt.setStepSize(_step_size);
      new_anh_ndt.setTransformationEpsilon(_trans_eps);

      pcl::PointCloud<pcl::PointXYZ>::Ptr dummy_scan_ptr(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::PointXYZ dummy_point;
      dummy_scan_ptr->push_back(dummy_point);
      new_anh_ndt.setInputSource(dummy_scan_ptr);

      new_anh_ndt.align(Eigen::Matrix4f::Identity());

      pthread_mutex_lock(&mutex);
      anh_ndt = new_anh_ndt;
      pthread_mutex_unlock(&mutex);
    }
#ifdef CUDA_FOUND
    else if (_method_type == MethodType::PCL_ANH_GPU)
    {
      std::shared_ptr<gpu::GNormalDistributionsTransform> new_anh_gpu_ndt_ptr =
          std::make_shared<gpu::GNormalDistributionsTransform>();

      new_anh_gpu_ndt_ptr->setResolution(_ndt_res);
      new_anh_gpu_ndt_ptr->setInputTarget(map_ptr);
      new_anh_gpu_ndt_ptr->setMaximumIterations(_max_iter);
      new_anh_gpu_ndt_ptr->setStepSize(_step_size);
      new_anh_gpu_ndt_ptr->setTransformationEpsilon(_trans_eps);

      pcl::PointCloud<pcl::PointXYZ>::Ptr dummy_scan_ptr(new pcl::PointCloud<pcl::PointXYZ>());

      pcl::PointXYZ dummy_point;
      dummy_scan_ptr->push_back(dummy_point);
      new_anh_gpu_ndt_ptr->setInputSource(dummy_scan_ptr);

      new_anh_gpu_ndt_ptr->align(Eigen::Matrix4f::Identity());

      pthread_mutex_lock(&mutex);
      anh_gpu_ndt_ptr = new_anh_gpu_ndt_ptr;
      pthread_mutex_unlock(&mutex);
    }
#endif
#ifdef USE_PCL_OPENMP
    else if (_method_type == MethodType::PCL_OPENMP)
    {
      pcl_omp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> new_omp_ndt;
      pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      new_omp_ndt.setResolution(_ndt_res);
      new_omp_ndt.setInputTarget(map_ptr);
      new_omp_ndt.setMaximumIterations(_max_iter);
      new_omp_ndt.setStepSize(_step_size);
      new_omp_ndt.setTransformationEpsilon(_trans_eps);

      new_omp_ndt.align(*output_cloud, Eigen::Matrix4f::Identity());

      pthread_mutex_lock(&mutex);
      omp_ndt = new_omp_ndt;
      pthread_mutex_unlock(&mutex);
    }
#endif
    map_loaded = 1;
  }
}

static void gnss_callback(const geometry_msgs::PoseStamped::ConstPtr& input)
{
  if(!_use_gnss) return;
  current_gnss_pose.x = input->pose.position.x;
  current_gnss_pose.y = input->pose.position.y;
  current_gnss_pose.z = input->pose.position.z;

  tf::Quaternion gnss_q(input->pose.orientation.x, input->pose.orientation.y, input->pose.orientation.z,
                        input->pose.orientation.w);
  tf::Matrix3x3 gnss_m(gnss_q);

  gnss_m.getRPY(current_gnss_pose.roll, current_gnss_pose.pitch, current_gnss_pose.yaw);

  double ndt_gnss_diff = hypot(current_gnss_pose.x - current_pose.x, current_gnss_pose.y - current_pose.y);

  #ifdef DEBUG
  std::cout << ndt_gnss_diff << " " << _gnss_pose_diff_thr << std::endl;
  #endif

  if(ndt_gnss_diff > _gnss_pose_diff_thr){
    matching_fail_cnt++;
  }
  else{
    matching_fail_cnt = 0;
  }
  
  if(matching_fail_cnt > 10){
    previous_score = 0.0;
    current_pose = current_gnss_pose;
    previous_pose = current_gnss_pose;
    
    current_velocity = 0.0;
    current_velocity_x = 0.0;
    current_velocity_y = 0.0;
    current_velocity_z = 0.0;
    angular_velocity = 0.0;

    matching_fail_cnt = 0;
  }

  previous_gnss_pose = current_gnss_pose;
}

static void initialpose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& input)
{
  tf::TransformListener listener;
  tf::StampedTransform transform;
  try
  {
    ros::Time now = ros::Time(0);
    listener.waitForTransform("/map", input->header.frame_id, now, ros::Duration(10.0));
    listener.lookupTransform("/map", input->header.frame_id, now, transform);
  }
  catch (tf::TransformException& ex)
  {
    ROS_ERROR("%s", ex.what());
  }

  tf::Quaternion q(input->pose.pose.orientation.x, input->pose.pose.orientation.y, input->pose.pose.orientation.z,
                   input->pose.pose.orientation.w);
  tf::Matrix3x3 m(q);

  current_pose.x = input->pose.pose.position.x + transform.getOrigin().x();
  current_pose.y = input->pose.pose.position.y + transform.getOrigin().y();
  current_pose.z = input->pose.pose.position.z + transform.getOrigin().z();

  m.getRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

  if (_get_height == true && map_loaded == 1)
  {
    double min_distance = DBL_MAX;
    double nearest_z = current_pose.z;
    for (const auto& p : map)
    {
      double distance = hypot(current_pose.x - p.x, current_pose.y - p.y);
      if (distance < min_distance)
      {
        min_distance = distance;
        nearest_z = p.z;
      }
    }
    current_pose.z = nearest_z;
  }

  previous_pose.x = current_pose.x;
  previous_pose.y = current_pose.y;
  previous_pose.z = current_pose.z;
  previous_pose.roll = current_pose.roll;
  previous_pose.pitch = current_pose.pitch;
  previous_pose.yaw = current_pose.yaw;

  current_velocity = 0.0;
  current_velocity_x = 0.0;
  current_velocity_y = 0.0;
  current_velocity_z = 0.0;
  angular_velocity = 0.0;

  current_accel = 0.0;
  current_accel_x = 0.0;
  current_accel_y = 0.0;
  current_accel_z = 0.0;

  offset_x = 0.0;
  offset_y = 0.0;
  offset_z = 0.0;
  offset_yaw = 0.0;

  previous_score = 0.0;

  init_pos_set = 1;
}

static double wrapToPm(double a_num, const double a_max)
{
  if (a_num >= a_max)
  {
    a_num -= 2.0 * a_max;
  }
  return a_num;
}

static double wrapToPmPi(const double a_angle_rad)
{
  return wrapToPm(a_angle_rad, M_PI);
}

static double calcDiffForRadian(const double lhs_rad, const double rhs_rad)
{
  double diff_rad = lhs_rad - rhs_rad;
  if (diff_rad >= M_PI)
    diff_rad = diff_rad - 2 * M_PI;
  else if (diff_rad < -M_PI)
    diff_rad = diff_rad + 2 * M_PI;
  return diff_rad;
}

static inline void ndt_matching(const sensor_msgs::PointCloud2::ConstPtr& input)
{ 
  static int match_cnt = 10;

  if (map_loaded != 1 || init_pos_set != 1) return;

  // Check inital matching is success or not
  if(_is_init_match_finished == false){

    if(previous_score < _init_match_threshold) match_cnt--;
    else match_cnt = 10;

    if(previous_score < _init_match_threshold && previous_score != 0.0 && match_cnt <0){
      _is_init_match_finished = true;
      #ifdef DEBUG
        std::cout<<"Success initial matching!"<<std::endl;
      #endif
    }
  }

  health_checker_ptr_->CHECK_RATE("topic_rate_filtered_points_slow", 8, 5, 1, "topic filtered_points subscribe rate slow.");

  matching_start = std::chrono::system_clock::now();

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion predict_q, ndt_q, current_q, localizer_q;

  pcl::PointXYZ p;
  pcl::PointCloud<pcl::PointXYZ> filtered_scan;

  ros::Time current_scan_time = input->header.stamp;
  static ros::Time previous_scan_time = current_scan_time;

  pcl::fromROSMsg(*input, filtered_scan);
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZ>(filtered_scan));
  int scan_points_num = filtered_scan_ptr->size();

  Eigen::Matrix4f t(Eigen::Matrix4f::Identity());   // base_link
  Eigen::Matrix4f t2(Eigen::Matrix4f::Identity());  // localizer

  std::chrono::time_point<std::chrono::system_clock> align_start, align_end, getFitnessScore_start,
      getFitnessScore_end;
  static double align_time, getFitnessScore_time = 0.0;

  pthread_mutex_lock(&mutex);

  if (_method_type == MethodType::PCL_GENERIC)
    ndt.setInputSource(filtered_scan_ptr);
  else if (_method_type == MethodType::PCL_ANH)
    anh_ndt.setInputSource(filtered_scan_ptr);
#ifdef CUDA_FOUND
  else if (_method_type == MethodType::PCL_ANH_GPU){
    anh_gpu_ndt_ptr->setInputSource(filtered_scan_ptr);
  }
#endif
#ifdef USE_PCL_OPENMP
  else if (_method_type == MethodType::PCL_OPENMP)
    omp_ndt.setInputSource(filtered_scan_ptr);
#endif

  // Guess the initial gross estimation of the transformation
  double diff_time = (current_scan_time - previous_scan_time).toSec();

  if (_offset == "linear")
  {
    offset_x = current_velocity_x * diff_time;
    offset_y = current_velocity_y * diff_time;
    offset_z = current_velocity_z * diff_time;
    offset_yaw = angular_velocity * diff_time;
  }
  else if (_offset == "quadratic")
  {
    offset_x = (current_velocity_x + current_accel_x * diff_time) * diff_time;
    offset_y = (current_velocity_y + current_accel_y * diff_time) * diff_time;
    offset_z = current_velocity_z * diff_time;
    offset_yaw = angular_velocity * diff_time;
  }
  else if (_offset == "zero")
  {
    offset_x = 0.0;
    offset_y = 0.0;
    offset_z = 0.0;
    offset_yaw = 0.0;
  }

  predict_pose.x = previous_pose.x + offset_x;
  predict_pose.y = previous_pose.y + offset_y;
  predict_pose.z = previous_pose.z + offset_z;
  predict_pose.roll = previous_pose.roll;
  predict_pose.pitch = previous_pose.pitch;
  predict_pose.yaw = previous_pose.yaw + offset_yaw;

  pose predict_pose_for_ndt;
  predict_pose_for_ndt = predict_pose;

  Eigen::Translation3f init_translation(predict_pose_for_ndt.x, predict_pose_for_ndt.y, predict_pose_for_ndt.z);
  Eigen::AngleAxisf init_rotation_x(predict_pose_for_ndt.roll, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf init_rotation_y(predict_pose_for_ndt.pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf init_rotation_z(predict_pose_for_ndt.yaw, Eigen::Vector3f::UnitZ());
  Eigen::Matrix4f init_guess = (init_translation * init_rotation_z * init_rotation_y * init_rotation_x) * tf_btol;

  pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (_method_type == MethodType::PCL_GENERIC)
  {
    align_start = std::chrono::system_clock::now();
    ndt.align(*output_cloud, init_guess);
    align_end = std::chrono::system_clock::now();

    has_converged = ndt.hasConverged();

    t = ndt.getFinalTransformation();
    iteration = ndt.getFinalNumIteration();

    getFitnessScore_start = std::chrono::system_clock::now();
    fitness_score = ndt.getFitnessScore();
    getFitnessScore_end = std::chrono::system_clock::now();

    trans_probability = ndt.getTransformationProbability();
  }
  else if (_method_type == MethodType::PCL_ANH)
  {
    align_start = std::chrono::system_clock::now();
    anh_ndt.align(init_guess);
    align_end = std::chrono::system_clock::now();

    has_converged = anh_ndt.hasConverged();

    t = anh_ndt.getFinalTransformation();
    iteration = anh_ndt.getFinalNumIteration();

    getFitnessScore_start = std::chrono::system_clock::now();
    fitness_score = anh_ndt.getFitnessScore();
    getFitnessScore_end = std::chrono::system_clock::now();

    trans_probability = anh_ndt.getTransformationProbability();
  }
#ifdef CUDA_FOUND
  else if (_method_type == MethodType::PCL_ANH_GPU)
  {
    align_start = std::chrono::system_clock::now();
    anh_gpu_ndt_ptr->align(init_guess);
    align_end = std::chrono::system_clock::now();

    has_converged = anh_gpu_ndt_ptr->hasConverged();

    t = anh_gpu_ndt_ptr->getFinalTransformation();
    iteration = anh_gpu_ndt_ptr->getFinalNumIteration();

    getFitnessScore_start = std::chrono::system_clock::now();
    fitness_score = anh_gpu_ndt_ptr->getFitnessScore();
    getFitnessScore_end = std::chrono::system_clock::now();

    trans_probability = anh_gpu_ndt_ptr->getTransformationProbability();      
  }
#endif
#ifdef USE_PCL_OPENMP
  else if (_method_type == MethodType::PCL_OPENMP)
  {
    align_start = std::chrono::system_clock::now();
    omp_ndt.align(*output_cloud, init_guess);
    align_end = std::chrono::system_clock::now();

    has_converged = omp_ndt.hasConverged();

    t = omp_ndt.getFinalTransformation();
    iteration = omp_ndt.getFinalNumIteration();

    getFitnessScore_start = std::chrono::system_clock::now();
    fitness_score = omp_ndt.getFitnessScore();
    getFitnessScore_end = std::chrono::system_clock::now();

    trans_probability = omp_ndt.getTransformationProbability();
  }
#endif
  align_time = std::chrono::duration_cast<std::chrono::microseconds>(align_end - align_start).count() / 1000.0;

  t2 = t * tf_btol.inverse();

  getFitnessScore_time =
      std::chrono::duration_cast<std::chrono::microseconds>(getFitnessScore_end - getFitnessScore_start).count() /
      1000.0;

  pthread_mutex_unlock(&mutex);

  tf::Matrix3x3 mat_l;  // localizer
  mat_l.setValue(static_cast<double>(t(0, 0)), static_cast<double>(t(0, 1)), static_cast<double>(t(0, 2)),
                  static_cast<double>(t(1, 0)), static_cast<double>(t(1, 1)), static_cast<double>(t(1, 2)),
                  static_cast<double>(t(2, 0)), static_cast<double>(t(2, 1)), static_cast<double>(t(2, 2)));

  // Update localizer_pose
  localizer_pose.x = t(0, 3);
  localizer_pose.y = t(1, 3);
  localizer_pose.z = t(2, 3);
  mat_l.getRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw, 1);

  tf::Matrix3x3 mat_b;  // base_link
  mat_b.setValue(static_cast<double>(t2(0, 0)), static_cast<double>(t2(0, 1)), static_cast<double>(t2(0, 2)),
                  static_cast<double>(t2(1, 0)), static_cast<double>(t2(1, 1)), static_cast<double>(t2(1, 2)),
                  static_cast<double>(t2(2, 0)), static_cast<double>(t2(2, 1)), static_cast<double>(t2(2, 2)));

  // Update ndt_pose
  ndt_pose.x = t2(0, 3);
  ndt_pose.y = t2(1, 3);
  ndt_pose.z = t2(2, 3);
  mat_b.getRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw, 1);

  // Calculate the difference between ndt_pose and predict_pose
  predict_pose_error = sqrt((ndt_pose.x - predict_pose_for_ndt.x) * (ndt_pose.x - predict_pose_for_ndt.x) +
                            (ndt_pose.y - predict_pose_for_ndt.y) * (ndt_pose.y - predict_pose_for_ndt.y) +
                            (ndt_pose.z - predict_pose_for_ndt.z) * (ndt_pose.z - predict_pose_for_ndt.z));

  if (predict_pose_error <= PREDICT_POSE_THRESHOLD)
  {
    use_predict_pose = 0;
  }
  else
  {
    use_predict_pose = 1;
  }
  use_predict_pose = 0;

  if (use_predict_pose == 0)
  {
    current_pose.x = ndt_pose.x;
    current_pose.y = ndt_pose.y;
    current_pose.z = ndt_pose.z;
    current_pose.roll = ndt_pose.roll;
    current_pose.pitch = ndt_pose.pitch;
    current_pose.yaw = ndt_pose.yaw;
  }
  else
  {
    current_pose.x = predict_pose_for_ndt.x;
    current_pose.y = predict_pose_for_ndt.y;
    current_pose.z = predict_pose_for_ndt.z;
    current_pose.roll = predict_pose_for_ndt.roll;
    current_pose.pitch = predict_pose_for_ndt.pitch;
    current_pose.yaw = predict_pose_for_ndt.yaw;
  }

  // Compute the velocity and acceleration
  diff_x = current_pose.x - previous_pose.x;
  diff_y = current_pose.y - previous_pose.y;
  diff_z = current_pose.z - previous_pose.z;
  diff_yaw = calcDiffForRadian(current_pose.yaw, previous_pose.yaw);
  diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

  const pose trans_current_pose = convertPoseIntoRelativeCoordinate(current_pose, previous_pose);

  current_velocity = (diff_time > 0) ? (diff / diff_time) : 0;
  current_velocity =  (trans_current_pose.x >= 0) ? current_velocity : -current_velocity;
  current_velocity_x = (diff_time > 0) ? (diff_x / diff_time) : 0;
  current_velocity_y = (diff_time > 0) ? (diff_y / diff_time) : 0;
  current_velocity_z = (diff_time > 0) ? (diff_z / diff_time) : 0;
  angular_velocity = (diff_time > 0) ? (diff_yaw / diff_time) : 0;

  current_velocity_smooth = (current_velocity + previous_velocity + previous_previous_velocity) / 3.0;
  if (std::fabs(current_velocity_smooth) < 0.2)
  {
    current_velocity_smooth = 0.0;
  }

  current_accel = (diff_time > 0) ? ((current_velocity - previous_velocity) / diff_time) : 0;
  current_accel_x = (diff_time > 0) ? ((current_velocity_x - previous_velocity_x) / diff_time) : 0;
  current_accel_y = (diff_time > 0) ? ((current_velocity_y - previous_velocity_y) / diff_time) : 0;
  current_accel_z = (diff_time > 0) ? ((current_velocity_z - previous_velocity_z) / diff_time) : 0;

  // Set values for publishing pose
  predict_q.setRPY(predict_pose.roll, predict_pose.pitch, predict_pose.yaw);

  ndt_q.setRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw);
  
  ndt_pose_msg.header.frame_id = "/map";
  ndt_pose_msg.header.stamp = current_scan_time;
  ndt_pose_msg.pose.position.x = ndt_pose.x;
  ndt_pose_msg.pose.position.y = ndt_pose.y;
  ndt_pose_msg.pose.position.z = ndt_pose.z;
  ndt_pose_msg.pose.orientation.x = ndt_q.x();
  ndt_pose_msg.pose.orientation.y = ndt_q.y();
  ndt_pose_msg.pose.orientation.z = ndt_q.z();
  ndt_pose_msg.pose.orientation.w = ndt_q.w();    

  current_q.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

  localizer_q.setRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw);
  
  localizer_pose_msg.header.frame_id = "/map";
  localizer_pose_msg.header.stamp = current_scan_time;
  localizer_pose_msg.pose.position.x = localizer_pose.x;
  localizer_pose_msg.pose.position.y = localizer_pose.y;
  localizer_pose_msg.pose.position.z = localizer_pose.z;
  localizer_pose_msg.pose.orientation.x = localizer_q.x();
  localizer_pose_msg.pose.orientation.y = localizer_q.y();
  localizer_pose_msg.pose.orientation.z = localizer_q.z();
  localizer_pose_msg.pose.orientation.w = localizer_q.w();

  health_checker_ptr_->CHECK_RATE("topic_rate_ndt_pose_slow", 8, 5, 1, "topic ndt_pose publish rate slow.");
  ndt_pose_pub.publish(ndt_pose_msg);
  rubis::sched::task_state_ = TASK_STATE_DONE;

  if(rubis::instance_mode_ && rubis::instance_ != RUBIS_NO_INSTANCE){
    rubis_ndt_pose_msg.instance = rubis::instance_;
    rubis_ndt_pose_msg.msg = ndt_pose_msg;
    rubis_ndt_pose_pub.publish(rubis_ndt_pose_msg);
  }

  localizer_pose_pub.publish(localizer_pose_msg);

  matching_end = std::chrono::system_clock::now();
  exe_time = std::chrono::duration_cast<std::chrono::microseconds>(matching_end - matching_start).count() / 1000.0;
  time_ndt_matching.data = exe_time;
  health_checker_ptr_->CHECK_MAX_VALUE("time_ndt_matching", time_ndt_matching.data, 50, 70, 100, "value time_ndt_matching is too high.");
  time_ndt_matching_pub.publish(time_ndt_matching);

  // Set values for /estimate_twist
  estimate_twist_msg.header.stamp = current_scan_time;
  estimate_twist_msg.header.frame_id = _baselink_frame;
  estimate_twist_msg.twist.linear.x = current_velocity;
  estimate_twist_msg.twist.linear.y = 0.0;
  estimate_twist_msg.twist.linear.z = 0.0;
  estimate_twist_msg.twist.angular.x = 0.0;
  estimate_twist_msg.twist.angular.y = 0.0;
  estimate_twist_msg.twist.angular.z = angular_velocity;

  estimate_twist_pub.publish(estimate_twist_msg);

  geometry_msgs::Vector3Stamped estimate_vel_msg;
  estimate_vel_msg.header.stamp = current_scan_time;
  estimate_vel_msg.vector.x = current_velocity;
  health_checker_ptr_->CHECK_MAX_VALUE("estimate_twist_linear", current_velocity, 5, 10, 15, "value linear estimated twist is too high.");
  health_checker_ptr_->CHECK_MAX_VALUE("estimate_twist_angular", angular_velocity, 5, 10, 15, "value linear angular twist is too high.");

  previous_score = fitness_score;
  if(!_is_matching_failed) _previous_success_score = previous_score;

  // Set values for /ndt_stat
  ndt_stat_msg.header.stamp = current_scan_time;
  ndt_stat_msg.exe_time = exe_time;
  ndt_stat_msg.iteration = iteration;
  ndt_stat_msg.score = fitness_score;
  ndt_stat_msg.velocity = current_velocity;
  ndt_stat_msg.acceleration = current_accel;
  ndt_stat_msg.use_predict_pose = 0;

  ndt_stat_pub.publish(ndt_stat_msg);
  
  // std::cout << "-----------------------------------------------------------------" << std::endl;
  // std::cout << "Sequence: " << input->header.seq << std::endl;
  // std::cout << "Timestamp: " << input->header.stamp << std::endl;
  // std::cout << "Frame ID: " << input->header.frame_id << std::endl;
  // std::cout << "Number of Scan Points: " << scan_ptr->size() << " points." << std::endl;
  // std::cout << "Number of Filtered Scan Points: " << scan_points_num << " points." << std::endl;
  // std::cout << "NDT has converged: " << has_converged << std::endl;
  // std::cout << "Fitness Score: " << fitness_score << std::endl;
  // std::cout << "Transformation Probability: " << trans_probability << std::endl;
  // std::cout << "Execution Time: " << exe_time << " ms." << std::endl;
  // std::cout << "Number of Iterations: " << iteration << std::endl;
  // std::cout << "(x,y,z,roll,pitch,yaw): " << std::endl;
  // std::cout << "(" << current_pose.x << ", " << current_pose.y << ", " << current_pose.z << ", " << current_pose.roll
  //           << ", " << current_pose.pitch << ", " << current_pose.yaw << ")" << std::endl;
  // std::cout << "Transformation Matrix: " << std::endl;
  // std::cout << t << std::endl;
  // std::cout << "Align time: " << align_time << std::endl;
  // std::cout << "Get fitness score time: " << getFitnessScore_time << std::endl;
  // std::cout << "-----------------------------------------------------------------" << std::endl;

  if(_publish_tf){
    // Send TF "/base_link" to "/map"
    if(!_is_matching_failed){
      transform.setOrigin(tf::Vector3(current_pose.x, current_pose.y, current_pose.z));
      transform.setRotation(current_q);
      
      br.sendTransform(tf::StampedTransform(transform, current_scan_time, "/map", _baselink_frame));
    }
    else{ // When matching is failed
      transform.setRotation(current_q);
      br.sendTransform(tf::StampedTransform(transform, current_scan_time, "/map", _baselink_frame));
    }
  }

  // Update previous
  previous_pose.x = current_pose.x;
  previous_pose.y = current_pose.y;
  previous_pose.z = current_pose.z;
  previous_pose.roll = current_pose.roll;
  previous_pose.pitch = current_pose.pitch;
  previous_pose.yaw = current_pose.yaw;

  previous_scan_time = current_scan_time;

  previous_previous_velocity = previous_velocity;
  previous_velocity = current_velocity;
  previous_velocity_x = current_velocity_x;
  previous_velocity_y = current_velocity_y;
  previous_velocity_z = current_velocity_z;
  previous_accel = current_accel;
  
  if(rubis::sched::is_task_ready_ == TASK_NOT_READY){
    rubis::sched::init_task();
    if(rubis::sched::gpu_profiling_flag_) rubis::sched::start_gpu_profiling();
  }  
}

static void points_callback(const sensor_msgs::PointCloud2::ConstPtr& input){
  rubis::instance_ = RUBIS_NO_INSTANCE;
  ndt_matching(input);
}

static void rubis_points_callback(const rubis_msgs::PointCloud2::ConstPtr& _input){
  sensor_msgs::PointCloud2::ConstPtr input = boost::make_shared<const sensor_msgs::PointCloud2>(_input->msg);
  rubis::instance_ = _input->instance;
  ndt_matching(input);
}

static void ins_stat_callback(const rubis_msgs::InsStat::ConstPtr& input){
  _is_ins_stat_received = true;
  _current_ins_stat_vel_x = input->vel_x;
  _current_ins_stat_vel_y = input->vel_y;
  _current_ins_stat_acc_x = input->acc_x;
  _current_ins_stat_acc_y = input->acc_y;
  _current_ins_stat_yaw = input->yaw;
  _current_ins_stat_linear_velocity = input->linear_velocity;
  _current_ins_stat_linear_acceleration = input->linear_acceleration;
  _current_ins_stat_angular_velocity = input->angular_velocity;
  return;
}

void* thread_func(void* args)
{
  ros::NodeHandle nh_map;
  ros::CallbackQueue map_callback_queue;
  nh_map.setCallbackQueue(&map_callback_queue);

  ros::Subscriber map_sub = nh_map.subscribe("points_map", 10, map_callback); 
  ros::Rate ros_rate(10);
  while (nh_map.ok())
  {
    map_callback_queue.callAvailable(ros::WallDuration());
    ros_rate.sleep();
  }

  return nullptr;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "modular_ndt_matching");
  pthread_mutex_init(&mutex, NULL);

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  health_checker_ptr_ = std::make_shared<autoware_health_checker::HealthChecker>(nh,private_nh);
  health_checker_ptr_->ENABLE();
  health_checker_ptr_->NODE_ACTIVATE();

  // Geting parameters
  init_params();

  try
  {
    tf::TransformListener base_localizer_listener;
    tf::StampedTransform  m_base_to_localizer;
    base_localizer_listener.waitForTransform(_baselink_frame, _localizer_frame, ros::Time(0), ros::Duration(1.0));
    base_localizer_listener.lookupTransform(_baselink_frame, _localizer_frame, ros::Time(0), m_base_to_localizer);

    _tf_x = m_base_to_localizer.getOrigin().x();
    _tf_y = m_base_to_localizer.getOrigin().y();
    _tf_z = m_base_to_localizer.getOrigin().z();

    tf::Quaternion b_l_q(
      m_base_to_localizer.getRotation().x(),
      m_base_to_localizer.getRotation().y(),
      m_base_to_localizer.getRotation().z(),
      m_base_to_localizer.getRotation().w()
    );

    tf::Matrix3x3 b_l_m(b_l_q);
    b_l_m.getRPY(_tf_roll, _tf_pitch, _tf_yaw);
  }
  catch (tf::TransformException& ex)
  {
    ROS_ERROR("%s", ex.what());
  }

  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "method_type: " << static_cast<int>(_method_type) << std::endl;
  std::cout << "use_gnss: " << _use_gnss << std::endl;
  std::cout << "queue_size: " << _queue_size << std::endl;
  std::cout << "offset: " << _offset << std::endl;
  std::cout << "get_height: " << _get_height << std::endl;
  std::cout << "localizer_frame: " << _localizer_frame << std::endl;
  std::cout << "gnss_reinit_fitness: " << _gnss_reinit_fitness << std::endl;
  std::cout << "(tf_x,tf_y,tf_z,tf_roll,tf_pitch,tf_yaw): (" << _tf_x << ", " << _tf_y << ", " << _tf_z << ", "
            << _tf_roll << ", " << _tf_pitch << ", " << _tf_yaw << ")" << std::endl;
  std::cout << "-----------------------------------------------------------------" << std::endl;

#ifndef CUDA_FOUND
  if (_method_type == MethodType::PCL_ANH_GPU)
  {
    std::cerr << "**************************************************************" << std::endl;
    std::cerr << "[ERROR]PCL_ANH_GPU is not built. Please use other method type." << std::endl;
    std::cerr << "**************************************************************" << std::endl;
    exit(1);
  }
#endif
#ifndef USE_PCL_OPENMP
  if (_method_type == MethodType::PCL_OPENMP)
  {
    std::cerr << "**************************************************************" << std::endl;
    std::cerr << "[ERROR]PCL_OPENMP is not built. Please use other method type." << std::endl;
    std::cerr << "**************************************************************" << std::endl;
    exit(1);
  }
#endif

  // Scheduling Setup
  int task_scheduling_flag;
  int task_profiling_flag;
  std::string task_response_time_filename;
  int rate;
  double task_minimum_inter_release_time;
  double task_execution_time;
  double task_relative_deadline;

  int gpu_scheduling_flag;
  int gpu_profiling_flag;
  std::string gpu_execution_time_filename;
  std::string gpu_response_time_filename;
  std::string gpu_deadline_filename;

  std::string node_name = ros::this_node::getName();
  private_nh.param<int>(node_name+"/task_scheduling_flag", task_scheduling_flag, 0);
  private_nh.param<int>(node_name+"/task_profiling_flag", task_profiling_flag, 0);
  private_nh.param<std::string>(node_name+"/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/ndt_matching.csv");
  private_nh.param<int>(node_name+"/rate", rate, 10);
  private_nh.param(node_name+"/task_minimum_inter_release_time", task_minimum_inter_release_time, (double)10);
  private_nh.param(node_name+"/task_execution_time", task_execution_time, (double)10);
  private_nh.param(node_name+"/task_relative_deadline", task_relative_deadline, (double)10);
  private_nh.param(node_name+"/gpu_scheduling_flag", gpu_scheduling_flag, 0);
  private_nh.param(node_name+"/gpu_profiling_flag", gpu_profiling_flag, 0);
  private_nh.param<std::string>(node_name+"/gpu_execution_time_filename", gpu_execution_time_filename, "~/Documents/gpu_profiling/test_ndt_matching_execution_time.csv");
  private_nh.param<std::string>(node_name+"/gpu_response_time_filename", gpu_response_time_filename, "~/Documents/gpu_profiling/test_ndt_matching_response_time.csv");
  private_nh.param<std::string>(node_name+"/gpu_deadline_filename", gpu_deadline_filename, "~/Documents/gpu_deadline/ndt_matching_gpu_deadline.csv");
  private_nh.param<int>(node_name+"/instance_mode", rubis::instance_mode_, 0);
  
  if(task_profiling_flag) rubis::sched::init_task_profiling(task_response_time_filename);
  if(gpu_profiling_flag) rubis::sched::init_gpu_profiling(gpu_execution_time_filename, gpu_response_time_filename);
  
  if( (_method_type == MethodType::PCL_ANH_GPU) && (gpu_scheduling_flag == 1) ){
    rubis::sched::init_gpu_scheduling("/tmp/ndt_matching", gpu_deadline_filename, 0);
  }    
  else if(_method_type != MethodType::PCL_ANH_GPU && gpu_scheduling_flag == 1){
    ROS_ERROR("GPU scheduling flag is true but type doesn't set to GPU!");
    exit(1);
  }

  Eigen::Translation3f tl_btol(_tf_x, _tf_y, _tf_z);                 // tl: translation
  Eigen::AngleAxisf rot_x_btol(_tf_roll, Eigen::Vector3f::UnitX());  // rot: rotation
  Eigen::AngleAxisf rot_y_btol(_tf_pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rot_z_btol(_tf_yaw, Eigen::Vector3f::UnitZ());
  tf_btol = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();

  // Publishers
  ndt_pose_pub = nh.advertise<geometry_msgs::PoseStamped>(_output_pose_topic, 10);

  // if(rubis::instance_mode_) rubis_ndt_pose_pub = nh.advertise<rubis_msgs::PoseStamped>("/rubis_" + _output_pose_topic,10);

  //debug
  std::string _output_pose_topic_rubis = "/rubis_" + _output_pose_topic;
  if(rubis::instance_mode_) rubis_ndt_pose_pub = nh.advertise<rubis_msgs::PoseStamped>(_output_pose_topic_rubis,10);

  // localizer_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/localizer_pose", 10);
  estimate_twist_pub = nh.advertise<geometry_msgs::TwistStamped>(_twist_topic, 10);
  time_ndt_matching_pub = nh.advertise<std_msgs::Float32>(_ndt_time_topic, 10);
  ndt_stat_pub = nh.advertise<autoware_msgs::NDTStat>(_ndt_stat_topic, 10);

  // Subscribers
  ros::Subscriber gnss_sub = nh.subscribe("gnss_pose", 10, gnss_callback); 

  ros::Subscriber map_sub = nh.subscribe("points_map", 1, map_callback);
  ros::Subscriber initialpose_sub = nh.subscribe("initialpose", 10, initialpose_callback); 

  ros::Subscriber points_sub;
  // if(rubis::instance_mode_) points_sub = nh.subscribe("rubis_" + _input_topic, _queue_size, rubis_points_callback);
  // else points_sub = nh.subscribe(_input_topic, _queue_size, points_callback);

  //debug
  std::string _input_pose_topic_rubis = "/rubis_" + _input_pose_topic;
  if(rubis::instance_mode_) points_sub = nh.subscribe(_input_pose_topic_rubis, _queue_size, rubis_points_callback);
  else points_sub = nh.subscribe(_input_topic, _queue_size, points_callback);

  ros::Subscriber ins_stat_sub = nh.subscribe("/ins_stat", 1, ins_stat_callback);
  
  pthread_t thread;
  pthread_create(&thread, NULL, thread_func, NULL);


  //debug
  std::cout<<"----------------modular_ndt_matching_debug start------------------- "<< std::endl;
  std::cout<<"output pose topic:         " << _output_pose_topic << std::endl;
  std::cout<<"output pose topic rubis:         " << _output_pose_topic_rubis << std::endl;
  std::cout<<"instance_mode param:          " <<  << std::endl;
  std::cout<<"----------------modular_ndt_matching_debug end--------------------- "<< std::endl;
  // SPIN  
  if(!task_scheduling_flag && !task_profiling_flag){
    ros::spin();
  }
  else{ 
    ros::Rate r(rate);

    // Initialize task ( Wait until first necessary topic is published )
    while(ros::ok()){
      if(map_loaded == 1) break;
      ros::spinOnce();
      r.sleep();      
    }
    
    map_sub.shutdown();

    // Executing task
    while(ros::ok()){
      if(task_profiling_flag) rubis::sched::start_task_profiling();        
      if(rubis::sched::task_state_ == TASK_STATE_READY){        
        if(task_scheduling_flag) rubis::sched::request_task_scheduling(task_minimum_inter_release_time, task_execution_time, task_relative_deadline); 
        if(gpu_profiling_flag || gpu_scheduling_flag) rubis::sched::start_job();
        rubis::sched::task_state_ = TASK_STATE_RUNNING;     
      }

      ros::spinOnce();

      if(task_profiling_flag) rubis::sched::stop_task_profiling(rubis::instance_, rubis::sched::task_state_);
      
      if(rubis::sched::task_state_ == TASK_STATE_DONE){      
        if(gpu_profiling_flag || gpu_scheduling_flag) rubis::sched::finish_job();        
        if(task_scheduling_flag) rubis::sched::yield_task_scheduling();        
        rubis::sched::task_state_ = TASK_STATE_READY;
      }
      
      r.sleep();
    }
  }

  return 0;
}
