#ifndef GNDT_H_
#define GNDT_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "Matrix.h"
#include "MatrixHost.h"
#include "MatrixDevice.h"
#include "common.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <stdio.h>
#include <rubis_sched/sched.hpp>

#define HTOD 0
#define DTOH 1
#define LAUNCH 2
#define GPU_PROFILING 1

using namespace rubis;

void start_profiling_execution_time();
void start_profiling_response_time();
void start_profiling_cpu_time();
void stop_cpu_profiling();
void stop_profiling(int id, int type);
void write_profiling_data(const char* id, float e_time, float r_time, int type);
void write_cpu_profiling_data(const char *id, long long int c_time);
void write_dummy_line();
void initialize_file(const char* execution_time_filename, const char* response_time_filename, const char* remain_time_filename);
void close_file();

namespace gpu {
class GRegistration {
public:
  GRegistration();
  GRegistration(const GRegistration &other);

  void align(const Eigen::Matrix<float, 4, 4> &guess);

  void setTransformationEpsilon(double trans_eps);

  double getTransformationEpsilon() const;

  void setMaximumIterations(int max_itr);

  int getMaximumIterations() const;

  Eigen::Matrix<float, 4, 4> getFinalTransformation() const;

  /* Set input Scanned point cloud.
   * Copy input points from the main memory to the GPU memory */
  void setInputSource(pcl::PointCloud<pcl::PointXYZI>::Ptr input);
  void setInputSource(pcl::PointCloud<pcl::PointXYZ>::Ptr input);

  /* Set input reference map point cloud.
   * Copy input points from the main memory to the GPU memory */
  void setInputTarget(pcl::PointCloud<pcl::PointXYZI>::Ptr input);
  void setInputTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr input);

  int getFinalNumIteration() const;

  bool hasConverged() const;

  virtual ~GRegistration();

protected:

  virtual void computeTransformation(const Eigen::Matrix<float, 4, 4> &guess);

  double transformation_epsilon_;
  int max_iterations_;

  //Original scanned point clouds
  float *x_, *y_, *z_;
  int points_number_;

  //Transformed point clouds
  float *trans_x_, *trans_y_, *trans_z_;

  bool converged_;
  int nr_iterations_;

  Eigen::Matrix<float, 4, 4> final_transformation_, transformation_, previous_transformation_;

  bool target_cloud_updated_;

  // Reference map point
  float *target_x_, *target_y_, *target_z_;
  int target_points_number_;

  bool is_copied_;
};
}

#endif
