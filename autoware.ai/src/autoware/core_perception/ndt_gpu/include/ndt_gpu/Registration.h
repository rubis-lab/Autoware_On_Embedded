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

/* For GPU Scheduling ====================*/
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <sys/ipc.h>
#include <sys/shm.h>


#define BUFFER_SIZE 1024
#define NS2MS(t) (t/1000000)
#define NS2US(t) (t/1000)
#define MS2NS(t) (t*1000000)
#define MS2US(t) (t*1000)

#define STOP -1
#define NONE 0
#define WAIT 1
#define RUN 2

typedef struct schedInfo{
    int pid;
    unsigned long long deadline;
    int state; // NONE = 0, WAIT = 1, RUN = 2
} SchedInfo;

static char task_filename_[BUFFER_SIZE];
static int scheduler_pid_;
static int key_id_;
static SchedInfo* sched_info_;
static key_t key_;
static int shmid_;
static sigset_t sigset_;
static int sig_;

void sig_handler(int signum);
void termination();
unsigned long long get_current_time_us();
void us_sleep(unsigned long long us);
void initialize_signal_handler();
void create_task_file();
void get_scheduler_pid();
void init_scheduling(char* task_filename, int key_id);
void request_scheduling(unsigned long long relative_deadline);
/* ========================================*/

#define HTOD 0
#define DTOH 1
#define LAUNCH 2
#define GPU_PROFILING 1

static FILE* fp;

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

  /* Added for GPU profiling */
  cudaEvent_t event_start, event_stop;
  
  int gid = 0;

  void start_profiling();
  void stop_profiling(int id, int type);
  void write_data(int id, float time, int type);
  void write_dummy_line();
  void initialize_file(const char name[]);
  void close_file();

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
