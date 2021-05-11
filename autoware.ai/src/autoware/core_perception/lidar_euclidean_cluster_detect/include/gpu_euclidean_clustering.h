#ifndef GPU_EUCLIDEAN_H_
#define GPU_EUCLIDEAN_H_

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

/* For GPU Scheduling ====================*/
#include <stdio.h>
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
static unsigned long long deadline_list_[40];
static int gpu_scheduling_flag_;
static unsigned long long absolute_deadline_;
static unsigned long long identical_deadline_;

void sig_handler(int signum);
void termination();
unsigned long long get_current_time_us();
void us_sleep(unsigned long long us);
void initialize_signal_handler();
void create_task_file();
void get_scheduler_pid();
void init_scheduling(char* task_filename, char* deadline_filename, int key_id);
void get_deadline_list(char* filename);
void set_identical_deadline(unsigned long long identical_deadline);
void set_absolute_deadline();
void initialize_sched_info();
void request_scheduling(int id);
/* ========================================*/

#define HTOD 0
#define DTOH 1
#define LAUNCH 2
#define GPU_PROFILING 1

/* Added for GPU profiling */
static cudaEvent_t e_event_start, e_event_stop, r_event_start, r_event_stop;

void start_profiling_execution_time();
void start_profiling_response_time();
void stop_profiling(int id, int type);
void write_profiling_data(int id, float e_time, float r_time, int type);
void write_dummy_line();
void initialize_file(const char execution_time_filename[], const char response_time_filename[], const char remain_time_filename[]);
void close_file();


class GpuEuclideanCluster
{
public:
  typedef struct
  {
    int index_value;
    std::vector<int> points_in_cluster;
  } GClusterIndex;

  typedef struct
  {
    float* x;
    float* y;
    float* z;
    int size;
  } SamplePointListXYZ;

  GpuEuclideanCluster();

  void setInputPoints(float* x, float* y, float* z, int size);
  void setThreshold(double threshold);
  void setMinClusterPts(int min_cluster_pts);
  void setMaxClusterPts(int max_cluster_pts);
  void extractClustersOld();
  void extractClusters();
  void extractClusters2();
  std::vector<GClusterIndex> getOutput();

  SamplePointListXYZ generateSample();

  ~GpuEuclideanCluster();

private:
  float *x_, *y_, *z_;
  int size_;
  double threshold_;
  int* cluster_indices_;
  int* cluster_indices_host_;
  int min_cluster_pts_;
  int max_cluster_pts_;
  int cluster_num_;

  void exclusiveScan(int* input, int ele_num, int* sum);
};

#endif
