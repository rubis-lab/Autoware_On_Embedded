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
static long long int deadline_list_[35];

void sig_handler(int signum);
void termination();
unsigned long long get_current_time_us();
void us_sleep(unsigned long long us);
void initialize_signal_handler();
void create_task_file();
void get_scheduler_pid();
void init_scheduling(char* task_filename, int key_id);
void request_scheduling(unsigned long long relative_deadline);
void get_deadlines(char* filename);
/* ========================================*/

#define HTOD 0
#define DTOH 1
#define LAUNCH 2
#define GPU_PROFILING 1

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

  /* Added for GPU profiling */
  cudaEvent_t event_start, event_stop;
  
  int gid = 0;

  void start_profiling();
  // void stop_profiling(int type);
  void stop_profiling(int id, int type);
  void write_data(int id, float time, int type);
  void write_dummy_line();
  void initialize_file(const char name[]);
  void close_file();


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
