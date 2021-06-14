#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"
#include <cuda_runtime.h>

#define HTOD 0
#define DTOH 1
#define LAUNCH 2
#define GPU_PROFILING 1
#define SLICING

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
    int scheduling_flag;
} SchedInfo;

static char task_filename_[BUFFER_SIZE];
static int scheduler_pid_;
static int key_id_;
static SchedInfo* sched_info_;
static key_t key_;
static int shmid_;
static sigset_t sigset_;
static int sig_;
static long long int deadline_list_[550];
static int gpu_scheduling_flag_;
static unsigned long long absolute_deadline_;
static unsigned long long identical_deadline_;
static int is_scheduled_;
void set_gpu_scheduling_flag(int gpu_scheduling_flag);

void sig_handler(int signum);
void termination();
unsigned long long get_current_time_us();
void us_sleep(unsigned long long us);
void initialize_signal_handler();
void create_task_file();
void get_scheduler_pid();
void init_scheduling(char* task_filename, const char deadline_filename[], int key_id);
void request_scheduling(int id);
void get_deadline_list(char* filename);
void set_identical_deadline(unsigned long long identical_deadline);
void set_absolute_deadline();
/* ========================================*/


#ifdef __cplusplus
extern "C" {
#endif

//extern int glob_id;
extern FILE* execution_time_fp;
extern FILE* response_time_fp;
extern FILE* remain_time_fp;

extern int bias_id;
extern int normalize_id;
extern int add_id;
extern int gpu_id;
extern int upsample_id;
extern int activation_id;
extern int im2col_id;
extern int max_id;
extern int copy_gpu_id;
extern int push_id;
extern int pull_id;
extern int cpu_id;
//int count_exec;
extern float htod_time;
extern float dtoh_time;
extern float launch_time;  
extern cudaEvent_t e_event_start, e_event_stop, r_event_start, r_event_stop;

void start_profiling_execution_time();
void start_profiling_response_time();
void start_profiling_cpu_time();
void stop_cpu_profiling();
void stop_profiling(int id, int type);
//void write_profiling_data(int id, float e_time, float r_time, int type);
void write_cpu_profiling_data(const char *id, long long int c_time);
void write_profiling_data(const char *id, float e_time, float r_time, int type);
void write_dummy_line();
void initialize_file(const char execution_time_filename[], const char response_time_filename[], const char remain_time_filename[]);
void close_file();

#ifdef __cplusplus
}
#endif






#ifdef GPU
void check_error(cudaError_t status);
cublasHandle_t blas_handle();
int *cuda_make_int_array(int *x, size_t n);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);
void file_write(char *filename,int id,float time,int type);

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif