#ifndef __RUBIS_LIB_PROF_H__
#define __RUBIS_LIB_PROF_H__


#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <rubis_lib/common_c.h>

#define rubis_lib_BUFFER_SIZE 1024

extern int instance_mode_;
extern unsigned long instance_;
extern unsigned long obj_instance_;

extern int task_profiling_flag_;
extern int gpu_profiling_flag_;

extern FILE* task_response_time_fp_;
extern FILE* seg_execution_time_fp_;
extern FILE* seg_response_time_fp_;

extern int is_gpu_profiling_started_;

extern struct timespec task_start_time_;
extern struct timespec task_end_time_;
extern unsigned long gpu_seg_response_time_;
extern unsigned long gpu_seg_execution_time_;
extern unsigned long cpu_seg_response_time_;
extern int is_gpu_profiling_ready_;

#ifdef __cplusplus
extern "C"{
#endif

void init_task_profiling(char* _task_response_time_filename);
void start_task_profiling();
void stop_task_profiling(unsigned long instance, int state);
void init_gpu_profiling(char* _execution_time_filename, char* _response_time_filename);
void start_profiling_cpu_seg_response_time();
void stop_profiling_cpu_seg_response_time(unsigned int cpu_seg_id, unsigned int iter);
void start_profiling_gpu_seg_response_time();
void start_profiling_gpu_seg_execution_time();
void stop_profiling_gpu_seg_time(unsigned int gpu_seg_id, unsigned int iter);
void stop_profiling_gpu_seg_time_with_remark(unsigned int gpu_seg_id, unsigned int iter, const char* remark);
unsigned long get_current_time_ns();
void start_job_profiling();
void finish_job_profiling(unsigned int cpu_seg_id);
void start_gpu_profiling();

#ifdef __cplusplus
}
#endif

#endif

