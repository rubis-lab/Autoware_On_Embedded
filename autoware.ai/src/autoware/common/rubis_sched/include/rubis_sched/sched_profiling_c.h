#ifndef __RUBIS_SCHED_PROF_H__
#define __RUBIS_SCHED_PROF_H__


#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#define RUBIS_SCHED_BUFFER_SIZE 1024

extern int task_profiling_flag_;
extern int gpu_profiling_flag_;

extern FILE* task_response_time_fp_;
extern FILE* seg_execution_time_fp_;
extern FILE* seg_response_time_fp_;

extern unsigned int cpu_seg_id_;
extern int is_gpu_profiling_started_;

extern struct timespec task_start_time_;
extern struct timespec task_end_time_;
extern unsigned long long gpu_seg_response_time_;
extern unsigned long long gpu_seg_execution_time_;
extern unsigned long long cpu_seg_response_time_;
extern int is_gpu_profiling_ready_;

#ifdef __cplusplus
extern "C"{
#endif

void init_task_profiling(char* _task_response_time_filename);
void start_task_profiling();
void stop_task_profiling();
void init_gpu_profiling(char* _execution_time_filename, char* _response_time_filename);
void start_profiling_cpu_seg_response_time();
void stop_profiling_cpu_seg_response_time();
void start_profiling_gpu_seg_response_time();
void start_profiling_gpu_seg_execution_time();
void stop_profiling_gpu_seg_time(unsigned int id);
void stop_profiling_gpu_seg_time_with_remark(unsigned int id, const char* remark);
unsigned long long get_current_time_ns();
void refresh_gpu_profiling();
void start_gpu_profiling();

#ifdef __cplusplus
}
#endif

#endif

