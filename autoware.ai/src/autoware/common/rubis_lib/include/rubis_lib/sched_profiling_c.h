#ifndef __RUBIS_LIB_PROF_H__
#define __RUBIS_LIB_PROF_H__

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define rubis_lib_BUFFER_SIZE 1024

extern unsigned long instance_;
extern unsigned long lidar_instance_;
extern unsigned long vision_instance_;

extern FILE *task_response_time_fp_;

extern struct timespec task_start_time_;
extern struct timespec task_end_time_;

#ifdef __cplusplus
extern "C" {
#endif

void init_task_profiling(char *_task_response_time_filename);
void start_task_profiling();
void stop_task_profiling(unsigned long instance, unsigned long lidar_instance_,
                         unsigned long vision_instance_);
unsigned long get_current_time_ns();

#ifdef __cplusplus
}
#endif

#endif
