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

#define rubis_lib_BUFFER_SIZE 1024

extern unsigned long instance_;
extern unsigned long obj_instance_;

extern FILE* task_response_time_fp_;

extern struct timespec task_start_time_;
extern struct timespec task_end_time_;

#ifdef __cplusplus
extern "C"{
#endif

void init_task_profiling(char* _task_response_time_filename);
void start_task_profiling();
void stop_task_profiling(unsigned long instance, int state);
unsigned long get_current_time_ns();

#ifdef __cplusplus
}
#endif

#endif

