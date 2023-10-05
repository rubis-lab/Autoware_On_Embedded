#ifndef __RUBIS_LIB_PROF_H__
#define __RUBIS_LIB_PROF_H__

#include <cstdlib>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace rubis {
extern unsigned long instance_;
extern unsigned long lidar_instance_;
extern unsigned long vision_instance_;

extern FILE *task_response_time_fp_;

extern struct timespec task_start_time_;
extern struct timespec task_end_time_;

void init_task_profiling(std::string task_reponse_time_filename);
void start_task_profiling();
void start_task_profiling_at_initial_node(long long tp_time_sec,
                                          long long tp_time_nsec);
void stop_task_profiling(unsigned long instance, unsigned long lidar_instance_,
                         unsigned long vision_instance_);
unsigned long get_current_time_ns();
} // namespace rubis

#endif