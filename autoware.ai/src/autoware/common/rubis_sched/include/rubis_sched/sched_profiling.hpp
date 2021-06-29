#ifndef __RUBIS_SCHED_PROF_H__
#define __RUBIS_SCHED_PROF_H__
#include <stdio.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <iostream>

namespace rubis {
namespace sched {
  static int task_profiling_flag_;
  static int gpu_profiling_flag_;

  static FILE* task_response_time_fp_;
  static FILE* seg_execution_time_fp_;
  static FILE* seg_response_time_fp_;

  static unsigned int cpu_seg_id_;
  static int is_gpu_profiling_started_;

  static struct timespec task_start_time_;
  static struct timespec task_end_time_;
  static unsigned long long gpu_seg_response_time_;
  static unsigned long long gpu_seg_execution_time_;
  static unsigned long long cpu_seg_response_time_;

  void init_task_profiling(std::string task_reponse_time_filename);
  void start_task_profiling();
  void stop_task_profiling();
  void init_gpu_profiling(std::string execution_time_filename, std::string response_time_filename);
  void start_profiling_cpu_seg_response_time();
  void stop_profiling_cpu_seg_response_time();
  void start_profiling_gpu_seg_response_time();
  void start_profiling_gpu_seg_execution_time();
  void stop_profiling_gpu_seg_time(unsigned int id, std::string remark = " ");
  unsigned long long get_current_time_ns();
  void refresh_gpu_profiling();

}
}

#endif