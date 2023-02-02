#ifndef __RUBIS_LIB_PROF_H__
#define __RUBIS_LIB_PROF_H__

#include <stdio.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <cstdlib>
#include <rubis_lib/common.hpp>

namespace rubis {
  extern int instance_mode_;
  extern unsigned long instance_;
  extern unsigned long obj_instance_;

namespace sched {
  extern int task_profiling_flag_;

  extern FILE* task_response_time_fp_;

  extern struct timespec task_start_time_;
  extern struct timespec task_end_time_;

  void init_task_profiling(std::string task_reponse_time_filename);
  void start_task_profiling();
  void stop_task_profiling(unsigned long instance, int state);
  unsigned long get_current_time_ns();

}
}

#endif