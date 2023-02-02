#include "rubis_lib/sched_profiling_c.h"ß

unsigned long instance_;
unsigned long obj_instance_;

int iter_;

FILE* task_response_time_fp_;

struct timespec task_start_time_;
struct timespec task_end_time_;

void init_task_profiling(char* _task_response_time_filename){
  char task_response_time_filename[rubis_lib_BUFFER_SIZE];
  char* user_name = getenv("USER_HOME");

  if(_task_response_time_filename[0] != '~'){
    strcpy(task_response_time_filename, _task_response_time_filename);
  }
  else{      
    strcpy(task_response_time_filename, user_name);
    strcat(task_response_time_filename, &_task_response_time_filename[1]);
  }

  task_response_time_fp_ = fopen(task_response_time_filename, "w+");
  if(task_response_time_fp_ == NULL){
      printf("Cannot create/open file: %s\n", task_response_time_filename);
      perror("Failed: ");
      exit(0);
  }

  chmod(task_response_time_filename, strtol("0777", 0, 8));
  fprintf(task_response_time_fp_, "iter,PID,start,end,instance\n");
}

void start_task_profiling(){
  clock_gettime(CLOCK_MONOTONIC, &task_start_time_);
}

void stop_task_profiling(unsigned long instance, int state){
  clock_gettime(CLOCK_MONOTONIC, &task_end_time_);
  fprintf(task_response_time_fp_, "%d,%d,%lld.%.9ld,%lld.%.9ld,%lu,%lu\n",iter_++, getpid(), (long long)task_start_time_.tv_sec, task_start_time_.tv_nsec, (long long)task_end_time_.tv_sec, task_end_time_.tv_nsec, instance, obj_instance_);
  fflush(task_response_time_fp_);
}

unsigned long get_current_time_ns(){
  struct timespec ts;
  unsigned long current_time;
  clock_gettime(CLOCK_REALTIME, &ts);
  current_time = ts.tv_sec%10000 * 1000000000 + ts.tv_nsec;
  return current_time;
}
