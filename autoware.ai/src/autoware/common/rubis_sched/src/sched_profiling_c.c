#include "rubis_sched/sched_profiling_c.h"

int task_profiling_flag_;
int gpu_profiling_flag_;

FILE* task_response_time_fp_;
FILE* seg_execution_time_fp_;
FILE* seg_response_time_fp_;

unsigned int cpu_seg_id_;
int is_gpu_profiling_started_;

struct timespec task_start_time_;
struct timespec task_end_time_;
unsigned long long gpu_seg_response_time_;
unsigned long long gpu_seg_execution_time_;
unsigned long long cpu_seg_response_time_;

void init_task_profiling(char* _task_response_time_filename){
  char task_response_time_filename[RUBIS_SCHED_BUFFER_SIZE];
  char* user_name = getenv("USER_HOME");

  if(_task_response_time_filename[0] != '~'){
    strcpy(task_response_time_filename, _task_response_time_filename);
  }
  else{      
    strcpy(task_response_time_filename, user_name);
    strcat(task_response_time_filename, &_task_response_time_filename[1]);
  }

  task_profiling_flag_ = 1;
  task_response_time_fp_ = fopen(task_response_time_filename, "w+");
  chmod(task_response_time_filename, strtol("0777", 0, 8));
}

void start_task_profiling(){
  if(task_profiling_flag_) clock_gettime(CLOCK_MONOTONIC, &task_start_time_);
}

void stop_task_profiling(){
  if(task_profiling_flag_){
    clock_gettime(CLOCK_MONOTONIC, &task_end_time_);
    fprintf(task_response_time_fp_, "%lld.%.9ld,%lld.%.9ld,%d\n",(long long)task_start_time_.tv_sec, task_start_time_.tv_nsec, (long long)task_end_time_.tv_sec, task_end_time_.tv_nsec, getpid());
    fflush(task_response_time_fp_);
  }
}

void init_gpu_profiling(char* _execution_time_filename, char* _response_time_filename){    
  char execution_time_filename[RUBIS_SCHED_BUFFER_SIZE];    
  char response_time_filename[RUBIS_SCHED_BUFFER_SIZE];    
  char* user_name = getenv("USER_HOME");


  if(_execution_time_filename[0] != '~'){
    strcpy(execution_time_filename, _execution_time_filename);
  }
  else{      
    strcpy(execution_time_filename, user_name);
    strcat(execution_time_filename, &_execution_time_filename[1]);
  }

  if(_response_time_filename[0] != '~'){
    strcpy(response_time_filename, _response_time_filename);
  }
  else{      
    strcpy(response_time_filename, user_name);
    strcat(response_time_filename, &_response_time_filename[1]);
  }

  gpu_profiling_flag_ = 1;
  seg_execution_time_fp_ = fopen(execution_time_filename, "w+");
  chmod(execution_time_filename, strtol("0777", 0, 8));
  fprintf(seg_execution_time_fp_, "ID, TIME, REMARK\n");

  seg_response_time_fp_ = fopen(response_time_filename, "w+");
  chmod(response_time_filename, strtol("0777", 0, 8));
  fprintf(seg_response_time_fp_, "ID, TIME, REMARK\n");
  cpu_seg_id_ = 0;
}

void start_profiling_cpu_seg_response_time(){
  if(gpu_profiling_flag_)
    cpu_seg_response_time_ = get_current_time_ns();
}

void stop_profiling_cpu_seg_response_time(){
  if(gpu_profiling_flag_){
    cpu_seg_response_time_ = get_current_time_ns() - cpu_seg_response_time_;
    fprintf(seg_response_time_fp_, "cpu_%u, %llu,  \n", cpu_seg_id_, cpu_seg_response_time_);
    cpu_seg_id_++;
  }    
}

void start_profiling_gpu_seg_response_time(){
  if(gpu_profiling_flag_) gpu_seg_response_time_ = get_current_time_ns();
}

void start_profiling_gpu_seg_execution_time(){
  if(gpu_profiling_flag_) gpu_seg_execution_time_ = get_current_time_ns();
}

void stop_profiling_gpu_seg_time(unsigned int id){
  if(gpu_profiling_flag_){
    unsigned long long current_time = get_current_time_ns();
    gpu_seg_response_time_ = current_time - gpu_seg_response_time_;
    gpu_seg_execution_time_ = current_time - gpu_seg_execution_time_;
    
    fprintf(seg_response_time_fp_, "gpu_%u, %llu\n", id, gpu_seg_response_time_);
    fprintf(seg_execution_time_fp_, "gpu_%u, %llu\n", id, gpu_seg_execution_time_);
  }
}

void stop_profiling_gpu_seg_time_with_remark(unsigned int id, const char* remark){
  if(gpu_profiling_flag_){
    unsigned long long current_time = get_current_time_ns();
    gpu_seg_response_time_ = current_time - gpu_seg_response_time_;
    gpu_seg_execution_time_ = current_time - gpu_seg_execution_time_;
    
    fprintf(seg_response_time_fp_, "gpu_%u, %llu, %s\n", id, gpu_seg_response_time_, remark);
    fprintf(seg_execution_time_fp_, "gpu_%u, %llu, %s\n", id, gpu_seg_execution_time_, remark);
  }
}

unsigned long long get_current_time_ns(){
  struct timespec ts;
  unsigned long long current_time;
  clock_gettime(CLOCK_REALTIME, &ts);
  current_time = ts.tv_sec%10000 * 1000000000 + ts.tv_nsec;
  return current_time;
}

void refresh_gpu_profiling(){
  if(is_gpu_profiling_started_ == 0) is_gpu_profiling_started_ = 1;
  else if(is_gpu_profiling_started_ == 1){
    stop_profiling_cpu_seg_response_time();
    fprintf(seg_response_time_fp_, "-1, -1, -1\n");
    fprintf(seg_execution_time_fp_, "-1, -1, -1\n");
    
    fflush(seg_response_time_fp_);
    fflush(seg_execution_time_fp_);
    cpu_seg_id_ = 0;
    start_profiling_cpu_seg_response_time();
  }
}