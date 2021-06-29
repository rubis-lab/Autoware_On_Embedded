#include "rubis_sched/sched_profiling.hpp"

namespace rubis{
namespace sched{

  void init_task_profiling(std::string task_reponse_time_filename){
    task_profiling_flag_ = 1;
    task_response_time_fp_ = fopen(task_reponse_time_filename.c_str(), "w+");
    chmod(task_reponse_time_filename.c_str(), strtol("0777", 0, 8));
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

  void init_gpu_profiling(std::string execution_time_filename, std::string response_time_filename){
    gpu_profiling_flag_ = 1;
    seg_execution_time_fp_ = fopen(execution_time_filename.c_str(), "w+");
    chmod(execution_time_filename.c_str(), strtol("0777", 0, 8));
    fprintf(seg_execution_time_fp_, "ID, TIME, REMARK\n");

    seg_response_time_fp_ = fopen(response_time_filename.c_str(), "w+");
    chmod(response_time_filename.c_str(), strtol("0777", 0, 8));
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

  void stop_profiling_gpu_seg_time(unsigned int id, std::string remark){
    if(gpu_profiling_flag_){
      unsigned long long current_time = get_current_time_ns();
      gpu_seg_response_time_ = current_time - gpu_seg_response_time_;
      gpu_seg_execution_time_ = current_time - gpu_seg_execution_time_;
      
      fprintf(seg_response_time_fp_, "gpu_%u, %llu, %s\n", id, gpu_seg_response_time_, remark.c_str());
      fprintf(seg_execution_time_fp_, "gpu_%u, %llu, %s\n", id, gpu_seg_execution_time_, remark.c_str());
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
    printf("refresh gpu profiling\n");
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

}// end namespace
}