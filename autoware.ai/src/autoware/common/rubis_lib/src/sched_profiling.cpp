#include "rubis_lib/sched_profiling.hpp"

namespace rubis{
  int instance_mode_;
  unsigned long instance_;
  unsigned long obj_instance_;
namespace sched{
  int task_profiling_flag_;
  int gpu_profiling_flag_;
  int iter_;

  FILE* task_response_time_fp_;
  FILE* seg_execution_time_fp_;
  FILE* seg_response_time_fp_;

  int is_gpu_profiling_started_;

  struct timespec task_start_time_;
  struct timespec task_end_time_;
  unsigned long gpu_seg_response_time_;
  unsigned long gpu_seg_execution_time_;
  unsigned long cpu_seg_response_time_;
  int is_gpu_profiling_ready_ = 0;

  void start_gpu_profiling(){
    is_gpu_profiling_ready_ = 1;
  }

  void init_task_profiling(std::string task_reponse_time_filename){
    if(task_reponse_time_filename.at(0) == '~'){
      task_reponse_time_filename.erase(0, 1);
      std::string user_home_str(std::getenv("USER_HOME"));
      task_reponse_time_filename =  user_home_str + task_reponse_time_filename;
    }

    task_profiling_flag_ = 1;
    task_response_time_fp_ = fopen(task_reponse_time_filename.c_str(), "w+");
    if(task_response_time_fp_ == NULL){
      std::cout<<"Cannot create/open file: "<<task_reponse_time_filename<<std::endl;
      perror("Failed: ");
      exit(0);
    }

    chmod(task_reponse_time_filename.c_str(), strtol("0777", 0, 8));
    fprintf(task_response_time_fp_, "iter,PID,start,end,instance,activation,obj_instance\n");
  }

  void start_task_profiling(){
    if(task_profiling_flag_) clock_gettime(CLOCK_MONOTONIC, &task_start_time_);
  }

  void stop_task_profiling(unsigned long instance, int state){
    if(task_profiling_flag_){
      int activation = 0;
      if(state == TASK_STATE_DONE) activation = 1;
      clock_gettime(CLOCK_MONOTONIC, &task_end_time_);
      fprintf(task_response_time_fp_, "%d,%d,%lld.%.9ld,%lld.%.9ld,%lu,%d,%lu\n",iter_++, getpid(), (long long)task_start_time_.tv_sec, task_start_time_.tv_nsec, (long long)task_end_time_.tv_sec, task_end_time_.tv_nsec, instance, activation, obj_instance_);
      fflush(task_response_time_fp_);
    }
  }

  void init_gpu_profiling(std::string execution_time_filename, std::string response_time_filename){
    if(execution_time_filename.at(0) == '~'){
      execution_time_filename.erase(0, 1);
      std::string user_home_str(std::getenv("USER_HOME"));
      execution_time_filename =  user_home_str + execution_time_filename;
    }

    if(response_time_filename.at(0) == '~'){
      response_time_filename.erase(0, 1);
      std::string user_home_str(std::getenv("USER_HOME"));
      response_time_filename =  user_home_str + response_time_filename;
    }

    gpu_profiling_flag_ = 1;
    seg_execution_time_fp_ = fopen(execution_time_filename.c_str(), "w+");
    chmod(execution_time_filename.c_str(), strtol("0777", 0, 8));
    fprintf(seg_execution_time_fp_, "ID,TIME,ITERATION, REMARK\n");

    seg_response_time_fp_ = fopen(response_time_filename.c_str(), "w+");
    chmod(response_time_filename.c_str(), strtol("0777", 0, 8));
    fprintf(seg_response_time_fp_, "ID,TIME,ITERATION, REMARK\n");
  }

  void start_profiling_cpu_seg_response_time(){
    if(gpu_profiling_flag_)
      cpu_seg_response_time_ = get_current_time_ns();
  }

  void stop_profiling_cpu_seg_response_time(unsigned int cpu_seg_id, unsigned int iter){
    if(gpu_profiling_flag_){
      cpu_seg_response_time_ = get_current_time_ns() - cpu_seg_response_time_;
      fprintf(seg_response_time_fp_, "cpu_%u,%llu,%u\n", cpu_seg_id, cpu_seg_response_time_, iter);
    }    
  }

  void start_profiling_gpu_seg_response_time(){
    if(gpu_profiling_flag_) gpu_seg_response_time_ = get_current_time_ns();
  }

  void start_profiling_gpu_seg_execution_time(){
    if(gpu_profiling_flag_) gpu_seg_execution_time_ = get_current_time_ns();
  }

  void stop_profiling_gpu_seg_time(unsigned int gpu_seg_id, unsigned int iter, std::string remark){
    if(gpu_profiling_flag_){
      unsigned long current_time = get_current_time_ns();
      gpu_seg_response_time_ = current_time - gpu_seg_response_time_;
      gpu_seg_execution_time_ = current_time - gpu_seg_execution_time_;
      
      fprintf(seg_response_time_fp_, "gpu_%u,%llu,%u,%s\n", gpu_seg_id, gpu_seg_response_time_, iter, remark.c_str());
      fprintf(seg_execution_time_fp_, "gpu_%u,%llu,%u,%s\n", gpu_seg_id, gpu_seg_execution_time_, iter, remark.c_str());
    }
  }

  unsigned long get_current_time_ns(){
    struct timespec ts;
    unsigned long current_time;
    clock_gettime(CLOCK_REALTIME, &ts);
    current_time = ts.tv_sec%10000 * 1000000000 + ts.tv_nsec;
    return current_time;
  }

  void start_job_profiling(){
    if(!gpu_profiling_flag_) return;
    
    if(is_gpu_profiling_started_ == 0) is_gpu_profiling_started_ = 1;
    else if( (is_gpu_profiling_started_ == 1) && (is_gpu_profiling_ready_ == 1) ){
      start_profiling_cpu_seg_response_time();
    }    
  }

  void finish_job_profiling(unsigned int cpu_seg_id){
    if(!gpu_profiling_flag_) return;
    
    if(is_gpu_profiling_started_ == 0) is_gpu_profiling_started_ = 1;
    else if( (is_gpu_profiling_started_ == 1) && (is_gpu_profiling_ready_ == 1) ){
      stop_profiling_cpu_seg_response_time(cpu_seg_id, 1);
      fprintf(seg_response_time_fp_, "-1, -1, -1\n");
      fprintf(seg_execution_time_fp_, "-1, -1, -1\n");
      
      fflush(seg_response_time_fp_);
      fflush(seg_execution_time_fp_);
    }    
  }


}// end namespace
}