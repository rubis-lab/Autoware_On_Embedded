#include "rubis_lib/sched_profiling.hpp"

namespace rubis{
  unsigned long instance_;
  unsigned long obj_instance_;

  int iter_;

  FILE* task_response_time_fp_;

  struct timespec task_start_time_;
  struct timespec task_end_time_;

  void init_task_profiling(std::string task_reponse_time_filename){
    if(task_reponse_time_filename.at(0) == '~'){
      task_reponse_time_filename.erase(0, 1);
      std::string user_home_str(std::getenv("USER_HOME"));
      task_reponse_time_filename =  user_home_str + task_reponse_time_filename;
    }

    task_response_time_fp_ = fopen(task_reponse_time_filename.c_str(), "w+");
    if(task_response_time_fp_ == NULL){
      std::cout<<"Cannot create/open file: "<<task_reponse_time_filename<<std::endl;
      perror("Failed: ");
      exit(0);
    }

    chmod(task_reponse_time_filename.c_str(), strtol("0777", 0, 8));
    fprintf(task_response_time_fp_, "iter,PID,start,end,instance,obj_instance\n");
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

}