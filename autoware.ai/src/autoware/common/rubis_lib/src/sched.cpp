#include "rubis_lib/sched.hpp"

// #define DEBUG 

namespace rubis{
namespace sched{

int key_id_;
int is_scheduled_;
std::string task_filename_;

int is_task_ready_ = TASK_NOT_READY;
int task_state_ = TASK_STATE_READY;

// system call hook to call SCHED_DEADLINE
int sched_setattr(pid_t pid, const struct sched_attr *attr, unsigned int flags){
	return syscall(__NR_sched_setattr, pid, attr, flags);
}

int sched_getattr(pid_t pid, struct sched_attr *attr, unsigned int size, unsigned int flags)
{
	return syscall(__NR_sched_getattr, pid, attr, size, flags);
}

bool set_sched_deadline(int _tid, __u64 _exec_time, __u64 _deadline, __u64 _period) {
    struct sched_attr attr;
    attr.size = sizeof(attr);
    attr.sched_flags = 0;
    attr.sched_nice = 0;
    attr.sched_priority = 0;

    attr.sched_policy = SCHED_DEADLINE; // 6
    attr.sched_runtime = _exec_time;
    attr.sched_deadline = _deadline;
    attr.sched_period = _period;

    int ret = sched_setattr(_tid, &attr, attr.sched_flags);
    if(ret < 0) {
        std::cerr << "[ERROR] sched_setattr failed. Are you root? (" << ret << ")" << std::endl;
        perror("sched_setattr");
        exit(-1);
        return false;
    } 
    // else {
    //     std::cerr << "[SCHED_DEADLINE] (" << _tid << ") exec_time: " << _exec_time << " _deadline: " << _deadline << " _period: " << _period << std::endl;
    // }
    return true;
}

void request_task_scheduling(double task_minimum_inter_release_time, double task_execution_time, double task_relative_deadline){
  sched::set_sched_deadline(gettid(), 
    static_cast<uint64_t>(task_execution_time), 
    static_cast<uint64_t>(task_relative_deadline), 
    static_cast<uint64_t>(task_minimum_inter_release_time)
  );
}

void yield_task_scheduling(){
  sched_yield();
}

void sig_handler(int signum){
  if(signum == SIGINT || signum == SIGTSTP || signum == SIGQUIT){
    termination();
  }
}

void termination(){
  printf("TERMINATION\n");
  if(remove(task_filename_.c_str())){
      printf("Cannot remove file %s\n", task_filename_);
      exit(1);
  }

  if(task_profiling_flag_){
    fclose(task_response_time_fp_);
  }
  exit(0);
}

void init_task(){
  is_task_ready_ = TASK_READY;
}

void disable_task(){
  is_task_ready_ = TASK_NOT_READY;
}

} // namespace sched
} // namespace rubiss