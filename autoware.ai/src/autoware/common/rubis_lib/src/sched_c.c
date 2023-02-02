#include "rubis_lib/sched_c.h"

// #define DEBUG

int key_id_;
int is_scheduled_;
char* task_filename_;

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

int set_sched_deadline(int _tid, __u64 _exec_time, __u64 _deadline, __u64 _period) {
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
        printf("[ERROR] sched_setattr failed. Are you root? (%d)\n", ret);
        perror("sched_setattr");
        exit(-1);
        return 0;
    } 
    return 1;
}

void request_task_scheduling(double task_minimum_inter_release_time, double task_execution_time, double task_relative_deadline){
  set_sched_deadline(gettid(), 
    (u_int64_t)(task_execution_time), 
    (u_int64_t)(task_relative_deadline), 
    (u_int64_t)(task_minimum_inter_release_time)
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
  
  if(remove(task_filename_)){
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
