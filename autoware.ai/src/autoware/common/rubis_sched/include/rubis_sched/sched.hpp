#ifndef __RUBIS_SCHED_H__
#define __RUBIS_SCHED_H__

#include <linux/kernel.h>
#include <linux/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <vector>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <iostream>

#include "rubis_sched/sched_profiling.hpp"

/* XXX use the proper syscall numbers */
#ifdef __x86_64__
#define __NR_sched_setattr		314
#define __NR_sched_getattr		315
#endif

#ifdef __i386__
#define __NR_sched_setattr		351
#define __NR_sched_getattr		352
#endif

#ifdef __arm__
#define __NR_sched_setattr		380
#define __NR_sched_getattr		381
#endif

#ifdef __aarch64__
#define __NR_sched_setattr		274
#define __NR_sched_getattr		275
#endif

#define SCHED_DEADLINE	6

#define gettid() syscall(__NR_gettid)
#define BUFFER_SIZE 1024
#define NS2MS(t) (t/1000000)
#define NS2US(t) (t/1000)
#define MS2NS(t) (t*1000000)
#define MS2US(t) (t*1000)

#define SCHEDULING_STATE_STOP -1
#define SCHEDULING_STATE_NONE 0
#define SCHEDULING_STATE_WAIT 1
#define SCHEDULING_STATE_RUN 2

#define TASK_NOT_READY 0
#define TASK_READY 1

// Task state
#define TASK_STATE_READY 0
#define TASK_STATE_RUNNING 1
#define TASK_STATE_DONE 2

#define GPU_SEG_LOOP_START 0
#define GPU_SEG_LOOP_MID 1
#define GPU_SEG_LOOP_END 2


namespace rubis {
namespace sched {

// contains thread-specific arguments
struct thr_arg {
    int thr_id;
    int exec_time;
};

// contains shared arguments among the threads and the vector of threads
struct task_arg {
    int option;
    int task_id;
    int parent;
    int deadline;
    int period;
    std::vector<thr_arg> thr_set;
};

struct sched_attr {
	__u32 size;

	__u32 sched_policy;
	__u64 sched_flags;

	/* SCHED_NORMAL, SCHED_BATCH */
	__s32 sched_nice;

	/* SCHED_FIFO, SCHED_RR */
	__u32 sched_priority;

	/*  SCHED_DEADLINE (nsec) */
	__u64 sched_runtime;
	__u64 sched_deadline;
	__u64 sched_period;
};

// GPU 
typedef struct  gpuSchedInfo{
    int pid;
    unsigned long deadline;
    int state; // NONE = 0, WAIT = 1, RUN = 2
    int scheduling_flag;
} GPUSchedInfo;

extern int key_id_;
extern int is_scheduled_;
extern int gpu_scheduling_flag_;
extern GPUSchedInfo* gpu_sched_info_;
extern int gpu_scheduler_pid_;
extern std::string task_filename_;
extern std::string gpu_deadline_filename_;
// extern unsigned long gpu_deadline_list_[1024];
extern unsigned long* gpu_deadline_list_;
extern unsigned int max_gpu_id_;
extern unsigned int gpu_seg_id_;
extern unsigned int cpu_seg_id_;
extern int task_state_;
extern int is_task_ready_;
extern int was_in_loop_;
extern int loop_cnt_;
extern int gpu_seg_cnt_in_loop_;

// Task scheduling
int sched_setattr(pid_t pid, const struct sched_attr *attr, unsigned int flags);
int sched_getattr(pid_t pid, struct sched_attr *attr, unsigned int size, unsigned int flags);
bool set_sched_deadline(int _tid, __u64 _exec_time, __u64 _deadline, __u64 _period);
void request_task_scheduling(double task_minimum_inter_release_time, double task_execution_time, double task_relative_deadline);
void yield_task_scheduling();
void init_task();
void disable_task();

// GPU scheduling
void init_gpu_scheduling(std::string task_filename, std::string gpu_deadline_filename, int key_id);
void get_deadline_list();
void sig_handler(int signum);
void termination();
unsigned long get_current_time_us();


void start_job();
void finish_job();

void request_gpu();
void request_gpu_in_loop(int flag);

void yield_gpu(std::string remark = "");
void yield_gpu_in_loop(int flag, std::string remark = "");
void print_loop_info(std::string tag);
void print_gpu_deadline_list();

} // namespace sched
} // namespace rubis

#endif


