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

#define STOP -1
#define NONE 0
#define WAIT 1
#define RUN 2

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

// GPU scheduling
typedef struct  gpuSchedInfo{
    int pid;
    unsigned long long deadline;
    int state; // NONE = 0, WAIT = 1, RUN = 2
    int scheduling_flag;
} GPUSchedInfo;

static int key_id_;
static int is_scheduled_;
static int gpu_scheduling_flag_;
static GPUSchedInfo* gpu_sched_info_;
static int gpu_scheduler_pid_;
static std::string task_filename_;
static std::string gpu_deadline_filename_;
static unsigned long long* gpu_deadline_list_;

// Task scheduling
int sched_setattr(pid_t pid, const struct sched_attr *attr, unsigned int flags);
int sched_getattr(pid_t pid, struct sched_attr *attr, unsigned int size, unsigned int flags);
bool set_sched_deadline(int _tid, __u64 _exec_time, __u64 _deadline, __u64 _period);
void request_task_scheduling(double task_minimum_inter_release_time, double task_execution_time, double task_relative_deadline);
void yield_task_scheduling();

// GPU scheduling
void init_gpu_scheduling(std::string task_filename, std::string gpu_deadline_filename, int key_id);
void get_deadline_list();
void sig_handler(int signum);
void termination();
unsigned long long get_current_time_us();
void request_gpu(unsigned int id);
void yield_gpu(unsigned int id, std::string remark = " ");

} // namespace sched
} // namespace rubis

#endif
