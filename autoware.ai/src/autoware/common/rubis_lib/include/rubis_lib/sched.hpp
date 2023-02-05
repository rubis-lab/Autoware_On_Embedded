#ifndef __RUBIS_LIB_H__
#define __RUBIS_LIB_H__

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

#include "rubis_lib/sched_profiling.hpp"

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

namespace rubis {

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

extern int key_id_;
extern int is_scheduled_;
extern std::string task_filename_;

int sched_setattr(pid_t pid, const struct sched_attr *attr, unsigned int flags);
int sched_getattr(pid_t pid, struct sched_attr *attr, unsigned int size, unsigned int flags);

bool set_sched_deadline(int pid, unsigned int exec_time, unsigned int deadline, unsigned int period);
bool set_sched_fifo(int pid, int priority);
bool set_sched_fifo(int pid, int priority, int child_priority);
bool set_sched_rr(int pid, int priority);
bool set_sched_rr(int pid, int priority, int child_priority);

bool init_task_scheduling(std::string policy, struct sched_attr attr);
void yield_task_scheduling();
struct sched_attr create_sched_attr(int priority, int exec_time, int deadline, int period);

void sig_handler(int signum);
void termination();
unsigned long get_current_time_us();

std::string get_cmd_output(const char* cmd);
std::vector<int> get_child_pids(int pid);
std::vector<std::string> tokenize_string(std::string s, std::string delimiter);

} // namespace rubis

#endif


