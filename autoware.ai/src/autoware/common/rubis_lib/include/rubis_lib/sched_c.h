#ifndef __RUBIS_LIB_H__
#define __RUBIS_LIB_H__

#include "rubis_lib/sched_profiling_c.h"

#include <linux/kernel.h>
#include <linux/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <stdlib.h>
#include <time.h>
#include <sys/ipc.h>
#include <string.h>
#include <time.h>
// #include <sched.h>
#include <signal.h>
#include <sys/shm.h>


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
    struct thr_arg* thr_set;
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
extern char* task_filename_;

// Task scheduling
// int sched_setattr(pid_t pid, const struct sched_attr *attr, unsigned int flags);
// int sched_getattr(pid_t pid, struct sched_attr *attr, unsigned int size, unsigned int flags);
int set_sched_deadline(int _tid, __u64 _exec_time, __u64 _deadline, __u64 _period);
void yield_task_scheduling();

void sig_handler(int signum);
void termination();
unsigned long get_current_time_us();

#endif