#ifndef __RUBIS_SCHED_H__
#define __RUBIS_SCHED_H__

#include <linux/kernel.h>
#include <linux/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <vector>

#define gettid() syscall(__NR_gettid)

namespace rubis {
namespace sched {

// struct sched_info {
//   int task_id;
//   int max_option;
//   std::string name;
//   std::string file;
//   double exec_time;
//   double deadline;
//   double period;
// };

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

// int sched_setattr(pid_t pid,
//     const struct sched_attr *attr,
//     unsigned int flags);

// int sched_getattr(pid_t pid,
//     struct sched_attr *attr,
//     unsigned int size,
//     unsigned int flags);

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
#define __NR_sched_setattr		380
#define __NR_sched_getattr		381
#endif

// #define BUFFER_SIZE 5000
// #define MAX_TOKEN_COUNT 128

// system call hook to call SCHED_DEADLINE
int sched_setattr(pid_t pid,
    const struct sched_attr *attr,
    unsigned int flags)
{
	return syscall(__NR_sched_setattr, pid, attr, flags);
}

int sched_getattr(pid_t pid,
    struct sched_attr *attr,
    unsigned int size,
    unsigned int flags)
{
	return syscall(__NR_sched_getattr, pid, attr, size, flags);
}

#define SCHED_DEADLINE	6
#include <iostream>

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

} // namespace sched
} // namespace rubis
#endif
