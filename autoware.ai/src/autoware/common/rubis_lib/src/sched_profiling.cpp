#include "rubis_lib/sched_profiling.hpp"

namespace rubis {
unsigned long instance_;
unsigned long lidar_instance_;
unsigned long vision_instance_;

int iter_;

FILE *task_response_time_fp_;

struct timespec task_start_time_;
struct timespec task_end_time_;
struct timespec topic_pub_time_;

void init_task_profiling(std::string task_reponse_time_filename) {
    if (task_reponse_time_filename.at(0) == '~') {
        task_reponse_time_filename.erase(0, 1);
        std::string user_home_str(std::getenv("USER_HOME"));
        task_reponse_time_filename = user_home_str + task_reponse_time_filename;
    }

    task_response_time_fp_ = fopen(task_reponse_time_filename.c_str(), "w+");
    if (task_response_time_fp_ == NULL) {
        std::cout << "Cannot create/open file: " << task_reponse_time_filename
                  << std::endl;
        perror("Failed: ");
        exit(0);
    }

    chmod(task_reponse_time_filename.c_str(), strtol("0777", 0, 8));
    fprintf(task_response_time_fp_,
            "iter,PID,start,end,instance,lidar_instance,topic_pub_time,vision_"
            "instasnce\n");
}

void start_task_profiling() {
    clock_gettime(CLOCK_REALTIME, &task_start_time_);
}

void start_task_profiling_at_initial_node(long long tp_time_sec,
                                          long long tp_time_nsec) {
    clock_gettime(CLOCK_REALTIME, &task_start_time_);
    topic_pub_time_.tv_sec = tp_time_sec;
    topic_pub_time_.tv_nsec = tp_time_nsec;
}

void stop_task_profiling(unsigned long instance, unsigned long lidar_instance_,
                         unsigned long vision_instance_) {
    clock_gettime(CLOCK_REALTIME, &task_end_time_);
    fprintf(task_response_time_fp_,
            "%d,%d,%lld.%.9ld,%lld.%.9ld,%lu,%lu,%lld.%.9ld,%lu\n", iter_++,
            getpid(), (long long)task_start_time_.tv_sec,
            task_start_time_.tv_nsec, (long long)task_end_time_.tv_sec,
            task_end_time_.tv_nsec, instance, lidar_instance_,
            (long long)topic_pub_time_.tv_sec, topic_pub_time_.tv_nsec,
            vision_instance_);
    fflush(task_response_time_fp_);
}

unsigned long get_current_time_ns() {
    struct timespec ts;
    unsigned long current_time;
    clock_gettime(CLOCK_REALTIME, &ts);
    current_time = ts.tv_sec % 10000 * 1000000000 + ts.tv_nsec;
    return current_time;
}

} // namespace rubis