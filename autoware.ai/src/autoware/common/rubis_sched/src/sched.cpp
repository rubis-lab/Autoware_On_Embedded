#include "rubis_sched/sched.hpp"

namespace rubis{
namespace sched{

int key_id_;
int is_scheduled_;
int gpu_scheduling_flag_;
GPUSchedInfo* gpu_sched_info_;
int gpu_scheduler_pid_;
std::string task_filename_;
std::string gpu_deadline_filename_;
// unsigned long long gpu_deadline_list_[1024];
unsigned long long* gpu_deadline_list_;
int max_gpu_id_ = 0;

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

void init_gpu_scheduling(std::string task_filename, std::string gpu_deadline_filename, int key_id){
  printf("init_gpu_scheduling\n");
  gpu_scheduling_flag_ = 1;
  task_filename_ = task_filename;
  gpu_deadline_filename_ = gpu_deadline_filename;
  key_id_ = key_id;

  // Get deadlines
  printf("deadline filename: %s\n", gpu_deadline_filename_.c_str());
  get_deadline_list();

  // Init signal handler
  signal(SIGINT, sig_handler);
  signal(SIGTSTP, sig_handler);
  signal(SIGQUIT, sig_handler);

  // Create task file
  FILE* task_fp;
  task_fp = fopen(task_filename_.c_str(), "w");
  if(task_fp == NULL){
      printf("Cannot create task file at %s\n", task_filename_.c_str());
      exit(1);
  }
  fprintf(task_fp, "%d\n", getpid());
  fprintf(task_fp, "%d", key_id_);
  fclose(task_fp);

  // Get pid of scheduler
  FILE* scheduler_fp;
  printf("Wait the scheduler...\n");

  while(1){
      scheduler_fp = fopen("/tmp/np_edf_scheduler", "r");
      if(scheduler_fp) break;
  }
  while(1){
      fscanf(scheduler_fp, "%d", &gpu_scheduler_pid_);
      if(gpu_scheduler_pid_ != 0) break;
  }
  printf("Scheduler pid: %d\n", gpu_scheduler_pid_);
  fclose(scheduler_fp);


  // Initialize scheduling information (shared memory data)
  FILE* sm_key_fp;
  sm_key_fp = fopen("/tmp/sm_key", "r");
  if(sm_key_fp == NULL){
      printf("Cannot open /tmp/sm_key\n");
      termination();
  }

  key_t key;
  int shmid;
  key = ftok("/tmp/sm_key", key_id_);
  shmid = shmget(key, sizeof(GPUSchedInfo), 0666|IPC_CREAT);
  gpu_sched_info_ = (GPUSchedInfo*)shmat(shmid, 0, 0);
  gpu_sched_info_->pid = getpid();
  gpu_sched_info_->state = NONE;
  gpu_sched_info_->scheduling_flag = 0;
  printf("Task [%d] is ready to work\n", getpid());

  
  return;
}

void sig_handler(int signum){
  if(signum == SIGINT || signum == SIGTSTP || signum == SIGQUIT){
    termination();
  }
}

void get_deadline_list(){
  FILE* fp;
  fp = fopen(gpu_deadline_filename_.c_str(), "r");
  if(fp==NULL){
	  fprintf(stderr, "Cannot find file %s\n", gpu_deadline_filename_.c_str());
	  exit(1);
  }
  char buf[1024];
  
  while(1){
    int id;
    if(!fgets(buf, 1024, fp)) break;
    strtok(buf, "\n");
    sscanf(buf, "%d, %*llu", &id);
    if(id > max_gpu_id_) max_gpu_id_ = id;
  }

  gpu_deadline_list_ = (unsigned long long *)malloc(sizeof(unsigned long long) * (max_gpu_id_+1));
  printf("file read is finished\n");

  rewind(fp);
  while(1){    
    int id;
    long long int deadline;
    if(!fgets(buf, 1024, fp)) break;
    sscanf(buf, "%d, %llu", &id, &deadline);
    gpu_deadline_list_[id] = deadline;
  }
  fclose(fp);

  printf("in function\n");
  for(int i = 0; i <= max_gpu_id_; i++){
    printf("%d\n", gpu_deadline_list_[i]);
  }

  return;  
}

void termination(){
  printf("TERMINATION\n");
	if(gpu_scheduling_flag_==1){
		gpu_sched_info_->state = STOP;
  	shmdt(gpu_sched_info_);
	}
  
  free(gpu_deadline_list_);
  if(remove(task_filename_.c_str())){
      printf("Cannot remove file %s\n", task_filename_);
      exit(1);
  }

  if(task_profiling_flag_){
    fclose(task_response_time_fp_);
  }

  if(gpu_profiling_flag_){
    fclose(seg_response_time_fp_);
    fclose(seg_execution_time_fp_);
  }

  exit(0);
}

void request_gpu(unsigned int id){
  if(id > max_gpu_id_){
    printf("[ERROR] GPU segment id bigger than max segment id!\n");
    exit(1);
  }
  stop_profiling_cpu_seg_response_time();
  if(gpu_scheduling_flag_==1){
    unsigned long long relative_deadline = gpu_deadline_list_[id];
    gpu_sched_info_->deadline = get_current_time_ns() + relative_deadline;
    gpu_sched_info_->state = WAIT;
  }
  
  start_profiling_gpu_seg_response_time();

  if(gpu_scheduling_flag_ == 1){
    while(1){
      kill(gpu_scheduler_pid_, SIGUSR1);
      if(gpu_sched_info_->scheduling_flag == 1) break;
    }
  }

  start_profiling_gpu_seg_execution_time();

  if(gpu_scheduling_flag_ == 1){
    gpu_sched_info_->state = RUN;
    gpu_sched_info_->deadline = -1;
  }
}

void yield_gpu(unsigned int id, std::string remark){
  if(gpu_scheduling_flag_==1){
    gpu_sched_info_->scheduling_flag = 0;
    gpu_sched_info_->state = NONE;
  }
  stop_profiling_cpu_seg_response_time();
  stop_profiling_gpu_seg_time(id, remark);
  start_profiling_cpu_seg_response_time();
}

} // namespace sched
} // namespace rubis