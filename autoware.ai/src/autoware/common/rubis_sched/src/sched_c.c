#include "rubis_sched/sched_c.h"

int key_id_;
int is_scheduled_;
int gpu_scheduling_flag_;
GPUSchedInfo* gpu_sched_info_;
int gpu_scheduler_pid_;
char* task_filename_;
char* gpu_deadline_filename_;
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

void init_gpu_scheduling(char* task_filename, char* gpu_deadline_filename, int key_id){
  printf("init_gpu_scheduling\n");
  gpu_scheduling_flag_ = 1;

  task_filename_ = (char *)malloc(strlen(task_filename) * sizeof(char));
  gpu_deadline_filename_ = (char *)malloc(strlen(gpu_deadline_filename) * sizeof(char));

  strcpy(task_filename_, task_filename);
  strcpy(gpu_deadline_filename_, gpu_deadline_filename);
  key_id_ = key_id;

  // Get deadlines
  printf("deadline filename: %s\n", gpu_deadline_filename_);
  get_deadline_list();

  // Init signal handler
  signal(SIGINT, sig_handler);
  signal(SIGTSTP, sig_handler);
  signal(SIGQUIT, sig_handler);

  // Create task file
  FILE* task_fp;
  task_fp = fopen(task_filename_, "w");
  if(task_fp == NULL){
      printf("Cannot create task file at %s\n", task_filename_);
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
  char gpu_deadline_filename[RUBIS_SCHED_BUFFER_SIZE];
  char* user_name = getenv("USER_HOME");

  if(gpu_deadline_filename_[0] != '~'){
    strcpy(gpu_deadline_filename, gpu_deadline_filename_);
  }
  else{
    strcpy(gpu_deadline_filename, user_name);
    strcat(gpu_deadline_filename, &gpu_deadline_filename_[1]);
  }  

  FILE* fp;
  fp = fopen(gpu_deadline_filename, "r");
  if(fp==NULL){
	  fprintf(stderr, "Cannot find file %s\n", gpu_deadline_filename);
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

  return;  
}

void termination(){
  printf("TERMINATION\n");
	if(gpu_scheduling_flag_==1){
		gpu_sched_info_->state = STOP;
  	shmdt(gpu_sched_info_);
	}
  
  free(gpu_deadline_list_);
  if(remove(task_filename_)){
      printf("Cannot remove file %s\n", task_filename_);
      exit(1);
  }

  if(task_profiling_flag_){
    fclose(task_response_time_fp_);
  }

  if(gpu_profiling_flag_){
    fclose(seg_response_time_fp_);
    fclose(seg_execution_time_fp_);
    free(task_filename_);
    free(gpu_deadline_filename_);
  }

  exit(0);
}

void request_gpu(unsigned int id){  
  stop_profiling_cpu_seg_response_time();
  if(gpu_scheduling_flag_==1){
    if(id > max_gpu_id_){
      printf("[ERROR] GPU segment id bigger than max segment id!\n");
      exit(1);
    }
    
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

void yield_gpu(unsigned int id){
  if(gpu_scheduling_flag_==1){
    gpu_sched_info_->scheduling_flag = 0;
    gpu_sched_info_->state = NONE;
  }
  stop_profiling_cpu_seg_response_time();
  stop_profiling_gpu_seg_time(id);
  start_profiling_cpu_seg_response_time();
}


void yield_gpu_with_remark(unsigned int id, const char* remark){
  if(gpu_scheduling_flag_==1){
    gpu_sched_info_->scheduling_flag = 0;
    gpu_sched_info_->state = NONE;
  }
  stop_profiling_cpu_seg_response_time();
  stop_profiling_gpu_seg_time_with_remark(id, remark);
  start_profiling_cpu_seg_response_time();
}
