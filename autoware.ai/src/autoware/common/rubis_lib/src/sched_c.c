#include "rubis_lib/sched_c.h"

// #define DEBUG

int key_id_;
int is_scheduled_;
int gpu_scheduling_flag_;
GPUSchedInfo* gpu_sched_info_;
int gpu_scheduler_pid_;
char* task_filename_;
char* gpu_deadline_filename_;
// unsigned long gpu_deadline_list_[1024];
unsigned long* gpu_deadline_list_;
unsigned int max_gpu_id_ = 0;
unsigned int cpu_seg_id_ = 0;
unsigned int gpu_seg_id_ = 0;

int is_task_ready_ = TASK_NOT_READY;
int task_state_ = TASK_STATE_READY;
int was_in_loop_ = 0;
int loop_cnt_ = 0;
int gpu_seg_cnt_in_loop_ = 0;

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
  gpu_sched_info_->state = SCHEDULING_STATE_NONE;
  gpu_sched_info_->scheduling_flag = 0;
  printf("Task [%d] is ready to work\n", getpid());

  gpu_seg_id_ = 0;
  cpu_seg_id_ = 0;
  
  return;
}

void sig_handler(int signum){
  if(signum == SIGINT || signum == SIGTSTP || signum == SIGQUIT){
    termination();
  }
}

void get_deadline_list(){
  char gpu_deadline_filename[rubis_lib_BUFFER_SIZE];
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

  char* buf;
  int cnt = 0;
  size_t len = 0;
  ssize_t n;

  getline(&buf, &len, fp); // skip first line
  while( (n = getline(&buf, &len, fp)) != -1 ){
    cnt++;
  }
  max_gpu_id_ = cnt;
  gpu_deadline_list_ = (unsigned long *)malloc(sizeof(unsigned long) * max_gpu_id_);
  rewind(fp);

  int idx = 0;
  
  getline(&buf, &len, fp); // skip first line
  while((n = getline(&buf, &len, fp)) != -1){
    unsigned long deadline;
    sscanf(buf, "gpu_%*d,%llu", &deadline);
    gpu_deadline_list_[idx++] = deadline;
  }

  free(buf); // free allocated memory at getline
  fclose(fp);  

  // print_gpu_deadline_list();

  return;  
}

void termination(){
  printf("TERMINATION\n");
	if(gpu_scheduling_flag_==1){
		gpu_sched_info_->state = SCHEDULING_STATE_STOP;
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

void start_job(){
  gpu_seg_id_ = 0;
  cpu_seg_id_ = 0;
  start_job_profiling();
}

void finish_job(){
  finish_job_profiling(cpu_seg_id_);
}

void request_gpu(){  
  if(is_task_ready_ != TASK_READY) return;
  if(was_in_loop_ == 1){
    was_in_loop_ = 0;
    loop_cnt_ = 0;
    gpu_seg_cnt_in_loop_ = 0;
  }

  stop_profiling_cpu_seg_response_time(cpu_seg_id_, 1);
  if(gpu_scheduling_flag_==1){
    unsigned long relative_deadline = gpu_deadline_list_[gpu_seg_id_];

    if(gpu_seg_id_ > max_gpu_id_){
      printf("[ERROR] %s - GPU segment id bigger than max segment id!\n", task_filename_);
      printf("gpu seg id: %d / max seg id: %d\n", gpu_seg_id_, max_gpu_id_);
      relative_deadline = 1000; // 1us
    }

    gpu_sched_info_->deadline = get_current_time_ns() + relative_deadline;
    gpu_sched_info_->state = SCHEDULING_STATE_WAIT;
  }
  
  start_profiling_gpu_seg_response_time();

  #ifdef DEBUG
    printf("request_gpu: %d\n", gpu_seg_id_);
  #endif

  if(gpu_scheduling_flag_ == 1){
    while(1){
      kill(gpu_scheduler_pid_, SIGUSR1);
      if(gpu_sched_info_->scheduling_flag == 1) break;
    }
  }

  start_profiling_gpu_seg_execution_time();

  if(gpu_scheduling_flag_ == 1){
    gpu_sched_info_->state = SCHEDULING_STATE_RUN;
    gpu_sched_info_->deadline = -1;
  }
}

void request_gpu_in_loop(int flag){
  if(is_task_ready_ != TASK_READY) return;
  was_in_loop_ = 1;

  if(flag == GPU_SEG_LOOP_START){
      loop_cnt_++;
    if(gpu_seg_cnt_in_loop_ == 0){
      gpu_seg_cnt_in_loop_ = 1;
    }
    else{
      gpu_seg_id_ = gpu_seg_id_ - gpu_seg_cnt_in_loop_;
      cpu_seg_id_ = cpu_seg_id_ - gpu_seg_cnt_in_loop_;
      gpu_seg_cnt_in_loop_ = 1;
    }
  }

  if(flag != GPU_SEG_LOOP_START){
    gpu_seg_cnt_in_loop_++;
  }

  stop_profiling_cpu_seg_response_time(cpu_seg_id_, loop_cnt_);
  if(gpu_scheduling_flag_==1){    
    unsigned long relative_deadline = gpu_deadline_list_[gpu_seg_id_];

    if(gpu_seg_id_ > max_gpu_id_){
      printf("[ERROR] %s - GPU segment id bigger than max segment id!\n", task_filename_);
      relative_deadline = 1000; // 1us
    }

    gpu_sched_info_->deadline = get_current_time_ns() + relative_deadline;
    gpu_sched_info_->state = SCHEDULING_STATE_WAIT;
  }
  
  start_profiling_gpu_seg_response_time();

  #ifdef DEBUG
    printf("request_gpu: %d\n", gpu_seg_id_);
  #endif

  if(gpu_scheduling_flag_ == 1){
    while(1){
      kill(gpu_scheduler_pid_, SIGUSR1);
      if(gpu_sched_info_->scheduling_flag == 1) break;
    }
  }

  start_profiling_gpu_seg_execution_time();

  if(gpu_scheduling_flag_ == 1){
    gpu_sched_info_->state = SCHEDULING_STATE_RUN;
    gpu_sched_info_->deadline = -1;
  }
}

void yield_gpu(){
  if(gpu_scheduling_flag_==1){
    gpu_sched_info_->scheduling_flag = 0;
    gpu_sched_info_->state = SCHEDULING_STATE_NONE;
  }

  #ifdef DEBUG
    printf("yield_gpu: %d\n", gpu_seg_id_);
  #endif

  stop_profiling_gpu_seg_time(gpu_seg_id_, 1);
  start_profiling_cpu_seg_response_time(1);

  gpu_seg_id_++;
  cpu_seg_id_++;
}

void yield_gpu_with_remark(const char* remark){
  if(gpu_scheduling_flag_==1){
    gpu_sched_info_->scheduling_flag = 0;
    gpu_sched_info_->state = SCHEDULING_STATE_NONE;
  }

  #ifdef DEBUG
    printf("yield_gpu: %d\n", gpu_seg_id_);
  #endif

  stop_profiling_gpu_seg_time_with_remark(gpu_seg_id_, 1, remark);
  start_profiling_cpu_seg_response_time();

  gpu_seg_id_++;
  cpu_seg_id_++;
}

void yield_gpu_in_loop(int flag){
  if(gpu_scheduling_flag_==1){
    gpu_sched_info_->scheduling_flag = 0;
    gpu_sched_info_->state = SCHEDULING_STATE_NONE;
  }

  #ifdef DEBUG
    printf("yield_gpu: %d\n", gpu_seg_id_);
  #endif

  stop_profiling_gpu_seg_time(gpu_seg_id_, loop_cnt_);
  start_profiling_cpu_seg_response_time();

  if(flag == GPU_SEG_LOOP_END){
    gpu_seg_id_ -= gpu_seg_cnt_in_loop_;
    cpu_seg_id_ -= gpu_seg_cnt_in_loop_;
  }
}

void yield_gpu_with_remark_in_loop(int flag, const char* remark){
  if(gpu_scheduling_flag_==1){
    gpu_sched_info_->scheduling_flag = 0;
    gpu_sched_info_->state = SCHEDULING_STATE_NONE;
  }

  #ifdef DEBUG
    printf("yield_gpu: %d\n", gpu_seg_id_);
  #endif

  stop_profiling_gpu_seg_time_with_remark(gpu_seg_id_, loop_cnt_, remark);
  start_profiling_cpu_seg_response_time();

  // if(flag == GPU_SEG_LOOP_END){
  //   gpu_seg_id_ = gpu_seg_id_ - gpu_seg_cnt_in_loop_ + 1;
  //   cpu_seg_id_ = cpu_seg_id_ - gpu_seg_cnt_in_loop_ + 1;
  // }
}

void init_task(){
  is_task_ready_ = TASK_READY;
}

void disable_task(){
  is_task_ready_ = TASK_NOT_READY;
}

void print_loop_info(const char* tag){
  printf("tag: %s\n",tag);
  printf("cpu_seg_id: %d\n",cpu_seg_id_);
  printf("gpu_seg_id: %d\n",gpu_seg_id_);
  printf("loop cnt: %d\n",loop_cnt_);
  printf("loop seg cnt: %d\n",gpu_seg_cnt_in_loop_);
}

void print_gpu_deadline_list(){
  printf("====================================\n[GPU deadline list]\n");
  printf("gpu_id\tdeadline\n");
  for(int i = 0; i < max_gpu_id_; i++){
    printf("%d\t%lu\n",i,gpu_deadline_list_[i]);
  }
  printf("====================================\n");
}
