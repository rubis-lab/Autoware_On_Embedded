int gpu_index = 0;

#ifdef GPU
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>

#define TINY 297
#define YOLO 1752

int count_dtoh = 0;
int count_htod = 0;

int push_id = 0; // 1
int pull_id = 2; // 1~4
int bias_id = 7; // 1~72
int normalize_id = 80; // 1~72
int add_id = 155; // 1~75
int gpu_id = 0; 
int copy_gpu_id = 260; // 1~104
int upsample_id = 263; // 1~2
int activation_id = 380; //1~116
int im2col_id = 460; //1~75

int cpu_id = 0;

int glob_id = 0;
float htod_time;
float dtoh_time;
float launch_time;

struct timeval startTime, endTime;
cudaEvent_t e_event_start, e_event_stop, r_event_start, r_event_stop;

FILE* execution_time_fp;
FILE* response_time_fp;
FILE* remain_time_fp;

#ifdef __cplusplus
extern "C" {
#endif
/* GPU Profiling */
void start_profiling_execution_time(){
	if(GPU_PROFILING == 1)
		cudaEventRecord(e_event_start, 0);
}

void start_profiling_response_time(){
	if(GPU_PROFILING == 1)
		cudaEventRecord(r_event_start, 0);
}

void start_profiling_cpu_time(){
  if(GPU_PROFILING == 1){
    cpu_id++;
		gettimeofday(&startTime, NULL);
  }
}

void stop_profiling(int id, int type){
	if(GPU_PROFILING == 1){		
		float e_time, r_time;
    char gpu_id_buf[BUFFER_SIZE];

		cudaEventRecord(e_event_stop, 0);
    cudaEventRecord(r_event_stop, 0);
		cudaEventSynchronize(e_event_stop);
    cudaEventSynchronize(r_event_stop);
		cudaEventElapsedTime(&e_time, e_event_start, e_event_stop);
    cudaEventElapsedTime(&r_time, r_event_start, r_event_stop);
    e_time = MS2US(e_time);
    r_time = MS2US(r_time);
		// write_data(gid, time, type);
    sprintf(gpu_id_buf,"g%d",id);

    //write_profiling_data(id, e_time, r_time, type);

    write_profiling_data(gpu_id_buf, e_time, r_time, type);
		// gid++;
	}
}

void stop_cpu_profiling(){
  if(GPU_PROFILING == 1){
    long long int elapsedTime;
    char cpu_id_buf[BUFFER_SIZE];

    gettimeofday(&endTime, NULL);
    elapsedTime = ((long long int)(endTime.tv_sec - startTime.tv_sec)) * 1000000ll + (endTime.tv_usec - startTime.tv_usec);

    sprintf(cpu_id_buf,"e%d",cpu_id);
    write_cpu_profiling_data(cpu_id_buf,elapsedTime);    
  }
}

void write_cpu_profiling_data(const char *id, long long int c_time){
  if(GPU_PROFILING == 1){
		fprintf(execution_time_fp, "%s, %02d\n", id, c_time);	
    fprintf(response_time_fp, "%s, %02d\n", id, c_time);    
	}
}

// void write_profiling_data(int id, float e_time, float r_time, int type){
// 	if(GPU_PROFILING == 1){
// 		fprintf(execution_time_fp, "%d, %f, %d\n", id, e_time, type);	
//     fprintf(response_time_fp, "%d, %f, %d\n", id, r_time, type);	
//     fprintf(remain_time_fp, "%d, %llu\n", id, absolute_deadline_ - get_current_time_us());	
// 	}
// }

void write_profiling_data(const char *id, float e_time, float r_time, int type){
	if(GPU_PROFILING == 1){
		fprintf(execution_time_fp, "%s, %f, %d\n", id, e_time, type);	
    fprintf(response_time_fp, "%s, %f, %d\n", id, r_time, type);	
    fprintf(remain_time_fp, "%s, %llu\n", id, absolute_deadline_ - get_current_time_us());	
	}
}

void write_dummy_line(){
	if(GPU_PROFILING == 1){  
    fprintf(execution_time_fp, "-1, -1, -1\n");						
		fflush(execution_time_fp);
		fprintf(response_time_fp, "-1, -1, -1\n");						
		fflush(response_time_fp);
    fprintf(remain_time_fp, "-1, -1\n");						
		fflush(remain_time_fp);
    cpu_id = 0;
	}

    push_id = 0; // 1
    pull_id = 2; // 1~4
    bias_id = 7; // 1~72
    normalize_id = 80; // 1~72
    add_id = 155; // 1~75
    gpu_id = 0; 
    copy_gpu_id = 260; // 1~104
    upsample_id = 263; // 1~2`
    activation_id = 380; //1~116
    im2col_id = 460; //1~75
}

void initialize_file(const char execution_time_filename[], const char response_time_filename[], const char remain_time_filename[]){
    cudaEventCreate(&e_event_start);
	cudaEventCreate(&e_event_stop);
    cudaEventCreate(&r_event_start);
	cudaEventCreate(&r_event_stop);

	if(GPU_PROFILING == 1){
		execution_time_fp = fopen(execution_time_filename, "w+");
		fprintf(execution_time_fp, "ID, TIME, TYPE\n");
    response_time_fp = fopen(response_time_filename, "w+");
		fprintf(response_time_fp, "ID, TIME, TYPE\n");
    remain_time_fp = fopen(remain_time_filename, "w+");
		fprintf(remain_time_fp, "ID, TIME\n");
	}
}

void close_file(){
	if(GPU_PROFILING == 1){
		fclose(execution_time_fp);
    fclose(response_time_fp);
    fclose(remain_time_fp);
  }
}

#ifdef __cplusplus
}
#endif

void file_write(char *filename,int id,float time,int type){
    FILE *f = fopen(filename, "w+");
    if(f == NULL){
        printf("Cannot open file!\n");
        return;
    } 
    fprintf(f, "%d,%f,%d\n", id,time,type);
    fclose(f);
}

void cuda_set_device(int n)
{
    gpu_index = n;
    cudaError_t status = cudaSetDevice(n);
    check_error(status);
}

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}

void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    } 
    if (status2 != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    } 
}

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}
#endif

cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        count_htod += 1;

        cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);

        check_error(status);

    } else {
        fill_gpu(n, 0, x_gpu, 1);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_random(float *x_gpu, size_t n)
{
    static curandGenerator_t gen[16];
    static int init[16] = {0};
    int i = cuda_get_device();
    if(!init[i]){
        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
        init[i] = 1;
    }
    curandGenerateUniform(gen[i], x_gpu, n);
    check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
{
    float *tmp = (float *) calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err/n));
    free(tmp);
    return err;
}

int *cuda_make_int_array(int *x, size_t n)
{
    int *x_gpu;
    size_t size = sizeof(int)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_free(float *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    count_htod += 1;
    if(count_htod > YOLO){
        push_id += 1;
        // set_absolute_deadline();       
        stop_cpu_profiling();        
        request_scheduling(push_id);        
    }
        

    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    if(count_htod > YOLO){
      stop_profiling(push_id, HTOD);      
      start_profiling_cpu_time();
    }
      

    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;

    pull_id += 1;
    stop_cpu_profiling();

    request_scheduling(pull_id);
    
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    
    stop_profiling(pull_id, DTOH);

    start_profiling_cpu_time();
    
    check_error(status);
}

float cuda_mag_array(float *x_gpu, size_t n)
{
    float *temp = (float *) calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    free(temp);
    return m;
}

void sig_handler(int signum){
  if(signum == SIGUSR1 || signum == SIGUSR2)
      return;
  else
      termination();    
}

void termination(){
  sched_info_->state = STOP;
  shmdt(sched_info_);
  if(remove(task_filename_)){
      printf("Cannot remove file %s\n", task_filename_);
      exit(1);
  }
  exit(0);
}

unsigned long long get_current_time_us(){
  struct timespec ts;
  unsigned long long current_time;
  clock_gettime(CLOCK_REALTIME, &ts);
  current_time = ts.tv_sec%10000 * 1000000 + ts.tv_nsec/1000;
  return current_time;
}

void us_sleep(unsigned long long us){
  struct timespec ts;
  ts.tv_sec = us/1000000;
  ts.tv_nsec = us%1000000*1000;
  nanosleep(&ts, NULL);
  return;
}

void initialize_signal_handler(){
  signal(SIGINT, sig_handler);
  signal(SIGTSTP, sig_handler);
  signal(SIGQUIT, sig_handler);
  signal(SIGUSR1, sig_handler);
  signal(SIGUSR2, sig_handler);
}

void create_task_file(){        
  FILE* task_fp;
  task_fp = fopen(task_filename_, "w");
  if(task_fp == NULL){
      printf("Cannot create task file at %s\n", task_filename_);
      exit(1);
  }
  fprintf(task_fp, "%d\n", getpid());
  fprintf(task_fp, "%d", key_id_);
  fclose(task_fp);
}

void get_scheduler_pid(){
  FILE* scheduler_fp;
  printf("Wait the scheduler...\n");
  while(1){
      scheduler_fp = fopen("/tmp/np_edf_scheduler", "r");
      if(scheduler_fp) break;
  }
  while(1){
      fscanf(scheduler_fp, "%d", &scheduler_pid_);
      if(scheduler_pid_ != 0) break;
  }
  printf("Scheduler pid: %d\n", scheduler_pid_);
  fclose(scheduler_fp);
}

void initialize_sched_info(){
  FILE* sm_key_fp;
  sm_key_fp = fopen("/tmp/sm_key", "r");
  if(sm_key_fp == NULL){
      printf("Cannot open /tmp/sm_key\n");
      termination();
  }

  key_ = ftok("/tmp/sm_key", key_id_);
  shmid_ = shmget(key_, sizeof(SchedInfo), 0666|IPC_CREAT);
  sched_info_ = (SchedInfo*)shmat(shmid_, 0, 0);
  sched_info_->pid = getpid();
  sched_info_->state = NONE;
}

void init_scheduling(char* task_filename, const char deadline_filename[], int key_id){
  gpu_scheduling_flag_ = 1;
  // Get deadline list
  get_deadline_list(deadline_filename);

  // Initialize key id for shared memory
  key_id_ = key_id;

  // Initialize signal handler
  initialize_signal_handler();

  // Create task file

  sprintf(task_filename_, "%s", task_filename);
  create_task_file();    

  // Get scheduler pid
  get_scheduler_pid();

  // Initialize scheduling information (shared memory data)
  initialize_sched_info();

  sigemptyset(&sigset_);
  sigaddset(&sigset_, SIGUSR1);
  sigaddset(&sigset_, SIGUSR2);
  sigprocmask(SIG_BLOCK, &sigset_, NULL);    

  // sigwait(&sigset_, &sig_);    
  // kill(scheduler_pid_, SIGUSR2);
  // sigprocmask(SIG_UNBLOCK, &sigset_, NULL);
  
  printf("Task [%d] is ready to work\n", getpid());
  // sigaddset(&sigset_, SIGUSR1);
  // sigprocmask(SIG_BLOCK, &sigset_, NULL);    

}

void request_scheduling(int id){  
  unsigned long long relative_deadline = deadline_list_[id];
  if(gpu_scheduling_flag_ == 0) return;
  if(identical_deadline_ != 0) sched_info_->deadline = absolute_deadline_;  
  else sched_info_->deadline = get_current_time_us() + relative_deadline;  

  sched_info_->state = WAIT;        
  // printf("Request schedule - deadline: %llu\n", sched_info_->deadline);

  start_profiling_response_time();
  while(1){
      kill(scheduler_pid_, SIGUSR1);
      if(!sigwait(&sigset_, &sig_)) break;
  }
  start_profiling_execution_time();
}

void get_deadline_list(char* filename){
  FILE* fp;
  fp = fopen(filename, "r");
  if(fp==NULL){
	  fprintf(stderr, "Cannot find file %s\n", filename);
	  exit(1);
  }
  char buf[1024];
  long long int deadline;
  for(int i = 0; i < sizeof(deadline_list_)/sizeof(long long int); i++){
    fgets(buf, 1024, fp);
    strtok(buf, "\n");
    sscanf(buf, "%*llu, %llu", &deadline);
    deadline_list_[i] = deadline;
  }
}

void set_identical_deadline(unsigned long long identical_deadline){
  identical_deadline_ = identical_deadline;
}

void set_absolute_deadline(){  
  absolute_deadline_ = get_current_time_us() + identical_deadline_;
}

#else
void cuda_set_device(int n){}

#endif
