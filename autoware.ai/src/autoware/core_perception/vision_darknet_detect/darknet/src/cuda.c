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


//int glob_id = 0;
float htod_time;
float dtoh_time;
float launch_time;  
cudaEvent_t event_start, event_stop;

FILE* fp;

#ifdef __cplusplus
extern "C" {
#endif
void start_profiling(){	
    cudaEventRecord(event_start, 0);
}

void stop_profiling(int type,char* ker_name,int id){		
    float time;
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&time, event_start, event_stop);
    write_data(id, time, type,ker_name);
    //glob_id++;	
}

void write_data(int id, float time, int type,char* ker_name){
    // fprintf(fp, "%d, %f, %d, %s\n", id, time, type, ker_name);	
    fprintf(fp, "%d, %f, %d\n", id, time, type);	
}

void write_dummy_line(){
    
    fprintf(fp, "-1, -1, -1\n");						
    fflush(fp);

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
    //glob_id = 0;
}

void initialize_file(const char name[]){
    cudaEventCreate(&event_stop);
    cudaEventCreate(&event_start);

    fp = fopen(name, "w+");
    fprintf(fp, "ID, TIME, TYPE\n");
}

void close_file(){
    fclose(fp);
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
        request_scheduling(deadline_list_[push_id]);
        start_profiling();
    }
        

    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    if(count_htod > YOLO)
        stop_profiling(HTOD,"push_arr",push_id);

    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;

    pull_id += 1;
    request_scheduling(deadline_list_[pull_id]);

    start_profiling();

    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);

    stop_profiling(DTOH,"pull_arr",pull_id);

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
  
void init_scheduling(char* task_filename, char* deadline_filename, int key_id){
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

void request_scheduling(unsigned long long relative_deadline){
  if(gpu_scheduling_flag_ == 0) return;
  sched_info_->deadline = get_current_time_us() + relative_deadline;  
  sched_info_->state = WAIT;        
  // printf("Request schedule - deadline: %llu\n", sched_info_->deadline);
  while(1){
      kill(scheduler_pid_, SIGUSR1);
      if(!sigwait(&sigset_, &sig_)) break;
  }
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

#else
void cuda_set_device(int n){}

#endif