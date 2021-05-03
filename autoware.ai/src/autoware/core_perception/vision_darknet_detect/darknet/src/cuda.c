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

// cudaEvent_t start, stop;
// float htod_time = 0;
// float dtoh_time = 0;
int count_dtoh = 0;
int count_htod = 0;
// int count_flag = 0;
// int count_exec = 0;

int glob_id = 0;
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

void stop_profiling(int type,char* ker_name,int count){		
    float time;
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&time, event_start, event_stop);
    write_data(glob_id, time, type,ker_name,count);
    glob_id++;	
}

void write_data(int id, float time, int type,char* ker_name,int count){
    fprintf(fp, "%d, %f, %d, %s, %d\n", id, time, type, ker_name, count);	
}

void write_dummy_line(){
    fprintf(fp, "-1, -1, -1\n");						
    fflush(fp);
    glob_id = 0;
}

// void initialize_file(){
//     cudaEventCreate(&event_stop);
//     cudaEventCreate(&event_start);

//     fp = fopen("/home/bkpark/prof_data/yolo_prof.csv", "w+");
//     fprintf(fp, "ID, TIME, TYPE\n");
// }
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
        
        //start_profiling();

        cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);

        //stop_profiling(HTOD,"make_arr",count_htod);

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
    if(count_htod > 297)
        start_profiling();

    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    if(count_htod > 297)
        stop_profiling(HTOD,"push_arr",count_htod);

    //htod_time += gpu_elapsed_time_ms1;
    /*
    if(count_htod >= 516){        
        htod_time = 0;
        count_htod = 0;
        count_flag = 1;
    }
    pid_t tid = syscall(__NR_gettid);


    if(count_flag && htod_time){
        char filename[100];
        sprintf(filename, "~/prof_data/yolo_mem_%d.csv", tid);
        
        FILE *f = fopen(filename, "a+");
        if(f == NULL){
            printf("Cannot open file!\n");
            return;
        } 
        
        fprintf(f, "htod time,%f\n", htod_time/1000);
        fclose(f);
        htod_time = 0;
    }*/
    // pid_t tid = syscall(__NR_gettid);
    // char filename[100];
    // sprintf(filename, "~/prof_data/yolo_mem_%d.csv", tid);
    // file_write(filename, count_dtoh, gpu_elapsed_time_ms1/1000,1);

    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;

    count_dtoh += 1;

    start_profiling();

    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);

    stop_profiling(DTOH,"pull_arr",count_dtoh);

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


#else
void cuda_set_device(int n){}

#endif