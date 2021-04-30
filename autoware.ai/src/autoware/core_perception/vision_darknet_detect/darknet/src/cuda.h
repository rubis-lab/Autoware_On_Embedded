#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"
#include <cuda_runtime.h>

#define HTOD 0
#define DTOH 1
#define LAUNCH 2
#define GPU_PROFILING 1





#ifdef __cplusplus
extern "C" {
#endif

extern int glob_id;
extern FILE* fp;

//int count_exec;
extern float htod_time;
extern float dtoh_time;
extern float launch_time;  
extern cudaEvent_t event_start, event_stop;


void start_profiling();
void stop_profiling(int type, char* ker_name, int count);
void write_data(int id, float time, int type, char* ker_name, int count);
void write_dummy_line();
//void initialize_file();
void initialize_file(const char name[]);
void close_file();

#ifdef __cplusplus
}
#endif






#ifdef GPU
void check_error(cudaError_t status);
cublasHandle_t blas_handle();
int *cuda_make_int_array(int *x, size_t n);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);
void file_write(char *filename,int id,float time,int type);

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif
