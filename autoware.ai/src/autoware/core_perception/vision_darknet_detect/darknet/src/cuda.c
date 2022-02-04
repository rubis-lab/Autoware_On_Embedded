int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>

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
    
    request_gpu();
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    yield_gpu_with_remark("cudaMalloc");

    check_error(status);
    if(x){
        request_gpu();
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        yield_gpu_with_remark("cuda_make_array");
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

    request_gpu();
    int i = cuda_get_device();
    yield_gpu_with_remark("cuda_get_device");

    if(!init[i]){
        request_gpu();
        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        yield_gpu_with_remark("curandCreateGenerator");

        request_gpu();
        curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
        yield_gpu_with_remark("curandSetPseudoRandomGeneratorSeed");

        init[i] = 1;
    }

    request_gpu();
    curandGenerateUniform(gen[i], x_gpu, n);
    yield_gpu_with_remark("curandGenerateUniform");

    check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
{
    float *tmp = calloc(n, sizeof(float));
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

    request_gpu();
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    yield_gpu_with_remark("cudaMalloc");

    check_error(status);
    if(x){
        request_gpu();
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        yield_gpu_with_remark("cuda_make_int_array");

        check_error(status);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_free(float *x_gpu)
{
    request_gpu();
    cudaError_t status = cudaFree(x_gpu);
    yield_gpu_with_remark("free");

    check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    request_gpu();
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    yield_gpu_with_remark("cuda_push_array");

    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;

    request_gpu();
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    yield_gpu_with_remark("cuda_pull_array");

    check_error(status);
}

float cuda_mag_array(float *x_gpu, size_t n)
{
    float *temp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    free(temp);
    return m;
}
#else
void cuda_set_device(int n){}

#endif
