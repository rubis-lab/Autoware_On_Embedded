#include "ndt_gpu/MatrixDevice.h"
#include "ndt_gpu/debug.h"
#include "rubis_sched/sched.hpp"

namespace gpu {
MatrixDevice::MatrixDevice(int rows, int cols) {
	rows_ = rows;
	cols_ = cols;
	offset_ = 1;
	fr_ = true;
	buffer_ = NULL;
	memAllocId = 0;
	memFreeId = 0;
}

void MatrixDevice::memAlloc()
{
	if (buffer_ != NULL && fr_) {
		//rubis::sched::request_gpu(1);
		checkCudaErrors(cudaFree(buffer_));
		//rubis::sched::yield_gpu(1,"free");
		buffer_ = NULL;
	}

	//rubis::sched::request_gpu(2);
	checkCudaErrors(cudaMalloc(&buffer_, sizeof(double) * rows_ * cols_ * offset_));
	//rubis::sched::yield_gpu(2,"cudaMalloc");

	//rubis::sched::request_gpu(3);
	checkCudaErrors(cudaMemset(buffer_, 0, sizeof(double) * rows_ * cols_ * offset_));
	//rubis::sched::yield_gpu(3,"cudaMemset");

	checkCudaErrors(cudaDeviceSynchronize());
	fr_ = true;
}

void MatrixDevice::memAlloc_free()
{
	if (buffer_ != NULL && fr_) {
		//rubis::sched::request_gpu(4);
		checkCudaErrors(cudaFree(buffer_));
		//rubis::sched::yield_gpu(4,"free");
		buffer_ = NULL;
	}
}

void MatrixDevice::memAlloc_malloc()
{
	//rubis::sched::request_gpu(5);
	checkCudaErrors(cudaMalloc(&buffer_, sizeof(double) * rows_ * cols_ * offset_));
	//rubis::sched::yield_gpu(5,"cudaMalloc");
}

void MatrixDevice::memAlloc_memset()
{
	//rubis::sched::request_gpu(6);
	checkCudaErrors(cudaMemset(buffer_, 0, sizeof(double) * rows_ * cols_ * offset_));
	//rubis::sched::yield_gpu(6,"cudaMemset");

	fr_ = true;
}

void MatrixDevice::memFree()
{
	if (fr_) {
		if (buffer_ != NULL) {
			//rubis::sched::request_gpu(7);
			checkCudaErrors(cudaFree(buffer_));
			//rubis::sched::yield_gpu(7,"free");
			buffer_ = NULL;
		}
	}
}


SquareMatrixDevice::SquareMatrixDevice(int size) :
	MatrixDevice(size, size)
{

}

}
