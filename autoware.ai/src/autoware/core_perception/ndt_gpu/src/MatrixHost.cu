#include "ndt_gpu/MatrixHost.h"
#include "ndt_gpu/debug.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "rubis_sched/sched.hpp"

namespace gpu {

MatrixHost::MatrixHost()
{
	fr_ = false;
}

MatrixHost::MatrixHost(int rows, int cols) {
	rows_ = rows;
	cols_ = cols;
	offset_ = 1;

	buffer_ = (double*)malloc(sizeof(double) * rows_ * cols_ * offset_);
	memset(buffer_, 0, sizeof(double) * rows_ * cols_ * offset_);
	fr_ = true;
}

MatrixHost::MatrixHost(int rows, int cols, int offset, double *buffer)
{
	rows_ = rows;
	cols_ = cols;
	offset_ = offset;
	buffer_ = buffer;
	fr_ = false;
}

MatrixHost::MatrixHost(const MatrixHost& other) {
	rows_ = other.rows_;
	cols_ = other.cols_;
	offset_ = other.offset_;
	fr_ = other.fr_;

	if (fr_) {
		buffer_ = (double*)malloc(sizeof(double) * rows_ * cols_ * offset_);
		memcpy(buffer_, other.buffer_, sizeof(double) * rows_ * cols_ * offset_);
	} else {
		buffer_ = other.buffer_;
	}
}

extern "C" __global__ void copyMatrixDevToDev(MatrixDevice input, MatrixDevice output) {
	int row = threadIdx.x;
	int col = threadIdx.y;
	int rows_num = input.rows();
	int cols_num = input.cols();

	if (row < rows_num && col < cols_num)
		output(row, col) = input(row, col);
}

bool MatrixHost::moveToGpu(MatrixDevice output) {
	if (rows_ != output.rows() || cols_ != output.cols())
		return false;

	if (offset_ == output.offset()) {
		//rubis::sched::request_gpu(8);
		checkCudaErrors(cudaMemcpy(output.buffer(), buffer_, sizeof(double) * rows_ * cols_ * offset_, cudaMemcpyHostToDevice));
		//rubis::sched::yield_gpu(8,"htod");
		return true;
	}
	else {
		double *tmp;

		//rubis::sched::request_gpu(9);
		checkCudaErrors(cudaMalloc(&tmp, sizeof(double) * rows_ * cols_ * offset_));
		//rubis::sched::request_gpu(9,"cudaMalloc");

		//rubis::sched::request_gpu(10);
		checkCudaErrors(cudaMemcpy(tmp, buffer_, sizeof(double) * rows_ * cols_ * offset_, cudaMemcpyHostToDevice));
		//rubis::sched::request_gpu(10,"htod");

		MatrixDevice tmp_output(rows_, cols_, offset_, tmp);

		dim3 block_x(rows_, cols_, 1);
		dim3 grid_x(1, 1, 1);

		//rubis::sched::request_gpu(11);
		copyMatrixDevToDev<<<grid_x, block_x>>>(tmp_output, output);
		//rubis::sched::yield_gpu(11,"copyMatrixDevToDev");

		checkCudaErrors(cudaDeviceSynchronize());

		//rubis::sched::request_gpu(12);
		checkCudaErrors(cudaFree(tmp));
		//rubis::sched::yield_gpu(12,"free");

		return true;
	}
}

bool MatrixHost::moveToHost(MatrixDevice input) {
	if (rows_ != input.rows() || cols_ != input.cols())
		return false;

	if (offset_ == input.offset()) {
		//rubis::sched::request_gpu(13);
		checkCudaErrors(cudaMemcpy(buffer_, input.buffer(), sizeof(double) * rows_ * cols_ * offset_, cudaMemcpyDeviceToHost));
		//rubis::sched::yield_gpu(13,"dtoh");
		return true;
	}
	else {
		double *tmp;

		//rubis::sched::request_gpu(14);
		checkCudaErrors(cudaMalloc(&tmp, sizeof(double) * rows_ * cols_ * offset_));
		//rubis::sched::yield_gpu(14,"cudaMalloc");

		MatrixDevice tmp_output(rows_, cols_, offset_, tmp);

		dim3 block_x(rows_, cols_, 1);
		dim3 grid_x(1, 1, 1);

		//rubis::sched::request_gpu(15);
		copyMatrixDevToDev << <grid_x, block_x >> >(input, tmp_output);
		//rubis::sched::yield_gpu(15,"copyMatrixDevToDev");

		checkCudaErrors(cudaDeviceSynchronize());

		//rubis::sched::request_gpu(16);
		checkCudaErrors(cudaMemcpy(buffer_, tmp, sizeof(double) * rows_ * cols_ * offset_, cudaMemcpyDeviceToHost));
		//rubis::sched::yield_gpu(16,"dtoh");

		//rubis::sched::request_gpu(17);
		checkCudaErrors(cudaFree(tmp));
		//rubis::sched::yield_gpu(17,"free");

		return true;
	}
}

MatrixHost &MatrixHost::operator=(const MatrixHost &other)
{
	rows_ = other.rows_;
	cols_ = other.cols_;
	offset_ = other.offset_;
	fr_ = other.fr_;

	if (fr_) {
		buffer_ = (double*)malloc(sizeof(double) * rows_ * cols_ * offset_);
		memcpy(buffer_, other.buffer_, sizeof(double) * rows_ * cols_ * offset_);
	} else {
		buffer_ = other.buffer_;
	}

	return *this;
}

void MatrixHost::debug()
{
	for (int i = 0; i < rows_; i++) {
		for (int j = 0; j < cols_; j++) {
			std::cout << buffer_[(i * cols_ + j) * offset_] << " ";
		}

		std::cout << std::endl;
	}

	std::cout << std::endl;
}

MatrixHost::~MatrixHost()
{
	if (fr_)
		free(buffer_);
}


SquareMatrixHost::SquareMatrixHost(int size) :
	 MatrixHost(size, size)
{

}

}
