#include "ndt_gpu/SymmetricEigenSolver.h"
#include "ndt_gpu/debug.h"
#include "rubis_lib/sched.hpp"

namespace gpu {

SymmetricEigensolver3x3::SymmetricEigensolver3x3(int offset)
{
	offset_ = offset;

	rubis::sched::request_gpu();
	checkCudaErrors(cudaMalloc(&buffer_, sizeof(double) * 18 * offset_));
	rubis::sched::yield_gpu("177_cudaMalloc");

	rubis::sched::request_gpu();
	checkCudaErrors(cudaMalloc(&maxAbsElement_, sizeof(double) * offset_));
	rubis::sched::yield_gpu("178_cudaMalloc");

	rubis::sched::request_gpu();
	checkCudaErrors(cudaMalloc(&norm_, sizeof(double) * offset_));
	rubis::sched::yield_gpu("179_cudaMalloc");

	rubis::sched::request_gpu();
	checkCudaErrors(cudaMalloc(&i02_, sizeof(int) * 2 * offset_));
	rubis::sched::yield_gpu("180_cudaMalloc");

	eigenvectors_ = NULL;
	eigenvalues_ = NULL;
	input_matrices_ = NULL;

	is_copied_ = false;
}

void SymmetricEigensolver3x3::setInputMatrices(double *input_matrices)
{
	input_matrices_ = input_matrices;
}

void SymmetricEigensolver3x3::setEigenvectors(double *eigenvectors)
{
	eigenvectors_ = eigenvectors;
}

void SymmetricEigensolver3x3::setEigenvalues(double *eigenvalues)
{
	eigenvalues_ = eigenvalues;
}

double* SymmetricEigensolver3x3::getBuffer() const
{
	return buffer_;
}

void SymmetricEigensolver3x3::memFree()
{
	if (!is_copied_) {
		if (buffer_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(buffer_));
			rubis::sched::yield_gpu("181_free");
			
			buffer_ = NULL;
		}

		if (maxAbsElement_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(maxAbsElement_));
			rubis::sched::yield_gpu("182_free");

			maxAbsElement_ = NULL;
		}

		if (norm_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(norm_));
			rubis::sched::yield_gpu("183_free");

			norm_ = NULL;
		}

		if (i02_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(i02_));
			rubis::sched::yield_gpu("184_free");

			i02_ = NULL;
		}
	}
}
}
