#include "ndt_gpu/SymmetricEigenSolver.h"
#include "ndt_gpu/debug.h"
#include "rubis_sched/sched.hpp"

namespace gpu {

SymmetricEigensolver3x3::SymmetricEigensolver3x3(int offset)
{
	offset_ = offset;

	//rubis::sched::request_gpu(176);
	checkCudaErrors(cudaMalloc(&buffer_, sizeof(double) * 18 * offset_));
	//rubis::sched::yield_gpu(176,"cudaMalloc");

	//rubis::sched::request_gpu(177);
	checkCudaErrors(cudaMalloc(&maxAbsElement_, sizeof(double) * offset_));
	//rubis::sched::yield_gpu(177,"cudaMalloc");

	//rubis::sched::request_gpu(178);
	checkCudaErrors(cudaMalloc(&norm_, sizeof(double) * offset_));
	//rubis::sched::yield_gpu(178,"cudaMalloc");

	//rubis::sched::request_gpu(179);
	checkCudaErrors(cudaMalloc(&i02_, sizeof(int) * 2 * offset_));
	//rubis::sched::yield_gpu(179,"cudaMalloc");

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
			//rubis::sched::request_gpu(180);
			checkCudaErrors(cudaFree(buffer_));
			//rubis::sched::yield_gpu(180,"free");
			
			buffer_ = NULL;
		}

		if (maxAbsElement_ != NULL) {
			//rubis::sched::request_gpu(181);
			checkCudaErrors(cudaFree(maxAbsElement_));
			//rubis::sched::yield_gpu(181,"free");

			maxAbsElement_ = NULL;
		}

		if (norm_ != NULL) {
			//rubis::sched::request_gpu(182);
			checkCudaErrors(cudaFree(norm_));
			//rubis::sched::yield_gpu(182,"free");

			norm_ = NULL;
		}

		if (i02_ != NULL) {
			//rubis::sched::request_gpu(183);
			checkCudaErrors(cudaFree(i02_));
			//rubis::sched::yield_gpu(183,"free");

			i02_ = NULL;
		}
	}
}
}
