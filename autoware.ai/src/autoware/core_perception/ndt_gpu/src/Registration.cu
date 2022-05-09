#include "ndt_gpu/Registration.h"
#include "ndt_gpu/debug.h"
#include <iostream>
#include <sys/stat.h>
#include "rubis_lib/sched.hpp"

//using namespace rubis::sched;

namespace gpu {
GRegistration::GRegistration()
{
	max_iterations_ = 0;
	x_ = y_ = z_ = NULL;
	points_number_ = 0;

	trans_x_ = trans_y_ = trans_z_ = NULL;

	converged_ = false;
	nr_iterations_ = 0;

	transformation_epsilon_ = 0;
	target_cloud_updated_ = true;
	target_points_number_ = 0;

	target_x_ = target_y_ = target_z_ = NULL;
	is_copied_ = false;	
}

GRegistration::GRegistration(const GRegistration &other)
{
	transformation_epsilon_ = other.transformation_epsilon_;
	max_iterations_ = other.max_iterations_;

	//Original scanned point clouds
	x_ = other.x_;
	y_ = other.y_;
	z_ = other.z_;

	points_number_ = other.points_number_;

	trans_x_ = other.trans_x_;
	trans_y_ = other.trans_y_;
	trans_z_ = other.trans_z_;

	converged_ = other.converged_;

	nr_iterations_ = other.nr_iterations_;
	final_transformation_ = other.final_transformation_;
	transformation_ = other.transformation_;
	previous_transformation_ = other.previous_transformation_;

	target_cloud_updated_ = other.target_cloud_updated_;

	target_x_ = other.target_x_;
	target_y_ = other.target_y_;
	target_z_ = other.target_z_;

	target_points_number_ = other.target_points_number_;
	is_copied_ = true;
}

GRegistration::~GRegistration()
{
	if (!is_copied_) {
		if (x_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(x_));
			rubis::sched::yield_gpu("102_free");
			x_ = NULL;
		}

		if (y_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(y_));
			rubis::sched::yield_gpu("103_free");

			y_ = NULL;
		}

		if (z_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(z_));
			rubis::sched::yield_gpu("104_free");
	
			z_ = NULL;
		}

		if (trans_x_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(trans_x_));
			rubis::sched::yield_gpu("105_free");

			trans_x_ = NULL;
		}

		if (trans_y_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(trans_y_));
			rubis::sched::yield_gpu("106_free");

			trans_y_ = NULL;
		}

		if (trans_z_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(trans_z_));
			rubis::sched::yield_gpu("107_free");
			
			trans_z_ = NULL;
		}

		if (target_x_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(target_x_));
			rubis::sched::yield_gpu("108_free");
			
			target_x_ = NULL;
		}

		if (target_y_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(target_y_));
			rubis::sched::yield_gpu("109_free");

			target_y_ = NULL;
		}

		if (target_z_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(target_z_));
			rubis::sched::yield_gpu("110_free");

			target_z_ = NULL;
		}
	}
}

void GRegistration::setTransformationEpsilon(double trans_eps)
{
	transformation_epsilon_ = trans_eps;
}

double GRegistration::getTransformationEpsilon() const
{
	return transformation_epsilon_;
}

void GRegistration::setMaximumIterations(int max_itr)
{
	max_iterations_ = max_itr;
}

int GRegistration::getMaximumIterations() const
{
	return max_iterations_;
}

Eigen::Matrix<float, 4, 4> GRegistration::getFinalTransformation() const
{
	return final_transformation_;
}

int GRegistration::getFinalNumIteration() const
{
	return nr_iterations_;
}

bool GRegistration::hasConverged() const
{
	return converged_;
}


template <typename T>
__global__ void convertInput(T *input, float *out_x, float *out_y, float *out_z, int point_num)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < point_num; i += stride) {
		T tmp = input[i];
		out_x[i] = tmp.x;
		out_y[i] = tmp.y;
		out_z[i] = tmp.z;
	}
}

void GRegistration::setInputSource(pcl::PointCloud<pcl::PointXYZI>::Ptr input)
{
	//Convert point cloud to float x, y, z
	if (input->size() > 0) {
		points_number_ = input->size();

		pcl::PointXYZI *tmp;

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZI) * points_number_));
		rubis::sched::yield_gpu("111_cudaMalloc");
		
		pcl::PointXYZI *host_tmp = input->points.data();

		// Pin the host buffer for accelerating the memory copy
#ifndef __aarch64__
		rubis::sched::request_gpu();
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZI) * points_number_, cudaHostRegisterDefault));
		rubis::sched::yield_gpu("112_htod");
#endif
		rubis::sched::request_gpu();
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZI) * points_number_, cudaMemcpyHostToDevice));
		rubis::sched::yield_gpu("113_htod");
		

		if (x_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(x_));
			rubis::sched::yield_gpu("114_free");
			x_ = NULL;
		}

		if (y_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(y_));
			rubis::sched::yield_gpu("115_free");
			y_ = NULL;
		}

		if (z_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(z_));
			rubis::sched::yield_gpu("116_free");
			z_ = NULL;
		}

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&x_, sizeof(float) * points_number_));
		rubis::sched::yield_gpu("117_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&y_, sizeof(float) * points_number_));
		rubis::sched::yield_gpu("118_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&z_, sizeof(float) * points_number_));
		rubis::sched::yield_gpu("119_cudaMalloc");

		int block_x = (points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : points_number_;
		int grid_x = (points_number_ - 1) / block_x + 1;

		rubis::sched::request_gpu();
		convertInput<pcl::PointXYZI><<<grid_x, block_x>>>(tmp, x_, y_, z_, points_number_);
		rubis::sched::yield_gpu("120_convertInput");

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());


		if (trans_x_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(trans_x_));
			rubis::sched::yield_gpu("121_free");
			trans_x_ = NULL;
		}

		if (trans_y_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(trans_y_));
			rubis::sched::yield_gpu("122_free");
			trans_y_ = NULL;
		}

		if (trans_z_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(trans_z_));
			rubis::sched::yield_gpu("123_free");
			trans_z_ = NULL;
		}

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&trans_x_, sizeof(float) * points_number_));
		rubis::sched::yield_gpu("124_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&trans_y_, sizeof(float) * points_number_));
		rubis::sched::yield_gpu("125_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&trans_z_, sizeof(float) * points_number_));
		rubis::sched::yield_gpu("126_cudaMalloc");

		// Initially, also copy scanned points to transformed buffers
		
		rubis::sched::request_gpu();
		checkCudaErrors(cudaMemcpy(trans_x_, x_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		rubis::sched::yield_gpu("127_dtod");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMemcpy(trans_y_, y_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		rubis::sched::yield_gpu("128_dtod");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMemcpy(trans_z_, z_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		rubis::sched::yield_gpu("129_dtod");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaFree(tmp));
		rubis::sched::yield_gpu("130_free");

		// Unpin host buffer
#ifndef __aarch64__
		rubis::sched::request_gpu();
		checkCudaErrors(cudaHostUnregister(host_tmp));
		rubis::sched::yield_gpu("131_free");
#endif
	}
}

void GRegistration::setInputSource(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
	//Convert point cloud to float x, y, z
	if (input->size() > 0) {
		points_number_ = input->size();

		pcl::PointXYZ *tmp;

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZ) * points_number_));
		rubis::sched::yield_gpu("132_cudaMalloc");

		pcl::PointXYZ *host_tmp = input->points.data();

		// Pin the host buffer for accelerating the memory copy
#ifndef __aarch64__
		rubis::sched::request_gpu();
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZ) * points_number_, cudaHostRegisterDefault));
		rubis::sched::yield_gpu("133_htod");
#endif			
		rubis::sched::request_gpu();
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZ) * points_number_, cudaMemcpyHostToDevice));
		rubis::sched::yield_gpu("134_htod");

		if (x_ != NULL) {			
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(x_));
			rubis::sched::yield_gpu("135_free");
			
			x_ = NULL;
		}

		if (y_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(y_));
			rubis::sched::yield_gpu("136_free");
			
			y_ = NULL;
		}

		if (z_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(z_));
			rubis::sched::yield_gpu("137_free");

			z_ = NULL;
		}

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&x_, sizeof(float) * points_number_));
		rubis::sched::yield_gpu("138_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&y_, sizeof(float) * points_number_));
		rubis::sched::yield_gpu("139_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&z_, sizeof(float) * points_number_));
		rubis::sched::yield_gpu("140_cudaMalloc");

		int block_x = (points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : points_number_;
		int grid_x = (points_number_ - 1) / block_x + 1;

		rubis::sched::request_gpu();
		convertInput<pcl::PointXYZ><<<grid_x, block_x>>>(tmp, x_, y_, z_, points_number_);
		rubis::sched::yield_gpu("141_convertInput");

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());		

		if (trans_x_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(trans_x_));
			rubis::sched::yield_gpu("142_free");

			trans_x_ = NULL;
		}

		if (trans_y_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(trans_y_));
			rubis::sched::yield_gpu("143_free");

			trans_y_ = NULL;
		}

		if (trans_z_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(trans_z_));
			rubis::sched::yield_gpu("144_free");

			trans_z_ = NULL;
		}

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&trans_x_, sizeof(float) * points_number_));
		rubis::sched::yield_gpu("145_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&trans_y_, sizeof(float) * points_number_));
		rubis::sched::yield_gpu("146_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&trans_z_, sizeof(float) * points_number_));
		rubis::sched::yield_gpu("147_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMemcpy(trans_x_, x_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));		
		rubis::sched::yield_gpu("148_dtod");
		
		rubis::sched::request_gpu();
		checkCudaErrors(cudaMemcpy(trans_y_, y_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		rubis::sched::yield_gpu("149_dtod");
		
		rubis::sched::request_gpu();
		checkCudaErrors(cudaMemcpy(trans_z_, z_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		rubis::sched::yield_gpu("150_dtod");
		
		rubis::sched::request_gpu();
		checkCudaErrors(cudaFree(tmp));
		rubis::sched::yield_gpu("151_free");

#ifndef __aarch64__
		rubis::sched::request_gpu();
		checkCudaErrors(cudaHostUnregister(host_tmp));
		rubis::sched::yield_gpu("152_free");
#endif
	}
}



//Set input MAP data
void GRegistration::setInputTarget(pcl::PointCloud<pcl::PointXYZI>::Ptr input)
{
	if (input->size() > 0) {
		target_points_number_ = input->size();

		pcl::PointXYZI *tmp;

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZI) * target_points_number_));
		rubis::sched::yield_gpu("153_cudaMalloc");

		pcl::PointXYZI *host_tmp = input->points.data();

#ifndef __aarch64__
		rubis::sched::request_gpu();
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZI) * target_points_number_, cudaHostRegisterDefault));
		rubis::sched::yield_gpu("154_htod");
#endif

		rubis::sched::request_gpu();	
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZI) * target_points_number_, cudaMemcpyHostToDevice));
		rubis::sched::yield_gpu("155_htod");

		if (target_x_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(target_x_));
			rubis::sched::yield_gpu("156_free");

			target_x_ = NULL;
		}

		if (target_y_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(target_y_));
			rubis::sched::yield_gpu("157_free");

			target_y_ = NULL;
		}

		if (target_z_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(target_z_));
			rubis::sched::yield_gpu("158_free");

			target_z_ = NULL;
		}

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&target_x_, sizeof(float) * target_points_number_));
		rubis::sched::yield_gpu("159_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&target_y_, sizeof(float) * target_points_number_));
		rubis::sched::yield_gpu("160_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&target_z_, sizeof(float) * target_points_number_));
		rubis::sched::yield_gpu("161_cudaMalloc");

		int block_x = (target_points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : target_points_number_;
		int grid_x = (target_points_number_ - 1) / block_x + 1;

		rubis::sched::request_gpu();
		convertInput<pcl::PointXYZI><<<grid_x, block_x>>>(tmp, target_x_, target_y_, target_z_, target_points_number_);
		rubis::sched::yield_gpu("162_convertInput");

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

#ifndef __aarch64__
		rubis::sched::request_gpu();
		checkCudaErrors(cudaHostUnregister(host_tmp));
		rubis::sched::yield_gpu("163_free");
#endif
		rubis::sched::request_gpu();
		checkCudaErrors(cudaFree(tmp));
		rubis::sched::yield_gpu("164_free");
	}
}

void GRegistration::setInputTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
	if (input->size() > 0) {
		target_points_number_ = input->size();

		pcl::PointXYZ *tmp;

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZ) * target_points_number_));
		rubis::sched::yield_gpu("165_cudaMalloc");

		pcl::PointXYZ *host_tmp = input->points.data();

#ifndef __aarch64__
		rubis::sched::request_gpu();
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZ) * target_points_number_, cudaHostRegisterDefault));
		rubis::sched::yield_gpu("166_htod");
#endif
		rubis::sched::request_gpu();
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZ) * target_points_number_, cudaMemcpyHostToDevice));
		rubis::sched::yield_gpu("167_htod");

		if (target_x_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(target_x_));
			rubis::sched::yield_gpu("168_free");

			target_x_ = NULL;
		}

		if (target_y_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(target_y_));
			rubis::sched::yield_gpu("169_free");

			target_y_ = NULL;
		}

		if (target_z_ != NULL) {
			rubis::sched::request_gpu();
			checkCudaErrors(cudaFree(target_z_));
			rubis::sched::yield_gpu("170_free");

			target_z_ = NULL;
		}

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&target_x_, sizeof(float) * target_points_number_));
		rubis::sched::yield_gpu("171_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&target_y_, sizeof(float) * target_points_number_));
		rubis::sched::yield_gpu("172_cudaMalloc");

		rubis::sched::request_gpu();
		checkCudaErrors(cudaMalloc(&target_z_, sizeof(float) * target_points_number_));
		rubis::sched::yield_gpu("173_cudaMalloc");

		int block_x = (target_points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : target_points_number_;
		int grid_x = (target_points_number_ - 1) / block_x + 1;

		rubis::sched::request_gpu();
		convertInput<pcl::PointXYZ><<<grid_x, block_x>>>(tmp, target_x_, target_y_, target_z_, target_points_number_);	
		rubis::sched::yield_gpu("174_convertInput");

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		rubis::sched::request_gpu();
		checkCudaErrors(cudaFree(tmp));
		rubis::sched::yield_gpu("175_free");
#ifndef __aarch64__
		rubis::sched::request_gpu();
		checkCudaErrors(cudaHostUnregister(host_tmp));
		rubis::sched::yield_gpu("176_free");
#endif
	}
}

void GRegistration::align(const Eigen::Matrix<float, 4, 4> &guess)
{
	converged_ = false;

	final_transformation_ = transformation_ = previous_transformation_ = Eigen::Matrix<float, 4, 4>::Identity();

	computeTransformation(guess);
}

void GRegistration::computeTransformation(const Eigen::Matrix<float, 4, 4> &guess) {
	printf("Unsupported by Registration\n");
}
}



