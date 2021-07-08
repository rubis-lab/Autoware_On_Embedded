#include "ndt_gpu/Registration.h"
#include "ndt_gpu/debug.h"
#include <iostream>
#include <sys/stat.h>
#include "rubis_sched/sched.hpp"

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
			//rubis::sched::request_gpu(101);
			checkCudaErrors(cudaFree(x_));
			//rubis::sched::yield_gpu(101,"free");
			x_ = NULL;
		}

		if (y_ != NULL) {
			//rubis::sched::request_gpu(102);
			checkCudaErrors(cudaFree(y_));
			//rubis::sched::yield_gpu(102,"free");

			y_ = NULL;
		}

		if (z_ != NULL) {
			//rubis::sched::request_gpu(103);
			checkCudaErrors(cudaFree(z_));
			//rubis::sched::yield_gpu(103,"free");
	
			z_ = NULL;
		}

		if (trans_x_ != NULL) {
			//rubis::sched::request_gpu(104);
			checkCudaErrors(cudaFree(trans_x_));
			//rubis::sched::yield_gpu(104,"free");

			trans_x_ = NULL;
		}

		if (trans_y_ != NULL) {
			//rubis::sched::request_gpu(405);
			checkCudaErrors(cudaFree(trans_y_));
			//rubis::sched::yield_gpu(405,"free");

			trans_y_ = NULL;
		}

		if (trans_z_ != NULL) {
			//rubis::sched::request_gpu(106);
			checkCudaErrors(cudaFree(trans_z_));
			//rubis::sched::yield_gpu(106,"free");
			
			trans_z_ = NULL;
		}

		if (target_x_ != NULL) {
			//rubis::sched::request_gpu(107);
			checkCudaErrors(cudaFree(target_x_));
			//rubis::sched::yield_gpu(107,"free");
			
			target_x_ = NULL;
		}

		if (target_y_ != NULL) {
			//rubis::sched::request_gpu(108);
			checkCudaErrors(cudaFree(target_y_));
			//rubis::sched::yield_gpu(108,"free");

			target_y_ = NULL;
		}

		if (target_z_ != NULL) {
			//rubis::sched::request_gpu(109);
			checkCudaErrors(cudaFree(target_z_));
			//rubis::sched::yield_gpu(109,"free");

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

		//rubis::sched::request_gpu(110);
		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZI) * points_number_));
		//rubis::sched::yield_gpu(110,"cudaMalloc");
		
		pcl::PointXYZI *host_tmp = input->points.data();

		// Pin the host buffer for accelerating the memory copy
#ifndef __aarch64__
		//rubis::sched::request_gpu(111);
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZI) * points_number_, cudaHostRegisterDefault));
		//rubis::sched::yield_gpu(111,"htod");
#endif
		//rubis::sched::request_gpu(112);
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZI) * points_number_, cudaMemcpyHostToDevice));
		//rubis::sched::yield_gpu(112,"htod");
		

		if (x_ != NULL) {
			//rubis::sched::request_gpu(113);
			checkCudaErrors(cudaFree(x_));
			//rubis::sched::yield_gpu(113,"free");
			x_ = NULL;
		}

		if (y_ != NULL) {
			//rubis::sched::request_gpu(114);
			checkCudaErrors(cudaFree(y_));
			//rubis::sched::yield_gpu(114,"free");
			y_ = NULL;
		}

		if (z_ != NULL) {
			//rubis::sched::request_gpu(115);
			checkCudaErrors(cudaFree(z_));
			//rubis::sched::yield_gpu(115,"free");
			z_ = NULL;
		}

		//rubis::sched::request_gpu(116);
		checkCudaErrors(cudaMalloc(&x_, sizeof(float) * points_number_));
		//rubis::sched::yield_gpu(116,"cudaMalloc");

		//rubis::sched::request_gpu(117);
		checkCudaErrors(cudaMalloc(&y_, sizeof(float) * points_number_));
		//rubis::sched::yield_gpu(117,"cudaMalloc");

		//rubis::sched::request_gpu(118);
		checkCudaErrors(cudaMalloc(&z_, sizeof(float) * points_number_));
		//rubis::sched::yield_gpu(118,"cudaMalloc");

		int block_x = (points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : points_number_;
		int grid_x = (points_number_ - 1) / block_x + 1;

		//rubis::sched::request_gpu(119);
		convertInput<pcl::PointXYZI><<<grid_x, block_x>>>(tmp, x_, y_, z_, points_number_);
		//rubis::sched::yield_gpu(119,"convertInput");

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());


		if (trans_x_ != NULL) {
			//rubis::sched::request_gpu(120);
			checkCudaErrors(cudaFree(trans_x_));
			//rubis::sched::yield_gpu(120,"free");
			trans_x_ = NULL;
		}

		if (trans_y_ != NULL) {
			//rubis::sched::request_gpu(121);
			checkCudaErrors(cudaFree(trans_y_));
			//rubis::sched::yield_gpu(121,"free");
			trans_y_ = NULL;
		}

		if (trans_z_ != NULL) {
			//rubis::sched::request_gpu(122);
			checkCudaErrors(cudaFree(trans_z_));
			//rubis::sched::yield_gpu(122,"free");
			trans_z_ = NULL;
		}

		//rubis::sched::request_gpu(123);
		checkCudaErrors(cudaMalloc(&trans_x_, sizeof(float) * points_number_));
		//rubis::sched::yield_gpu(123,"cudaMalloc");

		//rubis::sched::request_gpu(124);
		checkCudaErrors(cudaMalloc(&trans_y_, sizeof(float) * points_number_));
		//rubis::sched::yield_gpu(124,"cudaMalloc");

		//rubis::sched::request_gpu(125);
		checkCudaErrors(cudaMalloc(&trans_z_, sizeof(float) * points_number_));
		//rubis::sched::yield_gpu(125,"cudaMalloc");

		// Initially, also copy scanned points to transformed buffers
		
		//rubis::sched::request_gpu(126);
		checkCudaErrors(cudaMemcpy(trans_x_, x_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		//rubis::sched::yield_gpu(126,"dtod");

		//rubis::sched::request_gpu(127);
		checkCudaErrors(cudaMemcpy(trans_y_, y_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		//rubis::sched::yield_gpu(127,"dtod");

		//rubis::sched::request_gpu(128);
		checkCudaErrors(cudaMemcpy(trans_z_, z_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		//rubis::sched::yield_gpu(128,"dtod");

		//rubis::sched::request_gpu(129);
		checkCudaErrors(cudaFree(tmp));
		//rubis::sched::yield_gpu(129,"free");

		// Unpin host buffer
#ifndef __aarch64__
		//rubis::sched::request_gpu(130);
		checkCudaErrors(cudaHostUnregister(host_tmp));
		//rubis::sched::yield_gpu(130,"free");
#endif
	}
}

void GRegistration::setInputSource(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
	//Convert point cloud to float x, y, z
	if (input->size() > 0) {
		points_number_ = input->size();

		pcl::PointXYZ *tmp;

		//rubis::sched::request_gpu(131);
		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZ) * points_number_));
		//rubis::sched::yield_gpu(131,"cudaMalloc");

		pcl::PointXYZ *host_tmp = input->points.data();

		// Pin the host buffer for accelerating the memory copy
#ifndef __aarch64__
		//rubis::sched::request_gpu(132);
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZ) * points_number_, cudaHostRegisterDefault));
		//rubis::sched::yield_gpu(132,"htod");
#endif			
		//rubis::sched::request_gpu(133);
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZ) * points_number_, cudaMemcpyHostToDevice));
		//rubis::sched::yield_gpu(133,"htod");

		if (x_ != NULL) {			
			//rubis::sched::request_gpu(134);
			checkCudaErrors(cudaFree(x_));
			//rubis::sched::yield_gpu(134,"free");
			
			x_ = NULL;
		}

		if (y_ != NULL) {
			//rubis::sched::request_gpu(135);
			checkCudaErrors(cudaFree(y_));
			//rubis::sched::yield_gpu(135,"free");
			
			y_ = NULL;
		}

		if (z_ != NULL) {
			//rubis::sched::request_gpu(136);
			checkCudaErrors(cudaFree(z_));
			//rubis::sched::yield_gpu(136,"free");

			z_ = NULL;
		}

		//rubis::sched::request_gpu(137);
		checkCudaErrors(cudaMalloc(&x_, sizeof(float) * points_number_));
		//rubis::sched::yield_gpu(137,"cudaMalloc");

		//rubis::sched::request_gpu(138);
		checkCudaErrors(cudaMalloc(&y_, sizeof(float) * points_number_));
		//rubis::sched::yield_gpu(138,"cudaMalloc");

		//rubis::sched::request_gpu(139);
		checkCudaErrors(cudaMalloc(&z_, sizeof(float) * points_number_));
		//rubis::sched::yield_gpu(139,"cudaMalloc");

		int block_x = (points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : points_number_;
		int grid_x = (points_number_ - 1) / block_x + 1;

		//rubis::sched::request_gpu(140);
		convertInput<pcl::PointXYZ><<<grid_x, block_x>>>(tmp, x_, y_, z_, points_number_);
		//rubis::sched::yield_gpu(140,"convertInput");

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());		

		if (trans_x_ != NULL) {
			//rubis::sched::request_gpu(141);
			checkCudaErrors(cudaFree(trans_x_));
			//rubis::sched::yield_gpu(141,"free");

			trans_x_ = NULL;
		}

		if (trans_y_ != NULL) {
			//rubis::sched::request_gpu(142);
			checkCudaErrors(cudaFree(trans_y_));
			//rubis::sched::yield_gpu(142,"free");

			trans_y_ = NULL;
		}

		if (trans_z_ != NULL) {
			//rubis::sched::request_gpu(143);
			checkCudaErrors(cudaFree(trans_z_));
			//rubis::sched::yield_gpu(143,"free");

			trans_z_ = NULL;
		}

		//rubis::sched::request_gpu(144);
		checkCudaErrors(cudaMalloc(&trans_x_, sizeof(float) * points_number_));
		//rubis::sched::yield_gpu(144,"cudaMalloc");

		//rubis::sched::request_gpu(145);
		checkCudaErrors(cudaMalloc(&trans_y_, sizeof(float) * points_number_));
		//rubis::sched::yield_gpu(145,"cudaMalloc");

		//rubis::sched::request_gpu(146);
		checkCudaErrors(cudaMalloc(&trans_z_, sizeof(float) * points_number_));
		//rubis::sched::yield_gpu(146,"cudaMalloc");

		//rubis::sched::request_gpu(147);
		checkCudaErrors(cudaMemcpy(trans_x_, x_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));		
		//rubis::sched::yield_gpu(147,"dtod");
		
		//rubis::sched::request_gpu(148);
		checkCudaErrors(cudaMemcpy(trans_y_, y_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		//rubis::sched::yield_gpu(148,"dtod");
		
		//rubis::sched::request_gpu(149);
		checkCudaErrors(cudaMemcpy(trans_z_, z_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		//rubis::sched::yield_gpu(149,"dtod");
		
		//rubis::sched::request_gpu(150);
		checkCudaErrors(cudaFree(tmp));
		//rubis::sched::yield_gpu(150,"free");

#ifndef __aarch64__
		//rubis::sched::request_gpu(151);
		checkCudaErrors(cudaHostUnregister(host_tmp));
		//rubis::sched::yield_gpu(151,"free");
#endif
	}
}



//Set input MAP data
void GRegistration::setInputTarget(pcl::PointCloud<pcl::PointXYZI>::Ptr input)
{
	if (input->size() > 0) {
		target_points_number_ = input->size();

		pcl::PointXYZI *tmp;

		//rubis::sched::request_gpu(152);
		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZI) * target_points_number_));
		//rubis::sched::yield_gpu(152,"cudaMalloc");

		pcl::PointXYZI *host_tmp = input->points.data();

#ifndef __aarch64__
		//rubis::sched::request_gpu(153);
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZI) * target_points_number_, cudaHostRegisterDefault));
		//rubis::sched::yield_gpu(153,"htod");
#endif

		//rubis::sched::request_gpu(154);	
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZI) * target_points_number_, cudaMemcpyHostToDevice));
		//rubis::sched::yield_gpu(154,"htod");

		if (target_x_ != NULL) {
			//rubis::sched::request_gpu(155);
			checkCudaErrors(cudaFree(target_x_));
			//rubis::sched::yield_gpu(155,"free");

			target_x_ = NULL;
		}

		if (target_y_ != NULL) {
			//rubis::sched::request_gpu(156);
			checkCudaErrors(cudaFree(target_y_));
			//rubis::sched::yield_gpu(156,"free");

			target_y_ = NULL;
		}

		if (target_z_ != NULL) {
			//rubis::sched::request_gpu(157);
			checkCudaErrors(cudaFree(target_z_));
			//rubis::sched::yield_gpu(157,"free");

			target_z_ = NULL;
		}

		//rubis::sched::request_gpu(158);
		checkCudaErrors(cudaMalloc(&target_x_, sizeof(float) * target_points_number_));
		//rubis::sched::yield_gpu(158,"cudaMalloc");

		//rubis::sched::request_gpu(159);
		checkCudaErrors(cudaMalloc(&target_y_, sizeof(float) * target_points_number_));
		//rubis::sched::yield_gpu(159,"cudaMalloc");

		//rubis::sched::request_gpu(160);
		checkCudaErrors(cudaMalloc(&target_z_, sizeof(float) * target_points_number_));
		//rubis::sched::yield_gpu(160,"cudaMalloc");

		int block_x = (target_points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : target_points_number_;
		int grid_x = (target_points_number_ - 1) / block_x + 1;

		//rubis::sched::request_gpu(161);
		convertInput<pcl::PointXYZI><<<grid_x, block_x>>>(tmp, target_x_, target_y_, target_z_, target_points_number_);
		//rubis::sched::yield_gpu(161,"convertInput");

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

#ifndef __aarch64__
		//rubis::sched::request_gpu(162);
		checkCudaErrors(cudaHostUnregister(host_tmp));
		//rubis::sched::yield_gpu(162,"free");
#endif
		//rubis::sched::request_gpu(163);
		checkCudaErrors(cudaFree(tmp));
		//rubis::sched::yield_gpu(163,"free");
	}
}

void GRegistration::setInputTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
	if (input->size() > 0) {
		target_points_number_ = input->size();

		pcl::PointXYZ *tmp;

		//rubis::sched::request_gpu(164);
		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZ) * target_points_number_));
		//rubis::sched::yield_gpu(164,"cudaMalloc");

		pcl::PointXYZ *host_tmp = input->points.data();

#ifndef __aarch64__
		//rubis::sched::request_gpu(165);
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZ) * target_points_number_, cudaHostRegisterDefault));
		//rubis::sched::yield_gpu(165,"htod");
#endif
		//rubis::sched::request_gpu(166);
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZ) * target_points_number_, cudaMemcpyHostToDevice));
		//rubis::sched::yield_gpu(166,"htod");

		if (target_x_ != NULL) {
			//rubis::sched::request_gpu(167);
			checkCudaErrors(cudaFree(target_x_));
			//rubis::sched::yield_gpu(167,"free");

			target_x_ = NULL;
		}

		if (target_y_ != NULL) {
			//rubis::sched::request_gpu(168);
			checkCudaErrors(cudaFree(target_y_));
			//rubis::sched::yield_gpu(168,"free");

			target_y_ = NULL;
		}

		if (target_z_ != NULL) {
			//rubis::sched::request_gpu(169);
			checkCudaErrors(cudaFree(target_z_));
			//rubis::sched::yield_gpu(169,"free");

			target_z_ = NULL;
		}

		//rubis::sched::request_gpu(170);
		checkCudaErrors(cudaMalloc(&target_x_, sizeof(float) * target_points_number_));
		//rubis::sched::yield_gpu(170,"cudaMalloc");

		//rubis::sched::request_gpu(171);
		checkCudaErrors(cudaMalloc(&target_y_, sizeof(float) * target_points_number_));
		//rubis::sched::yield_gpu(171,"cudaMalloc");

		//rubis::sched::request_gpu(172);
		checkCudaErrors(cudaMalloc(&target_z_, sizeof(float) * target_points_number_));
		//rubis::sched::yield_gpu(172,"cudaMalloc");

		int block_x = (target_points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : target_points_number_;
		int grid_x = (target_points_number_ - 1) / block_x + 1;

		//rubis::sched::request_gpu(173);
		convertInput<pcl::PointXYZ><<<grid_x, block_x>>>(tmp, target_x_, target_y_, target_z_, target_points_number_);	
		//rubis::sched::yield_gpu(173,"convertInput");

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		//rubis::sched::request_gpu(174);
		checkCudaErrors(cudaFree(tmp));
		//rubis::sched::yield_gpu(174,"free");
#ifndef __aarch64__
		//rubis::sched::request_gpu(175);
		checkCudaErrors(cudaHostUnregister(host_tmp));
		//rubis::sched::yield_gpu(175,"free");
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



