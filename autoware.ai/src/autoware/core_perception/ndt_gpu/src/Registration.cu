#include "ndt_gpu/Registration.h"
#include "ndt_gpu/debug.h"
#include <iostream>

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

	if(GPU_PROFILING == 1){
		cudaEventCreate(&event_start);
		cudaEventCreate(&event_stop);
	}

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
			checkCudaErrors(cudaFree(x_));
			x_ = NULL;
		}

		if (y_ != NULL) {
			checkCudaErrors(cudaFree(y_));
			y_ = NULL;
		}

		if (z_ != NULL) {
			checkCudaErrors(cudaFree(z_));
			z_ = NULL;
		}

		if (trans_x_ != NULL) {
			checkCudaErrors(cudaFree(trans_x_));
			trans_x_ = NULL;
		}

		if (trans_y_ != NULL) {
			checkCudaErrors(cudaFree(trans_y_));
			trans_y_ = NULL;
		}

		if (trans_z_ != NULL) {
			checkCudaErrors(cudaFree(trans_z_));
			trans_z_ = NULL;
		}

		if (target_x_ != NULL) {
				checkCudaErrors(cudaFree(target_x_));
			target_x_ = NULL;
		}

		if (target_y_ != NULL) {
			checkCudaErrors(cudaFree(target_y_));
			target_y_ = NULL;
		}

		if (target_z_ != NULL) {
			checkCudaErrors(cudaFree(target_z_));
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

		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZI) * points_number_));

		pcl::PointXYZI *host_tmp = input->points.data();

		// Pin the host buffer for accelerating the memory copy
#ifndef __aarch64__
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZI) * points_number_, cudaHostRegisterDefault));
#endif
		start_profiling();
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZI) * points_number_, cudaMemcpyHostToDevice));
		stop_profiling(HTOD);

		if (x_ != NULL) {
			checkCudaErrors(cudaFree(x_));
			x_ = NULL;
		}

		if (y_ != NULL) {
			checkCudaErrors(cudaFree(y_));
			y_ = NULL;
		}

		if (z_ != NULL) {
			checkCudaErrors(cudaFree(z_));
			z_ = NULL;
		}

		checkCudaErrors(cudaMalloc(&x_, sizeof(float) * points_number_));
		checkCudaErrors(cudaMalloc(&y_, sizeof(float) * points_number_));
		checkCudaErrors(cudaMalloc(&z_, sizeof(float) * points_number_));

		int block_x = (points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : points_number_;
		int grid_x = (points_number_ - 1) / block_x + 1;

		start_profiling();
		convertInput<pcl::PointXYZI><<<grid_x, block_x>>>(tmp, x_, y_, z_, points_number_);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		stop_profiling(LAUNCH);


		if (trans_x_ != NULL) {
			checkCudaErrors(cudaFree(trans_x_));
			trans_x_ = NULL;
		}

		if (trans_y_ != NULL) {
			checkCudaErrors(cudaFree(trans_y_));
			trans_y_ = NULL;
		}

		if (trans_z_ != NULL) {
			checkCudaErrors(cudaFree(trans_z_));
			trans_z_ = NULL;
		}

		checkCudaErrors(cudaMalloc(&trans_x_, sizeof(float) * points_number_));
		checkCudaErrors(cudaMalloc(&trans_y_, sizeof(float) * points_number_));
		checkCudaErrors(cudaMalloc(&trans_z_, sizeof(float) * points_number_));

		// Initially, also copy scanned points to transformed buffers
		start_profiling();
		checkCudaErrors(cudaMemcpy(trans_x_, x_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		stop_profiling(DTOH);

		start_profiling();
		checkCudaErrors(cudaMemcpy(trans_y_, y_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		stop_profiling(DTOH);

		start_profiling();
		checkCudaErrors(cudaMemcpy(trans_z_, z_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		stop_profiling(DTOH);

		checkCudaErrors(cudaFree(tmp));

		// Unpin host buffer
#ifndef __aarch64__
		checkCudaErrors(cudaHostUnregister(host_tmp));
#endif
	}
}

void GRegistration::setInputSource(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
	//Convert point cloud to float x, y, z
	if (input->size() > 0) {
		points_number_ = input->size();

		pcl::PointXYZ *tmp;

		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZ) * points_number_));

		pcl::PointXYZ *host_tmp = input->points.data();

		// Pin the host buffer for accelerating the memory copy
#ifndef __aarch64__
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZ) * points_number_, cudaHostRegisterDefault));
#endif	
		start_profiling();
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZ) * points_number_, cudaMemcpyHostToDevice));
		stop_profiling(HTOD);

		if (x_ != NULL) {
			checkCudaErrors(cudaFree(x_));
			x_ = NULL;
		}

		if (y_ != NULL) {
			checkCudaErrors(cudaFree(y_));
			y_ = NULL;
		}

		if (z_ != NULL) {
			checkCudaErrors(cudaFree(z_));
			z_ = NULL;
		}

		checkCudaErrors(cudaMalloc(&x_, sizeof(float) * points_number_));
		checkCudaErrors(cudaMalloc(&y_, sizeof(float) * points_number_));
		checkCudaErrors(cudaMalloc(&z_, sizeof(float) * points_number_));

		int block_x = (points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : points_number_;
		int grid_x = (points_number_ - 1) / block_x + 1;

		start_profiling();
		convertInput<pcl::PointXYZ><<<grid_x, block_x>>>(tmp, x_, y_, z_, points_number_);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		stop_profiling(LAUNCH);

		if (trans_x_ != NULL) {
			checkCudaErrors(cudaFree(trans_x_));
			trans_x_ = NULL;
		}

		if (trans_y_ != NULL) {
			checkCudaErrors(cudaFree(trans_y_));
			trans_y_ = NULL;
		}

		if (trans_z_ != NULL) {
			checkCudaErrors(cudaFree(trans_z_));
			trans_z_ = NULL;
		}

		checkCudaErrors(cudaMalloc(&trans_x_, sizeof(float) * points_number_));
		checkCudaErrors(cudaMalloc(&trans_y_, sizeof(float) * points_number_));
		checkCudaErrors(cudaMalloc(&trans_z_, sizeof(float) * points_number_));

		start_profiling();
		checkCudaErrors(cudaMemcpy(trans_x_, x_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));		
		stop_profiling(DTOH);

		start_profiling();
		checkCudaErrors(cudaMemcpy(trans_y_, y_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		stop_profiling(DTOH);

		start_profiling();
		checkCudaErrors(cudaMemcpy(trans_z_, z_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		stop_profiling(DTOH);
		

		checkCudaErrors(cudaFree(tmp));
#ifndef __aarch64__
		checkCudaErrors(cudaHostUnregister(host_tmp));
#endif
	}
}



//Set input MAP data
void GRegistration::setInputTarget(pcl::PointCloud<pcl::PointXYZI>::Ptr input)
{
	if (input->size() > 0) {
		target_points_number_ = input->size();

		pcl::PointXYZI *tmp;

		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZI) * target_points_number_));

		pcl::PointXYZI *host_tmp = input->points.data();

#ifndef __aarch64__
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZI) * target_points_number_, cudaHostRegisterDefault));
#endif

		start_profiling();
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZI) * target_points_number_, cudaMemcpyHostToDevice));
		stop_profiling(HTOD);

		if (target_x_ != NULL) {
			checkCudaErrors(cudaFree(target_x_));
			target_x_ = NULL;
		}

		if (target_y_ != NULL) {
			checkCudaErrors(cudaFree(target_y_));
			target_y_ = NULL;
		}

		if (target_z_ != NULL) {
			checkCudaErrors(cudaFree(target_z_));
			target_z_ = NULL;
		}

		checkCudaErrors(cudaMalloc(&target_x_, sizeof(float) * target_points_number_));
		checkCudaErrors(cudaMalloc(&target_y_, sizeof(float) * target_points_number_));
		checkCudaErrors(cudaMalloc(&target_z_, sizeof(float) * target_points_number_));

		int block_x = (target_points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : target_points_number_;
		int grid_x = (target_points_number_ - 1) / block_x + 1;

		start_profiling();
		convertInput<pcl::PointXYZI><<<grid_x, block_x>>>(tmp, target_x_, target_y_, target_z_, target_points_number_);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		stop_profiling(LAUNCH);

#ifndef __aarch64__
		checkCudaErrors(cudaHostUnregister(host_tmp));
#endif
		checkCudaErrors(cudaFree(tmp));
	}
}

void GRegistration::setInputTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
	if (input->size() > 0) {
		target_points_number_ = input->size();

		pcl::PointXYZ *tmp;

		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZ) * target_points_number_));

		pcl::PointXYZ *host_tmp = input->points.data();

#ifndef __aarch64__
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZ) * target_points_number_, cudaHostRegisterDefault));
#endif
		start_profiling();
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZ) * target_points_number_, cudaMemcpyHostToDevice));
		stop_profiling(HTOD);

		if (target_x_ != NULL) {
			checkCudaErrors(cudaFree(target_x_));
			target_x_ = NULL;
		}

		if (target_y_ != NULL) {
			checkCudaErrors(cudaFree(target_y_));
			target_y_ = NULL;
		}

		if (target_z_ != NULL) {
			checkCudaErrors(cudaFree(target_z_));
			target_z_ = NULL;
		}

		checkCudaErrors(cudaMalloc(&target_x_, sizeof(float) * target_points_number_));
		checkCudaErrors(cudaMalloc(&target_y_, sizeof(float) * target_points_number_));
		checkCudaErrors(cudaMalloc(&target_z_, sizeof(float) * target_points_number_));

		int block_x = (target_points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : target_points_number_;
		int grid_x = (target_points_number_ - 1) / block_x + 1;

		start_profiling();
		convertInput<pcl::PointXYZ><<<grid_x, block_x>>>(tmp, target_x_, target_y_, target_z_, target_points_number_);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		stop_profiling(LAUNCH);

		checkCudaErrors(cudaFree(tmp));
#ifndef __aarch64__
		checkCudaErrors(cudaHostUnregister(host_tmp));
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

void GRegistration::start_profiling(){
	if(GPU_PROFILING == 1)
		cudaEventRecord(event_start, 0);
}

void GRegistration::stop_profiling(int type){
	if(GPU_PROFILING == 1){		
		float time;
		cudaEventRecord(event_stop, 0);
		cudaEventSynchronize(event_stop);
		cudaEventElapsedTime(&time, event_start, event_stop);
		write_data(gid, time, type);
		gid++;
	}
}

void GRegistration::write_data(int id, float time, int type){
	if(GPU_PROFILING == 1){
		fprintf(fp, "%d, %f, %d\n", id, time, type);				
	}
}

void GRegistration::write_dummy_line(){
	if(GPU_PROFILING == 1){
		fprintf(fp, "-1, -1, -1\n");						
		fflush(fp);
		gid = 0;
	}
}

void GRegistration::initialize_file(const char name[]){
	if(GPU_PROFILING == 1){
		fp = fopen(name, "w+");
		fprintf(fp, "ID, TIME, TYPE\n");		
	}
}



void GRegistration::close_file(){
	if(GPU_PROFILING == 1)
		fclose(fp);
}

}
