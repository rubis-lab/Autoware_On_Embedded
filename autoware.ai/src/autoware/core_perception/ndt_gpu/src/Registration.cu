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
		stop_profiling(35, HTOD);

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
		stop_profiling(36, LAUNCH);


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
		stop_profiling(37, DTOH);

		start_profiling();
		checkCudaErrors(cudaMemcpy(trans_y_, y_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		stop_profiling(38,DTOH);

		start_profiling();
		checkCudaErrors(cudaMemcpy(trans_z_, z_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		stop_profiling(39, DTOH);

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
		// long long int relative_deadline;
		// relative_deadline = MS2US(20);
		// fprintf(stderr, "sched1!!\n");
		// request_scheduling(relative_deadline);
		request_scheduling(deadline_list_[40]);
		start_profiling();
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZ) * points_number_, cudaMemcpyHostToDevice));
		stop_profiling(40, HTOD);

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

		request_scheduling(deadline_list_[41]);
		start_profiling();
		convertInput<pcl::PointXYZ><<<grid_x, block_x>>>(tmp, x_, y_, z_, points_number_);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());		
		stop_profiling(41, LAUNCH);

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

		request_scheduling(deadline_list_[42]);
		start_profiling();
		checkCudaErrors(cudaMemcpy(trans_x_, x_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));		
		stop_profiling(42, DTOH);

		request_scheduling(deadline_list_[43]);
		start_profiling();
		checkCudaErrors(cudaMemcpy(trans_y_, y_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		stop_profiling(43, DTOH);

		request_scheduling(deadline_list_[44]);
		start_profiling();
		checkCudaErrors(cudaMemcpy(trans_z_, z_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		stop_profiling(44, DTOH);
		

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
		stop_profiling(45, HTOD);

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
		stop_profiling(46, LAUNCH);

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
		request_scheduling(deadline_list_[47]);
		start_profiling();
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZ) * target_points_number_, cudaMemcpyHostToDevice));
		stop_profiling(47, HTOD);

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

		request_scheduling(deadline_list_[48]);
		start_profiling();
		convertInput<pcl::PointXYZ><<<grid_x, block_x>>>(tmp, target_x_, target_y_, target_z_, target_points_number_);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		stop_profiling(48, LAUNCH);

		checkCudaErrors(cudaFree(tmp));
#ifndef __aarch64__
		checkCudaErrors(cudaHostUnregister(host_tmp));
#endif
	}
}

void GRegistration::align(const Eigen::Matrix<float, 4, 4> &guess)
{
	set_absolute_deadline();
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

void GRegistration::stop_profiling(int id, int type){
	if(GPU_PROFILING == 1){		
		float time;
		cudaEventRecord(event_stop, 0);
		cudaEventSynchronize(event_stop);
		cudaEventElapsedTime(&time, event_start, event_stop);
		// write_data(gid, time, type);
    write_data(id, time, type);
		// gid++;
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
void sig_handler(int signum){
	if(signum == SIGUSR1 || signum == SIGUSR2)
		return;
	else
		termination();    
}
  
  void termination(){
	sched_info_->state = STOP;
	shmdt(sched_info_);
	if(remove(task_filename_)){
		printf("Cannot remove file %s\n", task_filename_);
		exit(1);
	}
	exit(0);
}
  
  unsigned long long get_current_time_us(){
	struct timespec ts;
	unsigned long long current_time;
	clock_gettime(CLOCK_REALTIME, &ts);
	current_time = ts.tv_sec%10000 * 1000000 + ts.tv_nsec/1000;
	return current_time;
}
  
  void us_sleep(unsigned long long us){
	struct timespec ts;
	ts.tv_sec = us/1000000;
	ts.tv_nsec = us%1000000*1000;
	nanosleep(&ts, NULL);
	return;
}
  
  void initialize_signal_handler(){
	signal(SIGINT, sig_handler);
	signal(SIGTSTP, sig_handler);
	signal(SIGQUIT, sig_handler);
	signal(SIGUSR1, sig_handler);
	signal(SIGUSR2, sig_handler);
}
  
  void create_task_file(){        
	FILE* task_fp;
	task_fp = fopen(task_filename_, "w");
	if(task_fp == NULL){
		printf("Cannot create task file at %s\n", task_filename_);
		exit(1);
	}
	fprintf(task_fp, "%d\n", getpid());
	fprintf(task_fp, "%d", key_id_);
	fclose(task_fp);
}
  
void get_scheduler_pid(){
	FILE* scheduler_fp;
	printf("Wait the scheduler...\n");
	while(1){
		scheduler_fp = fopen("/tmp/np_edf_scheduler", "r");
		if(scheduler_fp) break;
	}
	while(1){
		fscanf(scheduler_fp, "%d", &scheduler_pid_);
		if(scheduler_pid_ != 0) break;
	}
	printf("Scheduler pid: %d\n", scheduler_pid_);
	fclose(scheduler_fp);
}
  
void initialize_sched_info(){
	FILE* sm_key_fp;
	sm_key_fp = fopen("/tmp/sm_key", "r");
	if(sm_key_fp == NULL){
		printf("Cannot open /tmp/sm_key\n");
		termination();
	}
  
	key_ = ftok("/tmp/sm_key", key_id_);
	shmid_ = shmget(key_, sizeof(SchedInfo), 0666|IPC_CREAT);
	sched_info_ = (SchedInfo*)shmat(shmid_, 0, 0);
	sched_info_->pid = getpid();
	sched_info_->state = NONE;
}
  
void init_scheduling(char* task_filename, char* deadline_filename, int key_id){
	gpu_scheduling_flag_ = 1;
	// Get deadline list
	get_deadline_list(deadline_filename);

	// Initialize key id for shared memory
	key_id_ = key_id;
  
	// Initialize signal handler
	initialize_signal_handler();
  
	// Create task file
  
	sprintf(task_filename_, "%s", task_filename);
	create_task_file();    
  
	// Get scheduler pid
	get_scheduler_pid();
  
	// Initialize scheduling information (shared memory data)
	initialize_sched_info();
  
	sigemptyset(&sigset_);
	sigaddset(&sigset_, SIGUSR1);
	sigaddset(&sigset_, SIGUSR2);
	sigprocmask(SIG_BLOCK, &sigset_, NULL);    
  
	// sigwait(&sigset_, &sig_);    
	// kill(scheduler_pid_, SIGUSR2);
	// sigprocmask(SIG_UNBLOCK, &sigset_, NULL);
	
	printf("Task [%d] is ready to work\n", getpid());
	// sigaddset(&sigset_, SIGUSR1);
	// sigprocmask(SIG_BLOCK, &sigset_, NULL);    
  
}
  
void request_scheduling(unsigned long long relative_deadline){
	
	if(gpu_scheduling_flag_ == 0) return;

	if(absolute_deadline_ != 0) sched_info_->deadline = absolute_deadline_;  
	else sched_info_->deadline = get_current_time_us() + relative_deadline;  

	sched_info_->state = WAIT;        
	// printf("Request schedule - deadline: %llu\n", sched_info_->deadline);
	while(1){
		kill(scheduler_pid_, SIGUSR1);
		if(!sigwait(&sigset_, &sig_)) break;
	}
}

void get_deadline_list(char* filename){
  FILE* fp;
  fp = fopen(filename, "r");
  if(fp==NULL){
	  fprintf(stderr, "Cannot find file %s\n", filename);
	  exit(1);
  }
  char buf[1024];
  long long int deadline;
  for(int i = 0; i < sizeof(deadline_list_)/sizeof(long long int); i++){
    fgets(buf, 1024, fp);
    strtok(buf, "\n");
    sscanf(buf, "%*llu, %llu", &deadline);
    deadline_list_[i] = deadline;
  }
}

void set_identical_deadline(unsigned long long identical_deadline){
	identical_deadline_ = identical_deadline;
  }
  
  void set_absolute_deadline(){
	absolute_deadline_ = get_current_time_us() + identical_deadline_;
  }