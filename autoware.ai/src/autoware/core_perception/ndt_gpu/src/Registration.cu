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
		cudaEventCreate(&e_event_start);
		cudaEventCreate(&e_event_stop);
		cudaEventCreate(&r_event_start);
		cudaEventCreate(&r_event_stop);
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

		stop_cpu_profiling();
  		request_scheduling(1);
		checkCudaErrors(cudaMalloc(&tmp, sizeof(pcl::PointXYZI) * points_number_));
		stop_profiling(1, HTOD);
  		start_profiling_cpu_time();
		pcl::PointXYZI *host_tmp = input->points.data();

		// Pin the host buffer for accelerating the memory copy
#ifndef __aarch64__
		stop_cpu_profiling();
		request_scheduling(2);
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(pcl::PointXYZI) * points_number_, cudaHostRegisterDefault));
		stop_profiling(2, HTOD);
  		start_profiling_cpu_time();
#endif
		stop_cpu_profiling();
		request_scheduling(3);
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZI) * points_number_, cudaMemcpyHostToDevice));
		stop_profiling(3, HTOD);
  		start_profiling_cpu_time();

		if (x_ != NULL) {
			stop_cpu_profiling();
  			request_scheduling(4);
			checkCudaErrors(cudaFree(x_));
			stop_profiling(4, HTOD);
  			start_profiling_cpu_time();
			x_ = NULL;
		}

		if (y_ != NULL) {
			stop_cpu_profiling();
  			request_scheduling(5);
			checkCudaErrors(cudaFree(y_));
			stop_profiling(5, HTOD);
  			start_profiling_cpu_time();
			y_ = NULL;
		}

		if (z_ != NULL) {
			stop_cpu_profiling();
  			request_scheduling(6);
			checkCudaErrors(cudaFree(z_));
			stop_profiling(6, HTOD);
  			start_profiling_cpu_time();
			z_ = NULL;
		}

		stop_cpu_profiling();
  		request_scheduling(7);
		checkCudaErrors(cudaMalloc(&x_, sizeof(float) * points_number_));
		stop_profiling(7, HTOD);
  		start_profiling_cpu_time();

		stop_cpu_profiling();
  		request_scheduling(8);
		checkCudaErrors(cudaMalloc(&y_, sizeof(float) * points_number_));
		stop_profiling(8, HTOD);
  		start_profiling_cpu_time();

		stop_cpu_profiling();
  		request_scheduling(9);
		checkCudaErrors(cudaMalloc(&z_, sizeof(float) * points_number_));
		stop_profiling(9, HTOD);
  		start_profiling_cpu_time();

		int block_x = (points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : points_number_;
		int grid_x = (points_number_ - 1) / block_x + 1;

		stop_cpu_profiling();
		request_scheduling(10);
		convertInput<pcl::PointXYZI><<<grid_x, block_x>>>(tmp, x_, y_, z_, points_number_);
		stop_profiling(10, HTOD);
  		start_profiling_cpu_time();

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());


		if (trans_x_ != NULL) {
			stop_cpu_profiling();
  			request_scheduling(11);
			checkCudaErrors(cudaFree(trans_x_));
			stop_profiling(11, HTOD);
  			start_profiling_cpu_time();
			trans_x_ = NULL;
		}

		if (trans_y_ != NULL) {
			stop_cpu_profiling();
  			request_scheduling(12);
			checkCudaErrors(cudaFree(trans_y_));
			stop_profiling(12, HTOD);
  			start_profiling_cpu_time();
			trans_y_ = NULL;
		}

		if (trans_z_ != NULL) {
			stop_cpu_profiling();
  			request_scheduling(13);
			checkCudaErrors(cudaFree(trans_z_));
			stop_profiling(13, HTOD);
  			start_profiling_cpu_time();
			trans_z_ = NULL;
		}

		stop_cpu_profiling();
  		request_scheduling(14);
		checkCudaErrors(cudaMalloc(&trans_x_, sizeof(float) * points_number_));
		stop_profiling(14, HTOD);
  		start_profiling_cpu_time();

		stop_cpu_profiling();
  		request_scheduling(15);
		checkCudaErrors(cudaMalloc(&trans_y_, sizeof(float) * points_number_));
		stop_profiling(15, HTOD);
  		start_profiling_cpu_time();

		stop_cpu_profiling();
  		request_scheduling(16);
		checkCudaErrors(cudaMalloc(&trans_z_, sizeof(float) * points_number_));
		stop_profiling(16, HTOD);
  		start_profiling_cpu_time();

		// Initially, also copy scanned points to transformed buffers
		stop_cpu_profiling();
  		request_scheduling(17);
		checkCudaErrors(cudaMemcpy(trans_x_, x_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		stop_profiling(17, HTOD);
  		start_profiling_cpu_time();

		stop_cpu_profiling();
  		request_scheduling(18);
		checkCudaErrors(cudaMemcpy(trans_y_, y_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		stop_profiling(18, HTOD);
  		start_profiling_cpu_time();

		stop_cpu_profiling();
  		request_scheduling(19);
		checkCudaErrors(cudaMemcpy(trans_z_, z_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		stop_profiling(19, HTOD);
  		start_profiling_cpu_time();

		stop_cpu_profiling();
  		request_scheduling(20);
		checkCudaErrors(cudaFree(tmp));
		stop_profiling(20, HTOD);
  		start_profiling_cpu_time();

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
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZ) * points_number_, cudaMemcpyHostToDevice));

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

		convertInput<pcl::PointXYZ><<<grid_x, block_x>>>(tmp, x_, y_, z_, points_number_);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());		

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

		checkCudaErrors(cudaMemcpy(trans_x_, x_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));		

		checkCudaErrors(cudaMemcpy(trans_y_, y_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy(trans_z_, z_, sizeof(float) * points_number_, cudaMemcpyDeviceToDevice));
		

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

		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZI) * target_points_number_, cudaMemcpyHostToDevice));

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

		convertInput<pcl::PointXYZI><<<grid_x, block_x>>>(tmp, target_x_, target_y_, target_z_, target_points_number_);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

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
		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(pcl::PointXYZ) * target_points_number_, cudaMemcpyHostToDevice));

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

		convertInput<pcl::PointXYZ><<<grid_x, block_x>>>(tmp, target_x_, target_y_, target_z_, target_points_number_);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

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
}

/* GPU Profiling */
void start_profiling_execution_time(){
	if(GPU_PROFILING == 1)
		cudaEventRecord(e_event_start, 0);
}

void start_profiling_response_time(){
	if(GPU_PROFILING == 1)
		cudaEventRecord(r_event_start, 0);
}

void start_profiling_cpu_time(){
  if(GPU_PROFILING == 1){
    cpu_id++;
		gettimeofday(&startTime, NULL);
  }
}

void stop_profiling(int id, int type){
	if(GPU_PROFILING == 1){		
		float e_time, r_time;
    char gpu_id_buf[BUFFER_SIZE];
	if(gpu_scheduling_flag_==1){
    	sched_info_->scheduling_flag = 0;
    	sched_info_->state = NONE;
	}

		cudaEventRecord(e_event_stop, 0);
    cudaEventRecord(r_event_stop, 0);
		cudaEventSynchronize(e_event_stop);
    cudaEventSynchronize(r_event_stop);
		cudaEventElapsedTime(&e_time, e_event_start, e_event_stop);
    cudaEventElapsedTime(&r_time, r_event_start, r_event_stop);
    e_time = MS2US(e_time);
    r_time = MS2US(r_time);
		// write_data(gid, time, type);
    sprintf(gpu_id_buf,"g%d",id);

    //write_profiling_data(id, e_time, r_time, type);

    write_profiling_data(gpu_id_buf, e_time, r_time, type);
		// gid++;
	}
}

void stop_cpu_profiling(){
  if(GPU_PROFILING == 1){
    long long int elapsedTime;
    char cpu_id_buf[BUFFER_SIZE];

    gettimeofday(&endTime, NULL);
    elapsedTime = ((long long int)(endTime.tv_sec - startTime.tv_sec)) * 1000000ll + (endTime.tv_usec - startTime.tv_usec);

    sprintf(cpu_id_buf,"e%d",cpu_id);
    write_cpu_profiling_data(cpu_id_buf,elapsedTime);    
  }
}

void write_profiling_data(const char* id, float e_time, float r_time, int type){
	if(GPU_PROFILING == 1){
		fprintf(execution_time_fp, "%s, %f, %d\n", id, e_time, type);	
    fprintf(response_time_fp, "%s, %f, %d\n", id, r_time, type);	
    fprintf(remain_time_fp, "%s, %llu\n", id, absolute_deadline_ - get_current_time_us());
	}
}

void write_cpu_profiling_data(const char *id, long long int c_time){
  if(GPU_PROFILING == 1){
		fprintf(execution_time_fp, "%s, %02d\n", id, c_time);	
    fprintf(response_time_fp, "%s, %02d\n", id, c_time);    
	}
}


void write_dummy_line(){
	if(GPU_PROFILING == 1){  
    fprintf(execution_time_fp, "-1, -1, -1\n");						
		fflush(execution_time_fp);
		fprintf(response_time_fp, "-1, -1, -1\n");						
		fflush(response_time_fp);
    fprintf(remain_time_fp, "-1, -1\n");						
		fflush(remain_time_fp);
    cpu_id = 0;
	}
}

void initialize_file(const char* execution_time_filename, const char* response_time_filename, const char* remain_time_filename){
	if(GPU_PROFILING == 1){
		printf("ex filename: %s\n", execution_time_filename);
		execution_time_fp = fopen(execution_time_filename, "w+");
		fprintf(execution_time_fp, "ID, TIME, TYPE\n");
		response_time_fp = fopen(response_time_filename, "w+");
			fprintf(response_time_fp, "ID, TIME, TYPE\n");
		remain_time_fp = fopen(remain_time_filename, "w+");
		fprintf(remain_time_fp, "ID, TIME\n");
	}
}

void close_file(){
	if(GPU_PROFILING == 1){
		fclose(execution_time_fp);
    fclose(response_time_fp);
    fclose(remain_time_fp);
  }
}

void sig_handler(int signum){
  if(signum == SIGUSR1 || signum == SIGUSR2){
    is_scheduled_ = 1;
    return;    
  }
  else
      termination();    
}

void termination(){
	if(gpu_scheduling_flag_==1){
		sched_info_->state = STOP;
  	shmdt(sched_info_);
	}
  
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
	if(gpu_scheduling_flag_!=1) return;
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
  sched_info_->scheduling_flag = 0;
}

void init_scheduling(char* task_filename, const char* deadline_filename, int key_id){
	if(gpu_scheduling_flag_!=1) return;
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

void request_scheduling(int id){  
  if(gpu_scheduling_flag_ == 1){
		unsigned long long relative_deadline = deadline_list_[id];  
		if(identical_deadline_ != 0) sched_info_->deadline = absolute_deadline_;  
		else sched_info_->deadline = get_current_time_us() + relative_deadline;  

		sched_info_->state = WAIT;        
		// printf("Request schedule - deadline: %llu\n", sched_info_->deadline);
  }

  start_profiling_response_time();

  if(gpu_scheduling_flag_ == 1){
		while(1){
			kill(scheduler_pid_, SIGUSR1);
			// if(!sigwait(&sigset_, &sig_)) break;
			// if(is_scheduled_ == 1) break;
			if(sched_info_->scheduling_flag == 1) break;
		}  
  }

  start_profiling_execution_time();

  if(gpu_scheduling_flag_==1){
		sched_info_->state = RUN;
		sched_info_->deadline = 0;
	}
}

void get_deadline_list(const char* filename){
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
    sscanf(buf, "%*s, %llu", &deadline);
    deadline_list_[i] = deadline;
  }
}

void set_identical_deadline(unsigned long long identical_deadline){
  identical_deadline_ = identical_deadline;
}

void set_absolute_deadline(){  
  absolute_deadline_ = get_current_time_us() + identical_deadline_;
}

void set_slicing_flag(int flag){
  slicing_flag_ = flag;
}

void set_gpu_scheduling_flag(int gpu_scheduling_flag){
	gpu_scheduling_flag_ = gpu_scheduling_flag;
}
