#ifndef GPU_EUCLIDEAN_H_
#define GPU_EUCLIDEAN_H_

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define HTOD 0
#define DTOH 1
#define LAUNCH 2
#define GPU_PROFILING 1

class GpuEuclideanCluster
{
public:
  typedef struct
  {
    int index_value;
    std::vector<int> points_in_cluster;
  } GClusterIndex;

  typedef struct
  {
    float* x;
    float* y;
    float* z;
    int size;
  } SamplePointListXYZ;

  GpuEuclideanCluster();

  void setInputPoints(float* x, float* y, float* z, int size);
  void setThreshold(double threshold);
  void setMinClusterPts(int min_cluster_pts);
  void setMaxClusterPts(int max_cluster_pts);
  void extractClustersOld();
  void extractClusters();
  void extractClusters2();
  std::vector<GClusterIndex> getOutput();

  SamplePointListXYZ generateSample();

  ~GpuEuclideanCluster();

  /* Added for GPU profiling */
  cudaEvent_t event_start, event_stop;
  
  int gid = 0;

  void start_profiling();
  void stop_profiling(int type);
  void write_data(int id, float time, int type);
  void write_dummy_line();
  void initialize_file(const char name[]);
  void close_file();


private:
  float *x_, *y_, *z_;
  int size_;
  double threshold_;
  int* cluster_indices_;
  int* cluster_indices_host_;
  int min_cluster_pts_;
  int max_cluster_pts_;
  int cluster_num_;

  void exclusiveScan(int* input, int ele_num, int* sum);
};

#endif
