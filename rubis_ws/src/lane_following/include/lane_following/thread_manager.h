#ifndef INCLUDE_LANE_FOLLOWING_THREAD_MANAGER_H__
#define INCLUDE_LANE_FOLLOWING_THREAD_MANAGER_H__

#include "debug.h"

#if DEBUG_ZONE_ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#endif

#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <cstdio>
#include <list>
#include <memory>

#include "find_lanes.h"
#include "thread_base.h"
#include "thread_worker.h"

struct tm_args {
	cv::String videoFile;
	int threadPoolSize;
	int pipelineInstNum;
	int maxFrameCnt;
	double speed;
	long int delay;
	bool bParallel;
	bool bGpuAccel;
	bool bVerbose;
	tm_args() {
		videoFile = "project_video.mp4";
		threadPoolSize = 8;
		pipelineInstNum = 1;
		maxFrameCnt = 100;
		speed = 1000;
		delay = 0;
		bParallel = false;
		bGpuAccel = false;
		bVerbose = false;
	}
};

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
class ThreadManager: public ThreadBase {
public:
	enum {
		MAX_PIPELINE_INST_NUM = 16
	};
	ThreadManager(tm_args *args);
	virtual ~ThreadManager();
	void Start();
	void Stop();
	void GetNextFrame();
	void ShowFrame(const cv::Mat &img);
	bool StartTask(ThreadWorker* thread, CompletedItem &complete_item,
			std::shared_ptr<LaneBase> &obj);
	void CompleteTask(std::shared_ptr<LaneBase> *obj);
	bool StartWarp();
	bool StartColorGradThresh(std::shared_ptr<WARP> &warp);
	bool StartFindLanes(std::shared_ptr<COLOR_GRAD_THRESH> &colorGradThresh);
	virtual void ProcessMsg(std::shared_ptr<ThreadMsg> &msg) override;
	virtual bool PreWorkInit() override;
	virtual bool PostWorkDeinit() override;
	void PrintAvgFuncDurations();
	long int getAvgDuration();
	double getAvgSpeed();
	ThreadWorker* getBusyThread(ThreadId threadId) {
		for (auto it = busyList.begin(); it != busyList.end(); it++) {
			if ((*it)->GetThreadId() == threadId) {
				ThreadWorker* r = *it;
				busyList.erase(it);
				return r;
			}
		}
		return NULL;
	}
	static void stSignalHandler(int signum) {
		threadManager->SignalHandler(signum);
	}
	void SignalHandler(int signum) {
#if DEBUG_ZONE_ROS
		MotorPublisher(0, 0.5);
#endif
		EndThread();
	}
#if DEBUG_ZONE_ROS
	void zedCallback(const sensor_msgs::ImageConstPtr& img);
	void MotorPublisher(double speed, double angle);
#endif

private:
	struct ReadyTask {
		std::shared_ptr<LaneBase> obj;
		ReadyTask(){
		}
		ReadyTask(const ReadyTask &d){
			obj = d.obj;
		}
		bool operator()(const ReadyTask &first, const ReadyTask &second){
			return first.obj->getFrameIndex() < second.obj->getFrameIndex();
		}
	};


	static ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>* threadManager;
	tm_args args;

	cv::VideoCapture videoCap;
	cv::Mat frameImg;

	std::list<ThreadWorker*> freeList; // list of free threads
	std::list<ThreadWorker*> busyList; // list of busy threads

	std::shared_ptr<WARP> warp[MAX_PIPELINE_INST_NUM];
	std::shared_ptr<COLOR_GRAD_THRESH> colorGradThresh[MAX_PIPELINE_INST_NUM];
	std::shared_ptr<FIND_LANES> findLanes[MAX_PIPELINE_INST_NUM];

	int frameCnt;
	int pipelineFrameCnt;
	int processedFrameCnt;

	bool bWarpTaskReady;
	bool bColorGradThreshTaskReady;
	bool bFindLanesTaskReady;
	bool bStartWarpTaskReady;

	std::chrono::system_clock::time_point frameEndTime;
	std::chrono::system_clock::time_point warpEndTime;
	long int frameDuration; // usec
	long int procDuration; // usec
	std::vector<long int> frameDurations;
	std::vector<long int> procDurations;
	std::vector<double> speeds;
	double lastAngle;
	FindLanes::LaneHistory laneHistory;

	cv::VideoWriter outVideoWr;
#if DEBUG_ZONE_RAW_VIDEO
	cv::VideoWriter rawVideoWr;
#endif
#if DEBUG_ZONE_ROS
	cv::Mat zedImg;
	ros::NodeHandle nh;
	ros::Publisher speedPub;
	ros::Publisher servoPub;
	ros::Publisher durationPub;

	image_transport::ImageTransport it;
	image_transport::Subscriber zedSub;
#endif
};

#endif /* INCLUDE_LANE_FOLLOWING_THREAD_MANAGER_H__ */
