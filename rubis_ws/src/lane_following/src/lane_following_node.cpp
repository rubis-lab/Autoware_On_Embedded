#include <ros/ros.h>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stddef.h>
#include <csignal>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>

#include "lane_following/color_grad_thresh.h"
#include "lane_following/debug.h"
#include "lane_following/find_lanes.h"
#include "lane_following/lane_base.h"
#include "lane_following/thread_base.h"
#include "lane_following/thread_manager.h"
#include "lane_following/warp.h"


void Run(tm_args *args) {
	ThreadManager<Warp, ColorGradThresh, FindLanes>* threadManager = NULL;

	threadManager = new ThreadManager<Warp, ColorGradThresh, FindLanes>(args);

	std::shared_ptr<ThreadMsg> msg = std::make_shared<ThreadMsg>();
	msg->taskMsg = ThreadBase::THREAD_MSG_START;

	if (threadManager && !threadManager->ThreadCreate()) {
		PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR, "Error creating thread manager\n")
	}

	if (threadManager) {
		std::signal(SIGINT,
				ThreadManager<Warp, ColorGradThresh, FindLanes>::stSignalHandler);
		threadManager->AddMsg(msg);
		threadManager->WaitForThread();
	}

	delete threadManager;
	ThreadBase::ResetThreadRegister();
}

int main(int argc, char** argv) {
	ros::init(argc, argv, "lane_following_node",
			ros::init_options::NoSigintHandler);

	tm_args args;
	args.videoFile = "";
	args.threadPoolSize = 4;
	args.pipelineInstNum = 4;
	args.maxFrameCnt = -1;
	args.speed = 2000;
	args.delay = 0;
	args.bParallel = true;
	args.bGpuAccel = false;
	args.bVerbose = false;

	Run(&args);
	return 0;
}
