#include "lane_following/lane_base.h"

#include <opencv2/imgproc.hpp>
#include <list>
#include <vector>

LaneBase::LaneBase(std::string moduleName, int pipelineInstanceNum,
		bool bParallel, bool bGpuAccel, bool bVerbose) :
		moduleName(moduleName), pipelineInstanceNum(pipelineInstanceNum), bParallel(
				bParallel), bGpuAccel(bGpuAccel), bVerbose(bVerbose), frameIndex(
				-1), procStep(-1), taskState(TASK_STATE_UNDEFINED) {
}

LaneBase::~LaneBase() {
}

void LaneBase::Run(std::shared_ptr<ThreadMsg> &msg, ThreadBase* thread) {
	const char* name = getProcStepString(msg->procStep);
	auto time0 = std::chrono::high_resolution_clock::now();
	Process(msg, thread);
	auto time1 = std::chrono::high_resolution_clock::now();
	{
		std::lock_guard<std::recursive_mutex> lock(timestampsLock);
		timestamps.push_back(Timestamp(name, time0));
		timestamps.push_back(Timestamp(name, time1));
	}
}

void LaneBase::setParams(LaneBase* obj) {
	if (obj) {
		frameImg = obj->getFrameImg().clone();
		invPerspTf = obj->getInvPerspTf().clone();
		frameIndex = obj->getFrameIndex();
		startTime = obj->getStartTime();
	}
	taskState = TASK_STATE_INITIALIZED;
}

WarpBase::WarpBase(int pipelineInstanceNum, bool bParallel, bool bGpuAccel,
		bool bVerbose) :
		LaneBase("Warp", pipelineInstanceNum, bParallel, bGpuAccel, bVerbose) {
	msgObjType = MSG_OBJ_TYPE_WARP;
}

WarpBase::~WarpBase() {
	Deinit();
}

void WarpBase::Init() {
	int pos_x_lower = 280;
	int pos_y_lower = 670;
	int pos_x_upper = 504;
	int pos_y_upper = 470;
	int center_lower = 778;
	int center_upper = 718;

	src[0] = cv::Point(pos_x_upper, pos_y_upper);
	src[1] = cv::Point(2 * center_upper - pos_x_upper, pos_y_upper);
	src[2] = cv::Point(2 * center_lower - pos_x_lower, pos_y_lower);
	src[3] = cv::Point(pos_x_lower, pos_y_lower);

	int persp_center = frameImg.cols / 2;
	int persp_height = frameImg.rows;
	int persp_margin = frameImg.cols / 3;
	dst[0] = cv::Point2f(persp_center - persp_margin, 0); // Top left
	dst[1] = cv::Point2f(persp_center + persp_margin, 0); // Top right
	dst[2] = cv::Point2f(persp_center + persp_margin, persp_height); // Bottom right
	dst[3] = cv::Point2f(persp_center - persp_margin, persp_height); // Bottom left

	perspTf = getPerspectiveTransform(src, dst);
	invPerspTf = getPerspectiveTransform(dst, src);
}

void WarpBase::Deinit() {
	//frameImg.release();
}

void WarpBase::setParams(LaneBase* obj) {
	Deinit();
	Init();
	procStep = PROC_STEP_WARP;
	completedItemList.clear();
	completedItemList.addItem(procStep, TASK_STATE_INITIALIZED);
	LaneBase::setParams(this);
}

void WarpBase::NextStep() {
	completedItemList.rmCompleted();
	if (completedItemList.empty()) {
		procStep = -1;
		taskState = TASK_STATE_UNDEFINED;
	}
}

void WarpBase::Process(std::shared_ptr<ThreadMsg> &msg, ThreadBase* thread) {
	PRINT_DEBUG_MSG((DEBUG_ZONE_WARP || DEBUG_ZONE_PROCESS),
			"++[%ld]WarpBase[%d]::Process, procStep = %s, frameIndex = %d, img=%dX%d\n",
			thread ? thread->GetThreadId() : -1, pipelineInstanceNum,
			getProcStepString(msg->procStep), getFrameIndex(), frameImg.rows,
			frameImg.cols);
	if (msg->procStep == PROC_STEP_WARP) {
		RunWarp();
	}
	PRINT_DEBUG_MSG(DEBUG_ZONE_WARP || DEBUG_ZONE_PROCESS,
			"--[%ld]WarpBase::Process, procStep = %s, frameIndex = %d\n",
			thread ? thread->GetThreadId() : -1,
			getProcStepString(msg->procStep), getFrameIndex());
}

const char* WarpBase::getProcStepString(int proc_step) {
	switch (proc_step) {
	case PROC_STEP_WARP:
		return "Warp";
	default:
		return "";
	}
}

ColorGradThreshBase::ColorGradThreshBase(int pipelineInstanceNum,
		bool bParallel, bool bGpuAccel, bool bVerbose) :
		LaneBase("ColorGradThresh", pipelineInstanceNum, bParallel, bGpuAccel,
				bVerbose) {
	msgObjType = MSG_OBJ_TYPE_COLOR_GRAD_THRESH;
	// Thresholds
#if 0
	thresh.red[0] = 170;
	thresh.red[1] = 255;
	thresh.sat[0] = 170;
	thresh.sat[1] = 255;
	thresh.sobelx[0] = 20;
	thresh.sobelx[1] = 100;
#else
	thresh.red[0] = 190;
	thresh.red[1] = 240;
	thresh.sat[0] = 170;
	thresh.sat[1] = 250;
	thresh.sobelx[0] = 155;
	thresh.sobelx[1] = 255;
#endif
}

ColorGradThreshBase::~ColorGradThreshBase() {
	Deinit();
}

void ColorGradThreshBase::Init() {
}

void ColorGradThreshBase::Deinit() {
	outImg.release();
	warpImg.release();
}

void ColorGradThreshBase::setParams(LaneBase* obj) {
	completedItemList.clear();
	if (bParallel) { // && !bGpuAccel ?
		completedItemList.addItem(PROC_STEP_SPLIT_BGR, TASK_STATE_INITIALIZED);
		completedItemList.addItem(PROC_STEP_SPLIT_HLS, TASK_STATE_INITIALIZED);
		procStep = PROC_STEP_THRESH_RED;
	} else {
		completedItemList.addItem(PROC_STEP_SPLIT_BGR, TASK_STATE_INITIALIZED);
		procStep = PROC_STEP_SPLIT_BGR;
	}
	LaneBase::setParams(obj);
}

void ColorGradThreshBase::NextStep() {
	completedItemList.rmCompleted();
	if (completedItemList.empty()) {
		if (bParallel) { // && !bGpuAccel ?
			if (procStep == PROC_STEP_THRESH_RED) {
				completedItemList.addItem(PROC_STEP_THRESH_RED,
						TASK_STATE_INITIALIZED);
				completedItemList.addItem(PROC_STEP_THRESH_SAT,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_SOBEL_X;
			} else if (procStep == PROC_STEP_SOBEL_X) {
				completedItemList.addItem(PROC_STEP_SOBEL_X,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_THRESH_SOBEL_X;
			} else if (procStep == PROC_STEP_THRESH_SOBEL_X) {
				completedItemList.addItem(PROC_STEP_THRESH_SOBEL_X,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_COMB_THRESH;
			}	else if (procStep == PROC_STEP_COMB_THRESH) {
				completedItemList.addItem(PROC_STEP_COMB_THRESH,
						TASK_STATE_INITIALIZED);
				procStep = -1;
			} else {
				taskState = TASK_STATE_UNDEFINED;
			}
		} else {
			if (procStep == PROC_STEP_SPLIT_BGR) {
				completedItemList.addItem(PROC_STEP_THRESH_RED,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_THRESH_RED;
			} else if (procStep == PROC_STEP_THRESH_RED) {
#if DEBUG_ZONE_ALL_PROC_STEPS
				completedItemList.addItem(PROC_STEP_BGR_TO_HLS,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_BGR_TO_HLS;

			} else if (procStep == PROC_STEP_BGR_TO_HLS) {
#endif
				completedItemList.addItem(PROC_STEP_SPLIT_HLS,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_SPLIT_HLS;
			} else if (procStep == PROC_STEP_SPLIT_HLS) {
				completedItemList.addItem(PROC_STEP_THRESH_SAT,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_THRESH_SAT;
			} else if (procStep == PROC_STEP_THRESH_SAT) {
				completedItemList.addItem(PROC_STEP_SOBEL_X,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_SOBEL_X;
			} else if (procStep == PROC_STEP_SOBEL_X) {
#if DEBUG_ZONE_ALL_PROC_STEPS
				completedItemList.addItem(PROC_STEP_ABS_SOBEL_X,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_ABS_SOBEL_X;
			} else if (procStep == PROC_STEP_ABS_SOBEL_X) {
#endif
				completedItemList.addItem(PROC_STEP_THRESH_SOBEL_X,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_THRESH_SOBEL_X;
			} else if (procStep == PROC_STEP_THRESH_SOBEL_X) {
				completedItemList.addItem(PROC_STEP_COMB_THRESH,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_COMB_THRESH;
			} else {
				taskState = TASK_STATE_UNDEFINED;
			}
		}
	}
}

void ColorGradThreshBase::Process(std::shared_ptr<ThreadMsg> &msg,
		ThreadBase* thread) {
	PRINT_DEBUG_MSG((DEBUG_ZONE_COLOR_GRAD_THRESH || DEBUG_ZONE_PROCESS),
			"++[%ld]ColorGradThreshBase[%d]::Process, procStep = %s, frameIndex = %d\n",
			thread ? thread->GetThreadId() : -1, pipelineInstanceNum,
			getProcStepString(msg->procStep), getFrameIndex());

	if (msg->procStep == PROC_STEP_SPLIT_BGR) {
		// Separate BGR color channels
		SplitChannel(SPLIT_MODE_BGR);
#if DEBUG_ZONE_ALL_PROC_STEPS
	} else if (msg->procStep == PROC_STEP_BGR_TO_HLS) {
		// Convert to HLS color space
		CvtBGR2HLS();
#endif
	} else if (msg->procStep == PROC_STEP_SPLIT_HLS) {
#if !DEBUG_ZONE_ALL_PROC_STEPS
		// Convert to HLS color space
		CvtBGR2HLS();
#endif
		// Separate HLS color channels
		SplitChannel(SPLIT_MODE_HLS);
	} else if (msg->procStep == PROC_STEP_THRESH_RED) {
		// Threshold red channel
		ThresholdBinary(THRESH_MODE_RED);
	} else if (msg->procStep == PROC_STEP_THRESH_SAT) {
		// Threshold saturation channel
		ThresholdBinary(THRESH_MODE_SAT);
	} else if (msg->procStep == PROC_STEP_SOBEL_X) {
		// Take the derivative in x
		Sobelx();
#if DEBUG_ZONE_ALL_PROC_STEPS
	} else if (msg->procStep == PROC_STEP_ABS_SOBEL_X) {
#endif
		// Absolute x derivative to accentuate lines away from horizontal
		AbsSobelx();
	} else if (msg->procStep == PROC_STEP_THRESH_SOBEL_X) {
		// Threshold x gradient
		ThresholdBinary(THRESH_MODE_ABS_SOBELX);
	} else if (msg->procStep == PROC_STEP_COMB_THRESH) {
		// Combine three binary thresholds
		CombBinaries();
	}
	PRINT_DEBUG_MSG((DEBUG_ZONE_COLOR_GRAD_THRESH || DEBUG_ZONE_PROCESS),
			"--[%ld]ColorGradThreshBase[%d]::Process, procStep = %s, frameIndex = %d\n",
			thread ? thread->GetThreadId() : -1, pipelineInstanceNum,
			getProcStepString(msg->procStep), getFrameIndex());
}

const char* ColorGradThreshBase::getProcStepString(int proc_step) {
	switch (proc_step) {
	case PROC_STEP_SPLIT_BGR:
		return "SplitBGR";
	case PROC_STEP_THRESH_RED:
		return "ThreshRed";
#if DEBUG_ZONE_ALL_PROC_STEPS
	case PROC_STEP_BGR_TO_HLS:
		return "CvtBGR2HLS";
#endif
	case PROC_STEP_SPLIT_HLS:
		return "SplitHLS";
	case PROC_STEP_THRESH_SAT:
		return "ThreshSat";
	case PROC_STEP_SOBEL_X:
		return "Sobelx";
#if DEBUG_ZONE_ALL_PROC_STEPS
	case PROC_STEP_ABS_SOBEL_X:
		return "AbsSobelx";
#endif
	case PROC_STEP_THRESH_SOBEL_X:
		return "ThreshSobelx";
	case PROC_STEP_COMB_THRESH:
		return "CombThresh";
	default:
		return "";
	}
}
