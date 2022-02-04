#ifndef INCLUDE_LANE_FOLLOWING_LANE_BASE_H_
#define INCLUDE_LANE_FOLLOWING_LANE_BASE_H_

#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>
#include <opencv2/core/types.hpp>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include "completed_item.h"
#include "debug.h"
#include "thread_base.h"
#include "time_profiling.h"

class LaneBase: public MsgObj, public TimeProfiling {
public:
	LaneBase(std::string moduleName, int pipelineInstanceNum, bool bParallel,
			bool bGpuAccel, bool bVerbose);
	virtual ~LaneBase();
	void Run(std::shared_ptr<ThreadMsg> &msg, ThreadBase* thread);
	virtual void Init() = 0;
	virtual void Deinit() = 0;
	virtual void setParams(LaneBase* obj);
	virtual void NextStep() = 0;
	virtual void Process(std::shared_ptr<ThreadMsg> &msg,
			ThreadBase* thread) = 0;
	const virtual char* getProcStepString(int proc_step) = 0;
	const std::string& getModuleName() const {
		return moduleName;
	}
	cv::Mat& getFrameImg() {
		return frameImg;
	}
	int getPipelineInstanceNum() const {
		return pipelineInstanceNum;
	}
	int getFrameIndex() const {
		return frameIndex;
	}
	int getProcStep() const {
		return procStep;
	}
	virtual cv::Mat& getInvPerspTf() {
		return invPerspTf;
	}
	std::chrono::system_clock::time_point& getStartTime() {
		return startTime;
	}
	void setFrameIndex(int frameIndex) {
		this->frameIndex = frameIndex;
	}
	void setProcStep(int procStep) {
		this->procStep = procStep;
	}
	void setFrameImg(cv::Mat& frameImg) {
		this->frameImg = frameImg.clone();
	}
	void setInvPerspTf(cv::Mat& invPerspTf) {
		this->invPerspTf = invPerspTf.clone();
	}
	void setStartTime(std::chrono::system_clock::time_point& startTime) {
		this->startTime = startTime;
	}

	CompletedItemList completedItemList;
protected:
	cv::Mat frameImg;
	cv::Mat invPerspTf;
	std::string moduleName;
	int pipelineInstanceNum;
	bool bParallel;
	bool bGpuAccel;
	bool bVerbose;
	int frameIndex;
	int procStep;
	TASK_STATE taskState;
	std::chrono::system_clock::time_point startTime;
private:
	std::recursive_mutex timestampsLock;
};

class WarpBase: public LaneBase {
public:
	enum PROC_STEP {
		PROC_STEP_WARP
	};
	WarpBase(int pipelineInstanceNum, bool bParallel, bool bGpuAccel,
			bool bVerbose);
	virtual ~WarpBase();
	virtual void Init() override;
	virtual void Deinit() override;
	virtual void setParams(LaneBase* obj) override;
	virtual void NextStep() override;
	virtual void Process(std::shared_ptr<ThreadMsg> &msg, ThreadBase* thread)
			override;
	const virtual char* getProcStepString(int proc_step) override;
protected:
	cv::Mat perspTf;
	cv::Point2f src[4], dst[4];
	virtual void RunWarp() = 0;
};

class ColorGradThreshBase: public LaneBase {
public:
	enum SPLIT_MODE {
		SPLIT_MODE_BGR, SPLIT_MODE_HLS
	};
	enum THRESH_MODE {
		THRESH_MODE_RED, THRESH_MODE_SAT, THRESH_MODE_ABS_SOBELX
	};
	enum PROC_STEP {
		PROC_STEP_SPLIT_BGR,
#if DEBUG_ZONE_ALL_PROC_STEPS
		PROC_STEP_BGR_TO_HLS,
#endif
		PROC_STEP_SPLIT_HLS,
		PROC_STEP_THRESH_RED,
		PROC_STEP_THRESH_SAT,
		PROC_STEP_SOBEL_X,
#if DEBUG_ZONE_ALL_PROC_STEPS
		PROC_STEP_ABS_SOBEL_X,
#endif
		PROC_STEP_THRESH_SOBEL_X,
		PROC_STEP_COMB_THRESH
	};
	ColorGradThreshBase(int pipelineInstanceNum, bool bParallel, bool bGpuAccel,
			bool bVerbose);
	virtual ~ColorGradThreshBase();
	virtual void Init() override;
	virtual void Deinit() override;
	virtual void setParams(LaneBase* obj) override;
	virtual void NextStep() override;
	virtual void Process(std::shared_ptr<ThreadMsg> &msg, ThreadBase* thread)
			override;
	const virtual char* getProcStepString(int proc_step) override;
	cv::Mat& getOutImg() {
		return outImg;
	}
	cv::Mat& getWarpImg() {
		return warpImg;
	}
protected:
	cv::Mat outImg;
	cv::Mat warpImg;
	struct Thresholds {
		int red[2];
		int sat[2];
		int sobelx[2];
	} thresh;
	virtual void SplitChannel(SPLIT_MODE mode) = 0;
	virtual void CvtBGR2HLS() = 0;
	virtual void ThresholdBinary(THRESH_MODE mode) = 0;
	virtual void Sobelx() = 0;
	virtual void AbsSobelx() = 0;
	virtual void CombBinaries() = 0;
};

#endif /* INCLUDE_LANE_FOLLOWING_LANE_BASE_H_ */
