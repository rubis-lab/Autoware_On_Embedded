#ifndef INCLUDE_LANE_FOLLOWING_FIND_LANES_H__
#define INCLUDE_LANE_FOLLOWING_FIND_LANES_H__

#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>
#include <opencv2/core/types.hpp>
#include <memory>
#include <vector>

#include "debug.h"
#include "lane_base.h"

class ThreadBase;
struct ThreadMsg;

class FindLanes: public LaneBase {
public:
	enum LANE_MODE {
		LANE_MODE_LEFT, LANE_MODE_RIGHT
	};
	enum PROC_STEP {
		PROC_STEP_FIND_NONZERO,
		PROC_STEP_HISTOGRAM,
		PROC_STEP_PREP_OUT_IMG,
#if DEBUG_ZONE_ALL_PROC_STEPS
		PROC_STEP_WINDOW_SEARCH_LEFT,
		PROC_STEP_CALC_POLY_LEFT,
		PROC_STEP_WINDOW_SEARCH_RIGHT,
		PROC_STEP_CALC_POLY_RIGHT,
#endif
		PROC_STEP_LEFT_LANE,
		PROC_STEP_RIGHT_LANE,
		PROC_STEP_MAKE_OUT_IMG,
		PROC_STEP_STEERING
	};
	struct Line {
		bool found;
		int xBase;
		std::vector<float> fit;
		double angle;
		virtual void clear() {
			found = false;
			xBase = -1;
			fit.clear();
			angle = 0;
		}
	};
	struct LaneHistory {
		Line rightLine;
		Line leftLine;
		float laneWidth;
	};
	FindLanes(int pipelineInstanceNum, bool bParallel, bool bGpuAccel,
			bool bVerbose);
	virtual ~FindLanes();
	virtual void Init() override;
	virtual void Deinit() override;
	virtual void setParams(LaneBase* obj) override;
	virtual void NextStep() override;
	virtual void Process(std::shared_ptr<ThreadMsg> &msg, ThreadBase* thread)
			override;
	virtual const char* getProcStepString(int proc_step) override;
	cv::Mat& getOutImg() {
		return outImg;
	}
	double getSpeed() const {
		return speed;
	}
	double getSteeringAngle() const {
		return steeringAngle;
	}
	double getMaxSpeed() const {
		return maxSpeed;
	}
	const LaneHistory& getLaneHistory() const {
		return laneHistory;
	}
	void setFrameDuration(long int frameDuration) {
		this->frameDuration = frameDuration;
	}
	void setSpeed(double speed) {
		this->speed = speed;
	}
	void setLaneHistory(const LaneHistory& laneHistory) {
		this->laneHistory = laneHistory;
	}
	bool isDetected() const {
		return bDetected;
	}
private:
	cv::Mat img;
	cv::Mat warpImg;
	cv::Mat outImg;
	std::vector<cv::Point> nonzero;
	std::vector<int> histogram;
	long int frameDuration;
	double speed;
	double steeringAngle;
	double maxSpeed;
	bool bDetected;
	LaneHistory laneHistory;
	struct Hyperparams {
		int windowsNum;
		int margin;
		unsigned minPix;
		int windowHeight;
		void clear() {
			windowsNum = 0;
			margin = 0;
			minPix = 0;
			windowHeight = 0;
		}
	} hyperparams;
	struct LaneCurrent: public Line {
		std::vector<cv::Point> pts;
		std::vector<cv::Point> fittedPts;
		cv::Mat outImg;
		virtual void clear() {
			pts.clear();
			fittedPts.clear();
			outImg.release();
			Line::clear();
		}
	} leftLine, rightLine;
	void FindNonZero();
	void PrepOutImg();
	void MakeOutImg();
	void Histogram();
	void WindowSearch(LANE_MODE mode);
	void CalcPoly(LANE_MODE mode, bool bPredicted);
	void PlotPoly(LANE_MODE mode, bool bPredicted);
	void FitPoly(const cv::Mat& src_x, const cv::Mat& src_y, cv::Mat& dst,
			int order);
	void Steering();
};

#endif /* INCLUDE_LANE_FOLLOWING_FIND_LANES_H__ */
