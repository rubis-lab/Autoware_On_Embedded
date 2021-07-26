#include "lane_following/find_lanes.h"

#include <opencv2/core/base.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/operations.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <list>
#include <sstream>
#include <string>

#include "lane_following/completed_item.h"
#include "lane_following/thread_base.h"

FindLanes::FindLanes(int pipelineInstanceNum, bool bParallel, bool bGpuAccel,
		bool bVerbose) :
		LaneBase("FindLanes", pipelineInstanceNum, bParallel, bGpuAccel,
				bVerbose) {
	msgObjType = MSG_OBJ_TYPE_FIND_LANES;
	frameDuration = 1000000;
	speed = 1000;
	steeringAngle = 0.5;
	bDetected = false;
	maxSpeed = 0.75;
}

FindLanes::~FindLanes() {
	Deinit();
}

void FindLanes::Init() {
	// HYPERPARAMETERS
	// Number of sliding windows
	hyperparams.windowsNum = 32;
	// Width of the windows +/- margin
	hyperparams.margin = 100;
	// Minimum number of pixels found to re-center window
	hyperparams.minPix = 50;
	// Height of windows - based on windowsNum above and image shape
	hyperparams.windowHeight = img.rows / hyperparams.windowsNum;
}

void FindLanes::Deinit() {
	img.release();
	warpImg.release();
	outImg.release();
	nonzero.clear();
	histogram.clear();
	hyperparams.clear();
	leftLine.clear();
	rightLine.clear();
	steeringAngle = 0.5;
	bDetected = false;
}

void FindLanes::setParams(LaneBase* obj) {
	Deinit();
	ColorGradThreshBase* colorGradTf = dynamic_cast<ColorGradThreshBase*>(obj);
	if (colorGradTf) {
		img = colorGradTf->getOutImg().clone();
		warpImg = colorGradTf->getWarpImg().clone();
		Init();
		completedItemList.clear();
		if (bParallel) {
			completedItemList.addItem(PROC_STEP_FIND_NONZERO,
					TASK_STATE_INITIALIZED);
			completedItemList.addItem(PROC_STEP_HISTOGRAM,
					TASK_STATE_INITIALIZED);
			completedItemList.addItem(PROC_STEP_PREP_OUT_IMG,
					TASK_STATE_INITIALIZED);
			procStep = PROC_STEP_LEFT_LANE;
		} else {
			completedItemList.addItem(PROC_STEP_FIND_NONZERO,
					TASK_STATE_INITIALIZED);
			procStep = PROC_STEP_FIND_NONZERO;
		}
		LaneBase::setParams(obj);
	}
}

void FindLanes::NextStep() {
	completedItemList.rmCompleted();
	if (completedItemList.empty()) {
		if (bParallel) {
			if (procStep == PROC_STEP_LEFT_LANE) {
				completedItemList.addItem(PROC_STEP_LEFT_LANE,
						TASK_STATE_INITIALIZED);
				completedItemList.addItem(PROC_STEP_RIGHT_LANE,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_MAKE_OUT_IMG;
			} else if (procStep == PROC_STEP_MAKE_OUT_IMG) {
				completedItemList.addItem(PROC_STEP_MAKE_OUT_IMG,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_STEERING;
			} else if (procStep == PROC_STEP_STEERING) {
				completedItemList.addItem(PROC_STEP_STEERING,
						TASK_STATE_INITIALIZED);
				procStep = -1;
			} else {
				taskState = TASK_STATE_UNDEFINED;
			}
		} else {
			if (procStep == PROC_STEP_FIND_NONZERO) {
				completedItemList.addItem(PROC_STEP_HISTOGRAM,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_HISTOGRAM;
			} else if (procStep == PROC_STEP_HISTOGRAM) {
				completedItemList.addItem(PROC_STEP_PREP_OUT_IMG,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_PREP_OUT_IMG;
			} else if (procStep == PROC_STEP_PREP_OUT_IMG) {
#if DEBUG_ZONE_ALL_PROC_STEPS
				completedItemList.addItem(PROC_STEP_WINDOW_SEARCH_LEFT,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_WINDOW_SEARCH_LEFT;
			} else if (procStep == PROC_STEP_WINDOW_SEARCH_LEFT) {
				completedItemList.addItem(PROC_STEP_CALC_POLY_LEFT,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_CALC_POLY_LEFT;
			} else if (procStep == PROC_STEP_CALC_POLY_LEFT) {
				completedItemList.addItem(PROC_STEP_WINDOW_SEARCH_RIGHT,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_WINDOW_SEARCH_RIGHT;
			} else if (procStep == PROC_STEP_WINDOW_SEARCH_RIGHT) {
				completedItemList.addItem(PROC_STEP_CALC_POLY_RIGHT,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_CALC_POLY_RIGHT;
			} else if (procStep == PROC_STEP_CALC_POLY_RIGHT) {
#else
				completedItemList.addItem(PROC_STEP_LEFT_LANE,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_LEFT_LANE;
			} else if (procStep == PROC_STEP_LEFT_LANE) {
				completedItemList.addItem(PROC_STEP_RIGHT_LANE,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_RIGHT_LANE;
			} else if (procStep == PROC_STEP_RIGHT_LANE) {
#endif
				completedItemList.addItem(PROC_STEP_MAKE_OUT_IMG,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_MAKE_OUT_IMG;
			} else if (procStep == PROC_STEP_MAKE_OUT_IMG) {
				completedItemList.addItem(PROC_STEP_STEERING,
						TASK_STATE_INITIALIZED);
				procStep = PROC_STEP_STEERING;
			} else {
				taskState = TASK_STATE_UNDEFINED;
			}
		}
	}
}

void FindLanes::Process(std::shared_ptr<ThreadMsg> &msg, ThreadBase* thread) {
	PRINT_DEBUG_MSG((DEBUG_ZONE_FIND_LANES || DEBUG_ZONE_PROCESS),
			"++[%ld]FindLanes[%d]::Process, procStep = %s, frameIndex = %d\n",
			thread ? thread->GetThreadId() : -1, pipelineInstanceNum,
			getProcStepString(msg->procStep), getFrameIndex());

	if (msg->procStep == PROC_STEP_FIND_NONZERO) {
		FindNonZero();
	} else if (msg->procStep == PROC_STEP_HISTOGRAM) {
		Histogram();
	} else if (msg->procStep == PROC_STEP_PREP_OUT_IMG) {
		PrepOutImg();
#if DEBUG_ZONE_ALL_PROC_STEPS
	} else if (msg->procStep == PROC_STEP_WINDOW_SEARCH_LEFT) {
		if (leftLine.found) {
			if (bVerbose) {
				leftLine.outImg = cv::Mat::zeros(outImg.size(),
						outImg.type());
			}
			WindowSearch(LANE_MODE_LEFT);
		}
	} else if (msg->procStep == PROC_STEP_CALC_POLY_LEFT) {
		if (leftLine.found) {
			CalcPoly(LANE_MODE_LEFT, false);
		}
	} else if (msg->procStep == PROC_STEP_WINDOW_SEARCH_RIGHT) {
		if (rightLine.found) {
			if (bVerbose) {
				rightLine.outImg = cv::Mat::zeros(outImg.size(),
						outImg.type());
			}
			WindowSearch(LANE_MODE_RIGHT);
		}
	} else if (msg->procStep == PROC_STEP_CALC_POLY_RIGHT) {
		if (rightLine.found) {
			CalcPoly(LANE_MODE_RIGHT, false);
		}
#else
	} else if (msg->procStep == PROC_STEP_LEFT_LANE) {
		if (leftLine.found) {
			if (bVerbose) {
				leftLine.outImg = cv::Mat::zeros(outImg.size(), outImg.type());
			}
			WindowSearch(LANE_MODE_LEFT);
		}
		if (leftLine.found)
			CalcPoly(LANE_MODE_LEFT, false);
	} else if (msg->procStep == PROC_STEP_RIGHT_LANE) {
		if (rightLine.found) {
			if (bVerbose) {
				rightLine.outImg = cv::Mat::zeros(outImg.size(), outImg.type());
			}
			WindowSearch(LANE_MODE_RIGHT);
		}
		if (rightLine.found)
			CalcPoly(LANE_MODE_RIGHT, false);
#endif
	} else if (msg->procStep == PROC_STEP_MAKE_OUT_IMG) {
		MakeOutImg();
	} else if (msg->procStep == PROC_STEP_STEERING) {
		Steering();
	}
	PRINT_DEBUG_MSG((DEBUG_ZONE_FIND_LANES || DEBUG_ZONE_PROCESS),
			"--[%ld]FindLanes[%d]::Process, procStep = %s, frameIndex = %d\n",
			thread ? thread->GetThreadId() : -1, pipelineInstanceNum,
			getProcStepString(msg->procStep), getFrameIndex());

}

const char* FindLanes::getProcStepString(int proc_step) {
	switch (proc_step) {
	case PROC_STEP_FIND_NONZERO:
		return "FindNonZero";
	case PROC_STEP_HISTOGRAM:
		return "Histogram";
	case PROC_STEP_PREP_OUT_IMG:
		return "PrepOutImg";
	case PROC_STEP_LEFT_LANE:
		return "LeftLane";
	case PROC_STEP_RIGHT_LANE:
		return "RightLane";
#if DEBUG_ZONE_ALL_PROC_STEPS
	case PROC_STEP_WINDOW_SEARCH_LEFT:
		return "WindowSearchLeft";
	case PROC_STEP_CALC_POLY_LEFT:
		return "CalcPolyLeft";
	case PROC_STEP_WINDOW_SEARCH_RIGHT:
		return "WindowSearchRight";
	case PROC_STEP_CALC_POLY_RIGHT:
		return "CalcPolyRight";
#endif
	case PROC_STEP_MAKE_OUT_IMG:
		return "MakeOutImg";
	case PROC_STEP_STEERING:
		return "Steering";
	default:
		return "Undefined";
	}
}

void FindLanes::FindNonZero() {
	// Identify the x and y positions of all nonzero pixels in the image
	findNonZero(img, nonzero);
	reverse(nonzero.begin(), nonzero.end());
}

void FindLanes::PrepOutImg() {
	// Create an output image to draw on and  visualize the result
	if (bVerbose) {
#if DEBUG_ZONE_OUT_IMG
		std::vector<cv::Mat> channels;
		channels.push_back(img);
		channels.push_back(img);
		channels.push_back(img);
		merge(channels, outImg);
#else
		outImg = warpImg.clone();
#endif
	}
}

void FindLanes::MakeOutImg() {
	if (bVerbose) {
#if DEBUG_ZONE_OUT_IMG
		if (rightLine.found && leftLine.found) {
			addWeighted(rightLine.outImg, 1, leftLine.outImg, 1, 0, outImg);
		} else if (rightLine.found) {
			addWeighted(rightLine.outImg, 1, outImg, 0, 0, outImg);
		} else if (leftLine.found) {
			addWeighted(leftLine.outImg, 1, outImg, 0, 0, outImg);
		}
#endif
	}
}

void FindLanes::Histogram() {
	// Take a histogram of the bottom half of the image
	cv::Mat bottom_half = img(
			cv::Rect(cv::Point(0, img.rows / 2),
					cv::Point(img.cols, img.rows)));
	cv::Mat hist;
	reduce(bottom_half, hist, 0, CV_REDUCE_SUM, CV_32SC1);
	hist.row(0).copyTo(histogram);

	// Find the peak of the left and right halves of the histogram
	// These will be the starting cv::Point for the left and right lines
	int midPoint = histogram.size() / 2;
	leftLine.xBase = (max_element(histogram.begin(),
			histogram.begin() + midPoint)) - histogram.begin();
	rightLine.xBase = (max_element(histogram.begin() + midPoint,
			histogram.end())) - histogram.begin();
	if (histogram[leftLine.xBase] > 0) {
		leftLine.found = true;
	}
	if (histogram[rightLine.xBase] > 0) {
		rightLine.found = true;
	}
}

void FindLanes::WindowSearch(LANE_MODE mode) {
	// Current positions to be updated for each window in windowsNum
	unsigned y_current = 0;
	int x_current = (mode == LANE_MODE_LEFT) ? leftLine.xBase : rightLine.xBase;
	int count = 0;
	// Create empty list to receive lane pixel indices
	std::vector<cv::Point> lane_pts;
	bool bFound = false;

	// Step through the windows one by one
	for (int window = 0; window < hyperparams.windowsNum; window++) {
		// Identify window boundaries in x and y
		int win_y_low = img.rows - (window + 1) * hyperparams.windowHeight;
		int win_y_high = img.rows - window * hyperparams.windowHeight;
		int win_x_low = x_current - hyperparams.margin;
		int win_x_high = x_current + hyperparams.margin;

		// Draw the window on the visualization image
		if (bVerbose) {
			cv::Mat &img_src =
					(mode == LANE_MODE_LEFT) ?
							leftLine.outImg : rightLine.outImg;
			cv::Rect rect = cv::Rect(cv::Point(win_x_low, win_y_low),
					cv::Point(win_x_high, win_y_high));
			rectangle(img_src, rect, cv::Scalar(255, 0, 0), 2);
		}

		// Identify the nonzero pixels in x and y within the window
		std::vector<int> good_inds;
		for (unsigned i = y_current; i < nonzero.size(); i++) {
			if ((nonzero[i].y >= win_y_low) && (nonzero[i].y < win_y_high)) {
				if ((nonzero[i].x >= win_x_low)
						&& (nonzero[i].x < win_x_high)) {
					good_inds.push_back(i);
				}
#if DEBUG_ZONE_OUT_IMG
				// Color in lane regions
				if (bVerbose) {
					cv::Mat &img_src =
							(mode == LANE_MODE_LEFT) ?
									leftLine.outImg : rightLine.outImg;
					img_src.at<cv::Vec3b>(cv::Point(nonzero[i].x, nonzero[i].y)) =
							cv::Vec3b(0, 255, 0);
				}
#endif
			} else {
				y_current = i;
				break;
			}
		}

		// If you found > minPix pixels, re-center next window
		if (good_inds.size() > hyperparams.minPix) {
			int sum_x = 0;
			int sum_y = 0;
			for (unsigned i = 0; i < good_inds.size(); i++) {
				//lane_pts.push_back(cv::Point(nonzero[good_inds[i]].x, nonzero[good_inds[i]].y));
				sum_x += nonzero[good_inds[i]].x;
				sum_y += nonzero[good_inds[i]].y;
			}
			int x_avg = sum_x / good_inds.size();
			int y_avg = sum_y / good_inds.size();
			lane_pts.push_back(cv::Point(x_avg, y_avg));
			x_current = x_avg;
			count = 0;
		} else {
			count++;
			if (count == hyperparams.windowsNum / 4) {
				break;
			}
		}
	}

	if (lane_pts.size() > 3) {
		bFound = true;
	}

	switch (mode) {
	case LANE_MODE_LEFT:
		leftLine.found = bFound;
		leftLine.pts = lane_pts;
		break;
	case LANE_MODE_RIGHT:
		rightLine.found = bFound;
		rightLine.pts = lane_pts;
		break;
	}
}

void FindLanes::CalcPoly(LANE_MODE mode, bool bPredicted) {
	std::vector<cv::Point> &lane_pts =
			(mode == LANE_MODE_LEFT) ? leftLine.pts : rightLine.pts;

	// Extract left and right line pixel positions
	std::vector<float> x_pts, y_pts, fit;
	for (auto it : lane_pts) {
		x_pts.push_back(it.x);
		y_pts.push_back(it.y);
	}

	// Fit a second order polynomial
	cv::Mat src_x = cv::Mat(x_pts.size(), 1, CV_32F, x_pts.data());
	cv::Mat src_y = cv::Mat(y_pts.size(), 1, CV_32F, y_pts.data());
	cv::Mat dst = cv::Mat(3, 1, CV_32F);
	FitPoly(src_y, src_x, dst, 2);
	dst.col(0).copyTo(fit);

	// Calculate polynomial
	std::vector<cv::Point> fittedpts;
	for (float i = img.rows - 1; i >= 0; i--) {
		fittedpts.push_back(
				cv::Point(fit[0] * pow(i, 2) + fit[1] * i + fit[2], i));
	}

	switch (mode) {
	case LANE_MODE_LEFT:
		leftLine.fit = fit;
		leftLine.fittedPts = fittedpts;
		break;
	case LANE_MODE_RIGHT:
		rightLine.fit = fit;
		rightLine.fittedPts = fittedpts;
		break;
	}
#if DEBUG_ZONE_OUT_IMG
	//PlotPoly(mode, bPredicted);
#endif
}

void FindLanes::PlotPoly(LANE_MODE mode, bool bPredicted) {
	std::vector<cv::Point> &fittedpts =
			(mode == LANE_MODE_LEFT) ? leftLine.fittedPts : rightLine.fittedPts;
	// Plot polynomial on the lane line
	if (bVerbose) {
		const cv::Point *pts = (const cv::Point*) fittedpts.data();
		int npts = fittedpts.size();
		if (bPredicted) {
			polylines(outImg, &pts, &npts, 1, false, cv::Scalar(0, 0, 255), 2);
		} else {
			cv::Mat &img_src =
					(mode == LANE_MODE_LEFT) ?
							leftLine.outImg : rightLine.outImg;
			polylines(img_src, &pts, &npts, 1, false, cv::Scalar(0, 255, 255),
					20);
		}
	}
}

void FindLanes::FitPoly(const cv::Mat& src_x, const cv::Mat& src_y,
		cv::Mat& dst, int order) {
	CV_Assert(src_x.rows > 0);
	CV_Assert(src_x.rows > 0);
	CV_Assert(src_y.rows > 0);
	CV_Assert(src_x.cols == 1);
	CV_Assert(src_y.cols == 1);
	CV_Assert(dst.cols == 1);
	CV_Assert(dst.rows == (order + 1));
	CV_Assert(order >= 1);

	cv::Mat X;
	X = cv::Mat::zeros(src_x.rows, order + 1, CV_32FC1);
	cv::Mat copy;
	// Vandermonde matrix
	for (int i = 0, p = order; (i <= order) && (p >= 0); i++, p--) {
		copy = src_x.clone();
		pow(copy, p, copy);
		cv::Mat M = X.col(i);
		copy.col(0).copyTo(M);
	}
	// Scale X to improve condition number and solve
	cv::Mat scale;
	reduce(X.mul(X), scale, 0, CV_REDUCE_SUM);
	sqrt(scale, scale);
	for (int i = 0; i < X.rows; i++) {
		cv::Mat M = X.row(i);
		M /= scale;
	}
	// Linear least-squares
	cv::Mat C;
	solve(X.t() * X, X.t() * src_y, C, cv::DECOMP_LU);
	C /= scale.t();
	C.copyTo(dst);
}

void FindLanes::Steering() {
	float car_pos_x = img.cols / 2;
	float car_pos_y = img.rows - 1;

	LANE_MODE laneMode = LANE_MODE_LEFT;
	float displacement = 0;

	bool bSwap = false;

	if (leftLine.found && rightLine.found
			&& (rightLine.fittedPts[0].x - leftLine.fittedPts[0].x
					> hyperparams.margin * 2)) {
		// Calculate line angles
		rightLine.angle =
				atan2(
						(rightLine.fittedPts[0].x
								- rightLine.fittedPts[img.rows / 3].x),
						(rightLine.fittedPts[0].y
								- rightLine.fittedPts[img.rows / 3].y));
		leftLine.angle = atan2(
				(leftLine.fittedPts[0].x - leftLine.fittedPts[img.rows / 3].x),
				(leftLine.fittedPts[0].y - leftLine.fittedPts[img.rows / 3].y));
		// Check if lines are parallel
		bool bParallel = false;
		if (std::abs(rightLine.angle - leftLine.angle) > 15 * 3.14 / 180) {
			bParallel = false;
		} else {
			bParallel = true;
		}
		// Choose lane mode according to history, discard wrong line if not parallel
		if (laneHistory.leftLine.found && laneHistory.rightLine.found) {
			if (std::abs(rightLine.angle - laneHistory.rightLine.angle)
					> std::abs(leftLine.angle - laneHistory.leftLine.angle)) {
				if (!bParallel) {
					rightLine.found = false;
				}
				laneMode = LANE_MODE_LEFT;
			} else {
				if (!bParallel) {
					leftLine.found = false;
				}
				laneMode = LANE_MODE_RIGHT;
			}
		} else if (laneHistory.leftLine.found) {
			if (!bParallel) {
				rightLine.found = false;
			}
			laneMode = LANE_MODE_LEFT;
		} else {
			if (!bParallel) {
				leftLine.found = false;
			}
			laneMode = LANE_MODE_RIGHT;
		}
		if (laneMode == LANE_MODE_LEFT && !leftLine.found) {
			bSwap = true;
		} else if (laneMode == LANE_MODE_RIGHT && !rightLine.found) {
			bSwap = true;
		}

		// Record history and find displacement
		if (leftLine.found && rightLine.found) {
			laneHistory.leftLine.found = leftLine.found;
			laneHistory.rightLine.found = rightLine.found;
			laneHistory.leftLine.xBase = leftLine.fittedPts[0].x;
			laneHistory.rightLine.xBase = rightLine.fittedPts[0].x;
			laneHistory.leftLine.fit = leftLine.fit;
			laneHistory.rightLine.fit = rightLine.fit;
			laneHistory.leftLine.angle = leftLine.angle;
			laneHistory.rightLine.angle = rightLine.angle;
			laneHistory.laneWidth = rightLine.fittedPts[0].x
					- leftLine.fittedPts[0].x;
			displacement = (
					laneMode == LANE_MODE_LEFT ?
							laneHistory.laneWidth / 2 :
							-laneHistory.laneWidth / 2);
			bDetected = true;
		}
	}
	if (!bDetected) {
		float x = (leftLine.found ? leftLine.fittedPts[0].x :
					rightLine.found ? rightLine.fittedPts[0].x : -1);
		if (x < 0) {
			bDetected = false;
			return;
		}
		// Check according to history which line we found
		if (laneHistory.leftLine.found && laneHistory.rightLine.found) {
			if (abs(laneHistory.leftLine.xBase - x)
					> abs(laneHistory.rightLine.xBase - x)) {
				if (!rightLine.found) {
					bSwap = true;
				}
				rightLine.found = true;
				leftLine.found = false;
			} else {
				if (!leftLine.found) {
					bSwap = true;
				}
				leftLine.found = true;
				rightLine.found = false;
			}
		} else {
			if ((laneHistory.leftLine.found && !leftLine.found)
					|| (laneHistory.rightLine.found && !rightLine.found)) {
				bSwap = true;
			}
			leftLine.found = laneHistory.leftLine.found;
			rightLine.found = laneHistory.rightLine.found;
		}
		// Record history and find displacement
		if (leftLine.found) {
			laneHistory.leftLine.xBase = x;
			laneHistory.leftLine.found = true;
			laneHistory.rightLine.found = false;
			laneHistory.leftLine.fit = leftLine.fit;
			laneMode = LANE_MODE_LEFT;
			displacement = laneHistory.laneWidth / 2;
			if (x > car_pos_x) {
				displacement += x - car_pos_x;
			}
		} else {
			laneHistory.rightLine.xBase = x;
			laneHistory.rightLine.found = true;
			laneHistory.leftLine.found = false;
			laneHistory.rightLine.fit = rightLine.fit;
			laneMode = LANE_MODE_RIGHT;
			displacement = -laneHistory.laneWidth / 2;
			if (x < car_pos_x) {
				displacement -= car_pos_x - x;
			}
		}
		bDetected = true;
	}
	// Prevent line confusion
	if (bSwap) {
		rightLine.fittedPts.swap(leftLine.fittedPts);
		rightLine.fit.swap(leftLine.fit);
	}

	// Calculate predicted line from found line
	std::vector<cv::Point> &fittedpts = (
			laneMode == LANE_MODE_LEFT ?
					leftLine.fittedPts : rightLine.fittedPts);
	std::vector<cv::Point> predictedPts;
	for (auto it : fittedpts) {
		predictedPts.push_back(cv::Point(it.x + displacement, it.y));
	}
	if (laneMode == LANE_MODE_LEFT) {
		rightLine.fittedPts = predictedPts;
	} else {
		leftLine.fittedPts = predictedPts;
	}

	// Find target point
	float speed_m_p_s = speed * 0.00025;
	float speed_m_p_us = speed_m_p_s / 1000000;
	float speed_pix_p_us = speed_m_p_us * img.rows;
	float div =
			(laneMode == LANE_MODE_LEFT ? leftLine.fit[0] : rightLine.fit[0]);
	float N = FLT_MAX;
	if (div != 0) {
		N = std::abs(0.0005 / div);
	}
	if (N > 12) {
		N = 12;
	} else if (N < 6) {
		N = 6;
	}
	float constraint = frameDuration * N * speed_pix_p_us;

	float sum = 0;
	float dst_x = predictedPts[0].x;
	float dst_y = predictedPts[0].y;
	bool dstFound = false;
	for (unsigned i = 1; i < predictedPts.size(); i++) {
		if (!dstFound) {
			dst_x = predictedPts[i].x;
			dst_y = predictedPts[i].y;
		}
		sum += sqrt(
				pow(predictedPts[i].x - predictedPts[i - 1].x, 2)
						+ pow(predictedPts[i].y - predictedPts[i - 1].y, 2));
		if (sum >= constraint) {
			dstFound = true;
		}
	}

#if 0
	float max_pix_p_frame = sum / 3; // 3 frames is min
	float min_f_p_s = speed_pix_p_us * 1000000 / max_pix_p_frame;
	speed *= N / 9;
#endif

	// Find steering angle
	float offset = dst_x - car_pos_x;
	double angle_rad = atan2(offset, car_pos_y - dst_y);
	double angle_deg = angle_rad * 180 / 3.14;
	double min_angle = -90;
	double max_angle = 90;
	angle_deg = angle_deg < min_angle ? min_angle : angle_deg;
	angle_deg = angle_deg > max_angle ? max_angle : angle_deg;

	// Map (-90, 90) to (0, 1)
	steeringAngle = (angle_deg - min_angle) * (1 - 0) / (max_angle - min_angle)
			+ 0;

	// VISUALIZE
	if (bVerbose) {
		// Plot found and predicted lines
#if 1
		PlotPoly(LANE_MODE_LEFT, true);
		PlotPoly(LANE_MODE_RIGHT, true);
#endif
		// Offset and angle
		cv::Point p1(car_pos_x, car_pos_y);
		cv::Point p2(car_pos_x, dst_y);
		cv::Point p3(dst_x, dst_y);

		cv::line(outImg, p1, p2, cv::Scalar::all(255), 2);
		cv::line(outImg, p2, p3, cv::Scalar::all(255), 2);
		cv::arrowedLine(outImg, p1, p3, cv::Scalar::all(255), 2);

		// Frame duration indicator
		float pix_p_N_frames = speed_pix_p_us * frameDuration * N;
		cv::Point p4 = p1;
		for (int i = 1; i <= 3; i++) {
			float y = pix_p_N_frames * i / 3;
			cv::Point p5(car_pos_x, car_pos_y - y);
			cv::Scalar color = (
					i == 1 ? cv::Scalar(0, 0, 255) :
					i == 2 ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0));
			cv::line(outImg, p4, p5, color, 2);
			p4 = p5;
		}

#if 1
		warpPerspective(outImg, outImg, invPerspTf, outImg.size());
		addWeighted(outImg, 0.6, frameImg, 0.4, 0, outImg);
#endif

		int count = 1;
		bool bPrint = true;
		while (bPrint) {
			std::string name;
			std::string unit;
			double value;

			switch (count) {
			case 1:
				name = "Speed";
				unit = "m/s";
				value = speed_m_p_s;
				break;
			case 2:
				name = "Angle ";
				unit = "deg";
				value = angle_deg;
				break;
			case 3:
				name = "Offset";
				unit = "m";
				value = (offset / img.cols) * 1.4;
				break;
			case 4:
				name = "Moment FPS";
				unit = "";
				value = (double) 1000000 / frameDuration;
				break;
			case 5:
				name = "Moment SPF";
				unit = "";
				value = (double) frameDuration / 1000000;
				break;
			default:
				bPrint = false;
				break;
			}
			if (bPrint) {
				std::stringstream ss;
				ss << std::left << name << " = " << std::setw(6)
						<< std::setprecision(2) << value << " " << unit;
				putText(outImg, ss.str(), cv::Point(15, 30 * count++),
				CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
				ss.str(std::string());
			}
		}
	}

#if 1
	double max_pix_p_frame = sum / 3; // 3 frames is min
	double fps = 1000000 / frameDuration;
	maxSpeed = fps * max_pix_p_frame / 720;
#endif
}

