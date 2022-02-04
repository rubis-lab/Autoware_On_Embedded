#include "lane_following/color_grad_thresh.h"

#include <opencv2/core/hal/interface.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "lane_following/debug.h"
#include "lane_following/warp.h"

ColorGradThresh::ColorGradThresh(int pipelineInstanceNum, bool bParallel,
		bool bGpuAccel, bool bVerbose) :
		ColorGradThreshBase(pipelineInstanceNum, bParallel, bGpuAccel, bVerbose) {
}

ColorGradThresh::~ColorGradThresh() {
	Deinit();
}

void ColorGradThresh::Init() {
	ColorGradThreshBase::Init();
}

void ColorGradThresh::Deinit() {
	binarySrc.clear();
	binaryDst.clear();
	ColorGradThreshBase::Deinit();
}

void ColorGradThresh::setParams(LaneBase* obj) {
	Deinit();
	Warp* warp = dynamic_cast<Warp*>(obj);
	if (warp) {
		warpImg = warp->getOutImg().clone();
		ColorGradThreshBase::setParams(obj);
	}
}

void ColorGradThresh::SplitChannel(SPLIT_MODE mode) {
	switch (mode) {
	case SPLIT_MODE_BGR: {
		// Separate BRG color channels
		std::vector<cv::Mat> bgrchannel;
		split(warpImg, bgrchannel);
		if (bgrchannel.size() != 3) {
			PRINT_DEBUG_MSG(DEBUG_ZONE_COLOR_GRAD_THRESH,
					"ColorGradTf::SplitChannel error bgrchannel.size()=%lu\n",
					bgrchannel.size());
		}
		binarySrc.threshRed = bgrchannel[2];
		break;
	}
	case SPLIT_MODE_HLS: {
		// Separate HLS color channels
		std::vector<cv::Mat> hlschannel;
		split(hls, hlschannel);
		if (hlschannel.size() != 3) {
			PRINT_DEBUG_MSG(DEBUG_ZONE_COLOR_GRAD_THRESH,
					"ColorGradTf::SplitChannel error hlschannel.size()=%lu\n",
					hlschannel.size());
		}
		binarySrc.absSobelx = hlschannel[1];
		binarySrc.threshSat = hlschannel[2];
		hls.release();
		break;
	}
	}
}

void ColorGradThresh::CvtBGR2HLS() {
	// Convert to HLS color space
	cvtColor(warpImg, hls, cv::COLOR_BGR2HLS);
}

void ColorGradThresh::ThresholdBinary(THRESH_MODE mode) {
	switch (mode) {
	case THRESH_MODE_RED:
		Threshold(binarySrc.threshRed, thresh.red[0], thresh.red[1],
				binaryDst.threshRed);
		break;
	case THRESH_MODE_SAT:
		Threshold(binarySrc.threshSat, thresh.sat[0], thresh.sat[1],
				binaryDst.threshSat);
		break;
	case THRESH_MODE_ABS_SOBELX:
		Threshold(binarySrc.threshSobelx, thresh.sobelx[0], thresh.sobelx[1],
				binaryDst.threshSobelx);
		break;
	}
}

void ColorGradThresh::Threshold(const cv::Mat &src, int lowerb, int upperb,
		cv::Mat &dst) {
	dst = cv::Mat::zeros(src.size(), src.type());
	inRange(src, lowerb, upperb, dst);
}

void ColorGradThresh::Sobelx() {
	// Take the derivative in x
	Sobel(binarySrc.absSobelx, binaryDst.absSobelx, CV_64F, 1, 0);
}

void ColorGradThresh::AbsSobelx() {
	// Absolute x derivative to accentuate lines away from horizontal
	convertScaleAbs(binaryDst.absSobelx, binarySrc.threshSobelx);
}

void ColorGradThresh::CombBinaries() {
	// Combine three binary thresholds
	outImg = binaryDst.threshSat | (binaryDst.threshSobelx & binaryDst.threshRed);
}
