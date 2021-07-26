#include "lane_following/thread_worker.h"

#include "lane_following/debug.h"
#include "lane_following/find_lanes.h"
#include "lane_following/lane_base.h"

void ThreadWorker::ProcessMsg(std::shared_ptr<ThreadMsg> &msg) {
	switch (msg->taskMsg) {
	case TASK_MSG_RUN_WARP: {
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
				"++[%ld]ThreadWorker::ProcessMsg: TASK_MSG_RUN_WARP received\n",
				GetThreadId());
		WarpRun(msg);
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
				"--[%ld]ThreadWorker::ProcessMsg: TASK_MSG_RUN_WARP received\n",
				GetThreadId());
		break;
	}
	case TASK_MSG_RUN_COLOR_GRAD_THRESH: {
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
				"++[%ld]ThreadWorker::ProcessMsg: TASK_MSG_RUN_COLOR_GRAD_THRESH received\n",
				GetThreadId());
		ColorGradThreshRun(msg);
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
				"--[%ld]ThreadWorker::ProcessMsg: TASK_MSG_RUN_COLOR_GRAD_THRESH received\n",
				GetThreadId());
		break;
	}
	case TASK_MSG_RUN_FIND_LANES: {
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
				"++[%ld]ThreadWorker::ProcessMsg: TASK_MSG_RUN_FIND_LANES received\n",
				GetThreadId());
		FindLanesRun(msg);
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
				"--[%ld]ThreadWorker::ProcessMsg: TASK_MSG_RUN_FIND_LANES received\n",
				GetThreadId());
		break;
	}
	default:
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
				"[%ld]ThreadWorker::ProcessMsg: TASK_MSG_UNKNOWN received\n",
				GetThreadId());
	}
}

void ThreadWorker::WarpRun(std::shared_ptr<ThreadMsg> &msg) {
	std::shared_ptr<WarpBase> warp = std::dynamic_pointer_cast<WarpBase>(
			msg->msgObj);
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
			"++[%ld]ThreadWorker::WarpRun, warp = %p\n", GetThreadId(),
			warp.get());
	if (warp) {
		warp->Run(msg, this);
		std::shared_ptr<ThreadMsg> cmsg = std::make_shared<ThreadMsg>();
		cmsg->threadIdFrom = GetThreadId();
		cmsg->msgObj = warp;
		cmsg->taskMsg = TASK_MSG_COMPLETE_WARP;
		cmsg->procStep = msg->procStep;
		SendMsg(msg->threadIdFrom, cmsg);
	}
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
			"--[%ld]ThreadWorker::WarpRun, warp = %p\n", GetThreadId(),
			warp.get());
}

void ThreadWorker::ColorGradThreshRun(std::shared_ptr<ThreadMsg> &msg) {
	std::shared_ptr<ColorGradThreshBase> colorGradThresh =
			std::dynamic_pointer_cast<ColorGradThreshBase>(msg->msgObj);
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
			"++[%ld]ThreadWorker::ColorGradThreshRun, colorGradThresh = %p\n",
			GetThreadId(), colorGradThresh.get());
	if (colorGradThresh) {
		colorGradThresh->Run(msg, this);
		std::shared_ptr<ThreadMsg> cmsg = std::make_shared<ThreadMsg>();
		cmsg->threadIdFrom = GetThreadId();
		cmsg->msgObj = colorGradThresh;
		cmsg->taskMsg = TASK_MSG_COMPLETE_COLOR_GRAD_THRESH;
		cmsg->procStep = msg->procStep;
		SendMsg(msg->threadIdFrom, cmsg);
	}
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
			"--[%ld]ThreadWorker::ColorGradThreshRun, colorGradThresh = %p\n",
			GetThreadId(), colorGradThresh.get());
}

void ThreadWorker::FindLanesRun(std::shared_ptr<ThreadMsg> &msg) {
	std::shared_ptr<FindLanes> findLanes = std::dynamic_pointer_cast<FindLanes>(
			msg->msgObj);
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
			"++[%ld]ThreadWorker::FindLanesRun, findLanes = %p\n",
			GetThreadId(), findLanes.get());
	if (findLanes) {
		findLanes->Run(msg, this);
		std::shared_ptr<ThreadMsg> cmsg = std::make_shared<ThreadMsg>();
		cmsg->threadIdFrom = GetThreadId();
		cmsg->msgObj = findLanes;
		cmsg->taskMsg = TASK_MSG_COMPLETE_FIND_LANES;
		cmsg->procStep = msg->procStep;
		SendMsg(msg->threadIdFrom, cmsg);
	}
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER,
			"--[%ld]ThreadWorker::FindLanesRun, findLanes = %p\n",
			GetThreadId(), findLanes.get());
}

bool ThreadWorker::PreWorkInit() {
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER, "[%ld]PreWorkInit\n",
			GetThreadId());
	return true;
}
bool ThreadWorker::PostWorkDeinit() {
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_WORKER, "[%ld]PostWorkDeinit\n",
			GetThreadId());
	return true;
}
