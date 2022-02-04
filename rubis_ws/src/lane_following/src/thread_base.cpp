#include "../include/lane_following/thread_base.h"

#include <unistd.h>
#include <cstring>
#include <utility>

#include "lane_following/debug.h"

ThreadRegister::ThreadRegister() {
	currThreadId = 0;
}

ThreadRegister::~ThreadRegister() {
	Reset();
}

bool ThreadRegister::RegisterThread(ThreadId threadId,
		ThreadRegisterClientInterfase* thread) {
	std::lock_guard<std::recursive_mutex> lock(mapLock);
	std::pair<ThreadId, ThreadRegisterClientInterfase*> tp = std::make_pair(
			threadId, thread);
	auto ret = threadMap.insert(tp);
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE,
			"[%ld]ThreadRegister::RegisterThread, thread[%ld] registered\n",
			currThreadId, threadId);
	return ret.second;
}

bool ThreadRegister::UnregisterThread(ThreadId threadId) {
	std::lock_guard<std::recursive_mutex> lock(mapLock);
	auto thread = threadMap.find(threadId);
	if (thread != threadMap.end()) {
		threadMap.erase(thread);
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE,
				"[%ld]ThreadRegister::UnregisterThread, thread[%ld] unregistered\n",
				currThreadId, threadId);
		return true;
	} else {
		PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
				"[%ld]ThreadRegister::UnregisterThread error: thread[%ld] not found\n",
				currThreadId, threadId);
	}
	return false;
}

bool ThreadRegister::SendMsg(ThreadId threadId,
		std::shared_ptr<ThreadMsg>& msg) {
	std::lock_guard<std::recursive_mutex> lock(mapLock);
	auto thread = threadMap.find(threadId);
	if (thread != threadMap.end()) {
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE,
				"[%ld]ThreadRegister::SendMsg, thread[%ld] msg(%ud) sent\n",
				currThreadId, threadId, msg->taskMsg);
		return thread->second->AddMsg(msg);
	} else {
		PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
				"[%ld]ThreadRegister::SendMsg error: thread[%ld] not found, msg(%ud)\n",
				currThreadId, threadId, msg->taskMsg);
	}
	return false;
}

bool ThreadRegister::SendUniqueMsg(ThreadId threadId,
		std::shared_ptr<ThreadMsg>& msg) {
	std::lock_guard<std::recursive_mutex> lock(mapLock);
	auto thread = threadMap.find(threadId);
	if (thread != threadMap.end()) {
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE,
				"[%ld]ThreadRegister::SendMsg, thread[%ld] msg(%ud) sent\n",
				currThreadId, threadId, msg->taskMsg);
		return thread->second->AddUniqueMsg(msg);
	} else {
		PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
				"[%ld]ThreadRegister::SendMsg error: thread[%ld] not found, msg(%ud)\n",
				currThreadId, threadId, msg->taskMsg);
	}
	return false;
}

ThreadId ThreadRegister::GetNewThreadId() {
	std::lock_guard<std::recursive_mutex> lock(mapLock);
	return currThreadId++;
}

void ThreadRegister::Reset() {
	{
		std::lock_guard<std::recursive_mutex> lock(mapLock);
		for (auto it : threadMap) {
			it.second->Unregister();
		}
	}
	usleep(1000);
	{
		std::lock_guard<std::recursive_mutex> lock(mapLock);
		for (auto it : threadMap) {
			PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
					"ThreadRegister::~ThreadRegister error: thread[%ld] still running\n",
					it.first);
		}
	}
	currThreadId = 0;
}

ThreadQueue::ThreadQueue() :
		ready(false) {
}

ThreadQueue::~ThreadQueue() {
}

std::shared_ptr<ThreadMsg> ThreadQueue::GetMsg() {
	std::unique_lock<std::mutex> lock(conditionVariableLock);
	if (queue.empty()) {
		return nullptr;
	}
	std::shared_ptr<ThreadMsg> threadMsg = queue.front();
	queue.pop_front();
	if (!queue.empty()) {
		lock.unlock();
		conditionVariable.notify_all();
	}
	return threadMsg;
}

bool ThreadQueue::PutMsg(std::shared_ptr<ThreadMsg>& msg) {
	std::unique_lock<std::mutex> lock(conditionVariableLock);
	queue.push_back(msg);
	lock.unlock();
	conditionVariable.notify_all();
	return true;
}

bool ThreadQueue::PutUniqueMsg(std::shared_ptr<ThreadMsg>& msg) {
	std::unique_lock<std::mutex> lock(conditionVariableLock);
	for (auto it : queue) {
		if (it->taskMsg == msg->taskMsg) {
			return true;
		}
	}
	queue.push_back(msg);
	lock.unlock();
	conditionVariable.notify_all();
	return true;
}

bool ThreadQueue::Wait() {
	std::unique_lock<std::mutex> lock(conditionVariableLock);
	if (!queue.empty()) {
		return true;
	}
	conditionVariable.wait(lock);
	return true;
}

ThreadRegister ThreadBase::threadRegister;

ThreadBase::ThreadBase() {
	threadId = threadRegister.GetNewThreadId();
	threadHandle = 0;
}

ThreadBase::~ThreadBase() {
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE, "[%ld]ThreadBase::~ThreadBase\n",
			threadId);
	ThreadClose();
}

void* ThreadBase::stStartThread(void* arg) {
	ThreadBase* p = (ThreadBase*) arg;
	if (!p) {
		PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
				"ThreadBase::stStartThread error: ThreadBase* == NULL\n");
		return NULL;
	}
	return p->StartThread();
}

void* ThreadBase::StartThread() {
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE, "++[%ld]ThreadBase::StartThread\n",
			threadId);
	if (!PreWorkInit()) {
		PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
				"[%ld]ThreadBase::StartThread error: PreWorkInit() failed\n",
				threadId);
		return NULL;
	}
	Run();
	PostWorkDeinit();
	threadRegister.UnregisterThread(threadId);
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE, "--[%ld]ThreadBase::StartThread\n",
			threadId);
	return NULL; // doesn't return any params
}

void ThreadBase::EndThread() {
	std::shared_ptr<ThreadMsg> msg = std::make_shared<ThreadMsg>();
	msg->taskMsg = THREAD_MSG_EXIT;
	AddMsg(msg);
}

bool ThreadBase::ThreadOpen() {
	if (pthread_create(&threadHandle, NULL, stStartThread, (void*) this) != 0) {
		PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
				"[%ld]ThreadBase::ThreadOpen pthread_create error: %s\n",
				threadId, std::strerror(errno));
		return false;
	}
	return true;
}

bool ThreadBase::ThreadClose() {
	WaitForThread();
	return true;
}

ThreadBase* ThreadBase::ThreadCreate() {
	if (ThreadOpen()) {
		threadRegister.RegisterThread(threadId, this);
		return this;
	} else {
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE,
				"[%ld]ThreadBase::ThreadCreate error: ThreadOpen() failed\n",
				threadId);
	}
	return NULL;
}

void ThreadBase::WaitForThread() {
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE,
			"++[%ld]ThreadBase::WaitForThread, threadHandle=%ld\n", threadId,
			threadHandle);
	if (threadHandle != 0) {
		if (pthread_join(threadHandle, NULL) != 0) {
			PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
					"[%ld]ThreadBase::WaitForThread pthread_join error: %s\n",
					threadId, std::strerror(errno));
		}
		threadHandle = 0;
	}
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE,
			"--[%ld]ThreadBase::WaitForThread thread, threadHandle=%ld\n",
			threadId, threadHandle);
}

bool ThreadBase::SendMsg(ThreadId threadId, std::shared_ptr<ThreadMsg>& msg) {
	return threadRegister.SendMsg(threadId, msg);
}

void ThreadBase::Run() {
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE, "++[%ld]ThreadBase::Run\n",
			threadId);
	bool bExit = false;
	while (!bExit) {
		threadQueue.Wait();
		std::shared_ptr<ThreadMsg> msg = threadQueue.GetMsg();
		if (msg) {
			if (msg->taskMsg == THREAD_MSG_EXIT) {
				PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE,
						"[%ld]ThreadBase::Run, THREAD_MSG_EXIT received\n",
						threadId);
				while (true) {
					msg = threadQueue.GetMsg();
					if (!msg) {
						break;
					}
					ProcessMsg(msg);
				}
				bExit = true;
				break;
			}
			ProcessMsg(msg);
		}

	}
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_BASE, "--ThreadBase::Run thread[%ld]\n",
			threadId);
}

bool ThreadBase::AddMsg(std::shared_ptr<ThreadMsg>& msg) {
	return threadQueue.PutMsg(msg);
}

bool ThreadBase::AddUniqueMsg(std::shared_ptr<ThreadMsg>& msg) {
	return threadQueue.PutUniqueMsg(msg);
}

void ThreadBase::Unregister() {
	EndThread();
}

void ThreadBase::ResetThreadRegister() {
	threadRegister.Reset();
}
