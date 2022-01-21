#ifndef INCLUDE_LANE_FOLLOWING_THREAD_BASE_H_
#define INCLUDE_LANE_FOLLOWING_THREAD_BASE_H_

#include <pthread.h>
#include <stddef.h>
#include <condition_variable>
#include <list>
#include <map>
#include <memory>
#include <mutex>

typedef long ThreadId;

enum MSG_OBJ_TYPE {
	MSG_OBJ_TYPE_WARP, MSG_OBJ_TYPE_COLOR_GRAD_THRESH, MSG_OBJ_TYPE_FIND_LANES
};

struct MsgObj {
	MsgObj() {
		msgObjType = 0;
	}
	virtual ~MsgObj() {
	}
	unsigned int msgObjType; // MSG_OBJ_TYPE
};

struct ThreadMsg {
	std::shared_ptr<MsgObj> msgObj;
	unsigned int taskMsg;
	int procStep;
	ThreadId threadIdFrom;
	ThreadMsg() {
		taskMsg = (unsigned int) -1;
		procStep = -1;
		threadIdFrom = -1;
	}
	ThreadMsg(const ThreadMsg& d) {
		operator=(d);
	}
	ThreadMsg& operator=(const ThreadMsg& d) {
		msgObj = d.msgObj;
		taskMsg = d.taskMsg;
		procStep = d.procStep;
		threadIdFrom = d.threadIdFrom;
		return *this;
	}
};

class ThreadBase;

class ThreadRegisterClientInterfase {
public:
	ThreadRegisterClientInterfase() {
	}
	virtual ~ThreadRegisterClientInterfase() {
	}
	virtual bool AddMsg(std::shared_ptr<ThreadMsg> &msg) = 0;
	virtual bool AddUniqueMsg(std::shared_ptr<ThreadMsg> &msg) = 0;
	virtual void Unregister() = 0;
};

class ThreadRegister {
public:
	ThreadRegister();
	virtual ~ThreadRegister();
	bool RegisterThread(ThreadId threadId,
			ThreadRegisterClientInterfase* thread);
	bool UnregisterThread(ThreadId threadId);
	bool SendMsg(ThreadId threadId, std::shared_ptr<ThreadMsg> &msg);
	bool SendUniqueMsg(ThreadId threadId, std::shared_ptr<ThreadMsg> &msg);
	ThreadId GetNewThreadId();
	void Reset();
private:
	ThreadId currThreadId;
	std::map<ThreadId, ThreadRegisterClientInterfase*> threadMap;
	std::recursive_mutex mapLock;
};

class ThreadQueue {
public:
	ThreadQueue();
	~ThreadQueue();
	std::shared_ptr<ThreadMsg> GetMsg();
	bool PutMsg(std::shared_ptr<ThreadMsg> &msg);
	bool PutUniqueMsg(std::shared_ptr<ThreadMsg> &msg);
	bool Wait();
	size_t GetSize() {
		return queue.size();
	}
private:
	std::list<std::shared_ptr<ThreadMsg>> queue;
	std::recursive_mutex queueLock;
	std::mutex conditionVariableLock;
	std::condition_variable conditionVariable;
	bool ready;
};

class ThreadBase: public ThreadRegisterClientInterfase {
public:
	enum THREAD_MSG {
		THREAD_MSG_START, THREAD_MSG_EXIT, THREAD_MSG_SYS_MAX
	};
	ThreadBase();
	virtual ~ThreadBase();
	static void* stStartThread(void* arg);
	void* StartThread();
	void EndThread();
	bool ThreadOpen();
	bool ThreadClose();
	ThreadBase* ThreadCreate();
	void WaitForThread();
	static bool SendMsg(ThreadId threadId, std::shared_ptr<ThreadMsg> &msg);
	virtual void Run();
	virtual bool AddMsg(std::shared_ptr<ThreadMsg> &msg) override;
	virtual bool AddUniqueMsg(std::shared_ptr<ThreadMsg> &msg) override;
	virtual void Unregister() override;
	virtual void ProcessMsg(std::shared_ptr<ThreadMsg> &msg) {
	}
	virtual bool PreWorkInit() {
		return true;
	}
	virtual bool PostWorkDeinit() {
		return true;
	}
	inline ThreadId GetThreadId() {
		return threadId;
	}
	static void ResetThreadRegister();
private:
	static ThreadRegister threadRegister;
	ThreadId threadId;
	ThreadQueue threadQueue;
	pthread_t threadHandle;
};

#endif /* INCLUDE_LANE_FOLLOWING_THREAD_BASE_H_ */
