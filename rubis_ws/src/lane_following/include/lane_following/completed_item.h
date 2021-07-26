#ifndef INCLUDE_LANE_FOLLOWING_COMPLETED_ITEM_H_
#define INCLUDE_LANE_FOLLOWING_COMPLETED_ITEM_H_

#include <list>
#include <string>

enum TASK_STATE {
	TASK_STATE_INITIALIZED,
	TASK_STATE_RUNNING,
	TASK_STATE_COMPLETED,
	TASK_STATE_UNDEFINED
};

struct CompletedItem {
	int procStep;
	TASK_STATE taskState;
	CompletedItem() {
		procStep = -1;
		taskState = TASK_STATE_UNDEFINED;
	}
	CompletedItem(const CompletedItem& d) {
		operator=(d);
	}
	CompletedItem& operator=(const CompletedItem& d) {
		procStep = d.procStep;
		taskState = d.taskState;
		return *this;
	}
};

class CompletedItemList: public std::list<CompletedItem> {
public:
	void addItem(int procStep, TASK_STATE taskState) {
		for (auto it = begin(); it != end(); it++) {
			if (it->procStep == procStep) {
				it->procStep = procStep;
				it->taskState = taskState;
				return;
			}
		}
		emplace_back();
		back().procStep = procStep;
		back().taskState = taskState;
	}
	bool isCompleted() {
		for (auto it = begin(); it != end(); it++) {
			if (it->taskState != TASK_STATE_COMPLETED) {
				return false;
			}
		}
		return true;
	}
	bool isRunning() {
		for (auto it = begin(); it != end(); it++) {
			if (it->taskState != TASK_STATE_RUNNING) {
				return false;
			}
		}
		return true;
	}
	bool NeedToRun() {
		for (auto it = begin(); it != end(); it++) {
			if (it->taskState == TASK_STATE_INITIALIZED) {
				return true;
			}
		}
		return false;
	}
	CompletedItem* getUncompleted() {
		for (auto it = begin(); it != end(); it++) {
			if (it->taskState != TASK_STATE_COMPLETED) {
				return &(*it);
			}
		}
		return nullptr;
	}
	void rmCompleted() {
		for (auto it = begin(); it != end();) {
			if (it->taskState == TASK_STATE_COMPLETED) {
				auto it1 = it++;
				erase(it1);
			} else {
				it++;
			}
		}
	}
	std::string dump() {
		std::string ret = "\n++CompletedItemList dump: size = "
				+ std::to_string(size()) + "\n";
		int i = 0;
		for (auto it : *this) {
			ret += "CompletedItem[" + std::to_string(i++) + "]: procStep = "
					+ std::to_string(it.procStep) + ", taskState = "
					+ std::to_string(it.taskState) + "\n";
		}
		ret += "--CompletedItemList dump\n";
		return ret;
	}
};

#endif /* INCLUDE_LANE_FOLLOWING_COMPLETED_ITEM_H_ */
