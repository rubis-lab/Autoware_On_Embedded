#ifndef INCLUDE_LANE_FOLLOWING_TIME_PROFILING_H_
#define INCLUDE_LANE_FOLLOWING_TIME_PROFILING_H_

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

class TimeProfiling {
public:
	struct Timestamp {
		std::string name;
		std::chrono::system_clock::time_point time;
		Timestamp() {
		}
		Timestamp(const Timestamp &d) {
			operator=(d);
		}
		Timestamp(const std::string &name,
				const std::chrono::system_clock::time_point &time) :
				name(name), time(time) {
		}
		Timestamp &operator=(const Timestamp &d) {
			name = d.name;
			time = d.time;
			return *this;
		}
	};

	struct Duration {
		std::string name;
		long int duration;
		Duration(const std::string &name, long int duration) :
				name(name), duration(duration) {
		}
	};

	TimeProfiling() {
	}

	virtual ~TimeProfiling() {
	}

	void MakeDurations() {
		for (unsigned i = 0; i < timestamps.size(); i++, i++) {
			auto duration =
					std::chrono::duration_cast<std::chrono::microseconds>(
							timestamps[i + 1].time - timestamps[i].time);
			durations.push_back(Duration(timestamps[i].name, duration.count()));
		}
		SortDurations();
	}

	void AppendDurations(std::vector<Duration>& other_durations) {
		durations.insert(durations.end(), other_durations.begin(),
				other_durations.end());
		SortDurations();
	}

	void SortDurations() {
		std::sort(durations.begin(), durations.end(),
				[](const Duration &a, const Duration &b) {
					return a.name < b.name;
				});
	}

	std::vector<Timestamp>& getTimestamps() {
		return timestamps;
	}

	std::vector<Duration>& getDurations() {
		return durations;
	}

	void PrintTime(std::string moduleName, int pipelineInstanceNum,
			std::chrono::system_clock::time_point global_start,
			std::chrono::system_clock::time_point global_end) {
		std::cout << std::left << std::setw(16)
				<< "++" + moduleName + "[" + std::to_string(pipelineInstanceNum)
						+ "]" << std::setw(12) << "start_time" << std::setw(12)
				<< "end_time" << std::setw(16) << "duration (usec)"
				//<< std::setw(8) << "local %" << std::setw(8) << "global %"
				<< std::endl;

		auto total_duration = std::chrono::duration_cast<
				std::chrono::microseconds>(global_end - global_start);

		auto module_start =
				std::chrono::duration_cast<std::chrono::microseconds>(
						timestamps.front().time - global_start);
		auto module_end = std::chrono::duration_cast<std::chrono::microseconds>(
				timestamps.back().time - global_start);
		auto module_duration = std::chrono::duration_cast<
				std::chrono::microseconds>(
				timestamps.back().time - timestamps.front().time);
		for (unsigned i = 0; i < timestamps.size(); i++, i++) {
			auto start = std::chrono::duration_cast<std::chrono::microseconds>(
					timestamps[i].time - global_start);
			auto end = std::chrono::duration_cast<std::chrono::microseconds>(
					timestamps[i + 1].time - global_start);
			auto duration =
					std::chrono::duration_cast<std::chrono::microseconds>(
							timestamps[i + 1].time - timestamps[i].time);
			std::cout << std::left << std::setw(16) << timestamps[i].name
					<< std::setw(12) << start.count() << std::setw(12)
					<< end.count() << std::setw(16) << duration.count()
					//<< std::setw(8)
					//<< duration.count() * 100 / module_duration.count()
					//<< std::setw(8)
					//<< duration.count() * 100 / total_duration.count()
					<< std::endl;
		}

		std::cout << std::left << std::setw(16) << "--" + moduleName
				<< std::setw(12) << module_start.count() << std::setw(12)
				<< module_end.count() << std::setw(16)
				<< module_duration.count()
				//<< std::setw(8)
				//<< module_duration.count() * 100 / module_duration.count()
				//<< std::setw(8)
				//<< module_duration.count() * 100 / total_duration.count()
				<< std::endl << std::endl;
	}

	void PrintTime(std::string moduleName, int pipelineInstanceNum) {
		std::cout << std::left << std::setw(16)
				<< "++" + moduleName + "[" + std::to_string(pipelineInstanceNum)
						+ "]" << std::setw(12) << "start_time" << std::setw(12)
				<< "end_time" << std::setw(16) << "duration (usec)"
				//<< std::setw(8) << "local %"
				<< std::endl;

		auto module_start =
				std::chrono::duration_cast<std::chrono::microseconds>(
						timestamps.front().time.time_since_epoch());
		auto module_end = std::chrono::duration_cast<std::chrono::microseconds>(
				timestamps.back().time.time_since_epoch());
		auto module_duration = std::chrono::duration_cast<
				std::chrono::microseconds>(
				timestamps.back().time - timestamps.front().time);

		for (unsigned i = 0; i < timestamps.size(); i++, i++) {
			auto start = std::chrono::duration_cast<std::chrono::microseconds>(
					timestamps[i].time.time_since_epoch());
			auto end = std::chrono::duration_cast<std::chrono::microseconds>(
					timestamps[i + 1].time.time_since_epoch());
			auto duration =
					std::chrono::duration_cast<std::chrono::microseconds>(
							timestamps[i + 1].time - timestamps[i].time);
			std::cout << std::left << std::setw(16) << timestamps[i].name
					<< std::setw(12) << start.count() << std::setw(12)
					<< end.count() << std::setw(16) << duration.count()
					<< std::setw(8)
					<< duration.count() * 100 / module_duration.count()
					<< std::endl;
		}

		std::cout << std::left << std::setw(16) << "--" + moduleName
				<< std::setw(12) << module_start.count() << std::setw(12)
				<< module_end.count() << std::setw(16)
				<< module_duration.count()
				//<< std::setw(8)
				//<< module_duration.count() * 100 / module_duration.count()
				<< std::endl << std::endl;
	}

	void PrintAvgDurations(std::string moduleName) {
		std::cout << std::left << std::setw(20) << "++" + moduleName
				<< std::setw(12) << "sum (usec)" << std::setw(12)
				<< "avg (usec)" << std::endl;

		std::string prev_name = durations.front().name;
		long int total_sum = 0;
		long int avg_sum = 0;
		long int sum = 0;
		int cnt = 0;
		for (unsigned i = 0; i < durations.size(); i++) {
			total_sum += durations[i].duration;
			if (prev_name == durations[i].name) {
				sum += durations[i].duration;
				cnt++;
			}
			if (prev_name != durations[i].name || i == durations.size() - 1) {
				std::cout << std::left << std::setw(20) << prev_name
						<< std::setw(12) << sum << std::setw(12) << sum / cnt
						<< std::endl;
				avg_sum += sum / cnt;
				prev_name = durations[i].name;
				sum = durations[i].duration;
				cnt = 1;
			}
		}
		std::cout << std::left << std::setw(20) << "--" + moduleName
				<< std::setw(12) << total_sum << std::setw(12) << avg_sum
				<< std::endl << std::endl;
	}
protected:
	std::vector<Timestamp> timestamps;
	std::vector<Duration> durations;
};

#endif /* INCLUDE_LANE_FOLLOWING_TIME_PROFILING_H_ */
