#ifndef RUBIS_OMP_LOG_HPP_
#define RUBIS_OMP_LOG_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"

namespace rubis {
namespace sched_log {

struct sched_info {
  int task_id;
  int max_option;
  std::string name;
  std::string file;
  double exec_time;
  double deadline;
  double period;
};

struct sched_data {
  int thread_id;
  int iter;
  double start_time;
  double end_time;
  double resp_time;
};

class SchedLog {
public:
  SchedLog();
  SchedLog(sched_info _si);
  void add_entry(sched_data _sd);
  void add_entry(sched_data _sd, std::string _comment);

private:
  sched_info __si;
  std::shared_ptr<spdlog::logger> __log;

  std::string generate_header();
};

SchedLog::SchedLog() {
  return;
}

SchedLog::SchedLog(sched_info _si) {  
  __si = _si;

  std::ifstream fs(__si.file);
  __log = spdlog::basic_logger_mt<spdlog::async_factory>(__si.name, __si.file);
  if(fs) { // file exists
    std::cout << "Appending log to: " << __si.file << std::endl;
  } else {
    std::cout << "Log file created in: " << __si.file << std::endl;
    __log->info(generate_header());
  }
  fs.close();
  return;
}

std::string SchedLog::generate_header() {
  std::string h_str;
  h_str += "tid\t";
  h_str += "thr_id\t";
  h_str += "max_opt\t";
  h_str += "iter\t";
  h_str += "exec_t\t";
  h_str += "dead\t";
  h_str += "period\t";
  h_str += "start_t\t";
  h_str += "end_t\t";
  h_str += "resp_t\t";
  h_str += "slack\t";
  return h_str;
}

void SchedLog::add_entry(sched_data _sd) {
  std::string e_str = "";
  e_str += std::to_string(__si.task_id) + "\t";
  e_str += std::to_string(_sd.thread_id) + "\t";
  e_str += std::to_string(__si.max_option) + "\t";
  e_str += std::to_string(_sd.iter) + "\t";
  e_str += std::to_string(__si.exec_time) + "\t";  // exec_time
  e_str += std::to_string(__si.deadline) + "\t";
  e_str += std::to_string(__si.period) + "\t";
  e_str += std::to_string(_sd.start_time) + "\t";  // start_time
  e_str += std::to_string(_sd.end_time) + "\t";  // end_time
  e_str += std::to_string(_sd.resp_time) + "\t";
  e_str += std::to_string(0.0) + "\t";  // slack
  __log->info(e_str);
  return;
}

void SchedLog::add_entry(sched_data _sd, std::string _comment) {
  std::string e_str = "";
  e_str += std::to_string(__si.task_id) + "\t";
  e_str += std::to_string(_sd.thread_id) + "\t";
  e_str += std::to_string(__si.max_option) + "\t";
  e_str += std::to_string(_sd.iter) + "\t";
  e_str += std::to_string(__si.exec_time) + "\t";  // exec_time
  e_str += std::to_string(__si.deadline) + "\t";
  e_str += std::to_string(__si.period) + "\t";
  e_str += std::to_string(_sd.start_time) + "\t";  // start_time
  e_str += std::to_string(_sd.end_time) + "\t";  // end_time
  e_str += std::to_string(_sd.resp_time) + "\t";
  e_str += std::to_string(0.0) + "\t";  // slack
  e_str += _comment + "\t";  // slack
  __log->info(e_str);
  return;
}

} // namespace sched_log
} // namespace rubis
#endif