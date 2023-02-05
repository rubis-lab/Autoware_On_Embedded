#include "rubis_lib/sched.hpp"

// #define DEBUG 

namespace rubis{

int key_id_;
int is_scheduled_;
std::string task_filename_;

struct sched_attr create_sched_attr(int priority, int exec_time, int deadline, int period){
  struct sched_attr attr;

  attr.sched_priority = (__u32)priority;
  attr.sched_runtime = (__u64)exec_time;
  attr.sched_deadline = (__u64)deadline;
  attr.sched_period = (__u64)period;

  return attr;
}

// system call hook to call SCHED_DEADLINE
int sched_setattr(pid_t pid, const struct sched_attr *attr, unsigned int flags){
	return syscall(__NR_sched_setattr, pid, attr, flags);
}

int sched_getattr(pid_t pid, struct sched_attr *attr, unsigned int size, unsigned int flags)
{
	return syscall(__NR_sched_getattr, pid, attr, size, flags);
}

bool set_sched_fifo(int pid, int priority){
  struct sched_param sp = {.sched_priority = (int32_t) priority};
  int ret = sched_setscheduler(pid, SCHED_FIFO, &sp);
  if(ret == -1){
    perror("sched_setscheduler");
    return false;
  }
  return true;
}

bool set_sched_fifo(int pid, int priority, int child_priority){
  if(pid == 0) pid = getpid();
  bool output = set_sched_fifo(pid, priority);
  std::vector<int> child_pids = get_child_pids(pid);

  for(auto it = child_pids.begin(); it != child_pids.end(); it++){    
    int child_pid = *it;
    output = set_sched_fifo(child_pid, child_priority);
  }

  return output;
}

bool set_sched_rr(int pid, int priority){
  struct sched_param sp = {.sched_priority = (int32_t) priority};
  int ret = sched_setscheduler(pid, SCHED_RR, &sp);
  if(ret == -1){
    perror("sched_setscheduler");
    return false;
  }
  return true;
}

bool set_sched_rr(int pid, int priority, int child_priority){
  if(pid == 0) pid = getpid();
  bool output = set_sched_rr(pid, priority);
  std::vector<int> child_pids = get_child_pids(pid);

  for(auto it = child_pids.begin(); it != child_pids.end(); it++){    
    int child_pid = *it;
    output = set_sched_rr(child_pid, child_priority);
  }

  return output;
}

bool set_sched_deadline(int pid, unsigned int exec_time, unsigned int deadline, unsigned int period) {

  struct sched_attr attr;
  attr.size = sizeof(attr);
  attr.sched_flags = 0;
  attr.sched_nice = 0;
  attr.sched_priority = 0;

  attr.sched_policy = SCHED_DEADLINE; // 6
  attr.sched_runtime = (__u64)exec_time;
  attr.sched_deadline = (__u64)deadline;
  attr.sched_period = (__u64)period;

  int ret = sched_setattr(pid, &attr, attr.sched_flags);
  if(ret < 0) {
      std::cerr << "[ERROR] sched_setattr failed. Are you root? (" << ret << ")" << std::endl;
      perror("sched_setattr");
      exit(-1);
      return false;
  } 
  // else {
  //     std::cerr << "[SCHED_DEADLINE] (" << _pid << ") exec_time: " << _exec_time << " _deadline: " << _deadline << " _period: " << _period << std::endl;
  // }
  return true;
}

bool init_task_scheduling(std::string policy, struct sched_attr attr){
  if(policy.compare(std::string("NONE")) == 0){
    return true;
  }
  else if(policy.compare(std::string("SCHED_FIFO")) == 0){
    return set_sched_fifo(getpid(), attr.sched_priority, 10);
  }
  else if(policy.compare(std::string("SCHED_RR")) == 0){
    return set_sched_rr(getpid(), attr.sched_priority, 10);
  }
  else if(policy.compare(std::string("SCHED_DEADLINE")) == 0){
    return set_sched_deadline(getpid(), attr.sched_runtime, attr.sched_deadline, attr.sched_period);
  }
  else{
    std::cout<<"[ERROR] Invalidate scheduling policy: "<<policy<<std::endl;
  }

  return true;
}

void yield_task_scheduling(){
  sched_yield();
}

void sig_handler(int signum){
  if(signum == SIGINT || signum == SIGTSTP || signum == SIGQUIT){
    termination();
  }
}

void termination(){
  printf("TERMINATION\n");
  if(remove(task_filename_.c_str())){
      printf("Cannot remove file %s\n", task_filename_);
      exit(1);
  }

  fclose(task_response_time_fp_);
  
  exit(0);
}

std::string get_cmd_output(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}

std::vector<int> get_child_pids(int pid){
  std::vector<int> child_pids;
  std::string cmd = "ps -ef -T | grep " + std::to_string(pid);
  std::string s = get_cmd_output(cmd.c_str());

  std::vector<std::string> task_ps_info_vec = tokenize_string(s, "\n");

  for(auto it = task_ps_info_vec.begin(); it != task_ps_info_vec.end(); ++it){
    std::string task_ps_info = *it;
    if(task_ps_info.find(std::string("grep")) != std::string::npos) continue;
    if(task_ps_info.find(std::string("ps -ef -T")) != std::string::npos) continue;
    std::vector<std::string> parsed_task_ps_info = tokenize_string(task_ps_info, " ");
    int child_pid = std::stoi(parsed_task_ps_info[2]);
    if(child_pid == pid) continue;
    child_pids.push_back(child_pid);
  }

  return child_pids;
}

std::vector<std::string> tokenize_string(std::string s, std::string delimiter){
  std::vector<std::string> output;
  size_t pos = 0;

  while ((pos = s.find(delimiter)) != std::string::npos) {
      std::string token = s.substr(0, pos);
      if(token.size() != 0) output.push_back(token);
      s.erase(0, pos + delimiter.length());
  }  

  return output;
}

} // namespace rubis