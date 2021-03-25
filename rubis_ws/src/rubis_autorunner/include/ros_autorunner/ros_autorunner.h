#include <ros/ros.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <ros_autorunner/ros_autorunner_def.h>
#include <signal.h>

class StepInfo{
public:    
    int                 step_id;
    bool                is_prepared;
    std::string         pkg_name;
    std::string         target_name;
    bool                create_topic;
    bool                check_topic;
    int                 target_type;
    ros::Subscriber     sub;
};

typedef std::vector<ros::Subscriber>                 Sub_v;
typedef std::vector<ros::Subscriber>::iterator       Sub_it;
typedef std::vector<StepInfo>                        StepInfo_v;
typedef std::vector<StepInfo>::iterator              StepInfo_it;

class ROSAutorunner{
private:
    ros::NodeHandle     nh_;
    int                 total_step_num_; // Read from cfg  
    StepInfo_it         current_step_;                
private:
    void        run_node(int step_id);             // rosrun rosnode
    void        launch_script(int step_id);        // rosalaunch script
    void        print_step_info_list();
    std::string create_run_string(std::string pkg_name, std::string node_name);
    std::string create_launch_string(std::string pkg_name, std::string launch_name);
public:
    StepInfo_v          step_info_list_;
public:
    ROSAutorunner(){}
    void init(ros::NodeHandle nh, Sub_v sub_v);
    void Run();         // Execute each steps
};

class AutorunnerBase{
protected:    
    ros::NodeHandle     nh_;
    ROSAutorunner        ros_autorunner_;
protected:
    virtual void register_subscribers() = 0;
public:
    Sub_v               sub_v_;
public:
    AutorunnerBase(){ std::cout<<"\tBase default constructor"<<std::endl; }
    AutorunnerBase(ros::NodeHandle nh){ std::cout<<"\tBase nodehandle constructor"<<std::endl; }
};

static std::string                 terminate_script_path_;       
static void sig_handler(int signo);
