#include <ros_autorunner/ros_autorunner.h>

void ROSAutorunner::init(ros::NodeHandle nh, Sub_v sub_v){
    if(signal(SIGINT, sig_handler) == SIG_ERR){
        std::cout<<"\nCannot catch SIGINT\n";
    }

    nh_ = nh;
    if(!nh_.getParam("total_step_num", total_step_num_)){
        throw std::runtime_error("Cannot read total steps!");
        exit(1);
    }

    if(!nh_.getParam("terminate_script_path", terminate_script_path_)){
        throw std::runtime_error("Cannot read terminate script path!");
        exit(1);
    }
    
    step_info_list_.clear();
    step_info_list_.resize(total_step_num_);

    if(sub_v.size() != total_step_num_){
        ROS_WARN("Subscriber list size : %d / total_step_number : %d", (int)sub_v.size(), total_step_num_);
        ROS_ERROR("Subscriber list size is not equal to total step number");
        exit(1);
    }

    for(auto it = step_info_list_.begin(); it != step_info_list_.end(); ++it){
        std::string step_str;
        std::vector<std::string> step;
        StepInfo_it step_info = it;
        int idx = step_info - step_info_list_.begin();

        step_str = std::string("step_") + std::to_string(INT_TO_STEP(idx));
        if(!nh_.getParam(step_str, step)){
            throw std::runtime_error("There is no "+step_str+"!");
            exit(1);
        }

        step_info->step_id = idx;
        step_info->pkg_name = step[PKG_NAME];
        step_info->target_name = step[TARGET_NAME];
        step_info->is_prepared = false;

        // Create constraint topic or not
        if(step[CREATE_TOPIC] == "true")
            step_info->create_topic = true;
        else if(step[CREATE_TOPIC] == "false")
            step_info->create_topic = false;
        else{
            throw std::runtime_error("[Create Topic] is wrong ( only true or false )");
            exit(1);
        }

        // Check constraint topic or not
        if(step[CHECK_TOPIC] == "true")
            step_info->check_topic = true;
        else if(step[CHECK_TOPIC] == "false")
            step_info->check_topic = false;
        else{
            throw std::runtime_error("[Check Topic] is wrong ( only true or false )");
            exit(1);
        }

        // Target type
        if(step[TARGET_TYPE] == "RUN"){
            step_info->target_type = RUN;
        }
        else if(step[TARGET_TYPE] == "LAUNCH"){
            ROS_WARN("LAUNCH!");
            step_info->target_type = LAUNCH;
        }
        else{
            throw std::runtime_error("[Target Type] is wrong ( only RUN or LAUNCH )");
            exit(1);
        }

        if(step_info->create_topic)
            step_info->sub = sub_v[idx];
    }

    current_step_ = step_info_list_.begin();
    print_step_info_list();
}

void ROSAutorunner::Run(){
    if(current_step_ == step_info_list_.end()){
        return;
    }

    if(!current_step_->check_topic)
        current_step_->is_prepared = true;

    if(current_step_->is_prepared == true){
        ROS_WARN("[Step %d] Activated", INT_TO_STEP(current_step_->step_id));
        if(current_step_->target_type == RUN)
            run_node(current_step_->step_id);
        else if(current_step_->target_type == LAUNCH)
            launch_script(current_step_->step_id);
        else{
            throw std::runtime_error("[Target Type] is wrong");
            exit(1);
        }

        if(current_step_ != step_info_list_.begin()){
            StepInfo_it prev_step = current_step_ - 1;
            prev_step->sub.shutdown();
        }
        ++current_step_;
    }      

    return;
}

void ROSAutorunner::run_node(int step_id){
    std::string pkg_name = step_info_list_[step_id].pkg_name;
    std::string node_name = step_info_list_[step_id].target_name;
    std::string run_str = create_run_string(pkg_name, node_name);

    if(fork() == CHILD){
        system(run_str.c_str());
        exit(0);
    }
}

void ROSAutorunner::launch_script(int step_id){
    std::string pkg_name = step_info_list_[step_id].pkg_name;
    std::string node_name = step_info_list_[step_id].target_name;
    std::string launch_str = create_launch_string(pkg_name, node_name); 
    
    if(fork() == CHILD){
        system(launch_str.c_str());
        exit(0);
    }
}

std::string ROSAutorunner::create_run_string(std::string pkg_name, std::string node_name){
    return "rosrun " + pkg_name + " " + node_name;
}

std::string ROSAutorunner::create_launch_string(std::string pkg_name, std::string launch_name){
    return "roslaunch " + pkg_name + " " + launch_name;
}

void ROSAutorunner::print_step_info_list(){
    ROS_INFO("============================================================");        
    ROS_INFO("      < Step Infos >");
    ROS_INFO(" ");

    int step_cnt = 0;
    for(StepInfo_it it = step_info_list_.begin(); it != step_info_list_.end(); ++it){
        StepInfo_it step_info = it;
        ROS_INFO("  %d. Pacakge Name    : %s", INT_TO_STEP(step_cnt), step_info->pkg_name.c_str());
        ROS_INFO("      Target Name     : %s", step_info->target_name.c_str());
         ROS_INFO("     Target Type     : %s", step_info->target_type == RUN? "RUN" : "LAUNCH");
        ROS_INFO(" ");   
        step_cnt ++;
    }
    ROS_INFO("============================================================");      
}

void sig_handler(int signo){
    if (signo==SIGINT){
        ROS_WARN("Termiate all");
        system(terminate_script_path_.c_str());
        exit(0);   
    }
}