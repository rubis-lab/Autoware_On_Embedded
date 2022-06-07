#include <sys/wait.h>
#include <unistd.h>
#include <ros/ros.h>
#include <string>
#include <stdlib.h>
#include <sys/types.h>

static int pid;

void INT_handler(int sig){
    if (pid != 0){
        kill(-1 * (pid), SIGINT);
    }
    exit(0);
}

int main(int argc, char *argv[]){
    ros::init(argc, argv, "ntrip_client");
    ros::NodeHandle nh;

    //  register SIGINT handler
    struct sigaction action;
    action.sa_handler = INT_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    if (sigaction(SIGINT, &action, NULL) < 0){
        ROS_ERROR("ntrip : cannot install signal handler");
        exit(EXIT_FAILURE);
    }

    std::string file_path;
    ros::param::get("/ntrip_file_path", file_path);

    char ntrip_file_path_cstr[200];
    int str_len = file_path.length();
    if (str_len >= 200)
    {
        ROS_ERROR("ntrip : file path is too long");
        exit(0);
    }
    strcpy(ntrip_file_path_cstr, file_path.c_str());
    char *exe_argv[] = {"/bin/bash",
                        "-c",
                        ntrip_file_path_cstr,
                        NULL};

    while (ros::ok()){
        if ((pid = fork()) < 0){
            ROS_ERROR("ntrip : cannot create child process");
        }

        //  child process
        if (pid == 0){
            if (execvp(exe_argv[0], exe_argv) < 0){
                ROS_ERROR("ntrip : cannot execute script file");
            }
            exit(EXIT_FAILURE);
        }
        //  parent process
        else{
            int wstatus;
            while (waitpid(pid, &wstatus, WNOHANG) == 0){
                sleep(2);
            }
        }
    }
}