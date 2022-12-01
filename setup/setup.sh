#!/bin/bash
if [ -z "$1" ]
  then echo "Please proper arguments."
  echo "arg1: username"
  echo "arg2: Filepath of Autoware_On_Embedded"
  exit
fi

if [ -z "$2" ]
  then echo "Please proper arguments."
  echo "arg1: username"
  echo "arg2: Filepath of Autoware_On_Embedded"
  exit
fi

if [ ! -d $2/autoware.ai ]
    then echo "Cannot find autoware or rubis_ws directory at $2"
    exit
fi

if [[ $(id -u) -eq 0 ]]; then 
	echo "Please don't execute this script as root user. \n"
  exit
fi

if [ ! -d /home/$1/autoware.ai ]
    then ln -s $2/autoware.ai ~/autoware.ai
fi

if [ ! -d /home/$1/rubis_ws ]
    then ln -s $2/rubis_ws ~/rubis_ws
fi


if [ ! -d ~/Documents/profiling ]; then
    mkdir ~/Documents/profiling
    printf "~/Documents/profiling is created.\n"
fi

if [ ! -d ~/Documents/profiling/response_time ]; then
    mkdir ~/Documents/profiling/response_time
    printf "~/Documents/profiling/response_time is created.\n"
fi

if [ ! -d ~/Documents/gpu_profiling ]; then
    mkdir ~/Documents/gpu_profiling
    printf "~/Documents/gpu_profiling is created.\n"
fi

if [ ! -d ~/Documents/gpu_profiling ]; then
    mkdir ~/Documents/gpu_profiling
    printf "~/Documents/gpu_profiling is created.\n"
fi

if [ ! -d ~/Documents/gpu_deadline ]; then
    mkdir ~/Documents/gpu_deadline
    printf "~/Documents/gpu_deadline is created.\n"
fi

echo "Necessary directory paths are created to /home/${1}/Documents"

# sudo ./setup_bashrc.sh $1

