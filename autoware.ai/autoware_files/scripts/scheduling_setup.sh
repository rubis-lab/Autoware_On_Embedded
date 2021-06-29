if [[ $(id -u) -eq 0 ]]; then 
	echo "Please don't execute this script as root user. \n"
  exit
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

