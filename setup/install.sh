# Install ROS melodic
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl -y
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
if ! sudo apt install ros-melodic-desktop-full -y; then
    echo "[System] ROS Install Fail"
    exit 1
fi

source /opt/ros/melodic/setup.bash
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential python-rosdep -y
sudo rosdep init
rosdep update

# System Dependencies of Ubuntu 18.04 / ROS Melodic
sudo apt-get update -y
sudo apt install -y python-catkin-pkg python-rosdep ros-melodic-catkin
sudo apt install -y python3-pip python3-colcon-common-extensions python3-setuptools python3-vcstool
pip3 install -U setuptools

# Eigen Build
if [ ! -d "/usr/local/share/eigen3" ]; then
    wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
    mkdir eigen
    tar --strip-components=1 -xzvf eigen-3.3.7.tar.gz -C eigen
    cd eigen
    mkdir build
    cd build
    cmake ..
    if ! make; then
        cd ../..
        rm -rf eigen eigen-3.3.7.tar.gz
        echo "[System] Eigen Build Fail"
        exit 1
    fi
    sudo make install
    cd ../..
    rm -rf eigen eigen-3.3.7.tar.gz
    sudo rm /usr/lib/cmake/eigen3/*
    sudo cp /usr/local/share/eigen3/cmake/* /usr/lib/cmake/eigen3
fi

# Install rosdep
cd autoware.ai
rosdep update
if ! rosdep install -y --from-paths src --ignore-src --rosdistro melodic -r; then
    echo "[System] rosdep install Fail"
    exit 1
fi
cd ..

# Resolve OpenCV version issue
sudo apt-get install libopencv3.2 -y
if [ -d "/usr/include/opencv4" ]; then
    sudo cp setup/cv_bridgeConfig.cmake /opt/ros/melodic/share/cv_bridge/cmake
    sudo cp setup/image_geometryConfig.cmake /opt/ros/melodic/share/image_geometry/cmake
    sudo cp setup/grid_map_cvConfig.cmake /opt/ros/melodic/share/grid_map_cv/cmake
fi

# Autoware Build
if [ ! -d "autoware.ai/install" ]; then
    cd autoware.ai
    if [ -d "/usr/local/cuda" ]; then
        if ! AUTOWARE_COMPILE_WITH_CUDA=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release; then
            echo "[System] Autoware Build Fail"
            exit 1
        fi
    else
        if ! colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release; then
            echo "[System] Autoware Build Fail"
            exit 1
        fi
    fi
    cd ..
    ln -s $(pwd)/autoware.ai ~/autoware.ai
    source ~/autoware.ai/install/setup.bash
fi

# Build rubis_ws
if [ ! -d "rubis_ws/devel" ]; then
    sudo apt-get install ros-melodic-ackermann-msgs ros-melodic-serial ros-melodic-veldoyne ros-melodic-velodyne-driver -y
    cd rubis_ws/src
    catkin_init_workspace
    cd ..
    if ! catkin_make; then
        echo "[System] rubis_ws Build Fail"
        exit 1
    fi
    cd ..
    ln -s $(pwd)/rubis_ws ~/rubis_ws
    source ~/rubis_ws/devel/setup.bash
fi

# Build other files
sudo apt-get install -y ros-melodic-rosbridge-server net-tools
sudo apt-get install -y ros-melodic-can-msgs

# Kvaser Interface
sudo apt-add-repository ppa:astuff/kvaser-linux
sudo apt update
sudo apt install -y kvaser-canlib-dev kvaser-drivers-dkms

sudo apt install -y apt-transport-https
sudo sh -c 'echo "deb [trusted=yes] https://s3.amazonaws.com/autonomoustuff-repo/ $(lsb_release -sc) main" > /etc/apt/sources.list.d/autonomoustuff-public.list'
sudo apt update
sudo apt install -y ros-melodic-kvaser-interface

# Python packages
pip3 install --upgrade setuptools pip
pip3 install -U PyYAML
pip3 install rospkg matplotlib opencv-python pandas
cd setup/svl
python3 -m pip install -r requirements.txt --user .
cd ../..

# Allow Firewall
sudo ufw allow 9090
