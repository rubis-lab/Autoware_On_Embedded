# **Xavier Autoware**


## Environment

- L4T 4.3
  - Ubuntu 18.04
  - CUDA 10.2
  - OpenCV 4.1.1

- ROS Melodic
- Qt 5.9.5
- Eigen 3.3.7
- Autoware (master version)

## Install cuda
```
sudo apt-get install libopencv libopencv-python \
cuda-toolkit-10-2 libopencv-dev opencv-licenses
```

### Cuda Library path
In .bashrc
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```


## How to install ROS melodic
* ROS Melodic Install
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-melodic-desktop-full

echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential -y

sudo rosdep init
rosdep update
```

Make sure `$ROS_DISTRO` environment is `melodic`.

* Link to OpenCV4
Since we use OpenCV4, we should link ros to OpenCV4
Change line in `/opt/ros/melodic/share/grid_map_cv/cmake/grid_map_cvConfig.cmake`
from
`set(_include_dirs "include;/usr/include;/usr/include/opencv")`
to
`set(_include_dirs "include;/usr/include;/usr/include/opencv4")`

## How to build Autoware
* Prerequisite
```
sudo apt update
sudo apt install -y python-catkin-pkg python-rosdep ros-$ROS_DISTRO-catkin
sudo apt install -y python3-pip python3-colcon-common-extensions python3-setuptools python3-vcstool
pip3 install -U setuptools
```

* Eigen build
```
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
mkdir eigen
tar --strip-components=1 -xzvf eigen-3.3.7.tar.gz -C eigen
cd eigen
mkdir build
cd build
cmake ..
make
sudo make install
```

Older versions may already be installed. If `/usr/lib/cmake/eigen3/Eigen3Config.cmake` is older than 3.3.7 version, copy files in `/usr/local/share/eigen3/cmake` to `/usr/lib/cmake/eigen3`.

```
sudo rm /usr/lib/cmake/eigen3/*
sudo cp /usr/local/share/eigen3/cmake/* /usr/lib/cmake/eigen3
```

* Install dependent packages
```
cd {$WORKSPACE_DIR}/autoware.ai
rosdep update
rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO
```

* Autoware Build
For now, you should skip `trafficlight_recognizer` package.

```
# If you have CUDA
AUTOWARE_COMPILE_WITH_CUDA=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Build only some package
AUTOWARE_COMPILE_WITH_CUDA=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select $(pakcage name)

# Build without some package
AUTOWARE_COMPILE_WITH_CUDA=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-skip $(pakcage name)

# If you don't have CUDA
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Since Autoware recommend to use directory name 'autoware.ai', you should make soft link with autoware.ai to this repository
```
cd
ln -s ${WORKSPACE_DIR}/autoware.ai ~/autoware.ai
```

And it is recommned to add below sourcing command in your `~/.bashrc` file.
```
echo "source ~/autoware.ai/install/setup.bash" >> ~/.bashrc
```

## How to build package in rubis_ws

* Initialize ROS workspace
```
cd ${WORKSPACE_DIR}/rubis_ws/src
catkin_init_workspace
```

* Build rubis_ws packages
```
cd ${WORKSPACE_DIR}/rubis_ws
catkin_make
ln -s ${WORKSPACE_DIR}/rubis_ws ~/rubis_ws
echo "source ~/rubis_ws/devel/setup.bash" >> ~/.bashrc
```

## Create symoblic links
```
ln -s ${WORKSPACE_DIR}/autoware.ai ~/autoware.ai
ln -s ${WORKSPACE_DIR}/rubis_ws ~/rubis_ws
```

