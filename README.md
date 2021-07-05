# Autoware On Embedded
Autoware system for embedded boards

## Environment

- Ubuntu 18.04
- ROS Melodic

## How to install ROS melodic
* ROS Melodic Install
```
sudo apt update
sudo apt install -y python-catkin-pkg python-rosdep ros-$ROS_DISTRO-catkin
sudo apt install -y python3-pip python3-colcon-common-extensions python3-setuptools python3-vcstool
pip3 install -U setuptools
```

## How to build Autoware

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

* Install dependent packages
```
cd {$WORKSPACE_DIR}/autoware.ai
rosdep update
rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO
```

* Autoware Build
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

<!-- Since Autoware recommend to use directory name 'autoware.ai', you should make soft link with autoware.ai to this repository
```
cd
ln -s ${WORKSPACE_DIR}/RUBIS-SelfDriving ~/autoware.ai
```

And it is recommned to add below sourcing command in your `~/.bashrc` file.
```
source ~/autoware.ai/install/setup.bash
``` -->

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
```

## Launch script for additional setup
```
# Launch setup script
cd ${WORKSPACE_DIR}/setup

# USER_NAME: directory name of home
# WORTSPACE_PATH: path for Autoware_On_Embedded
./setup.sh ${USER_NAME} ${WORKSPACE_PATH}
sudo bash setup_bashrc.sh ${USER_NAME}
```


<!-- ## Create symoblic links
```
ln -s ${WORKSPACE_DIR}/autoware.ai ~/autoware.ai
ln -s ${WORKSPACE_DIR}/rubis_ws ~/rubis_ws
``` -->

## How to launch LGSVL scrips
* Setup environments
```
cd ${WORKSPACE_DIR}/autoware_files/lgsvl_file/scripts
pip3 install --user .
```

* Launch LGSVL scripts
```
sudo chomod 755 {TARGET_SCRIPTS}
./{TARGET_SCRIPTS}
```

---
### Created by spiraline
