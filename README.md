## Environment

- Ubuntu 18.04
- ROS Melodic
- CUDA 10.x
- OpenCV 4.x

# 1. Jetson TX2 초기화 및 우분투 설치

[https://wooono.tistory.com/311](https://wooono.tistory.com/311)를 참고함

### a. 필요 파일 다운로드

[https://developer.nvidia.com/embedded/linux-tegra-r3251](https://developer.nvidia.com/embedded/linux-tegra-r3251)(링크 안될 시 그냥 nvidia l4t 라고 검색해볼 것) 에서 L4T Driver Package와 Sample Root Filesystem을 다운로드

Tegra186_Linux_R32.5.1_aarch64.tbz2, Tegra_Linux_Sample-Root-Filesystem_R32.5.1_aarch64.tbz2 이렇게 두 개 파일 필요

두 파일을 홈 디렉토리로 옮긴 뒤 Tegra186_Linux_R32.5.1_aarch64.tbz2의 압축 해제, 파일 내부의 rootfs 폴더로 이동

```bash
cd Jetson_Linux_R32.5.2_aarch64/Linux_for_Tegra/rootfs
```

나머지 파일 Tegra_Linux_Sample-Root-Filesystem_R32.5.1_aarch64.tbz2를 해당 폴더 안에 압축해제

```bash
sudo tar -jxpf ../../../Tegra_Linux_Sample-Root-Filesystem_R32.5.1_aarch64.tbz2
```

그 후 상위 경로에서 apply_binaries.sh 실행

```bash
cd ..
sudo ./apply_binaries.sh
```

정상 실행 시 Success! 출력

****추가**

Linux_for_Tegra 안에 p2771-0000.conf.common 안에서 isolcpus=1-2 지우기

이렇게 해야 덴버가 켜짐

### b. TX2와 연결

TX2에 전원 넣고 컴퓨터와 usb-5핀으로 연결하고 Recovery 모드로 진입

꼭짓점 부분부터 1, 2, 3, 4번 버튼이라고 하면 

- 전원코드 뽑았다가 꽂기
- 4번 눌러서 전원 켜기
- 3번 누른채로 1번 누르고, 3초 뒤 3번 떼기

화면은 안 나옴, lsusb로 NVidia Corp.라는 장치가 연결되었는지 확인 

### c. Flash하기

a에서 준비한 Linux_for_Tegra 폴더에서 다음 명령어 실행해서 flash

```bash
 sudo ./flash.sh jetson-tx2 mmcblk0p1
```

약 10분 소요, 완료 시 TX2 자동 재부팅

# 2. 우분투 초기 설정 및 ROS, Autoware 설치

**우분투 초기 설정 시 MAXN 모드로 설정할 것**

이래야 코어 다 씀

## How to install ROS melodic

```bash
sudo apt-key del 421C365BD9FF1F717815A3895523BAEEB01FA116
sudo -E apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt clean && sudo apt update
sudo apt install
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl -y
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install ros-melodic-desktop-full -y
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential python-rosdep -y
sudo rosdep init
rosdep update
```

## How to build Autoware

### CUDA setting

```bash
sudo apt-get install cuda-toolkit-10-2 -y
echo "# Add 32-bit CUDA library & binary paths:" >> ~/.bashrc
echo "export PATH=/usr/local/cuda-10.2/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

### System Dependencies of Ubuntu 18.04 / ROS Melodic

```bash
sudo apt-get update
sudo apt install -y python-catkin-pkg python-rosdep ros-$ROS_DISTRO-catkin
sudo apt install -y python3-pip python3-colcon-common-extensions python3-setuptools python3-vcstool
pip3 install -U setuptools
```

### Eigen build

```bash
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

새로 밀고 하는거면 안해도 됨

```bash
sudo rm /usr/lib/cmake/eigen3/*
sudo cp /usr/local/share/eigen3/cmake/* /usr/lib/cmake/eigen3
```

### Download Autoware_On_Embedded, minicar branch

```bash
sudo apt install git
mkdir git
cd git
git clone --single-branch -b minicar https://github.com/rubis-lab/Autoware_On_Embedded.git
```

### Install dependent packages

```bash
cd git/Autoware_On_Embedded/autoware.ai
rosdep update
rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO
```

### Resolving OpenCV version issue

You should change some code to use OpenCV 4.x

```
sudo apt-get install libopencv3.2 -y
```

Change `set(_include_dirs "include;/usr/include;/usr/include/opencv")`

to `set(_include_dirs "include;/usr/include;/usr/include/opencv4")`

in below three files (`sudo` required)

- `/opt/ros/melodic/share/cv_bridge/cmake/cv_bridgeConfig.cmake`
- `/opt/ros/melodic/share/image_geometry/cmake/image_geometryConfig.cmake`
- `/opt/ros/melodic/share/grid_map_cv/cmake/grid_map_cvConfig.cmake`

### CUDA install

```bash
sudo apt-get install cuda-toolkit-10-2
echo "# Add 32-bit CUDA library & binary paths:" >> ~/.bashrc
echo "export PATH=/usr/local/cuda-10.2/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

### Autoware Build

```bash
cd git/Autoware_On_Embedded/autoware.ai
# If you have CUDA
AUTOWARE_COMPILE_WITH_CUDA=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
# Build only some package
AUTOWARE_COMPILE_WITH_CUDA=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select $(pakcage name)
# Build without some package
AUTOWARE_COMPILE_WITH_CUDA=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-skip $(pakcage name)
```

### rubis_ws build

```bash
sudo apt-get install ros-melodic-ackermann-msgs ros-melodic-serial -y

# Launch setup script
cd ${WORKSPACE_DIR}/setup
# USER_NAME: directory name of home
# WORTSPACE_PATH: path for Autoware_On_Embedded
./setup.sh ${USER_NAME} ${WORKSPACE_PATH}

cd ${WORKSPACE_DIR}/rubis_ws
catkin_make

```

### Download **Vision-Lane-Keeping, install dependency**

```bash
git clone https://github.com/Spiraline/Vision-Lane-Keeping.git
pip3 install rospkg
pip3 install scikit-build
pip3 install --upgrade pip setuptools wheel
pip3 install cmake
pip3 install opencv-python
echo "export OPENBLAS_CORETYPE=ARMV8 python3" >> ~/.bashrc
source ~/.bashrc
```

# 3. 추가 설정

## 한글 설정

(1) Laguage support - [install/remove language] - [Korean] 설치 (설치 안 될 시 재부팅?)

(2) Terminal: ibus-setup > [Input method]-[Add]-[Korean/Hangul] 추가

(3) Setting-[Region & Language]

(3-2) Input method: 한국어 (Hangul)

TX2에 할 경우 region & language가 사라져있는 경우도 있는데 아래 치면 나옴

```xml
gnome-control-center region
```

## LiDAR 설정

라이다-레이저측정장치-보드 연결

superkey → "Networks Connections"

Connections name: velodyne_interface (이름 이걸로 꼭 해줘야 함)

Method: manual

Address: 192.168.1.100 / Netmask: 255.255.255.0 / Gateway: 0.0.0.0 (ENTER&save)

sudo ifconfig eth0 192.168.3.100

주소창에 192.168.1.201 접속→확인

다시 터미널

```bash
sudo apt-get install ros-melodic-velodyne
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src/ && git clone https://github.com/ros-drivers/velodyne.git
rosdep install --from-paths src --ignore-src --rosdistro melodic -y
cd ~/catkin_ws/ && catkin_make
```

이후 runtime manager에서 키고 확인
