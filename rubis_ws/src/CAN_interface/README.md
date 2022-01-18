# RUBIS CAN Interface

## **Installation**

This package needs Kvaser interface package. And the kvaser_interface package depends on the Kvaser CANLIB API. You can install the Kvaser CANLIB from source directly from Kvaser, however the easiest way to install is using our ppa which distributes them as deb packages:

```
sudo apt-add-repository ppa:astuff/kvaser-linux
sudo apt update
sudo apt install kvaser-canlib-dev kvaser-drivers-dkms
```

Now that the dependencies are installed, we can install kvaser_interface:

```
sudo apt install apt-transport-https
sudo sh -c 'echo "deb [trusted=yes] https://s3.amazonaws.com/autonomoustuff-repo/ $(lsb_release -sc) main" > /etc/apt/sources.list.d/autonomoustuff-public.list'
sudo apt update
sudo apt install ros-$ROS_DISTRO-kvaser-interface
```

### **Now you are ready to build the RUBIS CAN interface package**

* Build packages
```
cd ${WORKSPACE_DIR}
catkin_make
```

* source packages
```
source ${WORKSPACE_DIR}/devel/setup.sh
```

# The `can_traslate` Node
This package generates two messages and translates CAN data from the ECU Gateway. You can use the messages by including the header <can_data_msgs/Car_output.h> and <can_data_msgs/Car_input.h>

## **SUBSCRIBE TOPIC**

<!-- ### can_tx [can_msgs::Frame]

This topic is subscribed by the node. It expects to have other nodes subscribe to it to receive data which are *sent by the CAN device*. -->

### car_ctrl_input [can_data_msgs::Car_ctrl_input] // Board -> Vehicle

This topic is subscribed to by the node. This topic contains a vehicle control data.
```
    int set_accel;
    int set_steering;
    double angle_filter_hz;
    double steering_angle; # steering wheel angle
    double acceleration;   
```

## **PUBLISH TOPIC**

<!-- ### can_rx [can_msgs::Frame]

This topic is published to by the node. It expects to have data published to it which are intended to be *received by the CAN device*. -->

### car_ctrl_output [can_data_msgs::Car_ctrl_output] // Vehicle -> Board

This topic is publish the translated data from the CAN.
```
    int accel_status;
    int steering_status;
    int steering_ctrl_override;
    int steering_error;
    int turn_indicator;
    int accel_override;
    int brake_override;
    double steering_angle; # steering wheel angle
    double real_speed;
    double ctrl_accel;
    int gear;
    int alive_count;
```




