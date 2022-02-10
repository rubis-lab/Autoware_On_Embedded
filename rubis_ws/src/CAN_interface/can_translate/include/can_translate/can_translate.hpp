#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <inttypes.h>
#include <cstdlib>
#include <string>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <cmath>


#include <net/if.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/in.h>
#include <netdb.h>

#include <linux/can.h>
#include <linux/can/raw.h>

#include <kvaser_interface/include/kvaser_interface/kvaser_interface.h>
#include <can_msgs/Frame.h>

#include <can_data_msgs/Car_ctrl_output.h>
#include <can_data_msgs/Car_ctrl_input.h>

struct INPUT_CTRL_DATA{
    int set_accel;
    int set_steering;
    double angle_filter_hz;
    double steering_angle;
    double acceleration;    
};

struct OUT_CTRL_DATA{
    int accel_status;
    int steering_status;
    int steering_ctrl_override;
    int steering_error;
    int turn_indicator;
    int accel_override;
    int brake_override;
    double steering_angle;
    double real_speed;
    double ctrl_accel;
    int gear;
    int alive_count;
};

struct OUT_WHEEL_DATA{
    double front_left;
    double front_right;
    double rear_left;
    double rear_right;
};

struct OUT_STEERING_SENSOR{
    double steering_angle;
    double steering_speed;
};

class CanDataReadWrite{
    private:
        can_data_msgs::Car_ctrl_output out_topic;
        can_msgs::Frame send;

        struct can_frame can_frame_write;
        struct can_frame can_frame_read;

        struct OUT_CTRL_DATA output_data;
        struct INPUT_CTRL_DATA input_data;        
        struct OUT_WHEEL_DATA wheel_data;
        struct OUT_STEERING_SENSOR steering_sensor_data;     

        int prev_alive;
        int read_ctrl_flag;
        int pub_flag;

        int hardware_id;
        int circuit_id;
        int bit_rate;
        bool is_debug;

        /* Data handling */
        int data_pub_flag;

        /* From the CAN DBC file */

        /* CAN input data profile */
        int input_len;
        int input_id;

        /* CAN output data profile */

        //Ionic AUTO_OUT
        int accel_status_idx;
        int accel_status_len;
        
        int steering_status_idx;
        int steering_status_len;

        int steering_ctrl_override_idx;
        int steering_ctrl_override_len;

        int steering_error_idx;
        int steering_error_len;

        int turn_indicator_idx;
        int turn_indicator_len;

        int accel_override_idx;
        int accel_override_len;

        int brake_override_idx;
        int brake_override_len;

        int steering_angle_idx;
        int steering_angle_len;

        int real_speed_idx;
        int real_speed_len;

        int ctrl_accel_idx;
        int ctrl_accel_len;

        int gear_idx;
        int gear_len;

        int alive_count_idx;
        int alive_count_len;

        // Wheel_Speed
        int fl_idx;
        int fl_len;

        int fr_idx;
        int fr_len;

        int rl_idx;
        int rl_len;

        int rr_idx;
        int rr_len;

        // Steering_Angle_Sensor
        int sensor_steering_angle_idx;
        int sensor_steering_angle_len;

        int sensor_steering_speed_idx;
        int sensor_steering_speed_len;

    public:
        // void init_CanDataReadWrite();
        void init_CanDataReadWrite(int _hardware_id, int _circuit_id, int _bit_rate, bool _is_debug);
        //functions for read 
        inline int SIGNEX(unsigned int _value, unsigned int _size);

        inline double extract_value_from_data(uint32_t idx, uint32_t length, uint8_t* data, int is_signed);

        inline void extractVariables(uint32_t msgID, uint8_t *data);

        inline void write_input_data_to_can_frame();
        
        void convertDataToTopic();

        void convertTopicToData(const can_data_msgs::Car_ctrl_input& msg);

        void can_raw_read_callback(const can_msgs::Frame::ConstPtr& msg);

        void ctrl_data_callback(const can_data_msgs::Car_ctrl_input::ConstPtr& msg);

        int get_flag();

        void set_flag(int set);

        can_data_msgs::Car_ctrl_output get_out_topic();
};