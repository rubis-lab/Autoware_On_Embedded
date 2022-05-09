#include <ros/ros.h>
#include "can_translate/can_translate.hpp"
#include "geometry_msgs/TwistStamped.h"

class CanDataReadWrite candata;
AS::CAN::KvaserCan can_reader, can_writer;
ros::Publisher pub_translated_data;

int hardware_id = 0;
int circuit_id = 0;
int bit_rate = 0;
bool is_debug = false;

void CanDataReadWrite::init_CanDataReadWrite(int _hardware_id, int _circuit_id, int _bit_rate, bool _is_debug){

    /* kvaser setting */
    this->hardware_id = _hardware_id;
    this->circuit_id = _circuit_id;
    this->bit_rate = _bit_rate;
    this->is_debug = _is_debug;
    
    this->prev_alive = 0;
    this->read_ctrl_flag = 0;

    /* Data handling */
    this->data_pub_flag = 0;

    /* From the CAN DBC file */

    /* CAN input data profile */
    this->input_len = 6;
    this->input_id = 342;

    /* CAN output data profile */

    //Ionic AUTO_OUT
    this->accel_status_idx = 0;
    this->accel_status_len = 1;
    
    this->steering_status_idx = 1;
    this->steering_status_len = 1;

    this->steering_ctrl_override_idx = 2;
    this->steering_ctrl_override_len = 1;

    this->steering_error_idx = 3;
    this->steering_error_len = 1;

    this->turn_indicator_idx = 4;
    this->turn_indicator_len = 2;

    this->accel_override_idx = 6;
    this->accel_override_len = 1;

    this->brake_override_idx = 7;
    this->brake_override_len = 1;

    this->steering_angle_idx = 8;
    this->steering_angle_len = 16;

    this->real_speed_idx = 24;
    this->real_speed_len = 8;

    this->ctrl_accel_idx = 32;
    this->ctrl_accel_len = 11;

    this->gear_idx = 43;
    this->gear_len = 4;

    this->alive_count_idx = 56;
    this->alive_count_len = 8;

    // Wheel_Speed
    this->fl_idx = 0;
    this->fl_len = 14;

    this->fr_idx = 16;
    this->fr_len = 14;

    this->rl_idx = 32;
    this->rl_len = 14;

    this->rr_idx = 48;
    this->rr_len = 14;

    // Steering_Angle_Sensor
    this->sensor_steering_angle_idx = 0;
    this->sensor_steering_angle_len = 16;

    this->sensor_steering_speed_idx = 16;
    this->sensor_steering_speed_len = 8;
}

can_data_msgs::Car_ctrl_output CanDataReadWrite::get_out_topic(){
    return this->out_topic;
}

int CanDataReadWrite::get_flag(){
    return this->data_pub_flag;
}

void CanDataReadWrite::set_flag(int set){
    this->data_pub_flag = set;
}

void CanDataReadWrite::write_input_data_to_can_frame(){
    this->can_frame_write.can_dlc = this->input_len;
    this->can_frame_write.can_id = this->input_id;
    
    uint8_t set_accel_steering;
    uint16_t offset_applyed_accel;
    int16_t offset_applyed_angle;
    uint8_t offset_applyed_fliter_hz;
    uint16_t intput_acceleration;
    
    if(this->input_data.set_steering == 0 && this->input_data.set_accel == 0){
        set_accel_steering = 0;
    }else if(this->input_data.set_steering == 1 && this->input_data.set_accel == 0){
        set_accel_steering = 1;
    }else if(this->input_data.set_steering == 0 && this->input_data.set_accel == 1){
        set_accel_steering = 2;
    }else if(this->input_data.set_steering == 1 && this->input_data.set_accel == 1){
        set_accel_steering = 3;
    }

    offset_applyed_fliter_hz = (uint8_t)(this->input_data.angle_filter_hz * 10);
    offset_applyed_angle = (int16_t)(this->input_data.steering_angle * 10);
    offset_applyed_accel = (uint16_t)((this->input_data.acceleration + 10.23)* 100);

    memcpy(this->can_frame_write.data, &set_accel_steering, 1);
    memcpy(this->can_frame_write.data + 1, &offset_applyed_fliter_hz, 1);
    memcpy(this->can_frame_write.data + 2, &offset_applyed_angle, 2);
    memcpy(this->can_frame_write.data + 4, &offset_applyed_accel, 2);    
}

inline int CanDataReadWrite::SIGNEX(unsigned int _value, unsigned int _size){
    int ret = 0;
    ret |= _value;
    ret <<= ((sizeof(int) * 8) - _size);
    ret >>= ((sizeof(int) * 8) - _size);
        return ret;
}

inline double CanDataReadWrite::extract_value_from_data(uint32_t idx, uint32_t length, uint8_t* data, int is_signed){    
    unsigned int unsigned_signal = 0;
    unsigned int tmp_signal = 0;    
    int signed_signal = 0;
    int len;
    for(len = length - 1; len >= 0; len--){
        int row = (idx + len) / 8;
        int col = (idx + len) % 8;
        tmp_signal = (data[row] & (1 << col)) ? 1 : 0;
        unsigned_signal |= tmp_signal;
        
        if (len != 0){
            unsigned_signal <<= 1;
        }
    }
    if(is_signed){
        signed_signal = SIGNEX(unsigned_signal, length);
        return (double)signed_signal;
    }
    return (double)unsigned_signal;
}

inline void CanDataReadWrite::extractVariables(uint32_t msgID, unsigned char *data){
    int len;
    double tmp_value;
    int signed_signal;
    unsigned int tmp_signal;
    unsigned int unsigned_signal;

    switch (msgID){
        /*
         IONIQ CAN Gateway
         */
        // case 0x710 : //Ionic AUTO_OUT, Socket interface   
        case 1808 : //Ionic AUTO_OUT
            this->read_ctrl_flag = 1;

            //Alive count
            tmp_value = extract_value_from_data(this->alive_count_idx, this->alive_count_len, data, 0);
            this->output_data.alive_count = (tmp_value * 1.000000) + 0.000000;

            if(this->prev_alive == this->output_data.alive_count){                
                break;
            }else{
                this->data_pub_flag = 1;
                this->prev_alive = this->output_data.alive_count;                
            }

            //IonicautoAccelDecelStatus
            tmp_value = extract_value_from_data(this->accel_status_idx, this->accel_status_len, data, 0); 
            this->output_data.accel_status = (int)((tmp_value * 1.000000) + 0.000000);
            
            //IonicautoSteeringStatus
            tmp_value = extract_value_from_data(this->steering_status_idx,this->steering_status_len, data, 0);
            this->output_data.steering_status = (int)((tmp_value * 1.000000) + 0.000000);
                        
            //IonicautoSteeringCtlOverride
            tmp_value = extract_value_from_data(this->steering_ctrl_override_idx, this->steering_ctrl_override_len, data, 0);
            this->output_data.steering_ctrl_override = (int)((tmp_value * 1.000000) + 0.000000);
            
            //IonicautoSteeringError
            tmp_value = extract_value_from_data(this->steering_error_idx, this->steering_error_len, data, 0);
            this->output_data.steering_error = (int)((tmp_value * 1.000000) + 0.000000);
            
            //IonicautoTrunIndicator
            tmp_value = extract_value_from_data(this->turn_indicator_idx, this->turn_indicator_len, data, 0);
            this->output_data.turn_indicator = (int)((tmp_value * 1.000000) + 0.000000);
            
            //IonicautoAPSOverride
            tmp_value = extract_value_from_data(this->accel_override_idx, this->accel_override_len, data, 0);
            this->output_data.accel_override = (int)((tmp_value * 1.000000) + 0.000000);
            
            //IonicautoBPSOverride
            tmp_value = extract_value_from_data(this->brake_override_idx, this->brake_override_len, data, 0);
            this->output_data.brake_override = (int)((tmp_value * 1.000000) + 0.000000);
            
            //IonicautoSteeringAngle            
            tmp_value = extract_value_from_data(this->steering_angle_idx, this->steering_angle_len, data, 1);            
            this->output_data.steering_angle = (tmp_value * 0.10000) + 0.000000;            
            
            //IonicautoRealSPEED
            tmp_value = extract_value_from_data(this->real_speed_idx, this->real_speed_len, data, 0);            
            this->output_data.real_speed = (tmp_value * 1.000000) + 0.000000;
            
            //IonicautoCtlAx
            tmp_value = extract_value_from_data(this->ctrl_accel_idx, this->ctrl_accel_len, data, 0);
            this->output_data.ctrl_accel = (tmp_value * 0.010000) - 10.230000;
            
            //IonicautoGear
            tmp_value = extract_value_from_data(this->gear_idx, this->gear_len, data, 0);
            this->output_data.gear = (int)((tmp_value * 1.000000) + 0.000000);
            break;
        
        // case 0x386: //Wheel_Speed, Socket
        case 902: //Wheel_Speed
            //Wheel_Speed_FL
            tmp_value = extract_value_from_data(this->fl_idx, this->fl_len, data, 0);
            this->wheel_data.front_left = (tmp_value * 0.03125) + 0.000000;

            //Wheel_Speed_FR
            tmp_value = extract_value_from_data(this->fr_idx, this->fr_len, data, 0);
            this->wheel_data.front_right = (tmp_value * 0.03125) + 0.000000;

            //Wheel_Speed_RL
            tmp_value = extract_value_from_data(this->rl_idx, this->rl_len, data, 0);
            this->wheel_data.rear_left = (tmp_value * 0.03125) + 0.000000;

            //Wheel_Speed_RR
            tmp_value = extract_value_from_data(this->rr_idx, this->rr_len, data, 0);
            this->wheel_data.rear_right = (tmp_value * 0.03125) + 0.000000;
            break;

        // case 0x2b0: //Steering_Angle_Sensor, Socket
        case 688: //Steering_Angle_Sensor
            //Steering_Angle                        
            tmp_value = extract_value_from_data(this->sensor_steering_angle_idx, this->sensor_steering_angle_len, data, 1);
            this->steering_sensor_data.steering_angle = (tmp_value * 0.100000) + 0.000000;

            //Steering_Speed
            tmp_value = extract_value_from_data(this->sensor_steering_speed_idx, this->sensor_steering_speed_len, data, 0);
            this->steering_sensor_data.steering_speed = (tmp_value * 4.00000) + 0.000000;
            break;
        default:
            break;
    }

    if(this->read_ctrl_flag){
        this->read_ctrl_flag = 0;
        convertDataToTopic();        
    }
}

void CanDataReadWrite::convertTopicToData(const can_data_msgs::Car_ctrl_input& msg){
    this->input_data.set_accel = msg.set_accel;
    this->input_data.set_steering = msg.set_steering;
    this->input_data.angle_filter_hz = msg.angle_filter_hz;
    this->input_data.steering_angle = msg.steering_angle;
    this->input_data.acceleration = msg.acceleration;
}

void CanDataReadWrite::convertDataToTopic(){    
    this->out_topic.accel_status = this->output_data.accel_status;
    this->out_topic.steering_status = this->output_data.steering_status;
    this->out_topic.steering_ctrl_override = this->output_data.steering_ctrl_override;
    this->out_topic.steering_error = this->output_data.steering_error;
    this->out_topic.turn_indicator = this->output_data.turn_indicator;
    this->out_topic.accel_override = this->output_data.accel_override;
    this->out_topic.brake_override = this->output_data.brake_override;
    this->out_topic.steering_angle = this->output_data.steering_angle;
    this->out_topic.real_speed = this->output_data.real_speed;
    this->out_topic.ctrl_accel = this->output_data.ctrl_accel;
    this->out_topic.gear = this->output_data.gear;
    this->out_topic.alive_count = this->output_data.alive_count;
}

// Translate the Vehicle control topic to the CAN raw data
void CanDataReadWrite::ctrl_data_callback(const can_data_msgs::Car_ctrl_input::ConstPtr& msg){
    can_data_msgs::Car_ctrl_input m = *msg.get();
    AS::CAN::ReturnStatuses ret;

    if (!can_writer.isOpen())
    {
        if(this->is_debug){
            ROS_INFO("Writer");
            ROS_INFO("%d, %d, %d",this->hardware_id, this->circuit_id, this->bit_rate);
        }   

        // Open the channel.
        ret = can_writer.open(this->hardware_id, this->circuit_id, this->bit_rate, false);                 

        if (ret != AS::CAN::ReturnStatuses::OK)
        {
            ROS_ERROR_THROTTLE(0.5, "Kvaser CAN Interface - Error opening writer: %d - %s", static_cast<int>(ret),
                                AS::CAN::KvaserCanUtils::returnStatusDesc(ret).c_str());
        }
    }

    if (can_writer.isOpen())
    {
        AS::CAN::CanMsg msg;
        convertTopicToData(m);

        write_input_data_to_can_frame();

        msg.id = this->can_frame_write.can_id;
        msg.dlc = this->can_frame_write.can_dlc;

        auto msg_size = AS::CAN::KvaserCanUtils::dlcToSize(this->can_frame_write.can_dlc);

        for (size_t i = 0; i < msg_size; ++i)
        {
            msg.data.push_back(this->can_frame_write.data[i]);
        }

        ret = can_writer.write(std::move(msg));

        if (ret != AS::CAN::ReturnStatuses::OK)
        {
            ROS_WARN_THROTTLE(0.5, "Kvaser CAN Interface - CAN send error: %d - %s", static_cast<int>(ret),
                            AS::CAN::KvaserCanUtils::returnStatusDesc(ret).c_str());
        }
    }
}

void can_read()
{
    AS::CAN::ReturnStatuses ret; 

    while (true)
    {
        if (!can_reader.isOpen())
        {
            if(is_debug){
                ROS_INFO("Reader");
                ROS_INFO("%d, %d, %d",hardware_id, circuit_id, bit_rate);
            }   
            ret = can_reader.open(hardware_id, circuit_id, bit_rate, false);

            if (ret != AS::CAN::ReturnStatuses::OK)
            {
                ROS_ERROR_THROTTLE(0.5, "Kvaser CAN Interface - Error opening reader: %d - %s", static_cast<int>(ret),
                                    AS::CAN::KvaserCanUtils::returnStatusDesc(ret).c_str());
                break;
            }
        }

        if (can_reader.isOpen())
        {
            AS::CAN::CanMsg msg;

            ret = can_reader.read(&msg);
            if (ret == AS::CAN::ReturnStatuses::OK)
            {
                // Only publish if msg is not CAN FD,
                // a wakeup message, a transmit acknowledgement,
                // a transmit request, a delay notification,
                // or a failed single-shot.
                if (!(msg.flags.fd_msg || msg.flags.wakeup_mode || msg.flags.tx_ack || msg.flags.tx_rq ||
                        msg.flags.msg_delayed || msg.flags.tx_nack))
                {
                    struct can_frame can_frame_read;
                    can_frame_read.can_id = msg.id;
                    can_frame_read.can_dlc = msg.dlc;
                    
                    for(int i = 0; i < 8; i++)
                        can_frame_read.data[i] = msg.data[i];

                    candata.extractVariables(can_frame_read.can_id, can_frame_read.data);
    
                    if(candata.get_flag()){
                        pub_translated_data.publish(candata.get_out_topic());
                        candata.set_flag(0);
                    }                    
                }
            }
            else
            {
                if (ret != AS::CAN::ReturnStatuses::NO_MESSAGES_RECEIVED)
                    ROS_WARN_THROTTLE(0.5, "Kvaser CAN Interface - Error reading CAN message: %d - %s", static_cast<int>(ret),
                                    AS::CAN::KvaserCanUtils::returnStatusDesc(ret).c_str());

                break;
            }
        }
    }
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "can_translate");
    ros::Subscriber sub_can_raw_data;
    ros::Subscriber sub_can_ctrl_data;
    ros::NodeHandle nh;
    
    sub_can_ctrl_data = nh.subscribe("/car_ctrl_input", 1, &CanDataReadWrite::ctrl_data_callback, &candata);
    pub_translated_data = nh.advertise<can_data_msgs::Car_ctrl_output>("/car_ctrl_output", 0);

    nh.param<int>("/can_translate/hardware_id", hardware_id, 0);
    nh.param<int>("/can_translate/circuit_id", circuit_id, 0);
    nh.param<int>("/can_translate/bit_rate", bit_rate, 0);
    nh.param<bool>("/can_translate/is_debug", is_debug, 0);
    
    candata.init_CanDataReadWrite(hardware_id, circuit_id, bit_rate, is_debug);

    ros::AsyncSpinner spinner(1);
    
    // Wait for time to be valid
    ros::Time::waitForValid();

    AS::CAN::ReturnStatuses ret;

    ROS_INFO("Initial Setup");
    ROS_INFO("hardware_id : %d",hardware_id);
    ROS_INFO("circuit_id : %d",circuit_id);
    ROS_INFO("bit_rate : %d",bit_rate);
    ROS_INFO("is_debug : %d",is_debug);

    // Open CAN reader channel
    ret = can_reader.open(hardware_id, circuit_id, bit_rate, false);
    
    if (ret == AS::CAN::ReturnStatuses::OK)
    {
        // Set up read callback
        ret = can_reader.registerReadCallback(can_read);

        if (ret == AS::CAN::ReturnStatuses::OK)
        {
            // Only start spinner if reader initialized OK
            spinner.start();
        }
        else
        {
            ROS_ERROR("Kvaser CAN Interface - Error registering reader callback: %d - %s", static_cast<int>(ret),
                        AS::CAN::KvaserCanUtils::returnStatusDesc(ret).c_str());
            ros::shutdown();
            return -1;
        }
    }
    else
    {
        ROS_ERROR("Kvaser CAN Interface - Error opening reader: %d - %s", static_cast<int>(ret),
                AS::CAN::KvaserCanUtils::returnStatusDesc(ret).c_str());
        ros::shutdown();
        return -1;
    }

    ros::waitForShutdown();

    if (can_reader.isOpen())
    {
        ret = can_reader.close();

        if (ret != AS::CAN::ReturnStatuses::OK)
            ROS_ERROR("Kvaser CAN Interface - Error closing reader: %d - %s", static_cast<int>(ret),
                        AS::CAN::KvaserCanUtils::returnStatusDesc(ret).c_str());
    }

    if (can_writer.isOpen())
    {
        ret = can_writer.close();

        if (ret != AS::CAN::ReturnStatuses::OK)
            ROS_ERROR("Kvaser CAN Interface - Error closing writer: %d - %s", static_cast<int>(ret),
                        AS::CAN::KvaserCanUtils::returnStatusDesc(ret).c_str());
    }

    return 0;
}
