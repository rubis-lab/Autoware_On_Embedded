#include <vector>
#include <ros/ros.h>
#include <ros/time.h>
#include <std_msgs/Header.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <autoware_msgs/RUBISTrafficSignal.h>
#include <autoware_msgs/RUBISTrafficSignalArray.h>
#include <autoware_msgs/ExtractedPosition.h>

int main(int argc, char* argv[]){
    // Initialize
    ros::init(argc, argv, "fake_traffic_signal_generator");
    ros::NodeHandle nh;
    double traffic_signal_rate;
    double remain_time = 0.0;
    std::vector<int> signal_seq;
    int current_signal_seq_idx = 0;
    double spin_rate = 10.0;

    ros::Rate rate(spin_rate);

    ros::Publisher traffic_signal_pub;
    ros::Publisher stop_line_rviz_pub;

    traffic_signal_pub = nh.advertise<autoware_msgs::RUBISTrafficSignalArray>("/v2x_traffic_signal", 10);
    stop_line_rviz_pub = nh.advertise<visualization_msgs::MarkerArray>("/stop_line_marker", 10);

    // Add Traffic Signal Info from yaml file
    XmlRpc::XmlRpcValue traffic_light_list;
    nh.getParam("/fake_traffic_signal_generator/traffic_light_list", traffic_light_list);

    // Add Traffic Signal Info from yaml file
    XmlRpc::XmlRpcValue stop_line_list;
    nh.getParam("/fake_traffic_signal_generator/stop_line_list", stop_line_list);

    // Get Traffic Signal Sequence
    nh.getParam("/fake_traffic_signal_generator/traffic_signal_sequence", signal_seq);

    // Make Visualization Marker msg
    visualization_msgs::MarkerArray stop_line_marker_array;

    for(int i=0; i<stop_line_list.size(); i++){
        visualization_msgs::Marker marker;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();

        int id = stop_line_list[i]["id"];
        marker.id = id;
        marker.ns = std::to_string(id);

        marker.pose.position.x = stop_line_list[i]["pose"]["x"];
        marker.pose.position.y = stop_line_list[i]["pose"]["y"];
        marker.pose.position.z = stop_line_list[i]["pose"]["z"];
        marker.pose.orientation.w = 1;

        marker.scale.x = 1;
        marker.scale.y = 1;
        marker.scale.z = 1;

        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0;

        marker.lifetime = ros::Duration();

        visualization_msgs::Marker text_marker;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.header.frame_id = "world";
        text_marker.ns = "text";
        text_marker.header.stamp = ros::Time::now();
        text_marker.id = -id;
        text_marker.text = "StopLine " + std::to_string(id);
        text_marker.pose = marker.pose;
        text_marker.scale.z = 2;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;

        stop_line_marker_array.markers.push_back(marker);
        stop_line_marker_array.markers.push_back(text_marker);
    }

    // Make signal msg
    autoware_msgs::RUBISTrafficSignalArray signal_msg;
    for(int i=0; i<stop_line_list.size(); i++){
        autoware_msgs::RUBISTrafficSignal sig;
        sig.id = stop_line_list[i]["tl_id"];
        signal_msg.signals.push_back(sig);
    }

    while(ros::ok()){
        if(remain_time < 0.05){ // check if remain time is 0
            nh.param<double>("/fake_traffic_signal_generator/traffic_signal_rate", traffic_signal_rate, 0.3);
            remain_time = 1 / traffic_signal_rate;

            for(int i=0; i<stop_line_list.size(); i++){
                signal_msg.signals.at(i).type = signal_seq.at(current_signal_seq_idx);

                if(signal_seq.at(current_signal_seq_idx) == 1){ // green
                    stop_line_marker_array.markers.at(2*i).color.r = 0.0f;
                    stop_line_marker_array.markers.at(2*i).color.g = 1.0f;
                }
                else if (signal_seq.at(current_signal_seq_idx) == 2){ // red
                    stop_line_marker_array.markers.at(2*i).color.r = 1.0f;
                    stop_line_marker_array.markers.at(2*i).color.g = 0.0f;
                }
            }
            stop_line_rviz_pub.publish(stop_line_marker_array);
            current_signal_seq_idx = (current_signal_seq_idx + 1) % signal_seq.size();
        }

        // update remain_time;
        for(int i=0; i<stop_line_list.size(); i++){
            signal_msg.signals.at(i).time = remain_time;
        }

        traffic_signal_pub.publish(signal_msg);

        remain_time -= (1 / spin_rate);
        rate.sleep();
    }

    return 0;
}