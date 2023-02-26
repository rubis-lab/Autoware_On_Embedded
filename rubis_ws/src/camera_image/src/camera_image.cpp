#include "ros/ros.h"
#include "opencv2/opencv.hpp"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "rubis_lib/sched.hpp"

// argv[0] : camera_id, argv[1] : frequency

static const bool DEBUG = false; // 디버깅 스위치
static const std::string OPENCV_WINDOW = "Raw Image Window";

template < typename T > std::string to_string( const T& n );

class CameraImage{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Publisher camera_image_pub_;

public:
    CameraImage(int par_camera_num,int par_frequency)
        : it_(nh_), camera_id(par_camera_num), frequency(par_frequency)
    { 
        createTopicName();  
        if(DEBUG) std::cout<<"topic_name : "<<topic_name<<std::endl;

        /* Dynamic topic name */
        // camera_image_pub_ = it_.advertise(topic_name,1);

        /* Static topic name */
        camera_image_pub_ = it_.advertise("/image_raw_origin", 1); 
        cap.open(camera_id);
    }

    ~CameraImage()
    {
        if(DEBUG) cv::destroyWindow(OPENCV_WINDOW);
    }

    void sendImage(); // image 퍼블리시
    std::string createTopicName();// topic이름 생성

private:
    int camera_id;
    int frequency;    
    cv::VideoCapture cap;
    std::string topic_name;
    sensor_msgs::ImagePtr msg; 
};

int main(int argc, char** argv){        
    ros::init(argc, argv, "camera_image");

    ros::NodeHandle pnh("~");
    std::string node_name = ros::this_node::getName();
    std::string output_topic_name = node_name + "/output_topic";

    std::string task_response_time_filename;
    pnh.param<std::string>(node_name+"/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/camera_image.csv");

    int rate, camera_id, frequency;
    pnh.param(node_name+"/rate", rate, 10);


    struct rubis::sched_attr attr;
    std::string policy;
    int priority, exec_time ,deadline, period;

    pnh.param(node_name+"/camera_id", camera_id,0);
    pnh.param(node_name+"/frequency", frequency, 10);
    pnh.param(node_name+"/task_scheduling_configs/policy", policy, std::string("NONE"));    
    pnh.param(node_name+"/task_scheduling_configs/priority", priority, 99);
    pnh.param(node_name+"/task_scheduling_configs/exec_time", exec_time, 0);
    pnh.param(node_name+"/task_scheduling_configs/deadline", deadline, 0);
    pnh.param(node_name+"/task_scheduling_configs/period", period, 0);
    
    attr = rubis::create_sched_attr(priority, exec_time, deadline, period);    
    rubis::init_task_scheduling(policy, attr);
    rubis::init_task_profiling(task_response_time_filename);

    ROS_INFO("camera_id : %d / frequency : %d",camera_id, frequency);
    // if(!frequency){
    //     ROS_INFO("Frequency is number more than 0");
    //     return 1;
    // }

    CameraImage cimage(camera_id, frequency);

    if(DEBUG) ROS_INFO("Start publishing");
   
    cimage.sendImage();
   
    if(DEBUG) ROS_INFO("Publishing done");

    return 0;
}





// 구현부

void CameraImage::sendImage(){
    ros::Rate loop_rate(frequency);
        cv::Mat frame;
        
    while(nh_.ok()){

        rubis::start_task_profiling();   

        cap >> frame;
        if(!frame.empty()){
            if(DEBUG) cv::imshow(OPENCV_WINDOW,frame);

            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
            msg->header.frame_id="camera";
            camera_image_pub_.publish(msg);
        }                        
        // int ckey = cv::waitKey(1);
        // if(ckey == 27)break;

        rubis::stop_task_profiling(0, 0);

        loop_rate.sleep();
    }
}

std::string CameraImage::createTopicName(){
    // topic_name =  "/cam"+ to_string(camera_id) +"/raw_image";
    topic_name = "image_raw_origin";
}

template < typename T > 
std::string to_string( const T& n )
{
    std::ostringstream stm ;
    stm << n ;
    return stm.str() ;
}
