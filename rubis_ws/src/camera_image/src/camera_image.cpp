#include "ros/ros.h"
#include "opencv2/opencv.hpp"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "rubis_lib/sched.hpp"

// argv[0] : camera_id, argv[1] : frequency

static const bool DEBUG = false; // 디버깅 스위치
static const std::string OPENCV_WINDOW = "Raw Image Window";

static int camera_id = 0;
static int frequency = 0;
static int task_scheduling_flag = 0;
static int task_profiling_flag = 0;
static std::string task_response_time_filename;
// static int rate = 0; // Frequency replaces rate
static double task_minimum_inter_release_time = 0;
static double task_execution_time = 0;
static double task_relative_deadline = 0;

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
        camera_image_pub_ = it_.advertise("/image_raw", 1); 
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

    pnh.param<int>("/camera_image/camera_id", camera_id, 0);
    pnh.param<int>("/camera_image/frequency", frequency, 10);
    pnh.param<int>("/camera_image/task_scheduling_flag", task_scheduling_flag, 0);
    pnh.param<int>("/camera_image/task_profiling_flag", task_profiling_flag, 0);
    pnh.param<std::string>("/camera_image/task_response_time_filename", task_response_time_filename, "~/Documents/profiling/response_time/camera_image.csv");
    // pnh.param<int>("/camera_image/rate", rate, 10); // Frequency replaces rate
    pnh.param("/camera_image/task_minimum_inter_release_time", task_minimum_inter_release_time, (double)100000000);
    pnh.param("/camera_image/task_execution_time", task_execution_time, (double)100000000);
    pnh.param("/camera_image/task_relative_deadline", task_relative_deadline, (double)100000000);
    
    if(task_profiling_flag) rubis::sched::init_task_profiling(task_response_time_filename);

    ROS_INFO("camera_id : %d / frequency : %d",camera_id, frequency);
    if(!frequency){
        ROS_INFO("Frequency is number more than 0");
        return 1;
    }

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

        if(task_profiling_flag) rubis::sched::start_task_profiling();   

        if(rubis::sched::task_state_ == TASK_STATE_READY){            
            if(task_scheduling_flag) rubis::sched::request_task_scheduling(task_minimum_inter_release_time, task_execution_time, task_relative_deadline);
            rubis::sched::task_state_ = TASK_STATE_RUNNING;
        }

        cap >> frame;
        if(!frame.empty()){
            if(DEBUG) cv::imshow(OPENCV_WINDOW,frame);

            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
            msg->header.frame_id="camera";
            camera_image_pub_.publish(msg);
            rubis::sched::is_task_ready_ = TASK_STATE_DONE;            
        }                        
        // int ckey = cv::waitKey(1);
        // if(ckey == 27)break;

        if(task_profiling_flag) rubis::sched::stop_task_profiling(0, rubis::sched::task_state_);

        if(rubis::sched::task_state_ == TASK_STATE_DONE){            
            if(task_scheduling_flag) rubis::sched::yield_task_scheduling();
            rubis::sched::task_state_ = TASK_STATE_READY;
        }
        loop_rate.sleep();
    }
}

std::string CameraImage::createTopicName(){
    // topic_name =  "/cam"+ to_string(camera_id) +"/raw_image";
    topic_name = "image_raw";
}

template < typename T > 
std::string to_string( const T& n )
{
    std::ostringstream stm ;
    stm << n ;
    return stm.str() ;
}
