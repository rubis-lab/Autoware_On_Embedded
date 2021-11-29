/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ********************
 *  v1.0: amc-nu (abrahammonrroy@yahoo.com)
 *
 * yolo3_node.cpp
 *
 *  Created on: April 4th, 2018
 */
#include "vision_darknet_detect.h"
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "../darknet/src/cuda.h"

#define SPIN_PROFILING

#if (CV_MAJOR_VERSION <= 2)
#include <opencv2/contrib/contrib.hpp>
#else
#include "gencolors.cpp"
#endif


namespace darknet
{
    uint32_t Yolo3Detector::get_network_height()
    {
        return darknet_network_->h;
    }
    uint32_t Yolo3Detector::get_network_width()
    {
        return darknet_network_->w;
    }
    void Yolo3Detector::load(std::string& in_model_file, std::string& in_trained_file, double in_min_confidence, double in_nms_threshold)
    {
        min_confidence_ = in_min_confidence;
        nms_threshold_ = in_nms_threshold;
        darknet_network_ = parse_network_cfg(&in_model_file[0]);
        load_weights(darknet_network_, &in_trained_file[0]);
        set_batch_network(darknet_network_, 1);

        layer output_layer = darknet_network_->layers[darknet_network_->n - 1];
        darknet_boxes_.resize(output_layer.w * output_layer.h * output_layer.n);
    }

    Yolo3Detector::~Yolo3Detector()
    {
        free_network(darknet_network_);
    }

    std::vector< RectClassScore<float> > Yolo3Detector::detect(image& in_darknet_image)
    {
        return forward(in_darknet_image);
    }

    image Yolo3Detector::convert_image(const sensor_msgs::ImageConstPtr& msg)
    {
        if (msg->encoding != sensor_msgs::image_encodings::BGR8)
        {
            ROS_ERROR("Unsupported encoding");
            exit(-1);
        }

        auto data = msg->data;
        uint32_t height = msg->height, width = msg->width, offset = msg->step - 3 * width;
        uint32_t i = 0, j = 0;
        image im = make_image(width, height, 3);

        for (uint32_t line = height; line; line--)
        {
            for (uint32_t column = width; column; column--)
            {
                for (uint32_t channel = 0; channel < 3; channel++)
                    im.data[i + width * height * channel] = data[j++] / 255.;
                i++;
            }
            j += offset;
        }

        if (darknet_network_->w == (int) width && darknet_network_->h == (int) height)
        {
            return im;
        }
        image resized = resize_image(im, darknet_network_->w, darknet_network_->h);
        free_image(im);
        return resized;
    }

    std::vector< RectClassScore<float> > Yolo3Detector::forward(image& in_darknet_image)
    {
        float * in_data = in_darknet_image.data;
        
        double time = what_time_is_it_now();
        
        float *prediction = network_predict(darknet_network_, in_data);


        layer output_layer = darknet_network_->layers[darknet_network_->n - 1];

        output_layer.output = prediction;
        int nboxes = 0;
        int num_classes = output_layer.classes;
        detection *darknet_detections = get_network_boxes(darknet_network_, darknet_network_->w, darknet_network_->h, min_confidence_, .5, NULL, 0, &nboxes);

        do_nms_sort(darknet_detections, nboxes, num_classes, nms_threshold_);

        std::vector< RectClassScore<float> > detections;

        for (int i = 0; i < nboxes; i++)
        {
            int class_id = -1;
            float score = 0.f;
            //find the class
            for(int j = 0; j < num_classes; ++j){
                if (darknet_detections[i].prob[j] >= min_confidence_){
                    if (class_id < 0) {
                        class_id = j;
                        score = darknet_detections[i].prob[j];
                    }
                }
            }
            //if class found
            if (class_id >= 0)
            {
                RectClassScore<float> detection;

                detection.x = darknet_detections[i].bbox.x - darknet_detections[i].bbox.w/2;
                detection.y = darknet_detections[i].bbox.y - darknet_detections[i].bbox.h/2;
                detection.w = darknet_detections[i].bbox.w;
                detection.h = darknet_detections[i].bbox.h;
                detection.score = score;
                detection.class_type = class_id;
                //std::cout << detection.toString() << std::endl;

                detections.push_back(detection);
            }
        }
        //std::cout << std::endl;
        return detections;
    }
}  // namespace darknet

///////////////////

void Yolo3DetectorNode::convert_rect_to_image_obj(std::vector< RectClassScore<float> >& in_objects, autoware_msgs::DetectedObjectArray& out_message)
{
    for (unsigned int i = 0; i < in_objects.size(); ++i)
    {
        {
            autoware_msgs::DetectedObject obj;

            obj.x = (in_objects[i].x /image_ratio_) - image_left_right_border_/image_ratio_;
            obj.y = (in_objects[i].y /image_ratio_) - image_top_bottom_border_/image_ratio_;
            obj.width = in_objects[i].w /image_ratio_;
            obj.height = in_objects[i].h /image_ratio_;
            if (in_objects[i].x < 0)
                obj.x = 0;
            if (in_objects[i].y < 0)
                obj.y = 0;
            if (in_objects[i].w < 0)
                obj.width = 0;
            if (in_objects[i].h < 0)
                obj.height = 0;

            obj.score = in_objects[i].score;
            if (use_coco_names_)
            {
                obj.label = in_objects[i].GetClassString();
            }
            else
            {
                if (in_objects[i].class_type < custom_names_.size())
                    obj.label = custom_names_[in_objects[i].class_type];
                else
                    obj.label = "unknown";
            }
            obj.valid = true;

            out_message.objects.push_back(obj);

        }
    }
}

void Yolo3DetectorNode::rgbgr_image(image& im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i)
    {
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

image Yolo3DetectorNode::convert_ipl_to_image(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(msg, "bgr8");//toCvCopy(image_source, sensor_msgs::image_encodings::BGR8);
    cv::Mat mat_image = cv_image->image;

    int network_input_width = yolo_detector_.get_network_width();
    int network_input_height = yolo_detector_.get_network_height();

    int image_height = msg->height,
            image_width = msg->width;

    IplImage ipl_image;
    cv::Mat final_mat;

    if (network_input_width!=image_width
        || network_input_height != image_height)
    {
        //final_mat = cv::Mat(network_input_width, network_input_height, CV_8UC3, cv::Scalar(0,0,0));
        image_ratio_ = (double ) network_input_width /  (double)mat_image.cols;

        cv::resize(mat_image, final_mat, cv::Size(), image_ratio_, image_ratio_);
        image_top_bottom_border_ = abs(final_mat.rows-network_input_height)/2;
        image_left_right_border_ = abs(final_mat.cols-network_input_width)/2;
        cv::copyMakeBorder(final_mat, final_mat,
                           image_top_bottom_border_, image_top_bottom_border_,
                           image_left_right_border_, image_left_right_border_,
                           cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    }
    else
        final_mat = mat_image;

    ipl_image = final_mat;

    unsigned char *data = (unsigned char *)ipl_image.imageData;
    int h = ipl_image.height;
    int w = ipl_image.width;
    int c = ipl_image.nChannels;
    int step = ipl_image.widthStep;
    int i, j, k;

    image darknet_image = make_image(w, h, c);

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                darknet_image.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    rgbgr_image(darknet_image);
    return darknet_image;
}

void Yolo3DetectorNode::image_callback(const sensor_msgs::ImageConstPtr& in_image_message)
{
    std::vector< RectClassScore<float> > detections;

    
    darknet_image_ = convert_ipl_to_image(in_image_message);

    detections = yolo_detector_.detect(darknet_image_);

    //Prepare Output message
    autoware_msgs::DetectedObjectArray output_message;
    output_message.header = in_image_message->header;

    convert_rect_to_image_obj(detections, output_message);

    publisher_objects_.publish(output_message);

    free(darknet_image_.data);
    
    if(is_task_ready_ == TASK_NOT_READY){
        init_task();
        if(gpu_profiling_flag_) start_gpu_profiling();
    }
    
    task_state_ = TASK_STATE_DONE;
}

void Yolo3DetectorNode::config_cb(const autoware_config_msgs::ConfigSSD::ConstPtr& param)
{
    score_threshold_ = param->score_threshold;
}

std::vector<std::string> Yolo3DetectorNode::read_custom_names_file(const std::string& in_names_path)
{
    std::ifstream file(in_names_path);
    std::string str;
    std::vector<std::string> names;
    while (std::getline(file, str))
    {
        names.push_back(str);
        // std::cout << str <<  std::endl;
    }
    return names;
}

std::string Yolo3DetectorNode::convert_to_absolute_path(std::string relative_path){
    std::string absolute_path;

    if(relative_path.at(0) == '~'){
      relative_path.erase(0, 1);
      std::string user_home_str(std::getenv("USER_HOME"));
      absolute_path =  user_home_str + relative_path;
    }
    return absolute_path;
}

void Yolo3DetectorNode::Run()
{    
    //ROS STUFF
    ros::NodeHandle private_node_handle("~");//to receive args   

    int key_id = 2;

    // Scheduling Setup
    int task_scheduling_flag;
    int task_profiling_flag;
    std::string task_response_time_filename_str;
    int rate;
    double task_minimum_inter_release_time;
    double task_execution_time;
    double task_relative_deadline;

    int gpu_scheduling_flag;
    int gpu_profiling_flag;
    std::string gpu_execution_time_filename_str;
    std::string gpu_response_time_filename_str;
    std::string gpu_deadline_filename_str;

    private_node_handle.param<int>("/vision_darknet_detect/task_scheduling_flag", task_scheduling_flag, 0);
    private_node_handle.param<int>("/vision_darknet_detect/task_profiling_flag", task_profiling_flag, 0);
    private_node_handle.param<std::string>("/vision_darknet_detect/task_response_time_filename", task_response_time_filename_str, "~/Documents/profiling/response_time/vision_darknet_detect.csv");
    private_node_handle.param<int>("/vision_darknet_detect/rate", rate, 10);
    private_node_handle.param("/vision_darknet_detect/task_minimum_inter_release_time", task_minimum_inter_release_time, (double)10);
    private_node_handle.param("/vision_darknet_detect/task_execution_time", task_execution_time, (double)10);
    private_node_handle.param("/vision_darknet_detect/task_relative_deadline", task_relative_deadline, (double)10);
    private_node_handle.param("/vision_darknet_detect/gpu_scheduling_flag", gpu_scheduling_flag, 0);
    private_node_handle.param("/vision_darknet_detect/gpu_profiling_flag", gpu_profiling_flag, 0);
    private_node_handle.param<std::string>("/vision_darknet_detect/gpu_execution_time_filename", gpu_execution_time_filename_str, "~/Documents/gpu_profiling/test_yolo_execution_time.csv");
    private_node_handle.param<std::string>("/vision_darknet_detect/gpu_response_time_filename", gpu_response_time_filename_str, "~/Documents/gpu_profiling/test_yolo_response_time.csv");
    private_node_handle.param<std::string>("/vision_darknet_detect/gpu_deadline_filename", gpu_deadline_filename_str, "~/Documents/gpu_deadline/yolo_gpu_deadline.csv");
    
    char* task_response_time_filename = strdup(task_response_time_filename_str.c_str());
    char* gpu_execution_time_filename = strdup(gpu_execution_time_filename_str.c_str());
    char* gpu_response_time_filename = strdup(gpu_response_time_filename_str.c_str());
    char* gpu_deadline_filename = strdup(gpu_deadline_filename_str.c_str());

    if(task_profiling_flag) init_task_profiling(task_response_time_filename);
    if(gpu_profiling_flag) init_gpu_profiling(gpu_execution_time_filename, gpu_response_time_filename);

    if(gpu_scheduling_flag){
        init_gpu_scheduling("/tmp/yolo", gpu_deadline_filename, key_id);
    }else if(gpu_scheduling_flag){
        ROS_ERROR("GPU scheduling flag is true but type doesn't set to GPU!");
        exit(1);
    }
    
    
    //RECEIVE IMAGE TOPIC NAME
    std::string image_raw_topic_str;
    if (private_node_handle.getParam("image_raw_node", image_raw_topic_str))
    {
        ROS_INFO("Setting image node to %s", image_raw_topic_str.c_str());
    }
    else
    {
        ROS_INFO("No image node received, defaulting to /image_raw, you can use _image_raw_node:=YOUR_TOPIC");
        image_raw_topic_str = "/image_raw";
    }

    std::string network_definition_file;
    std::string pretrained_model_file, names_file;
    if (private_node_handle.getParam("/vision_darknet_detect/network_definition_file", network_definition_file))
    {
        ROS_INFO("Network Definition File (Config): %s", network_definition_file.c_str());
    }
    else
    {
        ROS_INFO("No Network Definition File was received. Finishing execution.");
        return;
    }
    if (private_node_handle.getParam("/vision_darknet_detect/pretrained_model_file", pretrained_model_file))
    {
        ROS_INFO("Pretrained Model File (Weights): %s", pretrained_model_file.c_str());
    }
    else
    {
        ROS_INFO("No Pretrained Model File was received. Finishing execution.");
        return;
    }

    if (private_node_handle.getParam("names_file", names_file))
    {
        ROS_INFO("Names File: %s", names_file.c_str());
        use_coco_names_ = false;        
        std::string _names_file = convert_to_absolute_path(names_file);

        custom_names_ = read_custom_names_file(_names_file);
    }
    else
    {
        ROS_INFO("No Names file was received. Using default COCO names.");
        use_coco_names_ = true;
    }

    private_node_handle.param<float>("score_threshold", score_threshold_, 0.5);
    ROS_INFO("[%s] score_threshold: %f",__APP_NAME__, score_threshold_);

    private_node_handle.param<float>("nms_threshold", nms_threshold_, 0.45);
    ROS_INFO("[%s] nms_threshold: %f",__APP_NAME__, nms_threshold_);


    ROS_INFO("Initializing Yolo on Darknet...");
    std::string _network_definition_file = convert_to_absolute_path(network_definition_file);
    std::string _pretrained_model_file = convert_to_absolute_path(pretrained_model_file);

    yolo_detector_.load(_network_definition_file, _pretrained_model_file, score_threshold_, nms_threshold_);
    ROS_INFO("Initialization complete.");

    #if (CV_MAJOR_VERSION <= 2)
        cv::generateColors(colors_, 80);
    #else
        generateColors(colors_, 80);
    #endif

    publisher_objects_ = node_handle_.advertise<autoware_msgs::DetectedObjectArray>("/detection/image_detector/objects", 1);

    ROS_INFO("Subscribing to... %s", image_raw_topic_str.c_str());
    subscriber_image_raw_ = node_handle_.subscribe(image_raw_topic_str, 1, &Yolo3DetectorNode::image_callback, this);    
    std::string config_topic("/config");
    config_topic += "/Yolo3";
    subscriber_yolo_config_ = node_handle_.subscribe(config_topic, 1, &Yolo3DetectorNode::config_cb, this);

    ROS_INFO_STREAM( __APP_NAME__ << "" );

    if(!task_scheduling_flag && !task_profiling_flag){
        ros::spin();
    }
    else{
        ros::Rate r(rate);
        // Initialize task ( Wait until first necessary topic is published )
        while(ros::ok()){
            if(is_task_ready_) break;
            ros::spinOnce();
            r.sleep();      
        }

        // Executing task
        while(ros::ok()){
            if(task_state_ == TASK_STATE_READY){
                if(task_profiling_flag) start_task_profiling();                
                if(task_scheduling_flag) request_task_scheduling(task_minimum_inter_release_time, task_execution_time, task_relative_deadline); 
                if(gpu_profiling_flag || gpu_scheduling_flag) start_job();
                task_state_ = TASK_STATE_RUNNING;     
            }

            ros::spinOnce();

            if(task_state_ == TASK_STATE_DONE){
                if(gpu_profiling_flag || gpu_scheduling_flag) finish_job();
                if(task_profiling_flag) stop_task_profiling();
                if(task_scheduling_flag) yield_task_scheduling();
                task_state_ = TASK_STATE_READY;
            }
            
            r.sleep();
        }
    }
    ROS_INFO("END Yolo");

}