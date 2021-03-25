#include <ros/ros.h>
#include <autoware_msgs/DetectedObjectArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>


autoware_msgs::DetectedObjectArray obj_list;
geometry_msgs::PoseStamped pose;

void obj_cb(const autoware_msgs::DetectedObjectArray& msg){
    obj_list = msg;
    pose.header = obj_list.header;
    pose.pose = obj_list.objects[0].pose;

    std::cout<<"DetectedObject"<<std::endl;
    std::cout<<obj_list.header.frame_id<<std::endl;
    std::cout<<obj_list.objects[0].pose.position.x<<" "<<obj_list.objects[0].pose.position.y << " "
        << obj_list.objects[0].pose.position.z << " "<<obj_list.objects[0].pose.orientation.x <<" " 
        << obj_list.objects[0].pose.orientation.y << " " <<obj_list.objects[0].pose.orientation.z << " "
        << obj_list.objects[0].pose.orientation.w << std::endl;
    std::cout<<"========================================"<<std::endl;
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "TF_test");    
    ros::NodeHandle nh;
    ros::Subscriber obj_sub;
    ros::Rate rate(10);

    obj_sub = nh.subscribe("/detection/fusion_tools/objects", 2, obj_cb);

    geometry_msgs::PoseStamped trans_pose;    
    tf::TransformListener listener;    
    tf::StampedTransform transform;
    
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);
    

    
    
    while(ros::ok()){
        ros::spinOnce();
        try{
            listener.waitForTransform("/world", "/test",
                              ros::Time::now(), ros::Duration(0.01));
            listener.lookupTransform("/world", "/test", ros::Time::now(), transform);
            std::cout<<"Transform!"<<std::endl;

            listener.transformPose("/test", pose, trans_pose);
            std::cout<<"Trans Pose"<<std::endl;
            std::cout<<trans_pose.pose.position.x << " " << trans_pose.pose.position.y << " " << trans_pose.pose.position.z << " " <<
            trans_pose.pose.orientation.x << " " << trans_pose.pose.orientation.y << " " << trans_pose.pose.orientation.z << " " << 
            trans_pose.pose.orientation.w << std::endl;
        }
        catch(tf::TransformException& ex){
            ROS_ERROR("%s", ex.what());
        }
        
        


        rate.sleep();
    }    
    


    return 0;
}