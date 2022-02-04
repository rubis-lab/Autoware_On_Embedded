#include <iostream>
#include <algorithm>

#include <ros/ros.h>
#include <ros/time.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Point32.h>
#include <autoware_msgs/DetectedObject.h>
#include <autoware_msgs/DetectedObjectArray.h>

static std::vector<std::string> type_names;
void init_type_names();

std_msgs::Header create_header(int seq, std::string frame_id);

int main(int argc, char* argv[]){
    init_type_names();

    // Initialize
    ros::init(argc, argv, "fake_object_generator");
    ros::NodeHandle nh;
    ros::Rate rate(10);
    ros::Publisher fake_object_pub;
    ros::Publisher polygon_pub1;
    ros::Publisher polygon_pub2;
    // ros::Publisher bb_pub1;
    // ros::Publisher bb_pub2;

    XmlRpc::XmlRpcValue list;
    nh.getParam("/fake_object_generator/test", list);
    for(int i=0; i<list.size(); i++){
        std::cout<<type_names[list[i].getType()]<<std::endl;
        std::cout<<list[i]<<std::endl;
        std::cout<<type_names[list[i][0].getType()]<<std::endl;
        std::cout<<list[i][0]<<" "<<list[i][1]<<std::endl;
    }



    fake_object_pub = nh.advertise<autoware_msgs::DetectedObjectArray>("/tracked_objects", 10);
    polygon_pub1 = nh.advertise<geometry_msgs::PolygonStamped>("/fake_polygon1", 10);
    polygon_pub2 = nh.advertise<geometry_msgs::PolygonStamped>("/fake_polygon2", 10);
    int obstacle_num;
    std::vector<float> convex_hull_param1;
    std::vector<float> convex_hull_param2;
    std::string frame_id;
    nh.getParam("/fake_object_generator/obstacle_num", obstacle_num);
    nh.getParam("/fake_object_generator/convex_hull1", convex_hull_param1);
    nh.getParam("/fake_object_generator/convex_hull2", convex_hull_param2);    
    nh.getParam("/fake_object_generator/frame_id", frame_id);
    

    std::vector<geometry_msgs::Point32> convex_hull_points1;
    std::vector<geometry_msgs::Point32> convex_hull_points2;
    std::vector<float> x_vec1;
    std::vector<float> y_vec1;
    std::vector<float> z_vec1;
    std::vector<float> x_vec2;
    std::vector<float> y_vec2;
    std::vector<float> z_vec2;

    for(auto it = convex_hull_param1.begin(); it != convex_hull_param1.end(); ++it){
        int idx = it - convex_hull_param1.begin();
        geometry_msgs::Point32 point;
        if(idx%3 == 0){
            x_vec1.push_back(*it);
        }    
        else if(idx%3 == 1){
            y_vec1.push_back(*it);
        }
        else if(idx%3 == 2){
            z_vec1.push_back(*it);
            geometry_msgs::Point32 point;
            point.x = x_vec1[idx/3];
            point.y = y_vec1[idx/3];
            point.z = z_vec1[idx/3];    
            convex_hull_points1.push_back(point);
        }
    }

    for(auto it = convex_hull_param2.begin(); it != convex_hull_param2.end(); ++it){
        int idx = it - convex_hull_param2.begin();
        geometry_msgs::Point32 point;
        if(idx%3 == 0){
            x_vec2.push_back(*it);
        }    
        else if(idx%3 == 1){
            y_vec2.push_back(*it);
        }
        else if(idx%3 == 2){
            z_vec2.push_back(*it);
            geometry_msgs::Point32 point;
            point.x = x_vec2[idx/3];
            point.y = y_vec2[idx/3];
            point.z = z_vec2[idx/3];    
            convex_hull_points2.push_back(point);
        }
    }
    
    // Create Fake Object
    autoware_msgs::DetectedObjectArray fake_object_array;
    autoware_msgs::DetectedObject fake_object1;
    autoware_msgs::DetectedObject fake_object2;
    
    // ID
    fake_object1.id = 415;

    // Convex Hull
    for(auto it = convex_hull_points1.begin(); it != convex_hull_points1.end(); ++it){
        geometry_msgs::Point32 point = *it;
        fake_object1.convex_hull.polygon.points.push_back(*it);        
    }

    // Pose
    float max_x, min_x, max_y, min_y, max_z, min_z;
    max_x = *std::max_element(x_vec1.begin(), x_vec1.end());
    min_x = *std::min_element(x_vec1.begin(), x_vec1.end());
    max_y = *std::max_element(y_vec1.begin(), y_vec1.end());
    min_y = *std::min_element(y_vec1.begin(), y_vec1.end());
    max_z = *std::max_element(z_vec1.begin(), z_vec1.end());
    min_z = *std::min_element(z_vec1.begin(), z_vec1.end());

    fake_object1.pose.position.x = min_x + (max_x-min_x)/2.0;
    fake_object1.pose.position.y = min_y + (max_y-min_y)/2.0;
    fake_object1.pose.position.z = min_z + (max_z-min_z)/2.0;
    fake_object1.pose.orientation.x = 0;
    fake_object1.pose.orientation.y = 0;
    fake_object1.pose.orientation.z = 0;
    fake_object1.pose.orientation.w = 0;
    // Dimension
    fake_object1.dimensions.x = (max_x-min_x)/2.0;
    fake_object1.dimensions.y = (max_y-min_y)/2.0;
    fake_object1.dimensions.z = (max_z-min_z)/2.0;
    // Velocity
    fake_object1.velocity.linear.x = 0.5;
    fake_object1.velocity.linear.y = 0.5;
    fake_object1.velocity.linear.z = 0.5;
    // Polygon
    geometry_msgs::PolygonStamped polygon1;
    polygon1 = fake_object1.convex_hull;

    //////////////////////////
    
    // ID
    fake_object2.id = 416;

    // Convex Hull
    for(auto it = convex_hull_points2.begin(); it != convex_hull_points2.end(); ++it){
        geometry_msgs::Point32 point = *it;
        fake_object2.convex_hull.polygon.points.push_back(*it);        
    }

    // Pose
    max_x = *std::max_element(x_vec2.begin(), x_vec2.end());
    min_x = *std::min_element(x_vec2.begin(), x_vec2.end());
    max_y = *std::max_element(y_vec2.begin(), y_vec2.end());
    min_y = *std::min_element(y_vec2.begin(), y_vec2.end());
    max_z = *std::max_element(z_vec2.begin(), z_vec2.end());
    min_z = *std::min_element(z_vec2.begin(), z_vec2.end());

    fake_object2.pose.position.x = min_x + (max_x-min_x)/2.0;
    fake_object2.pose.position.y = min_y + (max_y-min_y)/2.0;
    fake_object2.pose.position.z = min_z + (max_z-min_z)/2.0;
    fake_object2.pose.orientation.x = 0;
    fake_object2.pose.orientation.y = 0;
    fake_object2.pose.orientation.z = 0;
    fake_object2.pose.orientation.w = 0;

    // Dimension
    fake_object2.dimensions.x = (max_x-min_x)/2.0;
    fake_object2.dimensions.y = (max_y-min_y)/2.0;
    fake_object2.dimensions.z = (max_z-min_z)/2.0;
    
    // Velocity
    fake_object2.velocity.linear.x = 0.5;
    fake_object2.velocity.linear.y = 0.5;
    fake_object2.velocity.linear.z = 0.5;

    // Polygon
    geometry_msgs::PolygonStamped polygon2;
    polygon2 = fake_object2.convex_hull;

    int seq = 0;



    while(1){        
        fake_object1.header = create_header(seq, frame_id);
        fake_object2.header = fake_object1.header;
        fake_object_array.objects.clear();
        fake_object_array.objects.push_back(fake_object1);
        fake_object_array.objects.push_back(fake_object2);
        fake_object_array.header = fake_object1.header;
        polygon1.header = fake_object1.header;
        polygon2.header = fake_object2.header;

        fake_object_pub.publish(fake_object_array);
        polygon_pub1.publish(polygon1);
        polygon_pub2.publish(polygon2);
        seq++;
        rate.sleep();
    }
    

    return 0;
}

std_msgs::Header create_header(int seq, std::string frame_id){
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.seq = seq;
    header.frame_id = frame_id;
    return header;
}

void init_type_names(){
    type_names.push_back(std::string("invalid"));
    type_names.push_back(std::string("bool"));
    type_names.push_back(std::string("int"));
    type_names.push_back(std::string("double"));
    type_names.push_back(std::string("string"));
    type_names.push_back(std::string("data_time"));
    type_names.push_back(std::string("base64"));
    type_names.push_back(std::string("array"));
    type_names.push_back(std::string("struct"));
}

