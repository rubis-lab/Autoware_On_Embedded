#include <iostream>
#include <algorithm>

#include <ros/ros.h>
#include <ros/time.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PolygonStamped.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <geometry_msgs/Point32.h>
#include <autoware_msgs/DetectedObject.h>
#include <autoware_msgs/DetectedObjectArray.h>

static std::vector<std::string> type_names;
void init_type_names();
void create_fake_objects(std::vector< std::vector<double> >& object_list, autoware_msgs::DetectedObjectArray& object_array_msg, jsk_recognition_msgs::PolygonArray& polygon_array_msg);
void add_objects(std_msgs::Header& header, autoware_msgs::DetectedObjectArray& input_msg, autoware_msgs::DetectedObjectArray& output_msg);



std_msgs::Header create_header(int seq, std::string frame_id);

int main(int argc, char* argv[]){
    init_type_names();

    // Initialize
    ros::init(argc, argv, "fake_object_generator");
    ros::NodeHandle nh;
    ros::Rate rate(10);
    ros::Publisher object_array_pub;
    ros::Publisher object_polygon_array_pub;    
    ros::Publisher person_polygon_array_pub;    
    
    object_array_pub = nh.advertise<autoware_msgs::DetectedObjectArray>("/detection/fusion_tools/objects", 10);
    object_polygon_array_pub = nh.advertise<jsk_recognition_msgs::PolygonArray>("/fake_object_polygons", 10);
    person_polygon_array_pub = nh.advertise<jsk_recognition_msgs::PolygonArray>("/fake_person_polygons", 10);

    // Load Parameters
    XmlRpc::XmlRpcValue xml_object_list;
    nh.getParam("/fake_object_generator/object_list", xml_object_list);
    std::vector< std::vector<double> > object_list;
    for(int i=0; i<xml_object_list.size(); i++){
        XmlRpc::XmlRpcValue xml_object = xml_object_list[i];
        std::vector<double> object;
        for(int j=0; j<xml_object.size(); j++)
            object.push_back((double)(xml_object[j]));
        object_list.push_back(object);
    }

    XmlRpc::XmlRpcValue xml_person_list;
    nh.getParam("/fake_object_generator/person_list", xml_person_list);
    std::vector< std::vector<double> > person_list;
    for(int i=0; i<xml_person_list.size(); i++){
        XmlRpc::XmlRpcValue xml_person = xml_person_list[i];
        std::vector<double> person;
        for(int j=0; j<xml_person.size(); j++)
            person.push_back((double)(xml_person[j]));
        person_list.push_back(person);
    }

    std::string frame_id = "/none";
    nh.getParam("/fake_object_generator/frame_id", frame_id);

    

    autoware_msgs::DetectedObjectArray object_array_msg;
    jsk_recognition_msgs::PolygonArray object_polygon_array_msg;
    create_fake_objects(object_list, object_array_msg, object_polygon_array_msg);
    
    autoware_msgs::DetectedObjectArray person_array_msg;
    jsk_recognition_msgs::PolygonArray person_polygon_array_msg;
    create_fake_objects(person_list, person_array_msg, person_polygon_array_msg);

    std::vector< std::vector<double> > empty_list;
    autoware_msgs::DetectedObjectArray empty_array_msg;
    jsk_recognition_msgs::PolygonArray empty_polygon_array_msg;
    create_fake_objects(empty_list, empty_array_msg, empty_polygon_array_msg);


    int seq = 0;
    int object_flag, person_flag;
    autoware_msgs::DetectedObjectArray final_object_msg;

    while(ros::ok()){
        nh.getParam("/fake_object_generator/object_flag", object_flag);
        nh.getParam("/fake_object_generator/person_flag", person_flag);

        std_msgs::Header header = create_header(seq, frame_id);

        object_polygon_array_msg.header = header;
        for(auto it = object_polygon_array_msg.polygons.begin(); it != object_polygon_array_msg.polygons.end(); ++it)
            (*it).header = header;      
        person_polygon_array_msg.header = header;
        for(auto it = person_polygon_array_msg.polygons.begin(); it != person_polygon_array_msg.polygons.end(); ++it)
            (*it).header = header;
        empty_polygon_array_msg.header = header;
        // for(auto it = fake_polygon_array_msg.polygons.begin(); it != fake_polygon_array_msg.polygons.end(); ++it)
        //     (*it).header = header;

        final_object_msg.header = header;
        final_object_msg.objects.clear();

        if(object_flag==1){
            add_objects(header, object_array_msg, final_object_msg);
            object_polygon_array_pub.publish(object_polygon_array_msg);
        }
        else{            
            object_polygon_array_pub.publish(empty_polygon_array_msg);
        }
        
        if(person_flag==1){
            add_objects(header, person_array_msg, final_object_msg);
            person_polygon_array_pub.publish(person_polygon_array_msg);
        }
        else{
            person_polygon_array_pub.publish(empty_polygon_array_msg);
        }

        object_array_pub.publish(final_object_msg);
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

void add_objects(std_msgs::Header& header, autoware_msgs::DetectedObjectArray& input_msg, autoware_msgs::DetectedObjectArray& output_msg){
    for(auto it = input_msg.objects.begin(); it != input_msg.objects.end(); ++it){
        (*it).header = header;
        output_msg.objects.push_back(*it);
    }
}



void create_fake_objects(std::vector< std::vector<double> >& object_list, autoware_msgs::DetectedObjectArray& object_array_msg, jsk_recognition_msgs::PolygonArray& polygon_array_msg){
    static int object_id = 415;

    for(auto itlist = object_list.begin(); itlist != object_list.end(); ++itlist){
        std::vector<double> object = *itlist;
        std::vector<double> x_vec;
        std::vector<double> y_vec;
        std::vector<double> z_vec;
        std::vector<geometry_msgs::Point32> convex_hull_points;
        autoware_msgs::DetectedObject object_msg;
        geometry_msgs::PolygonStamped polygon_msg;

        for(auto itobj = object.begin(); itobj != object.end(); ++itobj){            
            geometry_msgs::Point32 point;
            int obj_idx = itobj - object.begin();

            if(obj_idx%3 == 0)
                x_vec.push_back(*itobj);
            else if(obj_idx%3 == 1)
                y_vec.push_back(*itobj);
            else if(obj_idx%3 == 2){
                z_vec.push_back(*itobj);
                geometry_msgs::Point32 point;
                point.x = x_vec[obj_idx/3];
                point.y = y_vec[obj_idx/3];
                point.z = z_vec[obj_idx/3];    
                convex_hull_points.push_back(point);
            }
        }

        for(auto itconvex = convex_hull_points.begin(); itconvex != convex_hull_points.end(); ++itconvex){
            geometry_msgs::Point32 point = *itconvex;
            object_msg.convex_hull.polygon.points.push_back(*itconvex);
        }

        object_msg.id = object_id++;
        // Pose
        float max_x, min_x, max_y, min_y, max_z, min_z;
        max_x = *std::max_element(x_vec.begin(), x_vec.end());
        min_x = *std::min_element(x_vec.begin(), x_vec.end());
        max_y = *std::max_element(y_vec.begin(), y_vec.end());
        min_y = *std::min_element(y_vec.begin(), y_vec.end());
        max_z = *std::max_element(z_vec.begin(), z_vec.end());
        min_z = *std::min_element(z_vec.begin(), z_vec.end());
        
        object_msg.pose.position.x = min_x + (max_x-min_x)/2.0;
        object_msg.pose.position.y = min_y + (max_y-min_y)/2.0;
        object_msg.pose.position.z = min_z + (max_z-min_z)/2.0;
        object_msg.pose.orientation.x = 0.0132425152446;
        object_msg.pose.orientation.y = 0.00564379347071;
        object_msg.pose.orientation.z = -0.792667771434;
        object_msg.pose.orientation.w = 0.609483869775;
        // Dimension
        object_msg.dimensions.x = (max_x-min_x)/2.0;
        object_msg.dimensions.y = (max_y-min_y)/2.0;
        object_msg.dimensions.z = (max_z-min_z)/2.0;
        // Velocity
        object_msg.velocity.linear.x = 0.5;
        object_msg.velocity.linear.y = 0.5;
        object_msg.velocity.linear.z = 0.5;
        // Polygon
        polygon_msg = object_msg.convex_hull;
        
        // Create Msg
        object_array_msg.objects.push_back(object_msg);
        polygon_array_msg.polygons.push_back(polygon_msg);
    }
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

