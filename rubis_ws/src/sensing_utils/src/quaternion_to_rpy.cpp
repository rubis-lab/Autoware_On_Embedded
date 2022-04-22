#include <tf/tf.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_listener.h>

void cb(const sensor_msgs::ImuConstPtr msg){
    tf::Quaternion q(   msg->orientation.x,
                        msg->orientation.y,
                        msg->orientation.z,
                        msg->orientation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    roll *= 180/3.14;
    pitch *= 180/3.14;
    yaw *= 180/3.14;

    std::cout<<"IMU: "<<roll<<" "<<pitch<<" "<<yaw<<std::endl;
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "quaternion_to_rpy");

    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe("/imu_raw", 1, cb);

    tf::TransformListener listener;

    ros::Rate rate(10);
    while(nh.ok()){
        tf::StampedTransform tf;
        try{
            listener.lookupTransform("/map", "/base_link", ros::Time(0), tf);
            auto q = tf.getRotation();
            tf::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);

            roll *= 180/3.14;
            pitch *= 180/3.14;
            yaw *= 180/3.14;

            std::cout<<"map-base_link TF: "<<roll<<" "<<pitch<<" "<<yaw<<std::endl;
        }
        catch(tf::TransformException ex){

        }
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}