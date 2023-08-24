#include <ros/ros.h>
#include <std_msgs/String.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class MultiTopicSyncSubscriber {
public:
    MultiTopicSyncSubscriber() : nh("~") {
        topic1_sub.subscribe(nh, "/topic1", 1);
        topic2_sub.subscribe(nh, "/topic2", 1);
        topic3_sub.subscribe(nh, "/topic3", 1);

        typedef message_filters::sync_policies::ApproximateTime<std_msgs::String, std_msgs::String, std_msgs::String> MySyncPolicy;
        sync.reset(new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), topic1_sub, topic2_sub, topic3_sub));
        sync->registerCallback(boost::bind(&MultiTopicSyncSubscriber::callback, this, _1, _2, _3));
    }

    void callback(const std_msgs::String::ConstPtr& msg1, 
                  const std_msgs::String::ConstPtr& msg2,
                  const std_msgs::String::ConstPtr& msg3) {
        ROS_INFO("Received from topic1: %s", msg1->data.c_str());
        ROS_INFO("Received from topic2: %s", msg2->data.c_str());
        ROS_INFO("Received from topic3: %s", msg3->data.c_str());
    }

    void spin() {
        ros::spin();
    }

private:
    ros::NodeHandle nh;
    message_filters::Subscriber<std_msgs::String> topic1_sub;
    message_filters::Subscriber<std_msgs::String> topic2_sub;
    message_filters::Subscriber<std_msgs::String> topic3_sub;

    typedef message_filters::sync_policies::ApproximateTime<std_msgs::String, std_msgs::String, std_msgs::String> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    boost::shared_ptr<Sync> sync;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "multi_topic_sync_subscriber_class_example");

    MultiTopicSyncSubscriber instance;
    instance.spin();

    return 0;
}