import rospy
from geometry_msgs.msg import PoseStamped
from datetime import datetime
import csv
import keyboard

### TODO ###
file_name='2022-11-04_FMTC'
gnss_pose_topic_name='gnss_pose'
ndt_pose_topic_name='ndt_pose'
############

class Position:
    def __init__(self):
        self.x = -1
        self.y = -1
        self.is_updated = False
    
        return

    def update(self, x, y):
        self.x = x
        self.y = y
        self.is_updated = True

        return
    
    def clear(self):
        self.x = -1
        self.y = -1
        self.is_updated = False

        return


gnss_pose = Position()
ndt_pose = Position()

def gnss_pose_cb(msg):
    gnss_pose.update(msg.pose.position.x, msg.pose.position.y)

    return

def ndt_pose_cb(msg):
    ndt_pose.update(msg.pose.position.x, msg.pose.position.y)
    return

def main():
    rospy.init_node('transform_generator', anonymous=True)
    rate = rospy.Rate(100)
    rospy.Subscriber(ndt_pose_topic_name, PoseStamped, ndt_pose_cb)
    rospy.Subscriber(gnss_pose_topic_name, PoseStamped, gnss_pose_cb)


    pose_file = open('./data/'+file_name+'.csv', 'w')
    writer = csv.writer(pose_file)
    writer.writerow(['gnss_x','gnss_y','ndt_x','ndt_y'])

    print('[INFO] Start collection')

    while not rospy.is_shutdown():
        
        if gnss_pose.is_updated == True and ndt_pose.is_updated == True:
            writer.writerow([gnss_pose.x,gnss_pose.y,ndt_pose.x,ndt_pose.y])
            gnss_pose.clear()
            ndt_pose.clear()

        rate.sleep()
    

    return

if __name__ == '__main__':
    main()