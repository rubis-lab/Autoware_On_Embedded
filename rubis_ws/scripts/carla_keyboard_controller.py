from pynput import keyboard
import threading
import rospy
from geometry_msgs.msg import TwistStamped
from autoware_msgs.msg import ControlCommand
from autoware_msgs.msg import VehicleCmd
from carla_msgs.msg import CarlaEgoVehicleControl


# PI = 3.141592
THR_MIN = 0
THR_MAX = 1
STEER_MIN = -1
STEER_MAX = 1
BRAKE_MIN = 0
BRAKE_MAX = 1



current_pressed = set()
# global current_acc
# global current_steer
# current_acc = 0
# current_steer = 0
global current_thr
# 0 to 1
global current_steer
# -1 to 1
global current_brake
# 0 to 1
global reverse_toggle
# 0 "OR" 1

current_thr = 0
current_steer = 0
current_brake = 0
reverse_toggle = 0


def on_press(key):
    global current_thr
    global current_steer
    global current_brake
    global reverse_toggle

    current_pressed.add(key)
    # print('Key %s pressed' % current_pressed)

    if keyboard.KeyCode(char='w') in current_pressed:
        current_thr = THR_MAX

    if keyboard.KeyCode(char='s') in current_pressed:
        current_brake = BRAKE_MAX

    if keyboard.KeyCode(char='a') in current_pressed:
        current_steer = STEER_MIN

    if keyboard.KeyCode(char='d') in current_pressed:
        current_steer = STEER_MAX
    
    if keyboard.KeyCode(char='f') in current_pressed:
        current_steer = 0
        current_thr = 0

def keyboard_routine():
    print('W, A, S, D : Move, F : Stop')
    print('Press P to Quit')
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
        listener.join()

def on_release(key):
    global current_thr
    global current_brake
    global current_steer

    if key == keyboard.KeyCode(char='p'):
        print('\nYou pressed P. Quit!')
        return False
    if key == keyboard.KeyCode(char='a'):
        current_steer = 0        
    if key == keyboard.KeyCode(char='d'):
        current_steer = 0 
    if key == keyboard.KeyCode(char='s'):
        current_brake = 0 
    if key == keyboard.KeyCode(char='w'):
        current_thr = 0 
    if key in current_pressed:
        current_pressed.remove(key)
    

if __name__ == '__main__':
    keyboard_thread = threading.Thread(target=keyboard_routine)
    keyboard_thread.start()

    # twist_pub = rospy.Publisher('vehicle_cmd', VehicleCmd, queue_size=10)
    carla_control_pub = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, queue_size=10)

    rospy.init_node('keyboard_controller')
    rate = rospy.Rate(100)

    # twist_msg = TwistStamped()
    # ctrl_cmd = ControlCommand()
    # v_msg = VehicleCmd()

    carla_control_msg = CarlaEgoVehicleControl()
    
    while not rospy.is_shutdown():
        # twist_msg.twist.linear.x = current_acc
        # twist_msg.twist.angular.z = current_steer
        carla_control_msg.throttle = current_thr
        carla_control_msg.steer = current_steer
        carla_control_msg.brake = current_brake
        carla_control_msg.reverse = reverse_toggle
        # v_msg.twist_cmd = twist_msg
        # v_msg.ctrl_cmd = ctrl_cmd
        
        carla_control_pub.publish(carla_control_msg)

        rate.sleep()
