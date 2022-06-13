from pynput import keyboard
import threading
import rospy
from carla_msgs.msg import CarlaEgoVehicleControl

ACC_MIN = -0.5
ACC_MAX = 0.5
PI = 3.141592
STEER_MIN = -PI
STEER_MAX = PI

current_pressed = set()
global current_acc
global current_steer
current_acc = 0
current_steer = 0

def on_press(key):
    global current_acc
    global current_steer
    current_pressed.add(key)
    # print('Key %s pressed' % current_pressed)

    if keyboard.KeyCode(char='w') in current_pressed:
        current_acc = ACC_MAX

    if keyboard.KeyCode(char='s') in current_pressed:
        current_acc = ACC_MIN

    if keyboard.KeyCode(char='a') in current_pressed:
        current_steer = STEER_MAX

    if keyboard.KeyCode(char='d') in current_pressed:
        current_steer = STEER_MIN
    
    if keyboard.KeyCode(char='f') in current_pressed:
        current_steer = 0
        current_acc = 0

def keyboard_routine():
    print('W, A, S, D : Move, F : Stop')
    print('Press P to Quit')
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
        listener.join()

def on_release(key):
    global current_acc
    global current_steer
    if key == keyboard.KeyCode(char='p'):
        print('\nYou pressed P. Quit!')
        return False
    if key == keyboard.KeyCode(char='a'):
        current_steer = 0        
    if key == keyboard.KeyCode(char='d'):
        current_steer = 0 
    if key in current_pressed:
        current_pressed.remove(key)
    

if __name__ == '__main__':
    keyboard_thread = threading.Thread(target=keyboard_routine)
    keyboard_thread.start()

    twist_pub = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, queue_size=10)

    rospy.init_node('keyboard_controller')
    rate = rospy.Rate(100)

    carla_msg = CarlaEgoVehicleControl()
    
    while not rospy.is_shutdown():
        carla_msg.throttle = current_acc
        carla_msg.steer = -current_steer
        
        twist_pub.publish(carla_msg)

        rate.sleep()