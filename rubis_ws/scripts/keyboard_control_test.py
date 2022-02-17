from pynput import keyboard
import threading
import rospy
# from geometry_msgs.msg import TwistStamped
from autoware_msgs.msg import ControlCommandStamped

# ACC_MIN = -0.2
# ACC_MAX = 0.2
PI = 3.141592
STEER_MIN = -PI
STEER_MAX = PI

current_pressed = set()
global target_velocity_temp
global target_steering_angle_temp
global target_velocity
global target_steering_angle
target_velocity_temp = 0
target_steering_angle_temp = 0
target_velocity = 0
target_steering_angle = 0

def on_press(key):
    current_pressed.add(key)
    # print('Key %s pressed' % current_pressed)
    global target_velocity_temp
    global target_steering_angle_temp
    global target_velocity
    global target_steering_angle

    if keyboard.KeyCode(char='w') in current_pressed:
        target_velocity_temp += 1

    if keyboard.KeyCode(char='s') in current_pressed:
        target_velocity_temp -= 1

    if keyboard.KeyCode(char='a') in current_pressed:
        target_steering_angle_temp += STEER_MAX/2

    if keyboard.KeyCode(char='d') in current_pressed:
        target_steering_angle_temp += STEER_MIN/2
    
    if keyboard.KeyCode(char='r') in current_pressed:
        target_velocity = target_velocity_temp
        target_steering_angle = target_steering_angle_temp

    if keyboard.KeyCode(char='f') in current_pressed:
        target_velocity = 0
        target_steering_angle = 0

def print_temp():
    global target_velocity_temp
    global target_steering_angle_temp
    print('\n===========================================')
    print('W, A, S, D : control temp, F : Stop')
    print('R : control start, Q: print temp')
    print('Press P to Quit')
    print(f'target_velocity_temp: {target_velocity_temp}')
    print(f'target_steering_angle_temp: {target_steering_angle}')
    return

def print_target():
    global target_velocity
    global target_steering_angle
    print('\n===========================================')
    print('W, A, S, D : control temp, F : Stop')
    print('R : control start, Q: print temp')
    print('Press P to Quit')
    print(f'target_velocity: {target_velocity}')
    print(f'target_steering_angle: {target_steering_angle}')
    return

def keyboard_routine():
    global target_velocity_temp
    global target_steering_angle_temp
    global target_velocity
    global target_steering_angle
    print('\nW, A, S, D : control temp, F : Stop')
    print('R : control start, Q: print temp')
    print('Press P to Quit')
    
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
        listener.join()

def on_release(key):
    if key == keyboard.KeyCode(char='q'):
        print_temp()
    if key == keyboard.KeyCode(char='r'):
        print_target()
    if key == keyboard.KeyCode(char='p'):
        print('\nYou pressed P. Quit!')
        return False
    if key in current_pressed:
        current_pressed.remove(key)
    

if __name__ == '__main__':
    keyboard_thread = threading.Thread(target=keyboard_routine)
    keyboard_thread.start()

    ctrl_cmd_pub = rospy.Publisher('ctrl_cmd', ControlCommandStamped, queue_size=10)
    # twist_pub = rospy.Publisher('twist_cmd', TwistStamped, queue_size=10)

    rospy.init_node('keyboard_control_test')
    rate = rospy.Rate(100)

    ctrl_cmd_msg = ControlCommandStamped()
    # twist_msg = TwistStamped()
    
    while not rospy.is_shutdown():
        ctrl_cmd_msg.cmd.linear_velocity = target_velocity
        ctrl_cmd_msg.cmd.steering_angle = target_steering_angle
        
        ctrl_cmd_pub.publish(ctrl_cmd_msg)

        rate.sleep()
