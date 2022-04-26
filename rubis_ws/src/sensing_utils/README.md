# Sensing Utils
### yaw_offset_caclulator ###
Calculate yaw offset between **map TF** and **INS-D yaw**.
- How to use
1. Launch ndt_matching and conduct localization.
Note: The location of the vehicle should be fixed for whole process.
2. Launch INS-D module and check *Inertial_Labs/ins_data* topic is published.
```
roslaunch inertiallabs_ins ins.launch
rostopic echo /Inertial_Labs/ins_data
```
3. Execute *yaw_offset_calculator*.
```
rosrun sensing_utils yaw_offset_calculator
```
The result will be printed to the screen in 5 seconds.

### ins_sync_test ###
Debug node for INS-D sensor synchronization.

### quaternion_to_rpy ###
Sample source code for converting quaternion to rpy.

### odom_converter ###
Deprecated.
