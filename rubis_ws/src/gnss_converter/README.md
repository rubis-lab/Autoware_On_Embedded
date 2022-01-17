## How to use gnss_converter (/gnss_pose publisher)
### Case 1 : If you create a new map
* Record /Inertial_Labs/gps_data, /Inertial_Labs/ins_data, /ndt_pose, /ndt_stat data while driving through the map (rosbag)
```console
~$ rosbag record /ndt_pose /ndt_stat /Inertial_Labs/gps_data /Inertial_Labs/ins_data -O [NAME]
```
* Set /gnss_converter/calculate_tf of gnss_converter/cfg/gnsss_converter.yaml to true.
```yaml
/gnss_converter/calculate_tf : true
```
* Run the following code
```console
~$ roslaunch gnss_converter gnss_converter.launch
~$ rosbag play [rosbag file]
```
* Wait about 30 seconds and you will see a transformation matrix on the screen.
* The calculation result is transferred to gnss_converter/cfg/gnss_converter.yaml.
```yaml
# ========= position tf matrix =========
/gnss_converter/pos_tf : [-0.292395, -1.125976, -5.257557, 13690.052429, 1.616471, 0.198493, 13.630088, -10364.443012, -0.267041, -0.073857, -5.543469, 227.041684, 0.0, 0.0, 0.0, 1.0] 

# ========= orientation tf matrix =========
/gnss_converter/ori_tf : [0.276667, 15.435781, -17.992405, -0.632826, -0.004556, -2.592555, 6.993107, 0.099857, -0.010594, 0.505370, 3.094219, 0.039236, 0.0, 0.0, 0.0, 1.0]
```
* Move to Case 2
### Case 2 : If you've already done Case 1
* Set /gnss_converter/calculate_tf of gnss_converter/cfg/gnsss_converter.yaml to false.
* Execute gnss_converter launch file.
```console
~$ roslaunch gnss_converter gnss_converter.launch
```
