# How to use gnss_converter (/gnss_pose publisher)
### Case 1 : If you create a new map
* Record /Inertial_Labs/gps_data, /Inertial_Labs/ins_data, /ndt_pose, /ndt_stat data while driving through the map (rosbag)
```console
~$ rosbag record /ndt_pose /ndt_stat /Inertial_Labs/gps_data /Inertial_Labs/ins_data -O [NAME]
```
* Set /gnss_converter/calculate_tf of gnss_converter/cfg/gnsss_converter.yaml to true.
```yaml
/gnss_converter/calculate_tf : true
```
* Set /gnss_converter/bag_file_path of gnss_converter/cfg/gnsss_converter.yaml to recorded rosbag file.
```yaml
/gnss_converter/bag_file_path : /home/sunhokim/Documents/RUBIS/data/220111_pose_gnss_138ground.bag   # example
```
* Run the following code
```console
~$ roslaunch gnss_converter gnss_converter.launch
```
* Wait about 40 seconds and a image showing the route of the vehicle will appear on the screen.
* You can select the four points used to calculate the transformation matrix through the following method.
  - To change the image size, use track bar to select a value and press enter.
  - If you want to choose a point, press the number and click the point in the image. (RBUTTONDOWN)
  - If you want to end, press the ESC button.
  - **WARNING** : When performing angle transformation, linear transformation was assumed. However, since the angle has a value between -180 degrees and +180 degrees, if there is a discontinuous interval between the four points, the matrix is not properly calculated. Therefore, when selecting four points, it is necessary to make sure that there is no section where the change in angle is discontinuous. (If the first value of the ori_tf matrix is close to -1.0, the calculation was performed well.)
* After the previous process, the transformation matrix calculation result is displayed on the screen.
* The calculation result is transferred to gnss_converter/cfg/gnss_converter.yaml.
```yaml
# ========= position tf matrix =========
/gnss_converter/pos_tf : [-1.372640, -1.384772, -27.949656, 13944.211735, 0.410968, -0.109160, -11.105078, -9490.840625, -0.318483, -0.086243, -6.272676, 453.012563, 0.000000, -0.000000, 0.000000, 1.000000] 

# ========= orientation tf matrix =========
/gnss_converter/ori_tf : [-1.000448, 0.345540, 0.569044, 0.987319, 0.021691, -2.279683, -4.681593, -0.067717, 0.056971, 1.608331, -3.444783, -0.114943, 0.000000, -0.000000, 0.000000, 1.000000]
```
* Move to Case 2
### Case 2 : If you've already done Case 1
* Set /gnss_converter/calculate_tf of gnss_converter/cfg/gnsss_converter.yaml to false.
```yaml
/gnss_converter/calculate_tf : false
```
* Execute gnss_converter launch file.
```console
~$ roslaunch gnss_converter gnss_converter.launch
```
