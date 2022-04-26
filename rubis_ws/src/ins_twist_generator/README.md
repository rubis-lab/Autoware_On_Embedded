# ins_twist_generator
The *ins_twist_generator* convert velocity information in INS-D sensor data to Autoware format.

### How to Use
1. Calculate the yaw offset by using *yaw_offset_calculator* node.
2. Set the *yaw_offset* parameter in `ins_twist_generatr.launch` file.
3. Launch `ins_twist_generatr.launch`.


### Parameters
|Parameter|Type|Description|
-|-|-
|`yaw_offset`|*Double*|The yaw offset calculated from *yaw_offset_calculater* node.|

### Subscribed Topics
|Topic|Type|Description|
------|----|---------
`/Inertial_Labs/ins_data`|`inertiallabs_msgs/ins_data`|Calculate yaw value with yaw offset.
`/Inertial_Labs/sensor_data`|`inertiallabs_msgs/sensor_data`|Get angular velocity.
`/Inertial_Labs/gps_data`|`inertiallabs_msgs/gps_data`| Get linear velocity.

### Published Topics
|Topic|Type|Description|
------|----|---------
`/ins_twist`|`geometry_msgs/PoseStamped`| Accurate twist information.
`/ins_stat`|`rubis_msgs/InsStat`| InsStat information for Kalman filtering.