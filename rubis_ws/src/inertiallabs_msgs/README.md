# inertiallabs_msgs

  ins_data

  ```
std_msgs/Header             header
float64                     GPS_INS_Time
float64                     GPS_IMU_Time
std_msgs/UInt32             GPS_mSOW
geometry_msgs/Vector3       LLH
geometry_msgs/Vector3       YPR
geometry_msgs/Quaternion    OriQuat
geometry_msgs/Vector3       Vel_ENU
std_msgs/Int8               Solution_Status
geometry_msgs/Vector3       Pos_STD
float32                     Heading_STD
uint16                      USW

  ```

  gnss_data

  ```
std_msgs/Header          header
int8                     GNSS_info_1
int8                     GNSS_info_2
int8                     Number_Sat
float32                  GNSS_Velocity_Latency
int8                     GNSS_Angles_Position_Type
float32                  GNSS_Heading
float32                  GNSS_Pitch
float32                  GNSS_GDOP
float32                  GNSS_PDOP
float32                  GNSS_HDOP
float32                  GNSS_VDOP
float32                  GNSS_TDOP
uint8                    New_GNSS_Flags
float64                  Diff_Age

  ```

  gps_data

  ```

std_msgs/Header           header
geometry_msgs/Vector3     LLH
float32                   HorSpeed
float32                   SpeedDir
float32                   VerSpeed

  ```

  sensor_data

  ```

std_msgs/Header           header
geometry_msgs/Vector3     Mag
geometry_msgs/Vector3     Accel
geometry_msgs/Vector3     Gyro
float32                   Temp
float32                   Vinp
float32                   Pressure
float32                   Barometric_Height

  ```

  marine_data

  ```

  std_msgs/Header           header
float64                   Heave
float64                   Surge
float64                   Sway
float32                   Heave_velocity
float32                   Surge_velocity
float32                   Sway_velocity
float32                   Significant_wave_height

  ```
