;; Auto-generated. Do not edit!


(when (boundp 'autoware_msgs::VehicleCmd)
  (if (not (find-package "AUTOWARE_MSGS"))
    (make-package "AUTOWARE_MSGS"))
  (shadow 'VehicleCmd (find-package "AUTOWARE_MSGS")))
(unless (find-package "AUTOWARE_MSGS::VEHICLECMD")
  (make-package "AUTOWARE_MSGS::VEHICLECMD"))

(in-package "ROS")
;;//! \htmlinclude VehicleCmd.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass autoware_msgs::VehicleCmd
  :super ros::object
  :slots (_header _steer_cmd _accel_cmd _brake_cmd _lamp_cmd _gear_cmd _mode _twist_cmd _ctrl_cmd _emergency ))

(defmethod autoware_msgs::VehicleCmd
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:steer_cmd __steer_cmd) (instance autoware_msgs::SteerCmd :init))
    ((:accel_cmd __accel_cmd) (instance autoware_msgs::AccelCmd :init))
    ((:brake_cmd __brake_cmd) (instance autoware_msgs::BrakeCmd :init))
    ((:lamp_cmd __lamp_cmd) (instance autoware_msgs::LampCmd :init))
    ((:gear_cmd __gear_cmd) (instance autoware_msgs::Gear :init))
    ((:mode __mode) 0)
    ((:twist_cmd __twist_cmd) (instance geometry_msgs::TwistStamped :init))
    ((:ctrl_cmd __ctrl_cmd) (instance autoware_msgs::ControlCommand :init))
    ((:emergency __emergency) 0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _steer_cmd __steer_cmd)
   (setq _accel_cmd __accel_cmd)
   (setq _brake_cmd __brake_cmd)
   (setq _lamp_cmd __lamp_cmd)
   (setq _gear_cmd __gear_cmd)
   (setq _mode (round __mode))
   (setq _twist_cmd __twist_cmd)
   (setq _ctrl_cmd __ctrl_cmd)
   (setq _emergency (round __emergency))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:steer_cmd
   (&rest __steer_cmd)
   (if (keywordp (car __steer_cmd))
       (send* _steer_cmd __steer_cmd)
     (progn
       (if __steer_cmd (setq _steer_cmd (car __steer_cmd)))
       _steer_cmd)))
  (:accel_cmd
   (&rest __accel_cmd)
   (if (keywordp (car __accel_cmd))
       (send* _accel_cmd __accel_cmd)
     (progn
       (if __accel_cmd (setq _accel_cmd (car __accel_cmd)))
       _accel_cmd)))
  (:brake_cmd
   (&rest __brake_cmd)
   (if (keywordp (car __brake_cmd))
       (send* _brake_cmd __brake_cmd)
     (progn
       (if __brake_cmd (setq _brake_cmd (car __brake_cmd)))
       _brake_cmd)))
  (:lamp_cmd
   (&rest __lamp_cmd)
   (if (keywordp (car __lamp_cmd))
       (send* _lamp_cmd __lamp_cmd)
     (progn
       (if __lamp_cmd (setq _lamp_cmd (car __lamp_cmd)))
       _lamp_cmd)))
  (:gear_cmd
   (&rest __gear_cmd)
   (if (keywordp (car __gear_cmd))
       (send* _gear_cmd __gear_cmd)
     (progn
       (if __gear_cmd (setq _gear_cmd (car __gear_cmd)))
       _gear_cmd)))
  (:mode
   (&optional __mode)
   (if __mode (setq _mode __mode)) _mode)
  (:twist_cmd
   (&rest __twist_cmd)
   (if (keywordp (car __twist_cmd))
       (send* _twist_cmd __twist_cmd)
     (progn
       (if __twist_cmd (setq _twist_cmd (car __twist_cmd)))
       _twist_cmd)))
  (:ctrl_cmd
   (&rest __ctrl_cmd)
   (if (keywordp (car __ctrl_cmd))
       (send* _ctrl_cmd __ctrl_cmd)
     (progn
       (if __ctrl_cmd (setq _ctrl_cmd (car __ctrl_cmd)))
       _ctrl_cmd)))
  (:emergency
   (&optional __emergency)
   (if __emergency (setq _emergency __emergency)) _emergency)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; autoware_msgs/SteerCmd _steer_cmd
    (send _steer_cmd :serialization-length)
    ;; autoware_msgs/AccelCmd _accel_cmd
    (send _accel_cmd :serialization-length)
    ;; autoware_msgs/BrakeCmd _brake_cmd
    (send _brake_cmd :serialization-length)
    ;; autoware_msgs/LampCmd _lamp_cmd
    (send _lamp_cmd :serialization-length)
    ;; autoware_msgs/Gear _gear_cmd
    (send _gear_cmd :serialization-length)
    ;; int32 _mode
    4
    ;; geometry_msgs/TwistStamped _twist_cmd
    (send _twist_cmd :serialization-length)
    ;; autoware_msgs/ControlCommand _ctrl_cmd
    (send _ctrl_cmd :serialization-length)
    ;; int32 _emergency
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; autoware_msgs/SteerCmd _steer_cmd
       (send _steer_cmd :serialize s)
     ;; autoware_msgs/AccelCmd _accel_cmd
       (send _accel_cmd :serialize s)
     ;; autoware_msgs/BrakeCmd _brake_cmd
       (send _brake_cmd :serialize s)
     ;; autoware_msgs/LampCmd _lamp_cmd
       (send _lamp_cmd :serialize s)
     ;; autoware_msgs/Gear _gear_cmd
       (send _gear_cmd :serialize s)
     ;; int32 _mode
       (write-long _mode s)
     ;; geometry_msgs/TwistStamped _twist_cmd
       (send _twist_cmd :serialize s)
     ;; autoware_msgs/ControlCommand _ctrl_cmd
       (send _ctrl_cmd :serialize s)
     ;; int32 _emergency
       (write-long _emergency s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; autoware_msgs/SteerCmd _steer_cmd
     (send _steer_cmd :deserialize buf ptr-) (incf ptr- (send _steer_cmd :serialization-length))
   ;; autoware_msgs/AccelCmd _accel_cmd
     (send _accel_cmd :deserialize buf ptr-) (incf ptr- (send _accel_cmd :serialization-length))
   ;; autoware_msgs/BrakeCmd _brake_cmd
     (send _brake_cmd :deserialize buf ptr-) (incf ptr- (send _brake_cmd :serialization-length))
   ;; autoware_msgs/LampCmd _lamp_cmd
     (send _lamp_cmd :deserialize buf ptr-) (incf ptr- (send _lamp_cmd :serialization-length))
   ;; autoware_msgs/Gear _gear_cmd
     (send _gear_cmd :deserialize buf ptr-) (incf ptr- (send _gear_cmd :serialization-length))
   ;; int32 _mode
     (setq _mode (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; geometry_msgs/TwistStamped _twist_cmd
     (send _twist_cmd :deserialize buf ptr-) (incf ptr- (send _twist_cmd :serialization-length))
   ;; autoware_msgs/ControlCommand _ctrl_cmd
     (send _ctrl_cmd :deserialize buf ptr-) (incf ptr- (send _ctrl_cmd :serialization-length))
   ;; int32 _emergency
     (setq _emergency (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get autoware_msgs::VehicleCmd :md5sum-) "8e7481711780182446911d94fa3a384d")
(setf (get autoware_msgs::VehicleCmd :datatype-) "autoware_msgs/VehicleCmd")
(setf (get autoware_msgs::VehicleCmd :definition-)
      "Header header
autoware_msgs/SteerCmd steer_cmd
autoware_msgs/AccelCmd accel_cmd
autoware_msgs/BrakeCmd brake_cmd
autoware_msgs/LampCmd lamp_cmd
autoware_msgs/Gear gear_cmd
int32 mode
geometry_msgs/TwistStamped twist_cmd
autoware_msgs/ControlCommand ctrl_cmd
int32 emergency

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: autoware_msgs/SteerCmd
Header header
int32 steer

================================================================================
MSG: autoware_msgs/AccelCmd
Header header
int32 accel

================================================================================
MSG: autoware_msgs/BrakeCmd
Header header
int32 brake

================================================================================
MSG: autoware_msgs/LampCmd
Header header
int32 l
int32 r

================================================================================
MSG: autoware_msgs/Gear
uint8 NONE=0
uint8 PARK=1
uint8 REVERSE=2
uint8 NEUTRAL=3
uint8 DRIVE=4
uint8 LOW=5
uint8 gear
================================================================================
MSG: geometry_msgs/TwistStamped
# A twist with reference coordinate frame and timestamp
Header header
Twist twist

================================================================================
MSG: geometry_msgs/Twist
# This expresses velocity in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
================================================================================
MSG: autoware_msgs/ControlCommand
float64 linear_velocity
float64 linear_acceleration #m/s^2
float64 steering_angle

")



(provide :autoware_msgs/VehicleCmd "8e7481711780182446911d94fa3a384d")


