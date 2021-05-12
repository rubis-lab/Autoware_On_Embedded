;; Auto-generated. Do not edit!


(when (boundp 'autoware_config_msgs::ConfigTwistFilter)
  (if (not (find-package "AUTOWARE_CONFIG_MSGS"))
    (make-package "AUTOWARE_CONFIG_MSGS"))
  (shadow 'ConfigTwistFilter (find-package "AUTOWARE_CONFIG_MSGS")))
(unless (find-package "AUTOWARE_CONFIG_MSGS::CONFIGTWISTFILTER")
  (make-package "AUTOWARE_CONFIG_MSGS::CONFIGTWISTFILTER"))

(in-package "ROS")
;;//! \htmlinclude ConfigTwistFilter.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass autoware_config_msgs::ConfigTwistFilter
  :super ros::object
  :slots (_header _use_decision_maker _lateral_accel_limit _lateral_jerk_limit _lowpass_gain_linear_x _lowpass_gain_angular_z _lowpass_gain_steering_angle ))

(defmethod autoware_config_msgs::ConfigTwistFilter
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:use_decision_maker __use_decision_maker) nil)
    ((:lateral_accel_limit __lateral_accel_limit) 0.0)
    ((:lateral_jerk_limit __lateral_jerk_limit) 0.0)
    ((:lowpass_gain_linear_x __lowpass_gain_linear_x) 0.0)
    ((:lowpass_gain_angular_z __lowpass_gain_angular_z) 0.0)
    ((:lowpass_gain_steering_angle __lowpass_gain_steering_angle) 0.0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _use_decision_maker __use_decision_maker)
   (setq _lateral_accel_limit (float __lateral_accel_limit))
   (setq _lateral_jerk_limit (float __lateral_jerk_limit))
   (setq _lowpass_gain_linear_x (float __lowpass_gain_linear_x))
   (setq _lowpass_gain_angular_z (float __lowpass_gain_angular_z))
   (setq _lowpass_gain_steering_angle (float __lowpass_gain_steering_angle))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:use_decision_maker
   (&optional __use_decision_maker)
   (if __use_decision_maker (setq _use_decision_maker __use_decision_maker)) _use_decision_maker)
  (:lateral_accel_limit
   (&optional __lateral_accel_limit)
   (if __lateral_accel_limit (setq _lateral_accel_limit __lateral_accel_limit)) _lateral_accel_limit)
  (:lateral_jerk_limit
   (&optional __lateral_jerk_limit)
   (if __lateral_jerk_limit (setq _lateral_jerk_limit __lateral_jerk_limit)) _lateral_jerk_limit)
  (:lowpass_gain_linear_x
   (&optional __lowpass_gain_linear_x)
   (if __lowpass_gain_linear_x (setq _lowpass_gain_linear_x __lowpass_gain_linear_x)) _lowpass_gain_linear_x)
  (:lowpass_gain_angular_z
   (&optional __lowpass_gain_angular_z)
   (if __lowpass_gain_angular_z (setq _lowpass_gain_angular_z __lowpass_gain_angular_z)) _lowpass_gain_angular_z)
  (:lowpass_gain_steering_angle
   (&optional __lowpass_gain_steering_angle)
   (if __lowpass_gain_steering_angle (setq _lowpass_gain_steering_angle __lowpass_gain_steering_angle)) _lowpass_gain_steering_angle)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; bool _use_decision_maker
    1
    ;; float32 _lateral_accel_limit
    4
    ;; float32 _lateral_jerk_limit
    4
    ;; float32 _lowpass_gain_linear_x
    4
    ;; float32 _lowpass_gain_angular_z
    4
    ;; float32 _lowpass_gain_steering_angle
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; bool _use_decision_maker
       (if _use_decision_maker (write-byte -1 s) (write-byte 0 s))
     ;; float32 _lateral_accel_limit
       (sys::poke _lateral_accel_limit (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _lateral_jerk_limit
       (sys::poke _lateral_jerk_limit (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _lowpass_gain_linear_x
       (sys::poke _lowpass_gain_linear_x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _lowpass_gain_angular_z
       (sys::poke _lowpass_gain_angular_z (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _lowpass_gain_steering_angle
       (sys::poke _lowpass_gain_steering_angle (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; bool _use_decision_maker
     (setq _use_decision_maker (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float32 _lateral_accel_limit
     (setq _lateral_accel_limit (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _lateral_jerk_limit
     (setq _lateral_jerk_limit (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _lowpass_gain_linear_x
     (setq _lowpass_gain_linear_x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _lowpass_gain_angular_z
     (setq _lowpass_gain_angular_z (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _lowpass_gain_steering_angle
     (setq _lowpass_gain_steering_angle (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get autoware_config_msgs::ConfigTwistFilter :md5sum-) "b8c0018f21cee202f52983a44f92d9c3")
(setf (get autoware_config_msgs::ConfigTwistFilter :datatype-) "autoware_config_msgs/ConfigTwistFilter")
(setf (get autoware_config_msgs::ConfigTwistFilter :definition-)
      "Header header
bool use_decision_maker
float32 lateral_accel_limit
float32 lateral_jerk_limit
float32 lowpass_gain_linear_x
float32 lowpass_gain_angular_z
float32 lowpass_gain_steering_angle

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

")



(provide :autoware_config_msgs/ConfigTwistFilter "b8c0018f21cee202f52983a44f92d9c3")


