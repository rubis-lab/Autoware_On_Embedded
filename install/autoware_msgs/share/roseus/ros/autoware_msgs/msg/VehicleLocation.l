;; Auto-generated. Do not edit!


(when (boundp 'autoware_msgs::VehicleLocation)
  (if (not (find-package "AUTOWARE_MSGS"))
    (make-package "AUTOWARE_MSGS"))
  (shadow 'VehicleLocation (find-package "AUTOWARE_MSGS")))
(unless (find-package "AUTOWARE_MSGS::VEHICLELOCATION")
  (make-package "AUTOWARE_MSGS::VEHICLELOCATION"))

(in-package "ROS")
;;//! \htmlinclude VehicleLocation.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass autoware_msgs::VehicleLocation
  :super ros::object
  :slots (_header _lane_array_id _waypoint_index ))

(defmethod autoware_msgs::VehicleLocation
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:lane_array_id __lane_array_id) 0)
    ((:waypoint_index __waypoint_index) 0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _lane_array_id (round __lane_array_id))
   (setq _waypoint_index (round __waypoint_index))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:lane_array_id
   (&optional __lane_array_id)
   (if __lane_array_id (setq _lane_array_id __lane_array_id)) _lane_array_id)
  (:waypoint_index
   (&optional __waypoint_index)
   (if __waypoint_index (setq _waypoint_index __waypoint_index)) _waypoint_index)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; int32 _lane_array_id
    4
    ;; int32 _waypoint_index
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; int32 _lane_array_id
       (write-long _lane_array_id s)
     ;; int32 _waypoint_index
       (write-long _waypoint_index s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; int32 _lane_array_id
     (setq _lane_array_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _waypoint_index
     (setq _waypoint_index (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get autoware_msgs::VehicleLocation :md5sum-) "cba3770fc8eb8557ac8c63f4c0d3155b")
(setf (get autoware_msgs::VehicleLocation :datatype-) "autoware_msgs/VehicleLocation")
(setf (get autoware_msgs::VehicleLocation :definition-)
      "Header header
int32 lane_array_id
int32 waypoint_index

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



(provide :autoware_msgs/VehicleLocation "cba3770fc8eb8557ac8c63f4c0d3155b")


