;; Auto-generated. Do not edit!


(when (boundp 'autoware_msgs::RUBISTrafficSignalArray)
  (if (not (find-package "AUTOWARE_MSGS"))
    (make-package "AUTOWARE_MSGS"))
  (shadow 'RUBISTrafficSignalArray (find-package "AUTOWARE_MSGS")))
(unless (find-package "AUTOWARE_MSGS::RUBISTRAFFICSIGNALARRAY")
  (make-package "AUTOWARE_MSGS::RUBISTRAFFICSIGNALARRAY"))

(in-package "ROS")
;;//! \htmlinclude RUBISTrafficSignalArray.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass autoware_msgs::RUBISTrafficSignalArray
  :super ros::object
  :slots (_header _signals ))

(defmethod autoware_msgs::RUBISTrafficSignalArray
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:signals __signals) (let (r) (dotimes (i 0) (push (instance autoware_msgs::RUBISTrafficSignal :init) r)) r))
    )
   (send-super :init)
   (setq _header __header)
   (setq _signals __signals)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:signals
   (&rest __signals)
   (if (keywordp (car __signals))
       (send* _signals __signals)
     (progn
       (if __signals (setq _signals (car __signals)))
       _signals)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; autoware_msgs/RUBISTrafficSignal[] _signals
    (apply #'+ (send-all _signals :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; autoware_msgs/RUBISTrafficSignal[] _signals
     (write-long (length _signals) s)
     (dolist (elem _signals)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; autoware_msgs/RUBISTrafficSignal[] _signals
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _signals (let (r) (dotimes (i n) (push (instance autoware_msgs::RUBISTrafficSignal :init) r)) r))
     (dolist (elem- _signals)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get autoware_msgs::RUBISTrafficSignalArray :md5sum-) "8ae2769f49d9241a71af08054e4cc568")
(setf (get autoware_msgs::RUBISTrafficSignalArray :datatype-) "autoware_msgs/RUBISTrafficSignalArray")
(setf (get autoware_msgs::RUBISTrafficSignalArray :definition-)
      "Header header
RUBISTrafficSignal[] signals
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
MSG: autoware_msgs/RUBISTrafficSignal
int32 id
int32 type
float32 time
")



(provide :autoware_msgs/RUBISTrafficSignalArray "8ae2769f49d9241a71af08054e4cc568")


