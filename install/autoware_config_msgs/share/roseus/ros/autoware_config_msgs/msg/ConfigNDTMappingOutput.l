;; Auto-generated. Do not edit!


(when (boundp 'autoware_config_msgs::ConfigNDTMappingOutput)
  (if (not (find-package "AUTOWARE_CONFIG_MSGS"))
    (make-package "AUTOWARE_CONFIG_MSGS"))
  (shadow 'ConfigNDTMappingOutput (find-package "AUTOWARE_CONFIG_MSGS")))
(unless (find-package "AUTOWARE_CONFIG_MSGS::CONFIGNDTMAPPINGOUTPUT")
  (make-package "AUTOWARE_CONFIG_MSGS::CONFIGNDTMAPPINGOUTPUT"))

(in-package "ROS")
;;//! \htmlinclude ConfigNDTMappingOutput.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass autoware_config_msgs::ConfigNDTMappingOutput
  :super ros::object
  :slots (_header _filename _filter_res ))

(defmethod autoware_config_msgs::ConfigNDTMappingOutput
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:filename __filename) "")
    ((:filter_res __filter_res) 0.0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _filename (string __filename))
   (setq _filter_res (float __filter_res))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:filename
   (&optional __filename)
   (if __filename (setq _filename __filename)) _filename)
  (:filter_res
   (&optional __filter_res)
   (if __filter_res (setq _filter_res __filter_res)) _filter_res)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; string _filename
    4 (length _filename)
    ;; float32 _filter_res
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; string _filename
       (write-long (length _filename) s) (princ _filename s)
     ;; float32 _filter_res
       (sys::poke _filter_res (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; string _filename
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _filename (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float32 _filter_res
     (setq _filter_res (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get autoware_config_msgs::ConfigNDTMappingOutput :md5sum-) "ac31ee963c2f2d01d1d409a7749c20f6")
(setf (get autoware_config_msgs::ConfigNDTMappingOutput :datatype-) "autoware_config_msgs/ConfigNDTMappingOutput")
(setf (get autoware_config_msgs::ConfigNDTMappingOutput :definition-)
      "Header header
string filename
float32 filter_res

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



(provide :autoware_config_msgs/ConfigNDTMappingOutput "ac31ee963c2f2d01d1d409a7749c20f6")


