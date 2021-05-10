;; Auto-generated. Do not edit!


(when (boundp 'autoware_msgs::State)
  (if (not (find-package "AUTOWARE_MSGS"))
    (make-package "AUTOWARE_MSGS"))
  (shadow 'State (find-package "AUTOWARE_MSGS")))
(unless (find-package "AUTOWARE_MSGS::STATE")
  (make-package "AUTOWARE_MSGS::STATE"))

(in-package "ROS")
;;//! \htmlinclude State.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass autoware_msgs::State
  :super ros::object
  :slots (_header _vehicle_state _mission_state _behavior_state _motion_state ))

(defmethod autoware_msgs::State
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:vehicle_state __vehicle_state) "")
    ((:mission_state __mission_state) "")
    ((:behavior_state __behavior_state) "")
    ((:motion_state __motion_state) "")
    )
   (send-super :init)
   (setq _header __header)
   (setq _vehicle_state (string __vehicle_state))
   (setq _mission_state (string __mission_state))
   (setq _behavior_state (string __behavior_state))
   (setq _motion_state (string __motion_state))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:vehicle_state
   (&optional __vehicle_state)
   (if __vehicle_state (setq _vehicle_state __vehicle_state)) _vehicle_state)
  (:mission_state
   (&optional __mission_state)
   (if __mission_state (setq _mission_state __mission_state)) _mission_state)
  (:behavior_state
   (&optional __behavior_state)
   (if __behavior_state (setq _behavior_state __behavior_state)) _behavior_state)
  (:motion_state
   (&optional __motion_state)
   (if __motion_state (setq _motion_state __motion_state)) _motion_state)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; string _vehicle_state
    4 (length _vehicle_state)
    ;; string _mission_state
    4 (length _mission_state)
    ;; string _behavior_state
    4 (length _behavior_state)
    ;; string _motion_state
    4 (length _motion_state)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; string _vehicle_state
       (write-long (length _vehicle_state) s) (princ _vehicle_state s)
     ;; string _mission_state
       (write-long (length _mission_state) s) (princ _mission_state s)
     ;; string _behavior_state
       (write-long (length _behavior_state) s) (princ _behavior_state s)
     ;; string _motion_state
       (write-long (length _motion_state) s) (princ _motion_state s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; string _vehicle_state
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _vehicle_state (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _mission_state
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _mission_state (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _behavior_state
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _behavior_state (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _motion_state
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _motion_state (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get autoware_msgs::State :md5sum-) "937d7a6d5eba6aee9608f59553aebbe8")
(setf (get autoware_msgs::State :datatype-) "autoware_msgs/State")
(setf (get autoware_msgs::State :definition-)
      "Header header
string vehicle_state
string mission_state
string behavior_state
string motion_state

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



(provide :autoware_msgs/State "937d7a6d5eba6aee9608f59553aebbe8")


