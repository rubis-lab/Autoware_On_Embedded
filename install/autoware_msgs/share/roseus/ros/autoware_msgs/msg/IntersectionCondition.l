;; Auto-generated. Do not edit!


(when (boundp 'autoware_msgs::IntersectionCondition)
  (if (not (find-package "AUTOWARE_MSGS"))
    (make-package "AUTOWARE_MSGS"))
  (shadow 'IntersectionCondition (find-package "AUTOWARE_MSGS")))
(unless (find-package "AUTOWARE_MSGS::INTERSECTIONCONDITION")
  (make-package "AUTOWARE_MSGS::INTERSECTIONCONDITION"))

(in-package "ROS")
;;//! \htmlinclude IntersectionCondition.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass autoware_msgs::IntersectionCondition
  :super ros::object
  :slots (_header _intersectionID _intersectionDistance _isIntersection _riskyLeftTurn _riskyRightTurn ))

(defmethod autoware_msgs::IntersectionCondition
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:intersectionID __intersectionID) 0)
    ((:intersectionDistance __intersectionDistance) 0.0)
    ((:isIntersection __isIntersection) 0)
    ((:riskyLeftTurn __riskyLeftTurn) 0)
    ((:riskyRightTurn __riskyRightTurn) 0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _intersectionID (round __intersectionID))
   (setq _intersectionDistance (float __intersectionDistance))
   (setq _isIntersection (round __isIntersection))
   (setq _riskyLeftTurn (round __riskyLeftTurn))
   (setq _riskyRightTurn (round __riskyRightTurn))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:intersectionID
   (&optional __intersectionID)
   (if __intersectionID (setq _intersectionID __intersectionID)) _intersectionID)
  (:intersectionDistance
   (&optional __intersectionDistance)
   (if __intersectionDistance (setq _intersectionDistance __intersectionDistance)) _intersectionDistance)
  (:isIntersection
   (&optional __isIntersection)
   (if __isIntersection (setq _isIntersection __isIntersection)) _isIntersection)
  (:riskyLeftTurn
   (&optional __riskyLeftTurn)
   (if __riskyLeftTurn (setq _riskyLeftTurn __riskyLeftTurn)) _riskyLeftTurn)
  (:riskyRightTurn
   (&optional __riskyRightTurn)
   (if __riskyRightTurn (setq _riskyRightTurn __riskyRightTurn)) _riskyRightTurn)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; int32 _intersectionID
    4
    ;; float32 _intersectionDistance
    4
    ;; int8 _isIntersection
    1
    ;; int8 _riskyLeftTurn
    1
    ;; int8 _riskyRightTurn
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; int32 _intersectionID
       (write-long _intersectionID s)
     ;; float32 _intersectionDistance
       (sys::poke _intersectionDistance (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; int8 _isIntersection
       (write-byte _isIntersection s)
     ;; int8 _riskyLeftTurn
       (write-byte _riskyLeftTurn s)
     ;; int8 _riskyRightTurn
       (write-byte _riskyRightTurn s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; int32 _intersectionID
     (setq _intersectionID (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float32 _intersectionDistance
     (setq _intersectionDistance (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; int8 _isIntersection
     (setq _isIntersection (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _isIntersection 127) (setq _isIntersection (- _isIntersection 256)))
   ;; int8 _riskyLeftTurn
     (setq _riskyLeftTurn (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _riskyLeftTurn 127) (setq _riskyLeftTurn (- _riskyLeftTurn 256)))
   ;; int8 _riskyRightTurn
     (setq _riskyRightTurn (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _riskyRightTurn 127) (setq _riskyRightTurn (- _riskyRightTurn 256)))
   ;;
   self)
  )

(setf (get autoware_msgs::IntersectionCondition :md5sum-) "ec2240e9bbead2818bb2ecd5340b0db8")
(setf (get autoware_msgs::IntersectionCondition :datatype-) "autoware_msgs/IntersectionCondition")
(setf (get autoware_msgs::IntersectionCondition :definition-)
      "Header header
int32 intersectionID
float32 intersectionDistance
int8 isIntersection
int8 riskyLeftTurn
int8 riskyRightTurn
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



(provide :autoware_msgs/IntersectionCondition "ec2240e9bbead2818bb2ecd5340b0db8")


