; Auto-generated. Do not edit!


(cl:in-package autoware_config_msgs-msg)


;//! \htmlinclude ConfigSSD.msg.html

(cl:defclass <ConfigSSD> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (score_threshold
    :reader score_threshold
    :initarg :score_threshold
    :type cl:float
    :initform 0.0))
)

(cl:defclass ConfigSSD (<ConfigSSD>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ConfigSSD>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ConfigSSD)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name autoware_config_msgs-msg:<ConfigSSD> is deprecated: use autoware_config_msgs-msg:ConfigSSD instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <ConfigSSD>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:header-val is deprecated.  Use autoware_config_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'score_threshold-val :lambda-list '(m))
(cl:defmethod score_threshold-val ((m <ConfigSSD>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:score_threshold-val is deprecated.  Use autoware_config_msgs-msg:score_threshold instead.")
  (score_threshold m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ConfigSSD>) ostream)
  "Serializes a message object of type '<ConfigSSD>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'score_threshold))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ConfigSSD>) istream)
  "Deserializes a message object of type '<ConfigSSD>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'score_threshold) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ConfigSSD>)))
  "Returns string type for a message object of type '<ConfigSSD>"
  "autoware_config_msgs/ConfigSSD")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ConfigSSD)))
  "Returns string type for a message object of type 'ConfigSSD"
  "autoware_config_msgs/ConfigSSD")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ConfigSSD>)))
  "Returns md5sum for a message object of type '<ConfigSSD>"
  "9c20d382dda6d21d4020d239679f6abd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ConfigSSD)))
  "Returns md5sum for a message object of type 'ConfigSSD"
  "9c20d382dda6d21d4020d239679f6abd")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ConfigSSD>)))
  "Returns full string definition for message of type '<ConfigSSD>"
  (cl:format cl:nil "Header  header~%float32 score_threshold #minimum score required to keep the detection [0.0, 1.0] (default 0.6)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ConfigSSD)))
  "Returns full string definition for message of type 'ConfigSSD"
  (cl:format cl:nil "Header  header~%float32 score_threshold #minimum score required to keep the detection [0.0, 1.0] (default 0.6)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ConfigSSD>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ConfigSSD>))
  "Converts a ROS message object to a list"
  (cl:list 'ConfigSSD
    (cl:cons ':header (header msg))
    (cl:cons ':score_threshold (score_threshold msg))
))
