; Auto-generated. Do not edit!


(cl:in-package autoware_config_msgs-msg)


;//! \htmlinclude ConfigNDTMappingOutput.msg.html

(cl:defclass <ConfigNDTMappingOutput> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (filename
    :reader filename
    :initarg :filename
    :type cl:string
    :initform "")
   (filter_res
    :reader filter_res
    :initarg :filter_res
    :type cl:float
    :initform 0.0))
)

(cl:defclass ConfigNDTMappingOutput (<ConfigNDTMappingOutput>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ConfigNDTMappingOutput>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ConfigNDTMappingOutput)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name autoware_config_msgs-msg:<ConfigNDTMappingOutput> is deprecated: use autoware_config_msgs-msg:ConfigNDTMappingOutput instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <ConfigNDTMappingOutput>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:header-val is deprecated.  Use autoware_config_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'filename-val :lambda-list '(m))
(cl:defmethod filename-val ((m <ConfigNDTMappingOutput>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:filename-val is deprecated.  Use autoware_config_msgs-msg:filename instead.")
  (filename m))

(cl:ensure-generic-function 'filter_res-val :lambda-list '(m))
(cl:defmethod filter_res-val ((m <ConfigNDTMappingOutput>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:filter_res-val is deprecated.  Use autoware_config_msgs-msg:filter_res instead.")
  (filter_res m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ConfigNDTMappingOutput>) ostream)
  "Serializes a message object of type '<ConfigNDTMappingOutput>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'filename))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'filename))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'filter_res))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ConfigNDTMappingOutput>) istream)
  "Deserializes a message object of type '<ConfigNDTMappingOutput>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'filename) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'filename) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'filter_res) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ConfigNDTMappingOutput>)))
  "Returns string type for a message object of type '<ConfigNDTMappingOutput>"
  "autoware_config_msgs/ConfigNDTMappingOutput")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ConfigNDTMappingOutput)))
  "Returns string type for a message object of type 'ConfigNDTMappingOutput"
  "autoware_config_msgs/ConfigNDTMappingOutput")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ConfigNDTMappingOutput>)))
  "Returns md5sum for a message object of type '<ConfigNDTMappingOutput>"
  "ac31ee963c2f2d01d1d409a7749c20f6")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ConfigNDTMappingOutput)))
  "Returns md5sum for a message object of type 'ConfigNDTMappingOutput"
  "ac31ee963c2f2d01d1d409a7749c20f6")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ConfigNDTMappingOutput>)))
  "Returns full string definition for message of type '<ConfigNDTMappingOutput>"
  (cl:format cl:nil "Header header~%string filename~%float32 filter_res~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ConfigNDTMappingOutput)))
  "Returns full string definition for message of type 'ConfigNDTMappingOutput"
  (cl:format cl:nil "Header header~%string filename~%float32 filter_res~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ConfigNDTMappingOutput>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:length (cl:slot-value msg 'filename))
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ConfigNDTMappingOutput>))
  "Converts a ROS message object to a list"
  (cl:list 'ConfigNDTMappingOutput
    (cl:cons ':header (header msg))
    (cl:cons ':filename (filename msg))
    (cl:cons ':filter_res (filter_res msg))
))
