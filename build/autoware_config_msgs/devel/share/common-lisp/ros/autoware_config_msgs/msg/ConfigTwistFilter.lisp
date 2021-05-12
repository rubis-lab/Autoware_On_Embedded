; Auto-generated. Do not edit!


(cl:in-package autoware_config_msgs-msg)


;//! \htmlinclude ConfigTwistFilter.msg.html

(cl:defclass <ConfigTwistFilter> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (use_decision_maker
    :reader use_decision_maker
    :initarg :use_decision_maker
    :type cl:boolean
    :initform cl:nil)
   (lateral_accel_limit
    :reader lateral_accel_limit
    :initarg :lateral_accel_limit
    :type cl:float
    :initform 0.0)
   (lateral_jerk_limit
    :reader lateral_jerk_limit
    :initarg :lateral_jerk_limit
    :type cl:float
    :initform 0.0)
   (lowpass_gain_linear_x
    :reader lowpass_gain_linear_x
    :initarg :lowpass_gain_linear_x
    :type cl:float
    :initform 0.0)
   (lowpass_gain_angular_z
    :reader lowpass_gain_angular_z
    :initarg :lowpass_gain_angular_z
    :type cl:float
    :initform 0.0)
   (lowpass_gain_steering_angle
    :reader lowpass_gain_steering_angle
    :initarg :lowpass_gain_steering_angle
    :type cl:float
    :initform 0.0))
)

(cl:defclass ConfigTwistFilter (<ConfigTwistFilter>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ConfigTwistFilter>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ConfigTwistFilter)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name autoware_config_msgs-msg:<ConfigTwistFilter> is deprecated: use autoware_config_msgs-msg:ConfigTwistFilter instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <ConfigTwistFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:header-val is deprecated.  Use autoware_config_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'use_decision_maker-val :lambda-list '(m))
(cl:defmethod use_decision_maker-val ((m <ConfigTwistFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:use_decision_maker-val is deprecated.  Use autoware_config_msgs-msg:use_decision_maker instead.")
  (use_decision_maker m))

(cl:ensure-generic-function 'lateral_accel_limit-val :lambda-list '(m))
(cl:defmethod lateral_accel_limit-val ((m <ConfigTwistFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:lateral_accel_limit-val is deprecated.  Use autoware_config_msgs-msg:lateral_accel_limit instead.")
  (lateral_accel_limit m))

(cl:ensure-generic-function 'lateral_jerk_limit-val :lambda-list '(m))
(cl:defmethod lateral_jerk_limit-val ((m <ConfigTwistFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:lateral_jerk_limit-val is deprecated.  Use autoware_config_msgs-msg:lateral_jerk_limit instead.")
  (lateral_jerk_limit m))

(cl:ensure-generic-function 'lowpass_gain_linear_x-val :lambda-list '(m))
(cl:defmethod lowpass_gain_linear_x-val ((m <ConfigTwistFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:lowpass_gain_linear_x-val is deprecated.  Use autoware_config_msgs-msg:lowpass_gain_linear_x instead.")
  (lowpass_gain_linear_x m))

(cl:ensure-generic-function 'lowpass_gain_angular_z-val :lambda-list '(m))
(cl:defmethod lowpass_gain_angular_z-val ((m <ConfigTwistFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:lowpass_gain_angular_z-val is deprecated.  Use autoware_config_msgs-msg:lowpass_gain_angular_z instead.")
  (lowpass_gain_angular_z m))

(cl:ensure-generic-function 'lowpass_gain_steering_angle-val :lambda-list '(m))
(cl:defmethod lowpass_gain_steering_angle-val ((m <ConfigTwistFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:lowpass_gain_steering_angle-val is deprecated.  Use autoware_config_msgs-msg:lowpass_gain_steering_angle instead.")
  (lowpass_gain_steering_angle m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ConfigTwistFilter>) ostream)
  "Serializes a message object of type '<ConfigTwistFilter>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'use_decision_maker) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'lateral_accel_limit))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'lateral_jerk_limit))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'lowpass_gain_linear_x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'lowpass_gain_angular_z))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'lowpass_gain_steering_angle))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ConfigTwistFilter>) istream)
  "Deserializes a message object of type '<ConfigTwistFilter>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:slot-value msg 'use_decision_maker) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'lateral_accel_limit) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'lateral_jerk_limit) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'lowpass_gain_linear_x) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'lowpass_gain_angular_z) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'lowpass_gain_steering_angle) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ConfigTwistFilter>)))
  "Returns string type for a message object of type '<ConfigTwistFilter>"
  "autoware_config_msgs/ConfigTwistFilter")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ConfigTwistFilter)))
  "Returns string type for a message object of type 'ConfigTwistFilter"
  "autoware_config_msgs/ConfigTwistFilter")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ConfigTwistFilter>)))
  "Returns md5sum for a message object of type '<ConfigTwistFilter>"
  "b8c0018f21cee202f52983a44f92d9c3")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ConfigTwistFilter)))
  "Returns md5sum for a message object of type 'ConfigTwistFilter"
  "b8c0018f21cee202f52983a44f92d9c3")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ConfigTwistFilter>)))
  "Returns full string definition for message of type '<ConfigTwistFilter>"
  (cl:format cl:nil "Header header~%bool use_decision_maker~%float32 lateral_accel_limit~%float32 lateral_jerk_limit~%float32 lowpass_gain_linear_x~%float32 lowpass_gain_angular_z~%float32 lowpass_gain_steering_angle~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ConfigTwistFilter)))
  "Returns full string definition for message of type 'ConfigTwistFilter"
  (cl:format cl:nil "Header header~%bool use_decision_maker~%float32 lateral_accel_limit~%float32 lateral_jerk_limit~%float32 lowpass_gain_linear_x~%float32 lowpass_gain_angular_z~%float32 lowpass_gain_steering_angle~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ConfigTwistFilter>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     1
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ConfigTwistFilter>))
  "Converts a ROS message object to a list"
  (cl:list 'ConfigTwistFilter
    (cl:cons ':header (header msg))
    (cl:cons ':use_decision_maker (use_decision_maker msg))
    (cl:cons ':lateral_accel_limit (lateral_accel_limit msg))
    (cl:cons ':lateral_jerk_limit (lateral_jerk_limit msg))
    (cl:cons ':lowpass_gain_linear_x (lowpass_gain_linear_x msg))
    (cl:cons ':lowpass_gain_angular_z (lowpass_gain_angular_z msg))
    (cl:cons ':lowpass_gain_steering_angle (lowpass_gain_steering_angle msg))
))
