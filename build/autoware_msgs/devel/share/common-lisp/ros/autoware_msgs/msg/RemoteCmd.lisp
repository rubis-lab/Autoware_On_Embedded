; Auto-generated. Do not edit!


(cl:in-package autoware_msgs-msg)


;//! \htmlinclude RemoteCmd.msg.html

(cl:defclass <RemoteCmd> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (vehicle_cmd
    :reader vehicle_cmd
    :initarg :vehicle_cmd
    :type autoware_msgs-msg:VehicleCmd
    :initform (cl:make-instance 'autoware_msgs-msg:VehicleCmd))
   (control_mode
    :reader control_mode
    :initarg :control_mode
    :type cl:integer
    :initform 0))
)

(cl:defclass RemoteCmd (<RemoteCmd>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RemoteCmd>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RemoteCmd)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name autoware_msgs-msg:<RemoteCmd> is deprecated: use autoware_msgs-msg:RemoteCmd instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <RemoteCmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:header-val is deprecated.  Use autoware_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'vehicle_cmd-val :lambda-list '(m))
(cl:defmethod vehicle_cmd-val ((m <RemoteCmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:vehicle_cmd-val is deprecated.  Use autoware_msgs-msg:vehicle_cmd instead.")
  (vehicle_cmd m))

(cl:ensure-generic-function 'control_mode-val :lambda-list '(m))
(cl:defmethod control_mode-val ((m <RemoteCmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:control_mode-val is deprecated.  Use autoware_msgs-msg:control_mode instead.")
  (control_mode m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RemoteCmd>) ostream)
  "Serializes a message object of type '<RemoteCmd>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'vehicle_cmd) ostream)
  (cl:let* ((signed (cl:slot-value msg 'control_mode)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RemoteCmd>) istream)
  "Deserializes a message object of type '<RemoteCmd>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'vehicle_cmd) istream)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'control_mode) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RemoteCmd>)))
  "Returns string type for a message object of type '<RemoteCmd>"
  "autoware_msgs/RemoteCmd")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RemoteCmd)))
  "Returns string type for a message object of type 'RemoteCmd"
  "autoware_msgs/RemoteCmd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RemoteCmd>)))
  "Returns md5sum for a message object of type '<RemoteCmd>"
  "696e11e670c7366dad4c426f849c2368")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RemoteCmd)))
  "Returns md5sum for a message object of type 'RemoteCmd"
  "696e11e670c7366dad4c426f849c2368")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RemoteCmd>)))
  "Returns full string definition for message of type '<RemoteCmd>"
  (cl:format cl:nil "Header header~%autoware_msgs/VehicleCmd vehicle_cmd~%int32 control_mode~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: autoware_msgs/VehicleCmd~%Header header~%autoware_msgs/SteerCmd steer_cmd~%autoware_msgs/AccelCmd accel_cmd~%autoware_msgs/BrakeCmd brake_cmd~%autoware_msgs/LampCmd lamp_cmd~%autoware_msgs/Gear gear_cmd~%int32 mode~%geometry_msgs/TwistStamped twist_cmd~%autoware_msgs/ControlCommand ctrl_cmd~%int32 emergency~%~%================================================================================~%MSG: autoware_msgs/SteerCmd~%Header header~%int32 steer~%~%================================================================================~%MSG: autoware_msgs/AccelCmd~%Header header~%int32 accel~%~%================================================================================~%MSG: autoware_msgs/BrakeCmd~%Header header~%int32 brake~%~%================================================================================~%MSG: autoware_msgs/LampCmd~%Header header~%int32 l~%int32 r~%~%================================================================================~%MSG: autoware_msgs/Gear~%uint8 NONE=0~%uint8 PARK=1~%uint8 REVERSE=2~%uint8 NEUTRAL=3~%uint8 DRIVE=4~%uint8 LOW=5~%uint8 gear~%================================================================================~%MSG: geometry_msgs/TwistStamped~%# A twist with reference coordinate frame and timestamp~%Header header~%Twist twist~%~%================================================================================~%MSG: geometry_msgs/Twist~%# This expresses velocity in free space broken into its linear and angular parts.~%Vector3  linear~%Vector3  angular~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: autoware_msgs/ControlCommand~%float64 linear_velocity~%float64 linear_acceleration #m/s^2~%float64 steering_angle~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RemoteCmd)))
  "Returns full string definition for message of type 'RemoteCmd"
  (cl:format cl:nil "Header header~%autoware_msgs/VehicleCmd vehicle_cmd~%int32 control_mode~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: autoware_msgs/VehicleCmd~%Header header~%autoware_msgs/SteerCmd steer_cmd~%autoware_msgs/AccelCmd accel_cmd~%autoware_msgs/BrakeCmd brake_cmd~%autoware_msgs/LampCmd lamp_cmd~%autoware_msgs/Gear gear_cmd~%int32 mode~%geometry_msgs/TwistStamped twist_cmd~%autoware_msgs/ControlCommand ctrl_cmd~%int32 emergency~%~%================================================================================~%MSG: autoware_msgs/SteerCmd~%Header header~%int32 steer~%~%================================================================================~%MSG: autoware_msgs/AccelCmd~%Header header~%int32 accel~%~%================================================================================~%MSG: autoware_msgs/BrakeCmd~%Header header~%int32 brake~%~%================================================================================~%MSG: autoware_msgs/LampCmd~%Header header~%int32 l~%int32 r~%~%================================================================================~%MSG: autoware_msgs/Gear~%uint8 NONE=0~%uint8 PARK=1~%uint8 REVERSE=2~%uint8 NEUTRAL=3~%uint8 DRIVE=4~%uint8 LOW=5~%uint8 gear~%================================================================================~%MSG: geometry_msgs/TwistStamped~%# A twist with reference coordinate frame and timestamp~%Header header~%Twist twist~%~%================================================================================~%MSG: geometry_msgs/Twist~%# This expresses velocity in free space broken into its linear and angular parts.~%Vector3  linear~%Vector3  angular~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: autoware_msgs/ControlCommand~%float64 linear_velocity~%float64 linear_acceleration #m/s^2~%float64 steering_angle~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RemoteCmd>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'vehicle_cmd))
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RemoteCmd>))
  "Converts a ROS message object to a list"
  (cl:list 'RemoteCmd
    (cl:cons ':header (header msg))
    (cl:cons ':vehicle_cmd (vehicle_cmd msg))
    (cl:cons ':control_mode (control_mode msg))
))
