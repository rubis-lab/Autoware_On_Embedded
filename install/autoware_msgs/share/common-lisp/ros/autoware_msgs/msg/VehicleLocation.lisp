; Auto-generated. Do not edit!


(cl:in-package autoware_msgs-msg)


;//! \htmlinclude VehicleLocation.msg.html

(cl:defclass <VehicleLocation> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (lane_array_id
    :reader lane_array_id
    :initarg :lane_array_id
    :type cl:integer
    :initform 0)
   (waypoint_index
    :reader waypoint_index
    :initarg :waypoint_index
    :type cl:integer
    :initform 0))
)

(cl:defclass VehicleLocation (<VehicleLocation>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <VehicleLocation>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'VehicleLocation)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name autoware_msgs-msg:<VehicleLocation> is deprecated: use autoware_msgs-msg:VehicleLocation instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <VehicleLocation>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:header-val is deprecated.  Use autoware_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'lane_array_id-val :lambda-list '(m))
(cl:defmethod lane_array_id-val ((m <VehicleLocation>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:lane_array_id-val is deprecated.  Use autoware_msgs-msg:lane_array_id instead.")
  (lane_array_id m))

(cl:ensure-generic-function 'waypoint_index-val :lambda-list '(m))
(cl:defmethod waypoint_index-val ((m <VehicleLocation>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:waypoint_index-val is deprecated.  Use autoware_msgs-msg:waypoint_index instead.")
  (waypoint_index m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <VehicleLocation>) ostream)
  "Serializes a message object of type '<VehicleLocation>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let* ((signed (cl:slot-value msg 'lane_array_id)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'waypoint_index)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <VehicleLocation>) istream)
  "Deserializes a message object of type '<VehicleLocation>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'lane_array_id) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'waypoint_index) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<VehicleLocation>)))
  "Returns string type for a message object of type '<VehicleLocation>"
  "autoware_msgs/VehicleLocation")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'VehicleLocation)))
  "Returns string type for a message object of type 'VehicleLocation"
  "autoware_msgs/VehicleLocation")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<VehicleLocation>)))
  "Returns md5sum for a message object of type '<VehicleLocation>"
  "cba3770fc8eb8557ac8c63f4c0d3155b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'VehicleLocation)))
  "Returns md5sum for a message object of type 'VehicleLocation"
  "cba3770fc8eb8557ac8c63f4c0d3155b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<VehicleLocation>)))
  "Returns full string definition for message of type '<VehicleLocation>"
  (cl:format cl:nil "Header header~%int32 lane_array_id~%int32 waypoint_index~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'VehicleLocation)))
  "Returns full string definition for message of type 'VehicleLocation"
  (cl:format cl:nil "Header header~%int32 lane_array_id~%int32 waypoint_index~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <VehicleLocation>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <VehicleLocation>))
  "Converts a ROS message object to a list"
  (cl:list 'VehicleLocation
    (cl:cons ':header (header msg))
    (cl:cons ':lane_array_id (lane_array_id msg))
    (cl:cons ':waypoint_index (waypoint_index msg))
))
