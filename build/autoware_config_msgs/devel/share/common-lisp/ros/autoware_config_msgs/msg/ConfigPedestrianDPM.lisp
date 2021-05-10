; Auto-generated. Do not edit!


(cl:in-package autoware_config_msgs-msg)


;//! \htmlinclude ConfigPedestrianDPM.msg.html

(cl:defclass <ConfigPedestrianDPM> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (score_threshold
    :reader score_threshold
    :initarg :score_threshold
    :type cl:float
    :initform 0.0)
   (group_threshold
    :reader group_threshold
    :initarg :group_threshold
    :type cl:float
    :initform 0.0)
   (Lambda
    :reader Lambda
    :initarg :Lambda
    :type cl:integer
    :initform 0)
   (num_cells
    :reader num_cells
    :initarg :num_cells
    :type cl:integer
    :initform 0)
   (num_bins
    :reader num_bins
    :initarg :num_bins
    :type cl:integer
    :initform 0))
)

(cl:defclass ConfigPedestrianDPM (<ConfigPedestrianDPM>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ConfigPedestrianDPM>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ConfigPedestrianDPM)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name autoware_config_msgs-msg:<ConfigPedestrianDPM> is deprecated: use autoware_config_msgs-msg:ConfigPedestrianDPM instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <ConfigPedestrianDPM>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:header-val is deprecated.  Use autoware_config_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'score_threshold-val :lambda-list '(m))
(cl:defmethod score_threshold-val ((m <ConfigPedestrianDPM>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:score_threshold-val is deprecated.  Use autoware_config_msgs-msg:score_threshold instead.")
  (score_threshold m))

(cl:ensure-generic-function 'group_threshold-val :lambda-list '(m))
(cl:defmethod group_threshold-val ((m <ConfigPedestrianDPM>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:group_threshold-val is deprecated.  Use autoware_config_msgs-msg:group_threshold instead.")
  (group_threshold m))

(cl:ensure-generic-function 'Lambda-val :lambda-list '(m))
(cl:defmethod Lambda-val ((m <ConfigPedestrianDPM>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:Lambda-val is deprecated.  Use autoware_config_msgs-msg:Lambda instead.")
  (Lambda m))

(cl:ensure-generic-function 'num_cells-val :lambda-list '(m))
(cl:defmethod num_cells-val ((m <ConfigPedestrianDPM>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:num_cells-val is deprecated.  Use autoware_config_msgs-msg:num_cells instead.")
  (num_cells m))

(cl:ensure-generic-function 'num_bins-val :lambda-list '(m))
(cl:defmethod num_bins-val ((m <ConfigPedestrianDPM>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:num_bins-val is deprecated.  Use autoware_config_msgs-msg:num_bins instead.")
  (num_bins m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ConfigPedestrianDPM>) ostream)
  "Serializes a message object of type '<ConfigPedestrianDPM>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'score_threshold))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'group_threshold))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let* ((signed (cl:slot-value msg 'Lambda)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'num_cells)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'num_bins)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ConfigPedestrianDPM>) istream)
  "Deserializes a message object of type '<ConfigPedestrianDPM>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'score_threshold) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'group_threshold) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'Lambda) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'num_cells) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'num_bins) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ConfigPedestrianDPM>)))
  "Returns string type for a message object of type '<ConfigPedestrianDPM>"
  "autoware_config_msgs/ConfigPedestrianDPM")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ConfigPedestrianDPM)))
  "Returns string type for a message object of type 'ConfigPedestrianDPM"
  "autoware_config_msgs/ConfigPedestrianDPM")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ConfigPedestrianDPM>)))
  "Returns md5sum for a message object of type '<ConfigPedestrianDPM>"
  "4aad5f13bbefe1a8707af2b040e45167")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ConfigPedestrianDPM)))
  "Returns md5sum for a message object of type 'ConfigPedestrianDPM"
  "4aad5f13bbefe1a8707af2b040e45167")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ConfigPedestrianDPM>)))
  "Returns full string definition for message of type '<ConfigPedestrianDPM>"
  (cl:format cl:nil "Header header~%float32 score_threshold~%float32 group_threshold~%int32 Lambda~%int32 num_cells~%int32 num_bins~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ConfigPedestrianDPM)))
  "Returns full string definition for message of type 'ConfigPedestrianDPM"
  (cl:format cl:nil "Header header~%float32 score_threshold~%float32 group_threshold~%int32 Lambda~%int32 num_cells~%int32 num_bins~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ConfigPedestrianDPM>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ConfigPedestrianDPM>))
  "Converts a ROS message object to a list"
  (cl:list 'ConfigPedestrianDPM
    (cl:cons ':header (header msg))
    (cl:cons ':score_threshold (score_threshold msg))
    (cl:cons ':group_threshold (group_threshold msg))
    (cl:cons ':Lambda (Lambda msg))
    (cl:cons ':num_cells (num_cells msg))
    (cl:cons ':num_bins (num_bins msg))
))
