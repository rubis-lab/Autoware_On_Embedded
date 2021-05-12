; Auto-generated. Do not edit!


(cl:in-package autoware_msgs-msg)


;//! \htmlinclude IntersectionCondition.msg.html

(cl:defclass <IntersectionCondition> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (intersectionID
    :reader intersectionID
    :initarg :intersectionID
    :type cl:integer
    :initform 0)
   (intersectionDistance
    :reader intersectionDistance
    :initarg :intersectionDistance
    :type cl:float
    :initform 0.0)
   (isIntersection
    :reader isIntersection
    :initarg :isIntersection
    :type cl:fixnum
    :initform 0)
   (riskyLeftTurn
    :reader riskyLeftTurn
    :initarg :riskyLeftTurn
    :type cl:fixnum
    :initform 0)
   (riskyRightTurn
    :reader riskyRightTurn
    :initarg :riskyRightTurn
    :type cl:fixnum
    :initform 0))
)

(cl:defclass IntersectionCondition (<IntersectionCondition>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <IntersectionCondition>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'IntersectionCondition)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name autoware_msgs-msg:<IntersectionCondition> is deprecated: use autoware_msgs-msg:IntersectionCondition instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <IntersectionCondition>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:header-val is deprecated.  Use autoware_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'intersectionID-val :lambda-list '(m))
(cl:defmethod intersectionID-val ((m <IntersectionCondition>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:intersectionID-val is deprecated.  Use autoware_msgs-msg:intersectionID instead.")
  (intersectionID m))

(cl:ensure-generic-function 'intersectionDistance-val :lambda-list '(m))
(cl:defmethod intersectionDistance-val ((m <IntersectionCondition>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:intersectionDistance-val is deprecated.  Use autoware_msgs-msg:intersectionDistance instead.")
  (intersectionDistance m))

(cl:ensure-generic-function 'isIntersection-val :lambda-list '(m))
(cl:defmethod isIntersection-val ((m <IntersectionCondition>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:isIntersection-val is deprecated.  Use autoware_msgs-msg:isIntersection instead.")
  (isIntersection m))

(cl:ensure-generic-function 'riskyLeftTurn-val :lambda-list '(m))
(cl:defmethod riskyLeftTurn-val ((m <IntersectionCondition>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:riskyLeftTurn-val is deprecated.  Use autoware_msgs-msg:riskyLeftTurn instead.")
  (riskyLeftTurn m))

(cl:ensure-generic-function 'riskyRightTurn-val :lambda-list '(m))
(cl:defmethod riskyRightTurn-val ((m <IntersectionCondition>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:riskyRightTurn-val is deprecated.  Use autoware_msgs-msg:riskyRightTurn instead.")
  (riskyRightTurn m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <IntersectionCondition>) ostream)
  "Serializes a message object of type '<IntersectionCondition>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let* ((signed (cl:slot-value msg 'intersectionID)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'intersectionDistance))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let* ((signed (cl:slot-value msg 'isIntersection)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'riskyLeftTurn)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'riskyRightTurn)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <IntersectionCondition>) istream)
  "Deserializes a message object of type '<IntersectionCondition>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'intersectionID) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'intersectionDistance) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'isIntersection) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'riskyLeftTurn) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'riskyRightTurn) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<IntersectionCondition>)))
  "Returns string type for a message object of type '<IntersectionCondition>"
  "autoware_msgs/IntersectionCondition")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'IntersectionCondition)))
  "Returns string type for a message object of type 'IntersectionCondition"
  "autoware_msgs/IntersectionCondition")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<IntersectionCondition>)))
  "Returns md5sum for a message object of type '<IntersectionCondition>"
  "ec2240e9bbead2818bb2ecd5340b0db8")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'IntersectionCondition)))
  "Returns md5sum for a message object of type 'IntersectionCondition"
  "ec2240e9bbead2818bb2ecd5340b0db8")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<IntersectionCondition>)))
  "Returns full string definition for message of type '<IntersectionCondition>"
  (cl:format cl:nil "Header header~%int32 intersectionID~%float32 intersectionDistance~%int8 isIntersection~%int8 riskyLeftTurn~%int8 riskyRightTurn~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'IntersectionCondition)))
  "Returns full string definition for message of type 'IntersectionCondition"
  (cl:format cl:nil "Header header~%int32 intersectionID~%float32 intersectionDistance~%int8 isIntersection~%int8 riskyLeftTurn~%int8 riskyRightTurn~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <IntersectionCondition>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     4
     1
     1
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <IntersectionCondition>))
  "Converts a ROS message object to a list"
  (cl:list 'IntersectionCondition
    (cl:cons ':header (header msg))
    (cl:cons ':intersectionID (intersectionID msg))
    (cl:cons ':intersectionDistance (intersectionDistance msg))
    (cl:cons ':isIntersection (isIntersection msg))
    (cl:cons ':riskyLeftTurn (riskyLeftTurn msg))
    (cl:cons ':riskyRightTurn (riskyRightTurn msg))
))
