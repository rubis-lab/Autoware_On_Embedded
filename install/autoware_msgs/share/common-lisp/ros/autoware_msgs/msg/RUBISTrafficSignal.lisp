; Auto-generated. Do not edit!


(cl:in-package autoware_msgs-msg)


;//! \htmlinclude RUBISTrafficSignal.msg.html

(cl:defclass <RUBISTrafficSignal> (roslisp-msg-protocol:ros-message)
  ((id
    :reader id
    :initarg :id
    :type cl:integer
    :initform 0)
   (type
    :reader type
    :initarg :type
    :type cl:integer
    :initform 0)
   (time
    :reader time
    :initarg :time
    :type cl:float
    :initform 0.0))
)

(cl:defclass RUBISTrafficSignal (<RUBISTrafficSignal>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RUBISTrafficSignal>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RUBISTrafficSignal)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name autoware_msgs-msg:<RUBISTrafficSignal> is deprecated: use autoware_msgs-msg:RUBISTrafficSignal instead.")))

(cl:ensure-generic-function 'id-val :lambda-list '(m))
(cl:defmethod id-val ((m <RUBISTrafficSignal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:id-val is deprecated.  Use autoware_msgs-msg:id instead.")
  (id m))

(cl:ensure-generic-function 'type-val :lambda-list '(m))
(cl:defmethod type-val ((m <RUBISTrafficSignal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:type-val is deprecated.  Use autoware_msgs-msg:type instead.")
  (type m))

(cl:ensure-generic-function 'time-val :lambda-list '(m))
(cl:defmethod time-val ((m <RUBISTrafficSignal>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:time-val is deprecated.  Use autoware_msgs-msg:time instead.")
  (time m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RUBISTrafficSignal>) ostream)
  "Serializes a message object of type '<RUBISTrafficSignal>"
  (cl:let* ((signed (cl:slot-value msg 'id)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'type)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'time))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RUBISTrafficSignal>) istream)
  "Deserializes a message object of type '<RUBISTrafficSignal>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'id) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'type) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'time) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RUBISTrafficSignal>)))
  "Returns string type for a message object of type '<RUBISTrafficSignal>"
  "autoware_msgs/RUBISTrafficSignal")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RUBISTrafficSignal)))
  "Returns string type for a message object of type 'RUBISTrafficSignal"
  "autoware_msgs/RUBISTrafficSignal")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RUBISTrafficSignal>)))
  "Returns md5sum for a message object of type '<RUBISTrafficSignal>"
  "9019b7aea1c6b00cf12cfe69ccddab8f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RUBISTrafficSignal)))
  "Returns md5sum for a message object of type 'RUBISTrafficSignal"
  "9019b7aea1c6b00cf12cfe69ccddab8f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RUBISTrafficSignal>)))
  "Returns full string definition for message of type '<RUBISTrafficSignal>"
  (cl:format cl:nil "int32 id~%int32 type~%float32 time~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RUBISTrafficSignal)))
  "Returns full string definition for message of type 'RUBISTrafficSignal"
  (cl:format cl:nil "int32 id~%int32 type~%float32 time~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RUBISTrafficSignal>))
  (cl:+ 0
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RUBISTrafficSignal>))
  "Converts a ROS message object to a list"
  (cl:list 'RUBISTrafficSignal
    (cl:cons ':id (id msg))
    (cl:cons ':type (type msg))
    (cl:cons ':time (time msg))
))
