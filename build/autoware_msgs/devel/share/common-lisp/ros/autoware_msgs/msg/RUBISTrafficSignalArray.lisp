; Auto-generated. Do not edit!


(cl:in-package autoware_msgs-msg)


;//! \htmlinclude RUBISTrafficSignalArray.msg.html

(cl:defclass <RUBISTrafficSignalArray> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (signals
    :reader signals
    :initarg :signals
    :type (cl:vector autoware_msgs-msg:RUBISTrafficSignal)
   :initform (cl:make-array 0 :element-type 'autoware_msgs-msg:RUBISTrafficSignal :initial-element (cl:make-instance 'autoware_msgs-msg:RUBISTrafficSignal))))
)

(cl:defclass RUBISTrafficSignalArray (<RUBISTrafficSignalArray>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RUBISTrafficSignalArray>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RUBISTrafficSignalArray)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name autoware_msgs-msg:<RUBISTrafficSignalArray> is deprecated: use autoware_msgs-msg:RUBISTrafficSignalArray instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <RUBISTrafficSignalArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:header-val is deprecated.  Use autoware_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'signals-val :lambda-list '(m))
(cl:defmethod signals-val ((m <RUBISTrafficSignalArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-msg:signals-val is deprecated.  Use autoware_msgs-msg:signals instead.")
  (signals m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RUBISTrafficSignalArray>) ostream)
  "Serializes a message object of type '<RUBISTrafficSignalArray>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'signals))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'signals))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RUBISTrafficSignalArray>) istream)
  "Deserializes a message object of type '<RUBISTrafficSignalArray>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'signals) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'signals)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'autoware_msgs-msg:RUBISTrafficSignal))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RUBISTrafficSignalArray>)))
  "Returns string type for a message object of type '<RUBISTrafficSignalArray>"
  "autoware_msgs/RUBISTrafficSignalArray")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RUBISTrafficSignalArray)))
  "Returns string type for a message object of type 'RUBISTrafficSignalArray"
  "autoware_msgs/RUBISTrafficSignalArray")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RUBISTrafficSignalArray>)))
  "Returns md5sum for a message object of type '<RUBISTrafficSignalArray>"
  "8ae2769f49d9241a71af08054e4cc568")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RUBISTrafficSignalArray)))
  "Returns md5sum for a message object of type 'RUBISTrafficSignalArray"
  "8ae2769f49d9241a71af08054e4cc568")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RUBISTrafficSignalArray>)))
  "Returns full string definition for message of type '<RUBISTrafficSignalArray>"
  (cl:format cl:nil "Header header~%RUBISTrafficSignal[] signals~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: autoware_msgs/RUBISTrafficSignal~%int32 id~%int32 type~%float32 time~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RUBISTrafficSignalArray)))
  "Returns full string definition for message of type 'RUBISTrafficSignalArray"
  (cl:format cl:nil "Header header~%RUBISTrafficSignal[] signals~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: autoware_msgs/RUBISTrafficSignal~%int32 id~%int32 type~%float32 time~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RUBISTrafficSignalArray>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'signals) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RUBISTrafficSignalArray>))
  "Converts a ROS message object to a list"
  (cl:list 'RUBISTrafficSignalArray
    (cl:cons ':header (header msg))
    (cl:cons ':signals (signals msg))
))
