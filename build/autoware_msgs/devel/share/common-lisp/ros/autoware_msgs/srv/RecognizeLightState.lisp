; Auto-generated. Do not edit!


(cl:in-package autoware_msgs-srv)


;//! \htmlinclude RecognizeLightState-request.msg.html

(cl:defclass <RecognizeLightState-request> (roslisp-msg-protocol:ros-message)
  ((roi_image
    :reader roi_image
    :initarg :roi_image
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image)))
)

(cl:defclass RecognizeLightState-request (<RecognizeLightState-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RecognizeLightState-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RecognizeLightState-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name autoware_msgs-srv:<RecognizeLightState-request> is deprecated: use autoware_msgs-srv:RecognizeLightState-request instead.")))

(cl:ensure-generic-function 'roi_image-val :lambda-list '(m))
(cl:defmethod roi_image-val ((m <RecognizeLightState-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-srv:roi_image-val is deprecated.  Use autoware_msgs-srv:roi_image instead.")
  (roi_image m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RecognizeLightState-request>) ostream)
  "Serializes a message object of type '<RecognizeLightState-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'roi_image) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RecognizeLightState-request>) istream)
  "Deserializes a message object of type '<RecognizeLightState-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'roi_image) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RecognizeLightState-request>)))
  "Returns string type for a service object of type '<RecognizeLightState-request>"
  "autoware_msgs/RecognizeLightStateRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RecognizeLightState-request)))
  "Returns string type for a service object of type 'RecognizeLightState-request"
  "autoware_msgs/RecognizeLightStateRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RecognizeLightState-request>)))
  "Returns md5sum for a message object of type '<RecognizeLightState-request>"
  "15dc773b45f6bf3e4beeded009f5873b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RecognizeLightState-request)))
  "Returns md5sum for a message object of type 'RecognizeLightState-request"
  "15dc773b45f6bf3e4beeded009f5873b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RecognizeLightState-request>)))
  "Returns full string definition for message of type '<RecognizeLightState-request>"
  (cl:format cl:nil "sensor_msgs/Image roi_image~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RecognizeLightState-request)))
  "Returns full string definition for message of type 'RecognizeLightState-request"
  (cl:format cl:nil "sensor_msgs/Image roi_image~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RecognizeLightState-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'roi_image))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RecognizeLightState-request>))
  "Converts a ROS message object to a list"
  (cl:list 'RecognizeLightState-request
    (cl:cons ':roi_image (roi_image msg))
))
;//! \htmlinclude RecognizeLightState-response.msg.html

(cl:defclass <RecognizeLightState-response> (roslisp-msg-protocol:ros-message)
  ((class_id
    :reader class_id
    :initarg :class_id
    :type cl:fixnum
    :initform 0)
   (confidence
    :reader confidence
    :initarg :confidence
    :type cl:float
    :initform 0.0))
)

(cl:defclass RecognizeLightState-response (<RecognizeLightState-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RecognizeLightState-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RecognizeLightState-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name autoware_msgs-srv:<RecognizeLightState-response> is deprecated: use autoware_msgs-srv:RecognizeLightState-response instead.")))

(cl:ensure-generic-function 'class_id-val :lambda-list '(m))
(cl:defmethod class_id-val ((m <RecognizeLightState-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-srv:class_id-val is deprecated.  Use autoware_msgs-srv:class_id instead.")
  (class_id m))

(cl:ensure-generic-function 'confidence-val :lambda-list '(m))
(cl:defmethod confidence-val ((m <RecognizeLightState-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_msgs-srv:confidence-val is deprecated.  Use autoware_msgs-srv:confidence instead.")
  (confidence m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RecognizeLightState-response>) ostream)
  "Serializes a message object of type '<RecognizeLightState-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'class_id)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'confidence))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RecognizeLightState-response>) istream)
  "Deserializes a message object of type '<RecognizeLightState-response>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'class_id)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'confidence) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RecognizeLightState-response>)))
  "Returns string type for a service object of type '<RecognizeLightState-response>"
  "autoware_msgs/RecognizeLightStateResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RecognizeLightState-response)))
  "Returns string type for a service object of type 'RecognizeLightState-response"
  "autoware_msgs/RecognizeLightStateResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RecognizeLightState-response>)))
  "Returns md5sum for a message object of type '<RecognizeLightState-response>"
  "15dc773b45f6bf3e4beeded009f5873b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RecognizeLightState-response)))
  "Returns md5sum for a message object of type 'RecognizeLightState-response"
  "15dc773b45f6bf3e4beeded009f5873b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RecognizeLightState-response>)))
  "Returns full string definition for message of type '<RecognizeLightState-response>"
  (cl:format cl:nil "uint8 class_id~%float32 confidence~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RecognizeLightState-response)))
  "Returns full string definition for message of type 'RecognizeLightState-response"
  (cl:format cl:nil "uint8 class_id~%float32 confidence~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RecognizeLightState-response>))
  (cl:+ 0
     1
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RecognizeLightState-response>))
  "Converts a ROS message object to a list"
  (cl:list 'RecognizeLightState-response
    (cl:cons ':class_id (class_id msg))
    (cl:cons ':confidence (confidence msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'RecognizeLightState)))
  'RecognizeLightState-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'RecognizeLightState)))
  'RecognizeLightState-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RecognizeLightState)))
  "Returns string type for a service object of type '<RecognizeLightState>"
  "autoware_msgs/RecognizeLightState")