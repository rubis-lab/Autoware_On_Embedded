; Auto-generated. Do not edit!


(cl:in-package autoware_config_msgs-msg)


;//! \htmlinclude ConfigDecisionMaker.msg.html

(cl:defclass <ConfigDecisionMaker> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (auto_mission_reload
    :reader auto_mission_reload
    :initarg :auto_mission_reload
    :type cl:boolean
    :initform cl:nil)
   (auto_engage
    :reader auto_engage
    :initarg :auto_engage
    :type cl:boolean
    :initform cl:nil)
   (auto_mission_change
    :reader auto_mission_change
    :initarg :auto_mission_change
    :type cl:boolean
    :initform cl:nil)
   (use_fms
    :reader use_fms
    :initarg :use_fms
    :type cl:boolean
    :initform cl:nil)
   (disuse_vector_map
    :reader disuse_vector_map
    :initarg :disuse_vector_map
    :type cl:boolean
    :initform cl:nil)
   (sim_mode
    :reader sim_mode
    :initarg :sim_mode
    :type cl:boolean
    :initform cl:nil)
   (insert_stop_line_wp
    :reader insert_stop_line_wp
    :initarg :insert_stop_line_wp
    :type cl:boolean
    :initform cl:nil)
   (num_of_steer_behind
    :reader num_of_steer_behind
    :initarg :num_of_steer_behind
    :type cl:integer
    :initform 0)
   (change_threshold_dist
    :reader change_threshold_dist
    :initarg :change_threshold_dist
    :type cl:float
    :initform 0.0)
   (change_threshold_angle
    :reader change_threshold_angle
    :initarg :change_threshold_angle
    :type cl:float
    :initform 0.0)
   (goal_threshold_dist
    :reader goal_threshold_dist
    :initarg :goal_threshold_dist
    :type cl:float
    :initform 0.0)
   (goal_threshold_vel
    :reader goal_threshold_vel
    :initarg :goal_threshold_vel
    :type cl:float
    :initform 0.0)
   (stopped_vel
    :reader stopped_vel
    :initarg :stopped_vel
    :type cl:float
    :initform 0.0))
)

(cl:defclass ConfigDecisionMaker (<ConfigDecisionMaker>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ConfigDecisionMaker>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ConfigDecisionMaker)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name autoware_config_msgs-msg:<ConfigDecisionMaker> is deprecated: use autoware_config_msgs-msg:ConfigDecisionMaker instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:header-val is deprecated.  Use autoware_config_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'auto_mission_reload-val :lambda-list '(m))
(cl:defmethod auto_mission_reload-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:auto_mission_reload-val is deprecated.  Use autoware_config_msgs-msg:auto_mission_reload instead.")
  (auto_mission_reload m))

(cl:ensure-generic-function 'auto_engage-val :lambda-list '(m))
(cl:defmethod auto_engage-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:auto_engage-val is deprecated.  Use autoware_config_msgs-msg:auto_engage instead.")
  (auto_engage m))

(cl:ensure-generic-function 'auto_mission_change-val :lambda-list '(m))
(cl:defmethod auto_mission_change-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:auto_mission_change-val is deprecated.  Use autoware_config_msgs-msg:auto_mission_change instead.")
  (auto_mission_change m))

(cl:ensure-generic-function 'use_fms-val :lambda-list '(m))
(cl:defmethod use_fms-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:use_fms-val is deprecated.  Use autoware_config_msgs-msg:use_fms instead.")
  (use_fms m))

(cl:ensure-generic-function 'disuse_vector_map-val :lambda-list '(m))
(cl:defmethod disuse_vector_map-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:disuse_vector_map-val is deprecated.  Use autoware_config_msgs-msg:disuse_vector_map instead.")
  (disuse_vector_map m))

(cl:ensure-generic-function 'sim_mode-val :lambda-list '(m))
(cl:defmethod sim_mode-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:sim_mode-val is deprecated.  Use autoware_config_msgs-msg:sim_mode instead.")
  (sim_mode m))

(cl:ensure-generic-function 'insert_stop_line_wp-val :lambda-list '(m))
(cl:defmethod insert_stop_line_wp-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:insert_stop_line_wp-val is deprecated.  Use autoware_config_msgs-msg:insert_stop_line_wp instead.")
  (insert_stop_line_wp m))

(cl:ensure-generic-function 'num_of_steer_behind-val :lambda-list '(m))
(cl:defmethod num_of_steer_behind-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:num_of_steer_behind-val is deprecated.  Use autoware_config_msgs-msg:num_of_steer_behind instead.")
  (num_of_steer_behind m))

(cl:ensure-generic-function 'change_threshold_dist-val :lambda-list '(m))
(cl:defmethod change_threshold_dist-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:change_threshold_dist-val is deprecated.  Use autoware_config_msgs-msg:change_threshold_dist instead.")
  (change_threshold_dist m))

(cl:ensure-generic-function 'change_threshold_angle-val :lambda-list '(m))
(cl:defmethod change_threshold_angle-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:change_threshold_angle-val is deprecated.  Use autoware_config_msgs-msg:change_threshold_angle instead.")
  (change_threshold_angle m))

(cl:ensure-generic-function 'goal_threshold_dist-val :lambda-list '(m))
(cl:defmethod goal_threshold_dist-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:goal_threshold_dist-val is deprecated.  Use autoware_config_msgs-msg:goal_threshold_dist instead.")
  (goal_threshold_dist m))

(cl:ensure-generic-function 'goal_threshold_vel-val :lambda-list '(m))
(cl:defmethod goal_threshold_vel-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:goal_threshold_vel-val is deprecated.  Use autoware_config_msgs-msg:goal_threshold_vel instead.")
  (goal_threshold_vel m))

(cl:ensure-generic-function 'stopped_vel-val :lambda-list '(m))
(cl:defmethod stopped_vel-val ((m <ConfigDecisionMaker>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader autoware_config_msgs-msg:stopped_vel-val is deprecated.  Use autoware_config_msgs-msg:stopped_vel instead.")
  (stopped_vel m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ConfigDecisionMaker>) ostream)
  "Serializes a message object of type '<ConfigDecisionMaker>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'auto_mission_reload) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'auto_engage) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'auto_mission_change) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'use_fms) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'disuse_vector_map) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'sim_mode) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'insert_stop_line_wp) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'num_of_steer_behind)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'num_of_steer_behind)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'num_of_steer_behind)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'num_of_steer_behind)) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'change_threshold_dist))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'change_threshold_angle))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'goal_threshold_dist))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'goal_threshold_vel))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'stopped_vel))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ConfigDecisionMaker>) istream)
  "Deserializes a message object of type '<ConfigDecisionMaker>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:slot-value msg 'auto_mission_reload) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'auto_engage) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'auto_mission_change) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'use_fms) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'disuse_vector_map) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'sim_mode) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'insert_stop_line_wp) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'num_of_steer_behind)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'num_of_steer_behind)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'num_of_steer_behind)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'num_of_steer_behind)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'change_threshold_dist) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'change_threshold_angle) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'goal_threshold_dist) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'goal_threshold_vel) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'stopped_vel) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ConfigDecisionMaker>)))
  "Returns string type for a message object of type '<ConfigDecisionMaker>"
  "autoware_config_msgs/ConfigDecisionMaker")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ConfigDecisionMaker)))
  "Returns string type for a message object of type 'ConfigDecisionMaker"
  "autoware_config_msgs/ConfigDecisionMaker")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ConfigDecisionMaker>)))
  "Returns md5sum for a message object of type '<ConfigDecisionMaker>"
  "35bae6da1772c1676984cb320dd14d09")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ConfigDecisionMaker)))
  "Returns md5sum for a message object of type 'ConfigDecisionMaker"
  "35bae6da1772c1676984cb320dd14d09")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ConfigDecisionMaker>)))
  "Returns full string definition for message of type '<ConfigDecisionMaker>"
  (cl:format cl:nil "Header header~%bool auto_mission_reload~%bool auto_engage~%bool auto_mission_change~%bool use_fms~%bool disuse_vector_map~%bool sim_mode~%bool insert_stop_line_wp~%uint32 num_of_steer_behind~%float64 change_threshold_dist~%float64 change_threshold_angle~%float64 goal_threshold_dist~%float64 goal_threshold_vel~%float64 stopped_vel~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ConfigDecisionMaker)))
  "Returns full string definition for message of type 'ConfigDecisionMaker"
  (cl:format cl:nil "Header header~%bool auto_mission_reload~%bool auto_engage~%bool auto_mission_change~%bool use_fms~%bool disuse_vector_map~%bool sim_mode~%bool insert_stop_line_wp~%uint32 num_of_steer_behind~%float64 change_threshold_dist~%float64 change_threshold_angle~%float64 goal_threshold_dist~%float64 goal_threshold_vel~%float64 stopped_vel~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ConfigDecisionMaker>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     1
     1
     1
     1
     1
     1
     1
     4
     8
     8
     8
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ConfigDecisionMaker>))
  "Converts a ROS message object to a list"
  (cl:list 'ConfigDecisionMaker
    (cl:cons ':header (header msg))
    (cl:cons ':auto_mission_reload (auto_mission_reload msg))
    (cl:cons ':auto_engage (auto_engage msg))
    (cl:cons ':auto_mission_change (auto_mission_change msg))
    (cl:cons ':use_fms (use_fms msg))
    (cl:cons ':disuse_vector_map (disuse_vector_map msg))
    (cl:cons ':sim_mode (sim_mode msg))
    (cl:cons ':insert_stop_line_wp (insert_stop_line_wp msg))
    (cl:cons ':num_of_steer_behind (num_of_steer_behind msg))
    (cl:cons ':change_threshold_dist (change_threshold_dist msg))
    (cl:cons ':change_threshold_angle (change_threshold_angle msg))
    (cl:cons ':goal_threshold_dist (goal_threshold_dist msg))
    (cl:cons ':goal_threshold_vel (goal_threshold_vel msg))
    (cl:cons ':stopped_vel (stopped_vel msg))
))
