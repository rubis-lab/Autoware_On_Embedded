;; Auto-generated. Do not edit!


(when (boundp 'autoware_config_msgs::ConfigWaypointReplanner)
  (if (not (find-package "AUTOWARE_CONFIG_MSGS"))
    (make-package "AUTOWARE_CONFIG_MSGS"))
  (shadow 'ConfigWaypointReplanner (find-package "AUTOWARE_CONFIG_MSGS")))
(unless (find-package "AUTOWARE_CONFIG_MSGS::CONFIGWAYPOINTREPLANNER")
  (make-package "AUTOWARE_CONFIG_MSGS::CONFIGWAYPOINTREPLANNER"))

(in-package "ROS")
;;//! \htmlinclude ConfigWaypointReplanner.msg.html


(defclass autoware_config_msgs::ConfigWaypointReplanner
  :super ros::object
  :slots (_multi_lane_csv _replanning_mode _use_decision_maker _velocity_max _velocity_min _accel_limit _decel_limit _radius_thresh _radius_min _resample_mode _resample_interval _velocity_offset _end_point_offset _braking_distance _replan_curve_mode _replan_endpoint_mode _overwrite_vmax_mode _realtime_tuning_mode ))

(defmethod autoware_config_msgs::ConfigWaypointReplanner
  (:init
   (&key
    ((:multi_lane_csv __multi_lane_csv) "")
    ((:replanning_mode __replanning_mode) nil)
    ((:use_decision_maker __use_decision_maker) nil)
    ((:velocity_max __velocity_max) 0.0)
    ((:velocity_min __velocity_min) 0.0)
    ((:accel_limit __accel_limit) 0.0)
    ((:decel_limit __decel_limit) 0.0)
    ((:radius_thresh __radius_thresh) 0.0)
    ((:radius_min __radius_min) 0.0)
    ((:resample_mode __resample_mode) nil)
    ((:resample_interval __resample_interval) 0.0)
    ((:velocity_offset __velocity_offset) 0)
    ((:end_point_offset __end_point_offset) 0)
    ((:braking_distance __braking_distance) 0)
    ((:replan_curve_mode __replan_curve_mode) nil)
    ((:replan_endpoint_mode __replan_endpoint_mode) nil)
    ((:overwrite_vmax_mode __overwrite_vmax_mode) nil)
    ((:realtime_tuning_mode __realtime_tuning_mode) nil)
    )
   (send-super :init)
   (setq _multi_lane_csv (string __multi_lane_csv))
   (setq _replanning_mode __replanning_mode)
   (setq _use_decision_maker __use_decision_maker)
   (setq _velocity_max (float __velocity_max))
   (setq _velocity_min (float __velocity_min))
   (setq _accel_limit (float __accel_limit))
   (setq _decel_limit (float __decel_limit))
   (setq _radius_thresh (float __radius_thresh))
   (setq _radius_min (float __radius_min))
   (setq _resample_mode __resample_mode)
   (setq _resample_interval (float __resample_interval))
   (setq _velocity_offset (round __velocity_offset))
   (setq _end_point_offset (round __end_point_offset))
   (setq _braking_distance (round __braking_distance))
   (setq _replan_curve_mode __replan_curve_mode)
   (setq _replan_endpoint_mode __replan_endpoint_mode)
   (setq _overwrite_vmax_mode __overwrite_vmax_mode)
   (setq _realtime_tuning_mode __realtime_tuning_mode)
   self)
  (:multi_lane_csv
   (&optional __multi_lane_csv)
   (if __multi_lane_csv (setq _multi_lane_csv __multi_lane_csv)) _multi_lane_csv)
  (:replanning_mode
   (&optional __replanning_mode)
   (if __replanning_mode (setq _replanning_mode __replanning_mode)) _replanning_mode)
  (:use_decision_maker
   (&optional __use_decision_maker)
   (if __use_decision_maker (setq _use_decision_maker __use_decision_maker)) _use_decision_maker)
  (:velocity_max
   (&optional __velocity_max)
   (if __velocity_max (setq _velocity_max __velocity_max)) _velocity_max)
  (:velocity_min
   (&optional __velocity_min)
   (if __velocity_min (setq _velocity_min __velocity_min)) _velocity_min)
  (:accel_limit
   (&optional __accel_limit)
   (if __accel_limit (setq _accel_limit __accel_limit)) _accel_limit)
  (:decel_limit
   (&optional __decel_limit)
   (if __decel_limit (setq _decel_limit __decel_limit)) _decel_limit)
  (:radius_thresh
   (&optional __radius_thresh)
   (if __radius_thresh (setq _radius_thresh __radius_thresh)) _radius_thresh)
  (:radius_min
   (&optional __radius_min)
   (if __radius_min (setq _radius_min __radius_min)) _radius_min)
  (:resample_mode
   (&optional __resample_mode)
   (if __resample_mode (setq _resample_mode __resample_mode)) _resample_mode)
  (:resample_interval
   (&optional __resample_interval)
   (if __resample_interval (setq _resample_interval __resample_interval)) _resample_interval)
  (:velocity_offset
   (&optional __velocity_offset)
   (if __velocity_offset (setq _velocity_offset __velocity_offset)) _velocity_offset)
  (:end_point_offset
   (&optional __end_point_offset)
   (if __end_point_offset (setq _end_point_offset __end_point_offset)) _end_point_offset)
  (:braking_distance
   (&optional __braking_distance)
   (if __braking_distance (setq _braking_distance __braking_distance)) _braking_distance)
  (:replan_curve_mode
   (&optional __replan_curve_mode)
   (if __replan_curve_mode (setq _replan_curve_mode __replan_curve_mode)) _replan_curve_mode)
  (:replan_endpoint_mode
   (&optional __replan_endpoint_mode)
   (if __replan_endpoint_mode (setq _replan_endpoint_mode __replan_endpoint_mode)) _replan_endpoint_mode)
  (:overwrite_vmax_mode
   (&optional __overwrite_vmax_mode)
   (if __overwrite_vmax_mode (setq _overwrite_vmax_mode __overwrite_vmax_mode)) _overwrite_vmax_mode)
  (:realtime_tuning_mode
   (&optional __realtime_tuning_mode)
   (if __realtime_tuning_mode (setq _realtime_tuning_mode __realtime_tuning_mode)) _realtime_tuning_mode)
  (:serialization-length
   ()
   (+
    ;; string _multi_lane_csv
    4 (length _multi_lane_csv)
    ;; bool _replanning_mode
    1
    ;; bool _use_decision_maker
    1
    ;; float32 _velocity_max
    4
    ;; float32 _velocity_min
    4
    ;; float32 _accel_limit
    4
    ;; float32 _decel_limit
    4
    ;; float32 _radius_thresh
    4
    ;; float32 _radius_min
    4
    ;; bool _resample_mode
    1
    ;; float32 _resample_interval
    4
    ;; int32 _velocity_offset
    4
    ;; int32 _end_point_offset
    4
    ;; int32 _braking_distance
    4
    ;; bool _replan_curve_mode
    1
    ;; bool _replan_endpoint_mode
    1
    ;; bool _overwrite_vmax_mode
    1
    ;; bool _realtime_tuning_mode
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _multi_lane_csv
       (write-long (length _multi_lane_csv) s) (princ _multi_lane_csv s)
     ;; bool _replanning_mode
       (if _replanning_mode (write-byte -1 s) (write-byte 0 s))
     ;; bool _use_decision_maker
       (if _use_decision_maker (write-byte -1 s) (write-byte 0 s))
     ;; float32 _velocity_max
       (sys::poke _velocity_max (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _velocity_min
       (sys::poke _velocity_min (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _accel_limit
       (sys::poke _accel_limit (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _decel_limit
       (sys::poke _decel_limit (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _radius_thresh
       (sys::poke _radius_thresh (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _radius_min
       (sys::poke _radius_min (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; bool _resample_mode
       (if _resample_mode (write-byte -1 s) (write-byte 0 s))
     ;; float32 _resample_interval
       (sys::poke _resample_interval (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; int32 _velocity_offset
       (write-long _velocity_offset s)
     ;; int32 _end_point_offset
       (write-long _end_point_offset s)
     ;; int32 _braking_distance
       (write-long _braking_distance s)
     ;; bool _replan_curve_mode
       (if _replan_curve_mode (write-byte -1 s) (write-byte 0 s))
     ;; bool _replan_endpoint_mode
       (if _replan_endpoint_mode (write-byte -1 s) (write-byte 0 s))
     ;; bool _overwrite_vmax_mode
       (if _overwrite_vmax_mode (write-byte -1 s) (write-byte 0 s))
     ;; bool _realtime_tuning_mode
       (if _realtime_tuning_mode (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _multi_lane_csv
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _multi_lane_csv (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; bool _replanning_mode
     (setq _replanning_mode (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _use_decision_maker
     (setq _use_decision_maker (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float32 _velocity_max
     (setq _velocity_max (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _velocity_min
     (setq _velocity_min (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _accel_limit
     (setq _accel_limit (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _decel_limit
     (setq _decel_limit (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _radius_thresh
     (setq _radius_thresh (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _radius_min
     (setq _radius_min (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; bool _resample_mode
     (setq _resample_mode (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float32 _resample_interval
     (setq _resample_interval (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; int32 _velocity_offset
     (setq _velocity_offset (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _end_point_offset
     (setq _end_point_offset (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _braking_distance
     (setq _braking_distance (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; bool _replan_curve_mode
     (setq _replan_curve_mode (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _replan_endpoint_mode
     (setq _replan_endpoint_mode (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _overwrite_vmax_mode
     (setq _overwrite_vmax_mode (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _realtime_tuning_mode
     (setq _realtime_tuning_mode (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(setf (get autoware_config_msgs::ConfigWaypointReplanner :md5sum-) "377fcfc666c7fdd690ae1373651c353b")
(setf (get autoware_config_msgs::ConfigWaypointReplanner :datatype-) "autoware_config_msgs/ConfigWaypointReplanner")
(setf (get autoware_config_msgs::ConfigWaypointReplanner :definition-)
      "string multi_lane_csv
bool replanning_mode
bool use_decision_maker
float32 velocity_max
float32 velocity_min
float32 accel_limit
float32 decel_limit
float32 radius_thresh
float32 radius_min
bool resample_mode
float32 resample_interval
int32 velocity_offset
int32 end_point_offset
int32 braking_distance
bool replan_curve_mode
bool replan_endpoint_mode
bool overwrite_vmax_mode
bool realtime_tuning_mode

")



(provide :autoware_config_msgs/ConfigWaypointReplanner "377fcfc666c7fdd690ae1373651c353b")


