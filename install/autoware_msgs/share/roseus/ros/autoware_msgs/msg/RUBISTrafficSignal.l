;; Auto-generated. Do not edit!


(when (boundp 'autoware_msgs::RUBISTrafficSignal)
  (if (not (find-package "AUTOWARE_MSGS"))
    (make-package "AUTOWARE_MSGS"))
  (shadow 'RUBISTrafficSignal (find-package "AUTOWARE_MSGS")))
(unless (find-package "AUTOWARE_MSGS::RUBISTRAFFICSIGNAL")
  (make-package "AUTOWARE_MSGS::RUBISTRAFFICSIGNAL"))

(in-package "ROS")
;;//! \htmlinclude RUBISTrafficSignal.msg.html


(defclass autoware_msgs::RUBISTrafficSignal
  :super ros::object
  :slots (_id _type _time ))

(defmethod autoware_msgs::RUBISTrafficSignal
  (:init
   (&key
    ((:id __id) 0)
    ((:type __type) 0)
    ((:time __time) 0.0)
    )
   (send-super :init)
   (setq _id (round __id))
   (setq _type (round __type))
   (setq _time (float __time))
   self)
  (:id
   (&optional __id)
   (if __id (setq _id __id)) _id)
  (:type
   (&optional __type)
   (if __type (setq _type __type)) _type)
  (:time
   (&optional __time)
   (if __time (setq _time __time)) _time)
  (:serialization-length
   ()
   (+
    ;; int32 _id
    4
    ;; int32 _type
    4
    ;; float32 _time
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _id
       (write-long _id s)
     ;; int32 _type
       (write-long _type s)
     ;; float32 _time
       (sys::poke _time (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _id
     (setq _id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _type
     (setq _type (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float32 _time
     (setq _time (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get autoware_msgs::RUBISTrafficSignal :md5sum-) "9019b7aea1c6b00cf12cfe69ccddab8f")
(setf (get autoware_msgs::RUBISTrafficSignal :datatype-) "autoware_msgs/RUBISTrafficSignal")
(setf (get autoware_msgs::RUBISTrafficSignal :definition-)
      "int32 id
int32 type
float32 time
")



(provide :autoware_msgs/RUBISTrafficSignal "9019b7aea1c6b00cf12cfe69ccddab8f")


