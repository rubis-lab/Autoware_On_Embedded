;; Auto-generated. Do not edit!


(when (boundp 'autoware_msgs::RecognizeLightState)
  (if (not (find-package "AUTOWARE_MSGS"))
    (make-package "AUTOWARE_MSGS"))
  (shadow 'RecognizeLightState (find-package "AUTOWARE_MSGS")))
(unless (find-package "AUTOWARE_MSGS::RECOGNIZELIGHTSTATE")
  (make-package "AUTOWARE_MSGS::RECOGNIZELIGHTSTATE"))
(unless (find-package "AUTOWARE_MSGS::RECOGNIZELIGHTSTATEREQUEST")
  (make-package "AUTOWARE_MSGS::RECOGNIZELIGHTSTATEREQUEST"))
(unless (find-package "AUTOWARE_MSGS::RECOGNIZELIGHTSTATERESPONSE")
  (make-package "AUTOWARE_MSGS::RECOGNIZELIGHTSTATERESPONSE"))

(in-package "ROS")

(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))




(defclass autoware_msgs::RecognizeLightStateRequest
  :super ros::object
  :slots (_roi_image ))

(defmethod autoware_msgs::RecognizeLightStateRequest
  (:init
   (&key
    ((:roi_image __roi_image) (instance sensor_msgs::Image :init))
    )
   (send-super :init)
   (setq _roi_image __roi_image)
   self)
  (:roi_image
   (&rest __roi_image)
   (if (keywordp (car __roi_image))
       (send* _roi_image __roi_image)
     (progn
       (if __roi_image (setq _roi_image (car __roi_image)))
       _roi_image)))
  (:serialization-length
   ()
   (+
    ;; sensor_msgs/Image _roi_image
    (send _roi_image :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; sensor_msgs/Image _roi_image
       (send _roi_image :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; sensor_msgs/Image _roi_image
     (send _roi_image :deserialize buf ptr-) (incf ptr- (send _roi_image :serialization-length))
   ;;
   self)
  )

(defclass autoware_msgs::RecognizeLightStateResponse
  :super ros::object
  :slots (_class_id _confidence ))

(defmethod autoware_msgs::RecognizeLightStateResponse
  (:init
   (&key
    ((:class_id __class_id) 0)
    ((:confidence __confidence) 0.0)
    )
   (send-super :init)
   (setq _class_id (round __class_id))
   (setq _confidence (float __confidence))
   self)
  (:class_id
   (&optional __class_id)
   (if __class_id (setq _class_id __class_id)) _class_id)
  (:confidence
   (&optional __confidence)
   (if __confidence (setq _confidence __confidence)) _confidence)
  (:serialization-length
   ()
   (+
    ;; uint8 _class_id
    1
    ;; float32 _confidence
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _class_id
       (write-byte _class_id s)
     ;; float32 _confidence
       (sys::poke _confidence (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _class_id
     (setq _class_id (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; float32 _confidence
     (setq _confidence (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass autoware_msgs::RecognizeLightState
  :super ros::object
  :slots ())

(setf (get autoware_msgs::RecognizeLightState :md5sum-) "15dc773b45f6bf3e4beeded009f5873b")
(setf (get autoware_msgs::RecognizeLightState :datatype-) "autoware_msgs/RecognizeLightState")
(setf (get autoware_msgs::RecognizeLightState :request) autoware_msgs::RecognizeLightStateRequest)
(setf (get autoware_msgs::RecognizeLightState :response) autoware_msgs::RecognizeLightStateResponse)

(defmethod autoware_msgs::RecognizeLightStateRequest
  (:response () (instance autoware_msgs::RecognizeLightStateResponse :init)))

(setf (get autoware_msgs::RecognizeLightStateRequest :md5sum-) "15dc773b45f6bf3e4beeded009f5873b")
(setf (get autoware_msgs::RecognizeLightStateRequest :datatype-) "autoware_msgs/RecognizeLightStateRequest")
(setf (get autoware_msgs::RecognizeLightStateRequest :definition-)
      "sensor_msgs/Image roi_image

================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id
---
uint8 class_id
float32 confidence

")

(setf (get autoware_msgs::RecognizeLightStateResponse :md5sum-) "15dc773b45f6bf3e4beeded009f5873b")
(setf (get autoware_msgs::RecognizeLightStateResponse :datatype-) "autoware_msgs/RecognizeLightStateResponse")
(setf (get autoware_msgs::RecognizeLightStateResponse :definition-)
      "sensor_msgs/Image roi_image

================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id
---
uint8 class_id
float32 confidence

")



(provide :autoware_msgs/RecognizeLightState "15dc773b45f6bf3e4beeded009f5873b")


