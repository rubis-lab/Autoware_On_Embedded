
(cl:in-package :asdf)

(defsystem "autoware_msgs-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "RecognizeLightState" :depends-on ("_package_RecognizeLightState"))
    (:file "_package_RecognizeLightState" :depends-on ("_package"))
  ))