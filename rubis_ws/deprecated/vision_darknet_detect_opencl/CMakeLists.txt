cmake_minimum_required(VERSION 2.8.12)
project(vision_darknet_detect_opencl)

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
  add_definitions(-D__aarch64__)  
endif()

find_package(catkin REQUIRED COMPONENTS
  autoware_config_msgs
  autoware_msgs
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  rubis_lib
)

find_package(OpenCV REQUIRED)
find_package(OpenMP)

catkin_package()

set(CMAKE_CXX_FLAGS "-O3 -g -Wall ${CMAKE_CXX_FLAGS}")

add_compile_definitions(ARM)
add_compile_definitions(GPU)
add_compile_definitions(GPU_FAST)

add_library(vision_darknet_detect_opencl_lib SHARED
  darknet/src/activation_kernels.c
  darknet/src/avgpool_layer_kernels.c
  darknet/src/convolutional_kernels.c
  darknet/src/crop_layer_kernels.c
  darknet/src/col2im_kernels.c
  darknet/src/blas_kernels.c
  darknet/src/deconvolutional_kernels.c
  darknet/src/dropout_layer_kernels.c
  darknet/src/im2col_kernels.c
  darknet/src/maxpool_layer_kernels.c
  darknet/src/activation_layer.c
  darknet/src/activations.c
  darknet/src/avgpool_layer.c
  darknet/src/batchnorm_layer.c
  darknet/src/blas.c
  darknet/src/box.c
  darknet/src/col2im.c
  darknet/src/compare.c
  darknet/src/connected_layer.c
  darknet/src/convolutional_layer.c
  darknet/src/cost_layer.c
  darknet/src/crnn_layer.c
  darknet/src/crop_layer.c
  darknet/src/data.c
  darknet/src/deconvolutional_layer.c
  darknet/src/demo.c
  darknet/src/detection_layer.c
  darknet/src/dropout_layer.c
  darknet/src/gemm.c
  darknet/src/gru_layer.c
  darknet/src/im2col.c
  darknet/src/image.c
  darknet/src/iseg_layer.c
  darknet/src/layer.c
  darknet/src/list.c
  darknet/src/local_layer.c
  darknet/src/lstm_layer.c
  darknet/src/matrix.c
  darknet/src/maxpool_layer.c
  darknet/src/network.c
  darknet/src/normalization_layer.c
  darknet/src/option_list.c
  darknet/src/parser.c
  darknet/src/region_layer.c
  darknet/src/reorg_layer.c
  darknet/src/rnn_layer.c
  darknet/src/route_layer.c
  darknet/src/shortcut_layer.c
  darknet/src/softmax_layer.c
  darknet/src/system.c
  darknet/src/tree.c
  darknet/src/utils.c
  darknet/src/yolo_layer.c
  darknet/src/yolo4_layer.c
  darknet/src/upsample_layer.c
  darknet/src/logistic_layer.c
  darknet/src/l2norm_layer.c
  darknet/src/opencl.c
)

target_compile_definitions(vision_darknet_detect_opencl_lib PUBLIC -DGPU -DOPENCL)

target_include_directories(vision_darknet_detect_opencl_lib PRIVATE
  ${OpenCV_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/darknet
  ${PROJECT_SOURCE_DIR}/darknet/src
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/CL
)

target_link_libraries(vision_darknet_detect_opencl_lib
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Qt5Core_LIBRARIES}
  pthread
  OpenCL
)

add_dependencies(vision_darknet_detect_opencl_lib
  ${catkin_EXPORTED_TARGETS}
)

#ros node
add_executable(vision_darknet_detect_opencl
  src/vision_darknet_detect_opencl_node.cpp
  src/vision_darknet_detect_opencl.cpp
  src/vision_darknet_detect_opencl.h
)

target_compile_definitions(vision_darknet_detect_opencl PUBLIC -DGPU -DOPENCL)

target_include_directories(vision_darknet_detect_opencl PRIVATE
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/darknet
  ${PROJECT_SOURCE_DIR}/darknet/src
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/CL
)

target_link_libraries(vision_darknet_detect_opencl
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}  
  vision_darknet_detect_opencl_lib
)

add_dependencies(vision_darknet_detect_opencl
  ${catkin_EXPORTED_TARGETS}
)
install(
  TARGETS vision_darknet_detect_opencl_lib vision_darknet_detect_opencl
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY darknet/cfg/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/darknet/cfg/
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY darknet/data/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/darknet/data/
  PATTERN ".svn" EXCLUDE
)
