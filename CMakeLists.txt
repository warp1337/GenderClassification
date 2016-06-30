cmake_minimum_required(VERSION 2.8.4)
project( gender_classification )
find_package( OpenCV REQUIRED )


find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY )
add_definitions(-DENABLE_ASSERTS)

include(../dlib-18.18/dlib/cmake)
add_executable( gender_classification face_detection_and_crop.cpp FaceProcessing.cpp classification.cpp)
target_link_libraries( gender_classification ${OpenCV_LIBS} )
target_link_libraries( gender_classification dlib )
target_link_libraries( gender_classification ${Caffe_LIBRARIES})

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
