LIBS += -pthread
CONFIG += link_pkgconfig
PKGCONFIG += x11

QT += widgets

LIBS += -lboost_system -lglog

INCLUDEPATH += /home/sylar/dlib-18.18
LIBS += -L/home/sylar/dlib-18.18

INCLUDEPATH += /usr/include/flycapture
LIBS += -L/usr/lib
LIBS += -lflycapture

INCLUDEPATH += /home/sylar/caffe/include  /home/sylar/caffe/cmake_build/src /home/sylar/caffe/cmake_build/include
LIBS += -L/home/sylar/caffe/cmake_build/lib
LIBS += -lcaffe


INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /usr/local/cuda/include
LIBS += -L/usr/local/lib -L/usr/local/cuda-7.5/lib64 -lopencv_contrib -lopencv_core -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_video -lopencv_videostab -lcufft -lcublas -lnpps -lnppi -lnppc -lcudart -lrt -lpthread -lm -ldl


SOURCES += \
    classification.cpp \
    face_detection_and_crop.cpp \
    FaceProcessing.cpp \
    ../../dlib-18.18/dlib/all/source.cpp \
    mainwindow.cpp

HEADERS += \
    classification.h \
    FaceProcessing.h \
    mainwindow.h \
    FlyCaptureWrapper.h

FORMS += \
    mainwindow.ui
