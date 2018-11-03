add_library(readparameter SHARED Common/parameterReader.cpp)

include_directories(/usr/include/freetype2)
add_library(chineseputtext SHARED  Common/OpencvText.cpp)
target_link_libraries(chineseputtext freetype)

aux_source_directory(myfUtils MyfUtils_SRC)
add_library( myfUtils
        SHARED
        ${MyfUtils_SRC})
target_link_libraries( myfUtils
        ${OpenCV_LIBS})


add_library(dlibfacedetection SHARED  dlibface/FaceDetection.cpp)
target_link_libraries(dlibfacedetection readparameter dlib::dlib ${OpenCV_LIBS})

add_library(dlibdeepfaceprocess SHARED dlibface/deepFaceProcess.cpp)
target_link_libraries(dlibdeepfaceprocess readparameter dlibfacedetection dlib::dlib ${OpenCV_LIBS})

find_package(Protobuf REQUIRED)
add_library(dlibdeepfacedescriptormanager SHARED dlibface/faceDescriptorManager.cpp dlibface/protoFaceEncode.pb.cc)
target_link_libraries(dlibdeepfacedescriptormanager myfUtils readparameter dlibfacedetection dlibdeepfaceprocess dlib::dlib ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES})

